"""Qualify or train actual native ACP with released checkpoints and real demonstrations."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import traceback

import numpy as np
import torch
import yaml

from genedynamics.learning.baselines.acp.data import audit_native_labels, build_native_dataset, subset_store
from genedynamics.learning.baselines.acp.trainer import ACPTrainer, _rng_state, _gradient_state, deterministic_runtime
from genedynamics.solvers.single.acp.native import build_policy, load_author_payload

ROOT = Path(__file__).resolve().parents[2]


def _digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8*1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value):
    digest = hashlib.sha256()

    def update(item):
        if torch.is_tensor(item):
            digest.update(str(item.dtype).encode())
            digest.update(str(tuple(item.shape)).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            for element in item:
                update(element)
        else:
            digest.update(repr(item).encode())
    update(value)
    return digest.hexdigest()


def _seed(value):
    torch.manual_seed(value)
    np.random.seed(value)
    random.seed(value)


def _batch(dataset, indices):
    return torch.utils.data.default_collate([dataset[index] for index in indices])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("qualify", "train"))
    parser.add_argument("--config", type=Path, default=Path("configs/arm/flip_up/qualification/acp_native.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--optimizer-mode", choices=("upstream", "all_trainable"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = (args.output or Path(config["output"])).resolve()
    if (ROOT/"latex") == output or (ROOT/"latex") in output.parents:
        parser.error("native qualification evidence must be outside latex/")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / ("native_qualification.json" if args.action == "qualify" else "native_training.json")
    if report_path.exists():
        parser.error("qualification/training receipt exists; choose a fresh output directory")
    runtime_profile=deterministic_runtime()
    device = torch.device(args.device or config["qualification"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("native CUDA profile requested; CPU fallback is forbidden")
    torch.set_num_threads(1)
    seed = int(config["seed"])
    _seed(seed)
    checkpoint = ROOT / config["native"]["checkpoint"]
    dataset_path = ROOT / config["native"]["dataset"]
    source_root = ROOT / config["source"]["checkout"]
    submodule_root = ROOT / config["source"]["encoder_submodule"]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "config": config, "python": sys.version, "torch": str(torch.__version__),
              "device": str(device), "seed": seed, "stages": {}}
    report["runtime_profile"]=runtime_profile

    def record(name, value):
        report["stages"][name] = value
        (output/"native_progress.json").write_text(json.dumps(report,indent=2)+"\n")
        shown=value
        if isinstance(value,dict) and "omitted_trainable_parameters" in value:
            shown={**value,"omitted_trainable_parameters":len(value["omitted_trainable_parameters"])}
        print(json.dumps({"stage":name,"result":shown}),flush=True)

    try:
        asset_root = checkpoint.parents[2]
        manifest = json.loads((asset_root/"checkpoints.manifest.json").read_text())
        entry = next(item for item in manifest["members"] if (asset_root/item["path"]).resolve() == checkpoint.resolve())
        digest = _digest_file(checkpoint)
        if digest != entry["sha256"]:
            raise ValueError("actual author checkpoint SHA256 differs from download evidence")
        persistent = output/"author_checkpoint.ckpt"
        if not persistent.exists():
            try:
                os.link(checkpoint,persistent)
            except OSError:
                shutil.copy2(checkpoint,persistent)
        for name in ("checkpoints.manifest.json","flip_up.manifest.json"):
            shutil.copy2(asset_root/name,output/name)
        record("author_checkpoint",{"bytes":checkpoint.stat().st_size,"sha256":digest,
                                    "persistent_copy":str(persistent),"url":manifest["url"]})
        subset = ROOT/config["native"]["subset"]
        subset_report = subset_store(dataset_path,ROOT/config["native"]["inventory"],subset)
        record("released_demonstration_subset",subset_report)
        labels = audit_native_labels(subset,source_root=source_root)
        record("virtual_target_labels",labels)
        if any(item["source_port_max_error"] > 1e-10 for item in labels):
            raise AssertionError("native and ported real demonstration labels differ")
        payload = load_author_payload(checkpoint)
        cfg = payload["cfg"]
        policy,shape_meta = build_policy(cfg,checkpoint_weights=True)
        key = "ema_model" if config["native"]["use_ema"] else "model"
        policy.load_state_dict(payload["state_dicts"][key],strict=True)
        policy.to(device).eval()
        dataset = build_native_dataset(cfg,subset,source_root=source_root,submodule_root=submodule_root)
        if not len(dataset):
            raise ValueError("no actual native training sequences in the selected real episode")
        batch_size = args.batch_size or int(config["qualification"]["batch_size"])
        batch = _batch(dataset,list(range(batch_size)))
        record("native_dataset_chain",{"sequences":len(dataset),"micro_batch_size":batch_size,
            "obs_shapes":{key:list(value.shape) for key,value in batch["obs"]["sparse"].items()},
            "action_shape":list(batch["action"]["sparse"].shape),"native_shape_meta":shape_meta,
            "implementation":"author VirtualTargetDataset/SequenceSampler/raw_to_obs/raw_to_action19"})
        record("released_config_compatibility",{
            "policy":policy._native_checkpoint_compatibility,
            "dataset":dataset._native_checkpoint_compatibility,
            "active_legacy_features_are_rejected":True})
        if args.action == "qualify":
            upstream,_ = build_policy(cfg,upstream=True,checkpoint_weights=True,
                                      source_root=source_root,submodule_root=submodule_root)
            upstream.load_state_dict(payload["state_dicts"][key],strict=True)
            upstream.to(device).eval()
            if list(upstream.state_dict()) != list(policy.state_dict()):
                raise AssertionError("native checkpoint topology differs from original source")
            with torch.inference_mode():
                _seed(seed+1)
                expected = upstream.predict_action(batch["obs"])["sparse"]
                _seed(seed+1)
                actual = policy.predict_action(batch["obs"])["sparse"]
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            np.savez_compressed(output/"native_policy_outputs.npz",
                                upstream=expected.cpu().numpy(),ported=actual.cpu().numpy())
            record("actual_checkpoint_native_encoder_policy_parity",{
                "action_max_abs_error":float((actual-expected).abs().max().cpu()),
                "state_dict_keys":len(policy.state_dict()),"encoder":type(policy.obs_encoder).__name__,
                "vision_backbone":str(cfg.policy.obs_encoder.vision_encoder_cfg.model_name),
                "force_backbone":str(cfg.policy.obs_encoder.force_encoder_cfg.model_name),
                "weights":key,"native_inference_steps":int(policy.num_inference_steps)})
            upstream.train()
            policy.train()
            _seed(seed+2)
            expected_loss = upstream.compute_loss(batch,None)
            expected_loss.backward()
            selected = [name for name,_ in upstream.named_parameters()
                        if name in {"model_sparse.final_conv.1.weight", "obs_encoder.linear_projection.weight"}
                        or name.endswith("key_model_map.rgb_0.patch_embed.proj.weight")]
            reference_gradients = {name:dict(upstream.named_parameters())[name].grad.detach().cpu().clone()
                                   for name in selected}
            _seed(seed+2)
            actual_loss = policy.compute_loss(batch,None)
            actual_loss.backward()
            torch.testing.assert_close(actual_loss,expected_loss,rtol=1e-5,atol=1e-6)
            gradient_errors = {}
            for name,gradient in reference_gradients.items():
                own = dict(policy.named_parameters())[name].grad.detach().cpu()
                torch.testing.assert_close(own,gradient,rtol=1e-5,atol=1e-6)
                gradient_errors[name] = float((own-gradient).abs().max())
            record("actual_native_training_loss_and_gradients",{
                "loss":float(actual_loss.detach().cpu()),"loss_abs_error":float((actual_loss-expected_loss).abs().cpu()),
                "gradient_max_abs_errors":gradient_errors})
            policy.zero_grad(set_to_none=True)
            del upstream,expected_loss,actual_loss,reference_gradients,payload
            if device.type == "cuda": torch.cuda.empty_cache()
        # Start the continuation proof from the actual trained checkpoint. All train/EMA
        # weights and normalization statistics are preserved, and own new optimizer is explicit.
        optimizer_mode = args.optimizer_mode or config["native"]["optimizer_mode"]
        total_steps = int(config["qualification"]["total_schedule_steps"])
        trainer = ACPTrainer(policy,cfg,device=str(device),total_steps=total_steps,optimizer_mode=optimizer_mode)
        record("native_optimizer",{"mode":optimizer_mode,
             "omitted_trainable_parameters":trainer.omitted_trainable_parameters,
             "omission_is_in_pinned_author_source":optimizer_mode=="upstream",
             "new_optimizer_warmstart":True,"full_original_training":False})
        if args.action == "qualify":
            _seed(seed+3)
            prefix=[]
            for _ in range(int(config["qualification"]["continuation_prefix_steps"])):
                prefix.append(trainer.train_step(batch))
            trainer.data_state={"qualification_batch_indices":list(range(batch_size)),"cursor":trainer.global_step}
            continuation=output/"native_continuation.ckpt"
            trainer.save(continuation)
            expected_loss=trainer.train_step(batch)
            optimizer_reference=copy.deepcopy(trainer.optimizer.state_dict())
            expected={"model":_fingerprint(trainer.model.state_dict()),
                      "model_gradients":_fingerprint(_gradient_state(trainer.model)),
                      "ema":_fingerprint(trainer.ema_model.state_dict()),
                      "optimizer":_fingerprint(trainer.optimizer.state_dict()),
                      "scheduler":_fingerprint(trainer.scheduler.state_dict()),"rng":_fingerprint(_rng_state()),
                      "ema_optimization_step":trainer.ema.optimization_step}
            trainer.restore(continuation)
            actual_loss=trainer.train_step(batch)
            actual={"model":_fingerprint(trainer.model.state_dict()),
                    "model_gradients":_fingerprint(_gradient_state(trainer.model)),
                    "ema":_fingerprint(trainer.ema_model.state_dict()),
                    "optimizer":_fingerprint(trainer.optimizer.state_dict()),
                    "scheduler":_fingerprint(trainer.scheduler.state_dict()),"rng":_fingerprint(_rng_state()),
                    "ema_optimization_step":trainer.ema.optimization_step}
            optimizer_actual=trainer.optimizer.state_dict()
            optimizer_differences=[]
            if expected["optimizer"] != actual["optimizer"]:
                if _fingerprint(optimizer_reference["param_groups"]) != _fingerprint(optimizer_actual["param_groups"]):
                    optimizer_differences.append({"param_groups_continuous":optimizer_reference["param_groups"],
                                                  "param_groups_restored":optimizer_actual["param_groups"]})
                for parameter,state in optimizer_reference["state"].items():
                    for name,value in state.items():
                        own=optimizer_actual["state"][parameter][name]
                        if _fingerprint(value) != _fingerprint(own):
                            optimizer_differences.append({"parameter":parameter,"field":name,
                                "max_abs_error":float((value-own).abs().max().cpu()) if torch.is_tensor(value) else None})
            record("continuation_comparison",{"continuous_loss":expected_loss,"restored_loss":actual_loss,
                   "state_matches":{name:expected[name]==actual[name] for name in expected},
                   "continuous_state":expected,"restored_state":actual,
                   "optimizer_differences":optimizer_differences[:30]})
            if expected != actual or expected_loss != actual_loss:
                raise AssertionError("native ACP complete continuation differs from uninterrupted training")
            record("full_training_continuation",{"prefix_losses":prefix,"continuous_loss":expected_loss,
                 "restored_loss":actual_loss,"state_fingerprints":actual,"exact_match":True,
                 "optimizer_omitted_accumulated_gradients_restored":True,
                 "checkpoint_sha256":_digest_file(continuation),"checkpoint_bytes":continuation.stat().st_size,
                 "restored_data_state":trainer.data_state})
            report["status"]="native_assets_encoder_and_training_chain_qualified"
        else:
            if args.resume:
                trainer.restore(args.resume)
            losses=[]
            for _ in range(args.steps):
                state=trainer.data_state
                if "permutation" not in state or state["cursor"] >= len(state["permutation"]):
                    state.update(permutation=torch.randperm(len(dataset)).tolist(),cursor=0)
                    trainer.epoch+=1
                indices=state["permutation"][state["cursor"]:state["cursor"]+batch_size]
                state["cursor"]+=len(indices)
                losses.append(trainer.train_step(_batch(dataset,indices)))
            trainer.save(output/"trained_native.ckpt")
            record("training_updates",{"steps":args.steps,"losses":losses,"global_step":trainer.global_step,
                                         "optimizer_mode":optimizer_mode,"selected_episode_subset":True})
            report["status"]="native_training_updates_completed"
        report["limitations"]=["Selected real demonstration subset, not full300-epoch original training",
              "Original physical robot success-rate reproduction requires the author's hardware",
              "Cross-EBDM four-task adaptation and contribution comparisons are separate qualifications"]
        files=list((ROOT/"genedynamics/solvers/single/acp").rglob("*.py"))
        files+=list((ROOT/"genedynamics/learning/baselines/acp").rglob("*.py"))
        files+=list((ROOT/"scripts/baselines").glob("acp_native*.py"))
        files+=[args.config.resolve(),ROOT/"requirements/baselines/acp-native.txt",
                ROOT/"genedynamics/solvers/single/acp/SOURCE.json"]
        report["local_sha256"]={str(path.relative_to(ROOT)):_digest_file(path) for path in files}
        report["requirements_lock"]="native_requirements.lock.txt"
        import subprocess
        (output/"native_requirements.lock.txt").write_text(subprocess.check_output(
            [sys.executable,"-m","pip","freeze","--all"],text=True))
        report_path.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"status":report["status"],"receipt":str(report_path)},indent=2))
    except BaseException as error:
        report.update(status="failed",error=repr(error))
        failures=output/"native_failures"
        failures.mkdir(exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        (failures/f"{stamp}.json").write_text(json.dumps(report,indent=2)+"\n")
        (failures/f"{stamp}.log").write_text(traceback.format_exc())
        (output/"native_qualification.failed.json").write_text(json.dumps(report,indent=2)+"\n")
        (output/"native_failure.log").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
