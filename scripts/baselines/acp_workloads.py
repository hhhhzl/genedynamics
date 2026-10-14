"""Profile real native ACP and explicitly adapted source workloads, with charged physics.

Native uses the released RGB/7-kHz-force demonstration and author's EMA weights.
Adapted uses SHA-qualified public source windows and a trained state-history EMA;
its one-chunk Panda execution is a source compute workload, not a success trial.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
import yaml

from genedynamics.learning.baselines.acp.trainer import deterministic_runtime
from scripts.baselines.acp_native import ROOT, _batch, _digest_file, _seed

PREFIX=ROOT/"results/_development/p3_20261014/results"


def _sync(device):
    if device.type=="cuda":torch.cuda.synchronize(device)


def _memory(device):
    return {"rss_max_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0,
            "gpu_peak_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else 0}


def _start(device):
    _sync(device)
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def native(args,output,device):
    from genedynamics.learning.baselines.acp.data import build_native_dataset
    from genedynamics.solvers.single.acp.native import build_policy,load_author_payload
    config=yaml.safe_load(args.native_config.read_text())
    checkpoint=ROOT/config["native"]["checkpoint"]
    begun=time.perf_counter()
    payload=load_author_payload(checkpoint)
    cfg=payload["cfg"]
    policy,_=build_policy(cfg,checkpoint_weights=True)
    policy.load_state_dict(payload["state_dicts"]["ema_model"],strict=True)
    policy.to(device).eval()
    del payload
    gc.collect()
    dataset=build_native_dataset(cfg,ROOT/config["native"]["subset"],
        source_root=ROOT/config["source"]["checkout"],submodule_root=ROOT/config["source"]["encoder_submodule"])
    setup_seconds=time.perf_counter()-begun
    rows=[]
    for size in (1,2):
        started=_start(device)
        data_started=time.perf_counter()
        batch=_batch(dataset,list(range(size)))
        data_seconds=time.perf_counter()-data_started
        encoder_time=[]
        def before(module,inputs):
            _sync(device);encoder_time.append(time.perf_counter())
        def after(module,inputs,result):
            _sync(device);encoder_time[-1]=time.perf_counter()-encoder_time[-1]
        pre=policy.obs_encoder.register_forward_pre_hook(before)
        post=policy.obs_encoder.register_forward_hook(after)
        try:
            _seed(args.seed+size)
            inference_started=time.perf_counter()
            with torch.inference_mode():result=policy.predict_action(batch["obs"])["sparse"]
            _sync(device)
            inference_seconds=time.perf_counter()-inference_started
        finally:pre.remove();post.remove()
        np.savez_compressed(output/f"native_batch_{size}.npz",action=result.cpu().numpy())
        rows.append({"batch_size":size,"source_indices":list(range(size)),
            "data_decode_and_collate_seconds":data_seconds,"encoder_seconds":sum(encoder_time),
            "encoder_and_16ddim_seconds":inference_seconds,
            "pipeline_seconds":time.perf_counter()-started,
            "observation_shapes":{key:list(value.shape) for key,value in batch["obs"]["sparse"].items()},
            "action_shape":list(result.shape),"native_inference_steps":int(policy.num_inference_steps),
            "accounting":{"released_demonstration_windows_read":size,"policy_calls":1,
                          "encoder_calls":len(encoder_time),"denoiser_steps":16,"target_resets":0,"target_controls":0},
            **_memory(device)})
        print(json.dumps(rows[-1]),flush=True)
    return {"status":"actual_released_native_workloads_profiled","variant":"native_CLIP_forceViT_Mod1",
        "checkpoint_sha256":_digest_file(checkpoint),"checkpoint_bytes":checkpoint.stat().st_size,
        "released_dataset_subset":str(ROOT/config["native"]["subset"]),"setup_seconds":setup_seconds,
        "rows":rows,"limitations":"Selected real episode; policy compute only, no original hardware execution or success claim"}


def adapted(args,output,device):
    from genedynamics.learning.baselines.acp.task_training import SourceDemonstrations,build_policy,pose
    from genedynamics.solvers.single.acp import ACPSolver
    from genedynamics.experiments.baselines import instantiate_task
    checkpoint=args.adapted_checkpoint.resolve()
    qualification=json.loads(checkpoint.with_name("qualification.json").read_text())
    paths=[Path(row["path"]) for row in qualification["training"]["source_demonstrations"]]
    begun=time.perf_counter()
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    history_len=int(payload["history_len"])
    policy=build_policy(payload["shape_meta"],history_len)
    solver=ACPSolver(policy,payload["shape_meta"],device=str(device),raw_sample_interval_s=.02,
        slow_down_factor=1.,execution_horizon=4)
    solver.load_policy_state_dict(payload)
    dataset=SourceDemonstrations(paths,history_len=history_len,horizon=16)
    setup_seconds=time.perf_counter()-begun
    starts=[0]
    source_bases=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            source_bases.append(pose(data["public.tcp_position"][:1],data["public.tcp_rotation"][:1])[0])
            count=len(data["acp.reference_pose_world"])-16 if "acp.reference_pose_world" in data else len(data["public.reference_position"])-16
        starts.append(starts[-1]+count)
    rows=[]
    for size in (1,2):
        task=instantiate_task("T3",robot="panda",seed=args.seed+size,history_len=history_len)
        reset_started=time.perf_counter()
        public=task.reset(tuple(args.seed+size*10+index for index in range(size)),seed=args.seed+size)
        reset_seconds=time.perf_counter()-reset_started
        source_indices=starts[:size]
        started=_start(device)
        data_started=time.perf_counter()
        batch=dataset.batch(source_indices)
        data_seconds=time.perf_counter()-data_started
        current=pose(public["tcp_position"],public["tcp_rotation"])[:,None]
        _seed(args.seed+size)
        solver.reset()
        inference_started=time.perf_counter()
        prediction=solver.act({"sparse":batch["obs"]["sparse"],"current_tool_pose_world":current})
        _sync(device)
        inference_seconds=time.perf_counter()-inference_started
        execute_started=time.perf_counter()
        try:
            result=task.execute_acp_chunk(prediction)
            execution_error=None
            controls=result["executed_control_steps"]
        except (ValueError,RuntimeError,FloatingPointError) as error:
            execution_error=repr(error)
            controls=task.accounting["control_steps"]
            task.reject("adapted ACP source workload execution failure: "+str(error))
        execution_seconds=time.perf_counter()-execute_started
        # Active worlds remain visible as explicitly budget-terminated rows.
        if np.any(task.state.status==0):task.reject("source workload profile completed one declared four-waypoint chunk")
        np.savez_compressed(output/f"adapted_batch_{size}.npz",
            **{key:value.detach().cpu().numpy() if torch.is_tensor(value) else value for key,value in prediction.items()})
        row={"batch_size":size,"source_indices":source_indices,"data_collate_seconds":data_seconds,
            "encoder_and_16ddim_and_decode_seconds":inference_seconds,
            "cpu_reset_seconds":reset_seconds,"cpu_execution_seconds":execution_seconds,
            "pipeline_seconds":time.perf_counter()-started,
            "executed_control_rounds":controls,"execution_error":execution_error,
            "source_initial_pose_vs_live_reset_max_abs_error":float(np.max(np.abs(np.stack(source_bases[:size])-current[:,0]))),
            "accounting":{"source_windows_read":size,"policy_calls":1,"denoiser_steps":16,
                          "source_task":dict(task.accounting),"target_resets":0,"target_controls":0},
            "evaluation":task.evaluate(),**_memory(device)}
        rows.append(row)
        (output/"progress.json").write_text(json.dumps(rows,indent=2)+"\n")
        print(json.dumps(row),flush=True)
    return {"status":"actual_adapted_source_workloads_profiled","variant":"explicit_state_history_Mod1_adaptation",
        "checkpoint_sha256":_digest_file(checkpoint),"source_demonstrations":dataset.receipts,
        "setup_seconds":setup_seconds,"rows":rows,"target_interactions":0,
        "limitations":"Actual source windows drive frozen policy; commands execute on fresh nominal Panda source worlds. One-chunk budget rejections and physical failures are retained; no success-rate claim."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant",choices=("native","adapted"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--seed",type=int,default=18460)
    parser.add_argument("--native-config",type=Path,default=ROOT/"configs/arm/flip_up/qualification/acp_native.yaml")
    parser.add_argument("--adapted-checkpoint",type=Path,default=PREFIX/"arm/keyed_insertion_recovery/qualification/acp_adapted/level_state/seed_18422/policy_ema.pt")
    args=parser.parse_args()
    output=args.output.resolve()
    if ROOT/"latex" in output.parents:parser.error("reports belong outside latex/")
    output.mkdir(parents=True,exist_ok=False)
    deterministic_runtime();torch.set_num_threads(1)
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA workload requested without CUDA")
    report=native(args,output,device) if args.variant=="native" else adapted(args,output,device)
    report.update(device=str(device),seed=args.seed,source_sha256=_digest_file(Path(__file__)))
    (output/"workloads.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
