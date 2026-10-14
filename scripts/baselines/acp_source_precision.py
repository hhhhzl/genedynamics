"""Independent ACP source episode split, numerical diagnosis, training and gated rollout."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from genedynamics.learning.baselines.acp.task_training import (PUBLIC_FIELDS,SourceDemonstrations,
    build_policy,history_features,model_configuration,pose)
from genedynamics.learning.baselines.acp.trainer import ACPTrainer,deterministic_runtime,_rng_state,_gradient_state
from genedynamics.solvers.single.acp import ACPSolver,spatial
from scripts.baselines.acp_gt_audit import collect_live
from scripts.baselines.acp_native import ROOT,_digest_file,_fingerprint,_seed

PREFIX=ROOT/"results/_development/p3_20261014/results/arm/keyed_insertion_recovery/qualification"


def _save(path,value):path.write_text(json.dumps(value,indent=2)+"\n")


def _quantiles(values):
    values=np.asarray(values)
    return {"mean":float(values.mean()),"p50":float(np.quantile(values,.5)),
            "p95":float(np.quantile(values,.95)),"max":float(values.max())}


def validation(policy,dataset,*,windows=48,repeats=2,prefix=4,seed=18590):
    device=next(policy.parameters()).device
    indices=np.unique(np.linspace(0,len(dataset)-1,min(windows,len(dataset))).astype(int))
    output=[]
    begun=time.perf_counter()
    policy.eval()
    for repeat in range(repeats):
        for start in range(0,len(indices),16):
            selected=indices[start:start+16]
            _seed(seed+repeat*100+start)
            with torch.inference_mode():
                action=policy.predict_action(dataset.batch(selected)["obs"])["sparse"].cpu().numpy()
            expected=dataset.actions[selected].numpy()
            for local,index in enumerate(selected):
                for step in range(prefix):
                    pred,gt=action[local,step],expected[local,step]
                    row={"window":int(index),"repeat":repeat,"waypoint":step,
                        "reference_translation_m":float(np.linalg.norm(pred[:3]-gt[:3])),
                        "virtual_translation_m":float(np.linalg.norm(pred[9:12]-gt[9:12])),
                        "stiffness_abs_error_N_per_m":float(abs(pred[18]-gt[18])),
                        "predicted_stiffness_N_per_m":float(pred[18]),"label_stiffness_N_per_m":float(gt[18])}
                    for name,offset in (("reference",0),("virtual",9)):
                        a=spatial.pose9_to_SE3(pred[offset:offset+9])[:3,:3]
                        b=spatial.pose9_to_SE3(gt[offset:offset+9])[:3,:3]
                        row[name+"_rotation_rad"]=float(Rotation.from_matrix(a.T@b).magnitude())
                    output.append(row)
    keys=("reference_translation_m","virtual_translation_m","reference_rotation_rad","virtual_rotation_rad","stiffness_abs_error_N_per_m")
    return {"windows":indices.tolist(),"sampling_repeats":repeats,"waypoint_prefix":prefix,
        "metrics":{key:_quantiles([row[key] for row in output]) for key in keys},"rows":output,
        "wall_seconds":time.perf_counter()-begun,
        "rss_max_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0}


def passed(result,gate):
    m=result["metrics"]
    checks={"virtual_translation":m["virtual_translation_m"]["p95"]<=gate["virtual_translation_p95_m"],
        "reference_translation":m["reference_translation_m"]["p95"]<=gate["reference_translation_p95_m"],
        "rotation":max(m["reference_rotation_rad"]["p95"],m["virtual_rotation_rad"]["p95"])<=gate["rotation_p95_rad"],
        "stiffness":m["stiffness_abs_error_N_per_m"]["p95"]<=gate["stiffness_abs_error_p95_N_per_m"]}
    return all(checks.values()),checks


def diagnosis(policy,dataset):
    actions=dataset.actions
    normalizer=policy.sparse_normalizer["action"]
    normalized=normalizer.normalize(actions)
    reconstructed=normalizer.unnormalize(normalized)
    params=normalizer.params_dict
    # Exact native pose9 round-trip distinguishes coordinate errors from learning.
    poses=spatial.pose9_to_SE3(actions.numpy()[...,9:18])
    roundtrip=spatial.SE3_to_pose9(poses)
    source_ranges=actions.amax((0,1))-actions.amin((0,1))
    return {"action_min":actions.amin((0,1)).tolist(),"action_max":actions.amax((0,1)).tolist(),
        "action_scale":params["scale"].tolist(),"action_offset":params["offset"].tolist(),
        "normalized_min":normalized.amin((0,1)).tolist(),"normalized_max":normalized.amax((0,1)).tolist(),
        "normalizer_roundtrip_max_abs":float((actions-reconstructed.cpu()).abs().max()),
        "virtual_pose9_roundtrip_max_abs":float(np.max(np.abs(roundtrip-actions.numpy()[...,9:18]))),
        "constant_translation_channels":[int(i) for i in (0,1,2,9,10,11) if source_ranges[i]<1e-4],
        "normalized_outside_clip_fraction":float((normalized.abs()>1.00001).float().mean()),
        "scheduler":dict(policy.noise_scheduler.config),
        "label_time_contract":"public[t] before executed ACP command[t]; 16 future commands relative to current TCP[t]; first 4 commands executed at 20ms",
        "conditioning":"8 public TCP/wrench/twist histories plus current public goal and elapsed seconds; episode split before normalizer fitting",
        "VT_semantics":"native force/k displacement retained; VT-z can differ substantially from reference at k=200 N/m"}


def rollout(policy,meta,config,output):
    from genedynamics.experiments.baselines import instantiate_task
    solver=ACPSolver(policy,meta,device=str(next(policy.parameters()).device),raw_sample_interval_s=.02,
        slow_down_factor=1.,execution_horizon=config["training"]["execution_horizon"])
    task=instantiate_task("T3",robot="panda",history_len=config["training"]["history_len"])
    rows=[]
    for seed in config["source"]["closed_loop_seeds"]:
        history=[task.reset((seed,),seed=seed)];solver.reset();_seed(seed)
        started=time.perf_counter();chunks=0
        while task.state.status[0]==0:
            try:
                current=history[-1]
                pred=solver.act({"sparse":history_features(history,config["training"]["history_len"]),
                    "current_tool_pose_world":pose(current["tcp_position"],current["tcp_rotation"])[:,None]})
                result=task.execute_acp_chunk(pred);chunks+=1
                if not result["steps"]:raise RuntimeError("learned ACP produced no physical progress")
                history.extend({key:step.observation[key].copy() for key in PUBLIC_FIELDS} for step in result["steps"])
            except (ValueError,RuntimeError,FloatingPointError) as error:
                task.reject("ACP source validation policy execution error: "+str(error));break
        replay=task.replay_episode(task.records[-1])
        rows.append({"seed":seed,"chunks":chunks,"wall_seconds":time.perf_counter()-started,
            "evaluation_so_far":task.evaluate(),"charged_exact_replay":replay})
        _save(output/"source_rollout_progress.json",rows)
    return {"rows":rows,"evaluation":task.evaluate(),"target_interactions":0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("collect","train","diagnose"))
    parser.add_argument("--config",type=Path,default=ROOT/"configs/arm/keyed_insertion_recovery/qualification/acp_source_precision.yaml")
    parser.add_argument("--output",type=Path,default=PREFIX/"acp_source_precision/level_state/seed_18500")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--warmstart",type=Path)
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text());output=args.output.resolve()
    if ROOT/"latex" in output.parents:parser.error("evidence belongs outside latex/")
    output.mkdir(parents=True,exist_ok=True)
    config_copy=output/"predeclared_config.yaml"
    if config_copy.exists() and yaml.safe_load(config_copy.read_text())!=config:raise ValueError("cannot change predeclared source protocol in an existing result leaf")
    if not config_copy.exists():config_copy.write_text(yaml.safe_dump(config,sort_keys=False))
    if args.action=="collect":
        if (output/"collection.json").exists():raise ValueError("source collection already exists")
        rows=[]
        for seed,mode in zip(config["source"]["collection_seeds"],config["source"]["collection_modes"]):
            result=collect_live(output/f"source_seed_{seed}",mode=mode,seed=seed)
            rows.append(result);_save(output/"collection_progress.json",rows)
        _save(output/"collection.json",{"rows":rows,"target_interactions":0,
            "all_attempts_retained":True,"split":config["source"]})
        print(json.dumps({"source_attempts":len(rows),"source_successes":sum(row["valid_source_demo"] for row in rows)}));return
    deterministic_runtime();torch.set_num_threads(1);_seed(config["training"]["seed"])
    data_root=ROOT/config["source"]["data_root"] if "data_root" in config["source"] else output
    train_paths=[PREFIX/f"acp_adapted/level_state/seed_18420/live_gt_mode_{mode}/episode.npz" for mode in (0,1)]
    train_paths += [data_root/f"source_seed_{config['source']['collection_seeds'][i]}/episode.npz" for i in config["source"]["train_collection_indices"]]
    val_paths=[data_root/f"source_seed_{config['source']['collection_seeds'][i]}/episode.npz" for i in config["source"]["validation_collection_indices"]]
    profile=config["training"]
    train=SourceDemonstrations(train_paths,history_len=profile["history_len"],horizon=profile["horizon"])
    val=SourceDemonstrations(val_paths,history_len=profile["history_len"],horizon=profile["horizon"])
    meta,cfg=model_configuration(profile["history_len"],profile["horizon"],profile["seed"])
    policy=build_policy(meta,profile["history_len"])
    policy.set_normalizer(train.normalizer(rotation_residual=True))
    warmstart=args.warmstart or (ROOT/profile["warmstart"] if "warmstart" in profile else None)
    if warmstart:
        initial=torch.load(warmstart,map_location="cpu",weights_only=True)
        policy.load_state_dict(initial["state_dicts"]["ema_model"],strict=True)
        # A source-only optimizer warmstart; refit statistics use only train episodes.
        policy.set_normalizer(train.normalizer(rotation_residual=True))
    if args.action=="diagnose":
        path=PREFIX/"acp_adapted/level_state/seed_18422/policy_ema.pt"
        payload=torch.load(path,map_location="cpu",weights_only=True);policy.load_state_dict(payload["state_dicts"]["ema_model"],strict=True)
        policy.to(args.device).eval()
        report={"old_checkpoint":str(path),"train_diagnosis":diagnosis(policy,train),"validation_diagnosis":diagnosis(policy,val),
            "held_out_source_validation":validation(policy,val,windows=48,repeats=2),"target_interactions":0}
        _save(output/"old_2000_ema_diagnosis.json",report);print(json.dumps(report["held_out_source_validation"]["metrics"],indent=2));return
    if (output/"training.json").exists():raise ValueError("training receipt exists; use a fresh predeclared result leaf")
    device=torch.device(args.device)
    if device.type=="cuda":
        if not torch.cuda.is_available():raise RuntimeError("CUDA requested without CUDA")
        torch.cuda.set_per_process_memory_fraction(.23,device)
    trainer=ACPTrainer(policy,cfg,device=args.device,total_steps=profile["max_updates"]+2,optimizer_mode="all_trainable")
    if args.resume:trainer.restore(args.resume)
    report={"status":"running","train_episodes":train.receipts,"validation_episodes":val.receipts,
        "source_episode_split":True,"normalizer_fit_on_train_only":True,"diagnosis":diagnosis(trainer.model,train),
        "target_interactions":0,"candidates":[],"training_losses":[],"native_core_unchanged":True}
    report["source_optimizer_warmstart"]=None if warmstart is None else {"path":str(warmstart),"sha256":_digest_file(warmstart),"optimizer_reset":True}
    sampler=profile.get("sampler","uniform")
    low=np.flatnonzero(train.actions[:,0,18].numpy()<1000)
    high=np.flatnonzero(train.actions[:,0,18].numpy()>=1000)
    report["sampler"]={"type":sampler,"low_stiffness_windows":len(low),"other_windows":len(high)}
    def sample_indices():
        if sampler=="uniform":return np.random.randint(len(train),size=profile["batch_size"])
        if sampler!="balanced_current_stiffness" or not len(low) or not len(high):raise ValueError("invalid declared source sampler")
        count=profile["batch_size"]//2
        return np.random.permutation(np.r_[np.random.choice(low,count),np.random.choice(high,profile["batch_size"]-count)])
    _save(output/"training_attempt.json",report)
    begun=time.perf_counter();selected=None
    while trainer.global_step<profile["max_updates"]:
        indices=sample_indices()
        trainer.data_state={"source_window_indices":indices.tolist(),"updates":trainer.global_step}
        loss=trainer.train_step(train.batch(indices));report["training_losses"].append(loss)
        if not np.isfinite(loss):raise FloatingPointError("nonfinite source diffusion loss")
        if trainer.global_step%100==0:
            print(json.dumps({"update":trainer.global_step,"loss100":float(np.mean(report["training_losses"][-100:])),"seconds":time.perf_counter()-begun}),flush=True)
        if trainer.global_step%profile["validation_every"]==0:
            # Validation RNG cannot change the training minibatch/noise stream.
            state=_rng_state()
            result=validation(trainer.ema_model,val,windows=profile["validation_windows_per_episode"]*len(val_paths),
                repeats=profile["sampling_repeats"],prefix=config["gate"]["waypoint_prefix"])
            from genedynamics.learning.baselines.acp.trainer import _restore_rng
            _restore_rng(state)
            gate,checks=passed(result,config["gate"])
            candidate={"update":trainer.global_step,"gate_passed":gate,"gate_checks":checks,"validation":result}
            report["candidates"].append(candidate)
            checkpoint=output/f"ema_update_{trainer.global_step}.pt"
            torch.save({"state_dicts":{"ema_model":trainer.ema_model.state_dict()},"shape_meta":meta,
                "history_len":profile["history_len"],"variant":"state_history_source_precision_adapted"},checkpoint)
            candidate["checkpoint_sha256"]=_digest_file(checkpoint)
            trainer.save(output/"training_continuation.pt")
            _save(output/"training_progress.json",report)
            print(json.dumps({"source_validation_update":trainer.global_step,"gate_passed":gate,"metrics":result["metrics"]}),flush=True)
            if gate:selected=checkpoint;break
        if time.perf_counter()-begun>profile["max_attempt_seconds"]:
            report["budget_termination"]=True;break
    trainer.save(output/"training_continuation.pt")
    # Explicit one-step random-minibatch continuation proof, then restore prefix.
    def resume_update():
        indices=sample_indices()
        trainer.data_state={"source_window_indices":indices.tolist(),"updates":trainer.global_step}
        loss=trainer.train_step(train.batch(indices))
        return loss,{"model":_fingerprint(trainer.model.state_dict()),"gradients":_fingerprint(_gradient_state(trainer.model)),
            "ema":_fingerprint(trainer.ema_model.state_dict()),"optimizer":_fingerprint(trainer.optimizer.state_dict()),
            "lr":_fingerprint(trainer.scheduler.state_dict()),"rng":_fingerprint(_rng_state()),"data":_fingerprint(trainer.data_state)}
    expected=resume_update();trainer.restore(output/"training_continuation.pt");actual=resume_update()
    if expected!=actual:raise AssertionError("source training complete continuation differs")
    trainer.restore(output/"training_continuation.pt")
    report["complete_resume_exact"]=True;report["resume_fingerprints"]=actual[1]
    if selected:
        frozen=torch.load(selected,map_location="cpu",weights_only=True)
        trainer.ema_model.load_state_dict(frozen["state_dicts"]["ema_model"],strict=True);trainer.ema_model.to(args.device).eval()
        report["source_closed_loop"]=rollout(trainer.ema_model,meta,config,output)
        report["selected_checkpoint"]=str(selected)
        report["status"]="source_validation_gate_passed_and_charged_source_closed_loop_completed"
    else:report["status"]="source_fit_gate_not_met_no_closed_loop_or_target_calls"
    report.update(updates=trainer.global_step,wall_seconds=time.perf_counter()-begun,
        rss_max_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0,
        source_sha256={str(path.relative_to(ROOT)):_digest_file(path) for path in (Path(__file__),args.config.resolve(),ROOT/"genedynamics/learning/baselines/acp/task_training.py")})
    _save(output/"training.json",report)
    print(json.dumps({"status":report["status"],"updates":trainer.global_step,"output":str(output)}))


if __name__=="__main__":main()
