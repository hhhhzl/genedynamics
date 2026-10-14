"""Train an explicitly state-history ACP adaptation and charge every CPU target episode."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from genedynamics.learning.baselines.acp.task_training import (
    PUBLIC_FIELDS,SourceDemonstrations,build_policy,history_features,model_configuration,pose,
    sampling_profile,
)
from genedynamics.learning.baselines.acp.trainer import ACPTrainer,_rng_state,_gradient_state,deterministic_runtime
from genedynamics.solvers.single.acp import ACPSolver
from scripts.baselines.acp_native import _fingerprint
from genedynamics.experiments.framework.config import ExperimentConfig
from scripts.baselines import require_non_paper_output

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/arm/keyed_insertion_recovery/qualification/acp_adapted.yaml"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--development-root",type=Path,default=ROOT/"results/_development/p3_20261014")
    parser.add_argument("--steps",type=int)
    parser.add_argument("--seed",type=int,default=18420)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--demonstration",type=Path,action="append")
    parser.add_argument("--source-only",action="store_true")
    parser.add_argument("--rotation-residual",action="store_true")
    args=parser.parse_args()
    config=ExperimentConfig._load_yaml_dict(args.config.resolve(),seen=set())
    profile=config["acp"]
    route=ExperimentConfig(**{key:value for key,value in config.items() if key!="acp"})
    route.use_development_output_root(args.development_root)
    steps=args.steps or int(profile["training_steps"])
    if steps<1:parser.error("steps must be positive")
    output=require_non_paper_output(args.output or route.output_dir/"level_state"/f"seed_{args.seed}")
    output.mkdir(parents=True,exist_ok=False)
    runtime=deterministic_runtime()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    if torch.device(args.device).type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("explicit CUDA training requested without CUDA")
    demonstrations=args.demonstration or [route.output_dir.parent/"source_interface/level_nominal"/f"seed_{seed}/episode.npz" for seed in profile.get("source_demonstration_seeds",(18210,18211))]
    report={"status":"running","variant":"ACP-state-history-adapted","config":config,
            "seed":args.seed,"device":args.device,"runtime_profile":runtime,
            "permission":"C_source_robot_demonstrations","native_fft_or_human_demonstration_claim":False}
    (output/"attempt.json").write_text(json.dumps(report,indent=2)+"\n")
    started=time.monotonic()
    try:
        dataset=SourceDemonstrations(demonstrations,history_len=int(profile["history_len"]),horizon=int(profile["action_horizon"]))
        meta,cfg=model_configuration(int(profile["history_len"]),int(profile["action_horizon"]),args.seed)
        policy=build_policy(meta,int(profile["history_len"]))
        policy.set_normalizer(dataset.normalizer(rotation_residual=args.rotation_residual))
        trainer=ACPTrainer(policy,cfg,device=args.device,total_steps=steps+2,optimizer_mode="all_trainable")
        losses=[]
        def next_batch():
            indices=np.random.randint(0,len(dataset),size=int(profile["batch_size"]))
            trainer.data_state={"updates":trainer.global_step,"last_indices":indices.tolist(),"source_samples":len(dataset)}
            return dataset.batch(indices)
        for index in range(steps):
            losses.append(trainer.train_step(next_batch()))
            if not np.isfinite(losses[-1]):raise FloatingPointError("nonfinite real source demonstration diffusion loss")
            if (index+1)%25==0 or index==steps-1:
                progress={"stage":"source_training","updates":index+1,"last_loss":losses[-1],
                          "mean_recent_loss":float(np.mean(losses[-25:])),"wall_seconds":time.monotonic()-started}
                (output/"progress.json").write_text(json.dumps(progress,indent=2)+"\n")
                print(json.dumps(progress),flush=True)
        checkpoint=output/"training_continuation.pt"
        trainer.save(checkpoint)
        control_loss=trainer.train_step(next_batch())
        expected={"model":_fingerprint(trainer.model.state_dict()),"ema":_fingerprint(trainer.ema_model.state_dict()),
                  "model_gradients":_fingerprint(_gradient_state(trainer.model)),
                  "optimizer":_fingerprint(trainer.optimizer.state_dict()),"scheduler":_fingerprint(trainer.scheduler.state_dict()),
                  "rng":_fingerprint(_rng_state()),"data_state":_fingerprint(trainer.data_state)}
        trainer.restore(checkpoint)
        restored_loss=trainer.train_step(next_batch())
        actual={"model":_fingerprint(trainer.model.state_dict()),"ema":_fingerprint(trainer.ema_model.state_dict()),
                "model_gradients":_fingerprint(_gradient_state(trainer.model)),
                "optimizer":_fingerprint(trainer.optimizer.state_dict()),"scheduler":_fingerprint(trainer.scheduler.state_dict()),
                "rng":_fingerprint(_rng_state()),"data_state":_fingerprint(trainer.data_state)}
        if expected!=actual or control_loss!=restored_loss:
            (output/"resume_failure.json").write_text(json.dumps({"expected":expected,"actual":actual,
                "continuous_loss":control_loss,"restored_loss":restored_loss},indent=2)+"\n")
            raise AssertionError("adapted ACP complete continuation mismatch")
        # Deployment uses the frozen final EMA snapshot, with no target-dependent model selection.
        trainer.ema_model.eval()
        torch.save({"state_dicts":{"ema_model":trainer.ema_model.state_dict()},"shape_meta":meta,
                    "history_len":int(profile["history_len"]),"variant":"state-history-adapted"},output/"policy_ema.pt")
        report["training"]={"source_demonstrations":dataset.receipts,"source_windows":len(dataset),
             "updates":steps+1,"batch_size":int(profile["batch_size"]),"source_unet_down_dims":[256,512,1024],
             "mean_first25_loss":float(np.mean(losses[:25])),"mean_last25_loss":float(np.mean(losses[-25:])),
             "all_losses":losses,"optimizer_mode":"all_trainable_explicit_adaptation",
             "complete_resume_exact":True,"continuous_loss":control_loss,"restored_loss":restored_loss,
             "continuation_state_hashes":actual,"training_checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             "rotation_normalization":"source_bounded_identity_residual" if args.rotation_residual else "native_identity",
             "rotation_residual_scale":getattr(dataset,"rotation_residual_scale",None),
             "wall_seconds":time.monotonic()-started}
        report["actor_public_fields"]=list(PUBLIC_FIELDS)
        report["label_contract"]={"reference":"public source controller reference pose; recorded executed ACP commands when available",
             "virtual_target":"recorded executed ACP VT when available; otherwise posthoc native dim3 force-derived tool offset",
             "relative_base":"current public TCP pose; explicit adapted coordinate contract",
             "stiffness":"native k_max5000/min200/f_low0.5/high5/vel_tol999.002; full anisotropic K retained",
             "sensor_clock":"20ms task feedback; no 7000Hz FFT claim"}
        report["sampling_profile"]=sampling_profile(trainer.ema_model,dataset)
        files=[Path(__file__),ROOT/"genedynamics/learning/baselines/acp/task_training.py",ROOT/"genedynamics/learning/baselines/acp/trainer.py",args.config.resolve()]
        report["source_sha256"]={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        (output/"sampling_profile.json").write_text(json.dumps(report["sampling_profile"],indent=2)+"\n")
        if args.source_only:
            report["status"]="adapted_source_training_completed_without_target_interactions"
            report["target_interactions"]=0
            report["wall_seconds"]=time.monotonic()-started
            (output/"qualification.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({"status":report["status"],"sampling_profile":report["sampling_profile"]}),flush=True)
            return
        (output/"progress.json").write_text(json.dumps({"stage":"target_execution","training":report["training"]},indent=2)+"\n")
        from genedynamics.experiments.baselines import instantiate_task
        from genedynamics.learning.cross_embodiment.contracts import MechanismSpec
        target=instantiate_task(route.env_name,robot=str(profile["target_robot"]),seed=args.seed,
             mechanism_spec=MechanismSpec(**route.execution_env_params),history_len=int(profile["history_len"]))
        solver=ACPSolver(trainer.ema_model,meta,device=args.device,raw_sample_interval_s=.02,
                         slow_down_factor=1.,execution_horizon=int(profile["execution_horizon"]))
        execution=[]
        for target_seed in profile["target_seeds"]:
            history=[target.reset((int(target_seed),),seed=int(target_seed))]
            solver.reset()
            chunks=0
            while target.state.status[0]==0:
                try:
                    current=history[-1]
                    prediction=solver.act({"sparse":history_features(history,int(profile["history_len"])),
                        "current_tool_pose_world":pose(current["tcp_position"],current["tcp_rotation"])[:,None]})
                    result=target.execute_acp_chunk(prediction)
                    chunks+=1
                    for step in result["steps"]:history.append({key:step.observation[key].copy() for key in PUBLIC_FIELDS})
                    if not result["steps"]:raise RuntimeError("ACP chunk made no physical progress")
                except (ValueError,RuntimeError,FloatingPointError) as error:
                    target.reject(f"ACP adapted deployment failure: {error}")
                    execution.append({"seed":int(target_seed),"chunks":chunks,"error":str(error)})
                    break
            else:
                execution.append({"seed":int(target_seed),"chunks":chunks})
            report["target_evaluation"]=target.evaluate()
            (output/"target_progress.json").write_text(json.dumps({"execution":execution,"evaluation":report["target_evaluation"]},indent=2)+"\n")
            print(json.dumps({"stage":"target_episode","seed":target_seed,"evaluation":report["target_evaluation"]}),flush=True)
        report["execution"]=execution
        report["target_evaluation"]=target.evaluate()
        if report["target_evaluation"]["status_mismatches"]:
            raise AssertionError("CPU target evaluator/backend statuses disagree")
        report["status"]="adapted_source_training_and_charged_cpu_execution_completed"
        report["scope"]="source supervision and genuine learned ACP execution; all target episodes retained; development, not final comparison"
        report["wall_seconds"]=time.monotonic()-started
        files=[Path(__file__),ROOT/"genedynamics/learning/baselines/acp/task_training.py",ROOT/"genedynamics/learning/baselines/acp/trainer.py",args.config.resolve()]
        report["source_sha256"]={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        (output/"qualification.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"status":report["status"],"output":str(output),"target":report["target_evaluation"]},indent=2))
    except BaseException as error:
        report.update(status="failed",error=repr(error))
        (output/"qualification.failed.json").write_text(json.dumps(report,indent=2)+"\n")
        (output/"failure.log").write_text(traceback.format_exc())
        raise


if __name__=="__main__":main()
