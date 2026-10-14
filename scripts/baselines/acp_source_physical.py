"""Frozen-final-budget ACP source physical audit, independent of the sampling gate.

This reports a separate physical gate, keeps all four requested Panda episodes,
and physically replays every issued control. No training, model selection,
target embodiment or physics modification is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
import yaml

from genedynamics.experiments.baselines import instantiate_task
from genedynamics.learning.baselines.acp.task_training import PUBLIC_FIELDS,build_policy,history_features,pose
from genedynamics.learning.baselines.acp.trainer import deterministic_runtime
from genedynamics.solvers.single.acp import ACPSolver
from scripts.baselines.acp_native import ROOT,_digest_file,_seed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text())
    output=args.output.resolve()
    if ROOT/"latex" in output.parents:parser.error("source audit belongs outside latex/")
    output.mkdir(parents=True,exist_ok=False)
    seeds=tuple(config["source"]["closed_loop_seeds"])
    if len(seeds)!=4:raise ValueError("predeclared physical audit needs exactly four source seeds")
    checkpoint=args.checkpoint.resolve()
    report={"status":"predeclared","checkpoint":str(checkpoint),"checkpoint_sha256":_digest_file(checkpoint),
        "seed_cohort":list(seeds),"robot":"panda","source_only":True,"target_calls":0,
        "frozen_final_budget_EMA":True,"model_selection_or_training_permitted":False,
        "sampling_gate":"unchanged; retained separately even if this physical gate passes",
        "physical_gate":"all four requested episodes successful, zero status mismatches, exact charged replay",
        "budgets":{"collection_action_seconds":120,"replay_action_seconds":120,"wall_seconds":180},
        "scope":"development source physical audit; repeatedly developed validation is not a final independent test",
        "source_sha256":_digest_file(Path(__file__))}
    (output/"predeclared.json").write_text(json.dumps(report,indent=2)+"\n")
    deterministic_runtime();torch.set_num_threads(1)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    policy=build_policy(payload["shape_meta"],payload["history_len"])
    solver=ACPSolver(policy,payload["shape_meta"],device=args.device,raw_sample_interval_s=.02,
        slow_down_factor=1.,execution_horizon=config["training"]["execution_horizon"])
    solver.load_policy_state_dict(payload)
    task=instantiate_task("T3",robot="panda",history_len=payload["history_len"])
    begun=time.perf_counter();rows=[]
    for seed in seeds:
        history=[task.reset((seed,),seed=seed)];solver.reset();_seed(seed)
        episode_start=time.perf_counter();chunks=0;sampling_seconds=0.;physics_seconds=0.
        predictions=[]
        while task.state.status[0]==0:
            if time.perf_counter()-begun>160 or task.accounting["control_steps"]*task.config.control_dt>=120:
                task.reject("predeclared source physical audit budget exhausted");break
            try:
                current=history[-1]
                started=time.perf_counter()
                prediction=solver.act({"sparse":history_features(history,payload["history_len"]),
                    "current_tool_pose_world":pose(current["tcp_position"],current["tcp_rotation"])[:,None]})
                sampling_seconds+=time.perf_counter()-started
                predictions.append(prediction["raw_action"].detach().cpu().numpy()[0])
                started=time.perf_counter();result=task.execute_acp_chunk(prediction)
                physics_seconds+=time.perf_counter()-started;chunks+=1
                if not result["steps"]:raise RuntimeError("ACP source actor made no physical progress")
                history.extend({key:step.observation[key].copy() for key in PUBLIC_FIELDS} for step in result["steps"])
            except (ValueError,RuntimeError,FloatingPointError) as error:
                task.reject("ACP frozen source physical audit actor error: "+str(error));break
        evaluation=task.evaluate()
        record=task.records[-1]
        replay=task.replay_episode(record)
        arrays={"public."+key:np.stack([item[key][0] for item in history]) for key in PUBLIC_FIELDS}
        arrays["sampled_action19"]=np.asarray(predictions)
        if record.traces:
            arrays.update({"trace."+key:np.asarray([row[key] for row in record.traces]) for key in record.traces[0]})
        np.savez_compressed(output/f"source_episode_{seed}.npz",**arrays)
        rows.append({"seed":seed,"chunks":chunks,"sampling_seconds":sampling_seconds,
            "physics_seconds":physics_seconds,"episode_and_replay_seconds":time.perf_counter()-episode_start,
            "episode":evaluation["episodes"][-1],"charged_replay":replay,
            "trace_sha256":_digest_file(output/f"source_episode_{seed}.npz")})
        (output/"progress.json").write_text(json.dumps(rows,indent=2)+"\n")
        print(json.dumps(rows[-1]),flush=True)
    report["status"]="frozen_source_physical_audit_completed"
    report["episodes"]=rows;report["evaluation"]=task.evaluate()
    replay_accounting={}
    for row in rows:
        for key,value in row["charged_replay"]["accounting"].items():replay_accounting[key]=replay_accounting.get(key,0)+value
    report["physical_gate_passed"]=bool(report["evaluation"]["successes"]==4 and report["evaluation"]["status_mismatches"]==0
        and all(row["charged_replay"]["trace_exact"] and row["charged_replay"]["matches_status"] and row["charged_replay"]["max_pose_error"]==0 for row in rows))
    report["accounting"]={"collection":dict(task.accounting),"replay":replay_accounting,"target_calls":0}
    report["wall_seconds"]=time.perf_counter()-begun
    report["rss_max_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if torch.device(args.device).type=="cuda":report["gpu_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(torch.device(args.device))
    if _digest_file(checkpoint)!=report["checkpoint_sha256"]:raise AssertionError("frozen source checkpoint changed")
    (output/"physical_qualification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"physical_gate_passed":report["physical_gate_passed"],"evaluation":report["evaluation"],"accounting":report["accounting"]},indent=2))


if __name__=="__main__":main()
