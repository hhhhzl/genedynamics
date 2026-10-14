"""Diagnose source demonstrations against actual structured ACP CPU execution.

``raw`` replays post-hoc native labels from a successful hybrid teacher trace.
``live`` collects public-feedback, controller-consistent ACP source demonstrations.
Both routes use nominal Panda only; every reset, control, failure and replay is
retained. Neither route evaluates or selects a policy on target embodiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from genedynamics.experiments.baselines import instantiate_task
from genedynamics.learning.baselines.acp.task_training import pose
from genedynamics.learning.cross_embodiment.core.control_math import integrate_rotation
from genedynamics.solvers.single.acp.compliance import VirtualTargetEstimator,decode_stiffness_tool
from scripts.baselines.acp_native import ROOT,_digest_file
from scripts.baselines.qualify_tasks import PublicTeacher


def _estimator():
    return VirtualTargetEstimator(5000,200,.5,5,dim=3,characteristic_length=.02,vel_tol=999.002)


def _measurement(public):
    rotation=public["tcp_rotation"][0].T
    wrench=np.r_[rotation@public["measured_wrench"][0,:3],rotation@public["measured_wrench"][0,3:]]
    twist=np.r_[rotation@public["tcp_twist"][0,:3],rotation@public["tcp_twist"][0,3:]]
    return wrench,twist


def collect_live(output,*,mode,seed):
    output.mkdir(parents=True,exist_ok=False)
    task=instantiate_task("T3",robot="panda",seed=seed)
    public=task.reset((seed,),seed=seed)
    teacher=PublicTeacher(task,mode=mode)
    estimator=_estimator()
    observations={key:[value[0].copy()] for key,value in public.items()}
    references,virtuals,scalars=[],[],[]
    started=time.perf_counter()
    while task.state.status[0]==0:
        action=teacher.act(public)
        scalar,_,_=estimator.update(*_measurement(public))
        reference=pose(public["reference_position"],public["reference_rotation"])
        reference[0,:3,3]+=action[:3]*task.config.control_dt
        reference[0,:3,:3]=integrate_rotation(np,reference[0,:3,:3],action[3:6],task.config.control_dt)
        virtual=reference.copy()
        if action[12]>.01:
            reference[0,2,3]=public["tcp_position"][0,2]
            virtual[0,2,3]=reference[0,2,3]-action[12]/scalar
        current=pose(public["tcp_position"],public["tcp_rotation"])
        stiffness=decode_stiffness_tool(reference,virtual,np.array([scalar]),current)
        references.append(reference[0].copy());virtuals.append(virtual[0].copy());scalars.append(scalar)
        step=task.step_acp(dict(reference_pose_world=reference,virtual_target_pose_world=virtual,stiffness_tool=stiffness))
        public=step.observation
        for key,value in public.items():observations[key].append(value[0].copy())
    seconds=time.perf_counter()-started
    record=task.records[0]
    evaluation=task.evaluate()
    replay=task.replay_episode(record)
    arrays={"public."+key:np.array(value) for key,value in observations.items()}
    arrays.update({"acp.reference_pose_world":np.array(references),
        "acp.virtual_target_pose_world":np.array(virtuals),"acp.stiffness_scalar":np.array(scalars)})
    arrays.update({"trace."+key:np.asarray([row[key] for row in record.traces]) for key in record.traces[0]})
    np.savez_compressed(output/"episode.npz",**arrays)
    report={"kind":"live_public_feedback_teacher_via_structured_ACP","mode":mode,"seed":seed,
        "report":evaluation,"replay":replay,"execution_seconds":seconds,
        "rss_max_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "episode_sha256":_digest_file(output/"episode.npz"),
        "valid_source_demo":bool(evaluation["successes"]==1 and evaluation["status_mismatches"]==0
            and replay["matches_status"] and replay["trace_exact"] and replay["max_pose_error"]==0),
        "source_interaction_accounting":{"collection":dict(task.accounting),"audit_replay":replay["accounting"]},
        "force_intent":"public source desired normal force encoded as VT displacement force/k; explicit adapted expert",
        "native_human_dataset_claim":False,"target_interactions":0}
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def replay_raw(path,output,*,seed,offset):
    output.mkdir(parents=True,exist_ok=False)
    source_report=json.loads(path.with_name("report.json").read_text())
    if not source_report["valid_source_demo"] or source_report["episode_sha256"]!=_digest_file(path):
        raise ValueError("raw audit requires a SHA-qualified successful source demonstration")
    with np.load(path,allow_pickle=False) as data:
        public={key.removeprefix("public."):data[key].copy() for key in data.files if key.startswith("public.")}
    tool=pose(public["tcp_position"],public["tcp_rotation"])
    reference=pose(public["reference_position"],public["reference_rotation"])
    virtual=tool.copy()
    scalar=np.empty(len(tool))
    estimator=_estimator()
    for t in range(len(tool)):
        one={key:value[t:t+1] for key,value in public.items()}
        scalar[t],offset_tool,_=estimator.update(*_measurement(one))
        virtual[t,:3,3]+=tool[t,:3,:3]@offset_tool
    task=instantiate_task("T3",robot="panda",seed=seed)
    current=task.reset((seed,),seed=seed)
    started=time.perf_counter()
    issued=0
    for t in range(offset,len(tool)):
        if task.state.status[0]!=0:break
        base=pose(current["tcp_position"],current["tcp_rotation"])
        stiffness=decode_stiffness_tool(reference[t:t+1],virtual[t:t+1],scalar[t:t+1],base)
        current=task.step_acp(dict(reference_pose_world=reference[t:t+1],
            virtual_target_pose_world=virtual[t:t+1],stiffness_tool=stiffness)).observation
        issued+=1
    if task.state.status[0]==0:task.reject("raw source trace exhausted without success")
    report={"kind":"posthoc_source_GT19D_native_controller_audit","seed":seed,"offset":offset,
        "source_path":str(path),"source_sha256":_digest_file(path),"issued_commands":issued,
        "execution_seconds":time.perf_counter()-started,"rss_max_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "report":task.evaluate(),"target_interactions":0,
        "limitation":"Posthoc labels from a different hybrid controller need not form an ACP-consistent closed-loop expert"}
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind",choices=("raw","live"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=18450)
    parser.add_argument("--mode",type=int,choices=(0,1),default=0)
    parser.add_argument("--demonstration",type=Path)
    parser.add_argument("--offset",type=int,choices=(0,1),default=0)
    args=parser.parse_args()
    output=args.output.resolve()
    if ROOT/"latex" in output.parents:parser.error("audit evidence belongs outside latex/")
    if args.kind=="raw" and args.demonstration is None:parser.error("raw audit needs --demonstration")
    report=collect_live(output,mode=args.mode,seed=args.seed) if args.kind=="live" else replay_raw(
        args.demonstration,output,seed=args.seed,offset=args.offset)
    report["entrypoint_sha256"]=_digest_file(Path(__file__))
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
