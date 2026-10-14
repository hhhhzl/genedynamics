"""Known-body × hidden mechanism crossings, with counted context-only fitting."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import json
import hashlib
import numpy as np
from genedynamics.learning.cross_embodiment.contact import mating_contact_config,ContactMechanismSpec
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from genedynamics.learning.cross_embodiment.structured_response import fit_contact_twin,validate_twin
from scripts.cross_embodiment.calibrate_contact_twin import collect_probes
from scripts.cross_embodiment.collect_conditional_response import collect_one,candidate_bank
from scripts.cross_embodiment.run_theory_feasibility import costs,publish


def cell(job):
    name,robot,gain,delay,index,out,reuse,refined,prior=job;out=Path(out);config=mating_contact_config(robot);truth=ContactMechanismSpec(gain,delay)
    seed=(64000 if refined else 58000)+index*1000
    bank=candidate_bank()+[dict(style="impedance",scale=s,stiffness=k,bias=[0.,0.]) for s in ((1.,1.5,2.,2.5) if refined else (1.,1.5,2.)) for k in ((.5,.75,1.) if refined else (0.,1.))]
    train_seeds=list(range(seed,seed+(8 if refined else 2)))
    budget=len(bank)*len(train_seeds)*2
    contract=dict(config=asdict(config),diagnostic_truth=asdict(truth),candidates=bank,source_prior_sha256=load_bundle(prior)[1]["tensor_sha256"],
        train_seeds=train_seeds,test_seeds=list(range(seed+100,seed+105)),target_task_training_interactions=0,
        calibrated_model_episode_budget=budget,nominal_model_episode_budget=budget,common_command_regularization=.1,
        probe_budget=5,calibration_provenance=None if reuse is None else dict(path=reuse,sha256=hashlib.sha256(Path(reuse).read_bytes()).hexdigest()),
        thresholds=dict(success_fraction=.95,mode_fraction=1.),protocol="Panda/xArm7 factorial and UR5e heldout body; body topology known, prior frozen, no target task adaptation")
    if refined:contract["selection_protocol"]="model success/mode feasibility first, then common objective; 8 training reset groups, intermediate all-phase stiffness; frozen before new physical tests"
    if (out/"contract.json").exists():
        if json.loads((out/"contract.json").read_text())!=contract:raise ValueError("matrix contract mismatch")
    else:publish(out/"contract.json",contract)
    if not (out/"calibration.json").exists():
        if reuse is not None:publish(out/"calibration.json",json.loads(Path(reuse).read_text()))
        else:
            records,ledger=collect_probes(config,truth,out/"probes",seed=seed+200,tangential=True)
            estimate,fit=fit_contact_twin(config,records[:3]);validation=validate_twin(config,estimate,records[3:])
            publish(out/"calibration.json",dict(estimated=asdict(estimate),fit=fit,validation=validation,budget=ledger))
    calibration=json.loads((out/"calibration.json").read_text())
    if not calibration["validation"]["supported"]:raise RuntimeError(f"{name}: independent calibration rejected")
    def run(path,cfg,mech,i,s,m):return collect_one((str(path),asdict(cfg),mech,prior,i,bank[i],s,m,None))
    source={}
    for s in contract["train_seeds"]:
        for m in (0,1):
            row=run(out/f"source/s{s}_m{m}",mating_contact_config(),asdict(ContactMechanismSpec()),0,s,m)
            if not row["success"] or row["observed_mode"]!=m:raise RuntimeError("matrix source support failure")
            source[s,m]=np.asarray(row["response_signature"])
    selected={}
    for role,mech in (("calibrated",calibration["estimated"]),("nominal",asdict(ContactMechanismSpec()))):
        history=[]
        for i,param in enumerate(bank):
            rows=[run(out/f"{role}/c{i}/s{s}_m{m}",config,mech,i,s,m) for s in contract["train_seeds"] for m in (0,1)]
            values=[costs(row,source[s,m]) for row,(s,m) in zip(rows,[(s,m) for s in contract["train_seeds"] for m in (0,1)])]
            reg=.1*float(np.sum((np.asarray(param["bias"])/.01)**2))
            history.append({k:float(np.mean([v[k] for v in values]))+reg for k in ("task","joint")})
            if refined:history[-1]["unsupported_training_episodes"]=sum(not r["success"] or r["observed_mode"]!=r["mode"] for r in rows)
            if refined and (i+1)%4==0:print(dict(cell=name,role=role,candidates=i+1,total=len(bank)),flush=True)
        selected[role]={k:min(range(len(bank)),key=lambda i:((history[i]["unsupported_training_episodes"],history[i][k]) if refined else history[i][k])) for k in ("task","joint")}
        publish(out/f"{role}/search.json",dict(scores=history,winners=selected[role],model_episodes=budget))
    methods=dict(direct=0,calibrated=selected["calibrated"]["joint"],task_only=selected["calibrated"]["task"],nominal=selected["nominal"]["joint"])
    publish(out/"frozen.json",dict(methods=methods,parameters={n:bank[i] for n,i in methods.items()}))
    for i in sorted(set(methods.values())):
        for s in contract["test_seeds"]:
            for m in (0,1):run(out/f"evaluation/c{i}/s{s}_m{m}",config,asdict(truth),i,s,m)
    summaries={}
    for n,i in methods.items():
        rows=[load_bundle(out/f"evaluation/c{i}/s{s}_m{m}")[1] for s in contract["test_seeds"] for m in (0,1)]
        summaries[n]=dict(attempted=len(rows),successes=sum(r["success"] for r in rows),mode_matches=sum(r["observed_mode"]==r["mode"] for r in rows))
    r=summaries["calibrated"];report=dict(complete=True,passed=r["successes"]>=.95*r["attempted"] and r["mode_matches"]==r["attempted"],
        contract=contract,calibration=calibration,summaries=summaries,methods=methods,distinct_physical_target_evaluation_episodes=10*len(set(methods.values())))
    publish(out/"report.json",report);print(dict(cell=name,passed=report["passed"],summaries=summaries),flush=True);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);p.add_argument("--workers",type=int,default=2)
    p.add_argument("--prior",type=Path,required=True)
    p.add_argument("--refined",action="store_true");p.add_argument("--calibration-root",type=Path,help="explicit reuse only; default collects fresh counted probes");a=p.parse_args()
    definitions=[("panda_matched","panda",1.,0,None),("panda_derated","panda",.65,3,None),
        ("xarm7_matched","xarm7",1.,0,None),("xarm7_derated","xarm7",.65,3,None),
        ("ur5e_derated","ur5e",.65,3,None)]
    jobs=[(n,r,g,d,i,str(a.output/n),str(a.calibration_root/n/"calibration.json") if a.calibration_root else None,a.refined,str(a.prior)) for i,(n,r,g,d,_) in enumerate(definitions)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:reports=list(pool.map(cell,jobs))
    publish(a.output/"report.json",dict(complete=True,passed=all(r["passed"] for r in reports),cells={d[0]:r for d,r in zip(definitions,reports)},
        scope="finite body/mechanism factorial plus known-morphology heldout UR5e context-only calibration; no body-agnostic universal transfer"))


if __name__=="__main__":main()
