"""Complete CPU mechanism chain on a real rising dual-route contact guide."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from genedynamics.learning.cross_embodiment.contact import spatial_guide_config,ContactMechanismSpec
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from genedynamics.learning.cross_embodiment.structured_response import fit_contact_twin,validate_twin
from scripts.cross_embodiment.cpu_mechanism_validation import rollout
from scripts.cross_embodiment.calibrate_contact_twin import collect_probes
from scripts.cross_embodiment.collect_conditional_response import collect_one
from scripts.cross_embodiment.run_theory_feasibility import publish


def cost(row,config,source):
    task=.5*(not row["success"])+.25*min(np.linalg.norm(np.asarray(row["final_position"])-config.goal)/config.position_tolerance,1.)+.25*min(row["final_force_rmse"]/5.,1.)
    response=np.mean((np.asarray(row["response_signature"])-source)**2)/4.
    mode=row["mode"]!=row["observed_mode"]
    return dict(task=float(task),joint=float(.6*task+.25*response+.1*mode+.05*row["seconds"]/config.timeout_seconds))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);p.add_argument("--workers",type=int,default=4)
    p.add_argument("--prior",type=Path);p.add_argument("--seed-start",type=int,default=44000)
    p.add_argument("--tangential-limit",type=float,default=12.);p.add_argument("--approach-speed",type=float,default=.005)
    p.add_argument("--approach-limit",type=float,default=.01)
    a=p.parse_args();out=a.output;source_config=spatial_guide_config();config=spatial_guide_config("xarm7")
    source_config=replace(source_config,tangential_wrench_limit=a.tangential_limit,approach_speed=a.approach_speed,approach_integral_limit=a.approach_limit)
    config=replace(config,tangential_wrench_limit=a.tangential_limit,approach_speed=a.approach_speed,approach_integral_limit=a.approach_limit)
    truth=ContactMechanismSpec(.65,3,.5,.7,.04)
    bank=[dict(style="impedance",scale=s,stiffness=k,bias=[0.,0.]) for s in (1.,1.5,2.,2.5) for k in (0.,.5,1.)]
    contract=dict(source_config=asdict(source_config),target_config=asdict(config),diagnostic_truth=asdict(truth),
        source_train_seeds=[] if a.prior else [a.seed_start,a.seed_start+1],source_test_seeds=[a.seed_start+10,a.seed_start+11],target_train_seeds=[a.seed_start+100,a.seed_start+101],target_test_seeds=list(range(a.seed_start+200,a.seed_start+205)),
        candidates=bank,physical_thresholds=dict(success_fraction=.95,mode_fraction=1.),
        target_task_training_interactions=0,probe_budget=5,calibrated_model_episode_budget=48,nominal_model_episode_budget=48,
        protocol="3D rising guide with true central ridge; frozen source; same force/pose/hold/collision rules for all baselines")
    if (out/"contract.json").exists():
        if json.loads((out/"contract.json").read_text())!=contract:raise ValueError("spatial guide contract mismatch")
    else:publish(out/"contract.json",contract)
    for seed in contract["source_train_seeds"]:
        for mode in (0,1):
            path=out/f"source_teacher/s{seed}_m{mode}"
            row=load_bundle(path)[1] if (path/"metadata.json").exists() else rollout(source_config,ContactMechanismSpec(),seed=seed,mode=mode,output=path,record_carry=True)
            if not row["success"] or row["mode"]!=row["observed_mode"]:raise RuntimeError("source teacher physical gate failed")
    prior_path=a.prior or out/"source_prior"
    if not (prior_path/"metadata.json").exists():subprocess.run([sys.executable,"-m","scripts.cross_embodiment.train_phase_prior","--source",str(out/"source_teacher"),"--output",str(prior_path),"--phase-support","0","1","3","--backend","native"],check=True)
    prior=PhaseGaussianPrior(FrozenRegressor.load(prior_path,backend="native"));source={}
    for seed in contract["source_test_seeds"]:
        for mode in (0,1):
            path=out/f"source_qualification/s{seed}_m{mode}"
            row=load_bundle(path)[1] if (path/"metadata.json").exists() else rollout(source_config,ContactMechanismSpec(),seed=seed,mode=mode,prior=prior,output=path,record_carry=True)
            if not row["success"] or row["mode"]!=row["observed_mode"]:raise RuntimeError("frozen source physical gate failed")
            source.setdefault(mode,[]).append(row["response_signature"])
    source={m:np.mean(v,axis=0) for m,v in source.items()}
    if not (out/"calibration.json").exists():
        records,ledger=collect_probes(config,truth,out/"probes",seed=a.seed_start+300,tangential=True)
        estimate,fit=fit_contact_twin(config,records[:3]);validation=validate_twin(config,estimate,records[3:])
        publish(out/"calibration.json",dict(estimated=asdict(estimate),fit=fit,validation=validation,budget=ledger))
    calibration=json.loads((out/"calibration.json").read_text())
    if not calibration["validation"]["supported"]:raise RuntimeError("guide response calibration validation failed")
    jobs=[]
    for role,mech in (("calibrated_search",calibration["estimated"]),("nominal_search",asdict(ContactMechanismSpec()))):
        for i,param in enumerate(bank):
            for seed in contract["target_train_seeds"]:
                for mode in (0,1):jobs.append((str(out/f"{role}/c{i}/s{seed}_m{mode}"),asdict(config),mech,str(prior_path),i,param,seed,mode,None))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for number,_ in enumerate(pool.map(collect_one,jobs),1):
            if number%12==0:print(dict(stage="guide_model_search",episodes=number,total=len(jobs)),flush=True)
    winners={}
    for role in ("calibrated_search","nominal_search"):
        history=[]
        for i,param in enumerate(bank):
            rows=[load_bundle(out/f"{role}/c{i}/s{seed}_m{mode}")[1] for seed in contract["target_train_seeds"] for mode in (0,1)]
            scores=[cost(r,config,source[r["mode"]]) for r in rows]
            history.append(dict(parameters=param,successes=sum(r["success"] for r in rows),scores={k:float(np.mean([s[k] for s in scores])) for k in ("task","joint")}))
        winners[role]={k:min(range(len(bank)),key=lambda i:history[i]["scores"][k]) for k in ("task","joint")}
        publish(out/f"{role}/search.json",dict(history=history,winners=winners[role]))
    methods=dict(direct=0,calibrated=winners["calibrated_search"]["joint"],task_only=winners["calibrated_search"]["task"],nominal=winners["nominal_search"]["joint"])
    publish(out/"frozen.json",dict(methods=methods,source_sha256=prior.regressor.metadata["tensor_sha256"],parameters={name:bank[i] for name,i in methods.items()}))
    jobs=[(str(out/f"evaluation/c{i}/s{seed}_m{mode}"),asdict(config),asdict(truth),str(prior_path),i,bank[i],seed,mode,None)
        for i in sorted(set(methods.values())) for seed in contract["target_test_seeds"] for mode in (0,1)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(collect_one,jobs))
    evaluations={}
    for name,i in methods.items():
        rows=[load_bundle(out/f"evaluation/c{i}/s{seed}_m{mode}")[1] for seed in contract["target_test_seeds"] for mode in (0,1)]
        evaluations[name]=dict(attempted=len(rows),successes=sum(r["success"] for r in rows),mode_matches=sum(r["observed_mode"]==r["mode"] for r in rows),episodes=rows)
    e=evaluations["calibrated"]
    publish(out/"report.json",dict(complete=True,contract=contract,calibration=calibration,evaluations=evaluations,methods=methods,
        passed=e["successes"]>=.95*e["attempted"] and e["mode_matches"]==e["attempted"],
        source_teacher_successes=2*len(contract["source_train_seeds"]),frozen_source_successes=4,distinct_physical_target_evaluation_episodes=len(jobs),
        target_task_training_interactions=0,scope="CPU complete guided-contact mechanism chain; finite twin transport, frozen learned source; no neural-kernel superiority claim"))
    print({k:{x:v[x] for x in ("successes","attempted","mode_matches")} for k,v in evaluations.items()},flush=True)


if __name__=="__main__":main()
