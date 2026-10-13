"""Five independent counted calibrations AND independent transport searches.

Previously charged immutable probe records are reused with explicit provenance;
no physical probe is silently repeated. Each replicate has new optimization
resets and new frozen evaluation resets. Candidate budgets match task-only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment.contact import mating_contact_config,ContactMechanismSpec,ContactConfig
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from scripts.cross_embodiment.collect_conditional_response import candidate_bank,collect_one
from scripts.cross_embodiment.run_theory_feasibility import costs,publish


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--root",type=Path,default=Path("tmp/cross_embodiment/theory_feasibility"))
    p.add_argument("--workers",type=int,default=4);p.add_argument("--clusters",type=int,default=5)
    p.add_argument("--training-clusters",type=int,default=2)
    p.add_argument("--train-seed-start",type=int,default=38000);p.add_argument("--test-seed-start",type=int,default=39000)
    p.add_argument("--regularization",type=float,default=0.,help="common squared tangential command-change penalty, frozen before evaluation")
    a=p.parse_args();bank=candidate_bank();config=asdict(mating_contact_config("xarm7"));truth=asdict(ContactMechanismSpec(.65,3))
    prior="tmp/cross_embodiment/mechanism_revision/strict_phase_prior"
    calibrations=[a.root/"derated_contact/calibration.json",*[a.root/f"calibration_repeats/r{i}/calibration.json" for i in range(1,5)]]
    records=[json.loads(path.read_text()) for path in calibrations]
    if any(not r["validation"]["supported"] for r in records):raise ValueError("independent calibration rejection")
    contract=dict(config=config,candidates=bank,calibrations=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in calibrations],
        independent_training_resets=[list(range(a.train_seed_start+100*r,a.train_seed_start+100*r+a.training_clusters)) for r in range(5)],
        evaluation_resets=[list(range(a.test_seed_start+100*r,a.test_seed_start+100*r+a.clusters)) for r in range(5)],
        common_command_regularization=a.regularization,
        prior_sha256=load_bundle(prior)[1]["tensor_sha256"],
        thresholds=dict(success_fraction=.95,mode_fraction=1.),
        accounting=f"5 previously charged probes per calibration; {24*a.training_clusters} calibrated and same nominal model episodes per independent search; no target task fitting")
    if (a.output/"contract.json").exists():
        original=json.loads((a.output/"contract.json").read_text())
        normalized=dict(original,config=asdict(ContactConfig(**original["config"])))
        if normalized!=contract:raise ValueError("replicate contract mismatch")
        if original!=contract:publish(a.output/"resume_defaults.json",dict(original_contract_preserved=True,
            added_defaults={k:v for k,v in config.items() if k not in original["config"]},
            reason="explicit serialization of existing numerical defaults; no change of task, controller values, candidates or splits"))
    else:publish(a.output/"contract.json",contract)
    def job(path,cfg,mech,index,param,seed,mode):return (str(path),cfg,mech,prior,index,param,seed,mode,None)
    source_jobs=[];search_jobs=[]
    for r,record in enumerate(records):
        for seed in contract["independent_training_resets"][r]:
            for mode in (0,1):source_jobs.append(job(a.output/f"r{r}/source/s{seed}_m{mode}",asdict(mating_contact_config()),asdict(ContactMechanismSpec()),0,bank[0],seed,mode))
            for role,mech in (("calibrated_search",record["estimated"]),("nominal_search",asdict(ContactMechanismSpec()))):
                for index,param in enumerate(bank):
                    for mode in (0,1):search_jobs.append(job(a.output/f"r{r}/{role}/c{index}/s{seed}_m{mode}",config,mech,index,param,seed,mode))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for number,_ in enumerate(pool.map(collect_one,[*source_jobs,*search_jobs]),1):
            if number%24==0:print(dict(stage="independent_training",episodes=number,total=len(source_jobs)+len(search_jobs)),flush=True)
    methods=[];eval_jobs=[]
    for r in range(5):
        selected={}
        for role in ("calibrated_search","nominal_search"):
            history=[]
            for i,param in enumerate(bank):
                rows=[]
                for seed in contract["independent_training_resets"][r]:
                    for mode in (0,1):
                        m=load_bundle(a.output/f"r{r}/{role}/c{i}/s{seed}_m{mode}")[1]
                        source=load_bundle(a.output/f"r{r}/source/s{seed}_m{mode}")[1]["response_signature"]
                        rows.append(dict(**m,costs=costs(m,np.asarray(source))))
                history.append(dict(parameters=param,episodes=rows,means={k:float(np.mean([row["costs"][k] for row in rows])) for k in rows[0]["costs"]}))
            change=[float(np.sum((np.asarray(param["bias"])/.01)**2)) for param in bank]
            winners={k:min(range(len(bank)),key=lambda i:history[i]["means"][k]+a.regularization*change[i]) for k in ("task","joint")}
            publish(a.output/f"r{r}/{role}/search.json",dict(history=history,winners=winners,model_episodes=24*a.training_clusters,
                command_change_cost=change,common_command_regularization=a.regularization))
            selected[role]=winners
        method=dict(direct=0,calibrated=selected["calibrated_search"]["joint"],task_only=selected["calibrated_search"]["task"],nominal=selected["nominal_search"]["joint"])
        methods.append(method);publish(a.output/f"r{r}/frozen.json",dict(methods=method,parameters={name:bank[i] for name,i in method.items()}))
        for i in sorted(set(method.values())):
            for seed in contract["evaluation_resets"][r]:
                for mode in (0,1):eval_jobs.append(job(a.output/f"r{r}/evaluation/c{i}/s{seed}_m{mode}",config,truth,i,bank[i],seed,mode))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for number,_ in enumerate(pool.map(collect_one,eval_jobs),1):
            if number%24==0:print(dict(stage="frozen_evaluation",episodes=number,total=len(eval_jobs)),flush=True)
    replicates=[]
    for r,method in enumerate(methods):
        rows={name:[load_bundle(a.output/f"r{r}/evaluation/c{i}/s{seed}_m{mode}")[1] for seed in contract["evaluation_resets"][r] for mode in (0,1)] for name,i in method.items()}
        summary={name:dict(successes=sum(x["success"] for x in values),attempted=len(values),mode_matches=sum(x["observed_mode"]==x["mode"] for x in values)) for name,values in rows.items()}
        improvement=[]
        for seed in contract["evaluation_resets"][r]:
            per_method={name:np.mean([costs(x,np.zeros(24))["task"] for x in values if x["seed"]==seed]) for name,values in rows.items()}
            improvement.append(per_method["direct"]-per_method["calibrated"])
        replicates.append(dict(replica=r,methods=method,summary=summary,paired_reset_improvements=improvement,
            mean_improvement=float(np.mean(improvement)),task_only_equals_calibrated=method["task_only"]==method["calibrated"]))
    values=np.array([r["mean_improvement"] for r in replicates]);rng=np.random.default_rng(902)
    boot=[]
    for _ in range(10000):
        replicas=rng.integers(0,5,5)
        boot.append(np.mean([rng.choice(replicates[r]["paired_reset_improvements"],a.clusters).mean() for r in replicas]))
    accepted=all(r["summary"]["calibrated"]["successes"]>=.95*2*a.clusters and r["summary"]["calibrated"]["mode_matches"]==2*a.clusters for r in replicates)
    publish(a.output/"report.json",dict(complete=True,contract=contract,replicates=replicates,passed=accepted,
        independent_calibrations=5,independent_transport_optimizations=5,distinct_physical_evaluation_episodes=len(eval_jobs),
        paired_mean_improvement=float(values.mean()),hierarchical_bootstrap_95_percentile=np.quantile(boot,[.025,.975]).tolist(),
        scope="five calibration+optimization replicates; bootstrap is descriptive with only five top-level clusters; equality to strong baseline retained; conditional development evidence"))
    print(dict(passed=accepted,replicates=[r["summary"] for r in replicates]),flush=True)


if __name__=="__main__":main()
