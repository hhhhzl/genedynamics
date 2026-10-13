"""Frozen short-response prediction on actual noisy-feedback task occupancy."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from scripts.cross_embodiment.collect_conditional_response import collect_one,candidate_bank
from scripts.cross_embodiment.qualify_conditional_response import regression_metrics
from scripts.cross_embodiment.run_theory_feasibility import publish


def project(carry):
    carry=carry.copy()
    for lo in (83,101):
        u,_,v=np.linalg.svd(carry[:,lo:lo+9].reshape(-1,3,3));r=u@v
        u[:,:,-1]*=np.linalg.det(r)[:,None];carry[:,lo:lo+9]=(u@v).reshape(-1,9)
    carry[:,194]=(carry[:,194]>.5);carry[:,195:197]=np.maximum(np.rint(carry[:,195:197]),0.)
    return carry


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--study",type=Path,default=Path("tmp/cross_embodiment/theory_feasibility/derated_contact"))
    p.add_argument("--models",type=Path,default=Path("tmp/cross_embodiment/mechanism_closure_v2/conditional_response_fit"))
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args();study=json.loads((a.study/"report.json").read_text());model=FrozenRegressor.load(a.models/"short",backend="native")
    bank=candidate_bank();selected=bank.index(study["methods"]["calibrated_transport"])
    contract=dict(source_sha256=study["contract"]["source_sha256"],short_model_sha256=model.metadata["tensor_sha256"],
        seeds=list(range(47000,47005)),observation_noise_std=[.00002,.0002,.03],methods=dict(direct=0,calibrated=selected),
        prediction_thresholds={"1":dict(position_rmse=.002,rotation_matrix_rmse=.03,force_rmse=1.),
                               "3":dict(position_rmse=.003,rotation_matrix_rmse=.05,force_rmse=2.)},
        predictor="full learned carry; rotation projection and discrete event-counter projection; externally supplied command windows",
        noise_scope="task feedback every .2 s; high-rate impedance/encoder feedback exact",target_task_training_interactions=0)
    if (a.output/"contract.json").exists():
        if json.loads((a.output/"contract.json").read_text())!=contract:raise ValueError("noisy occupancy contract mismatch")
    else:publish(a.output/"contract.json",contract)
    jobs=[(str(a.output/f"physical/c{i}/s{s}_m{m}"),study["contract"]["config"],study["contract"]["diagnostic_truth"],
        "tmp/cross_embodiment/mechanism_revision/strict_phase_prior",i,bank[i],s,m,contract["observation_noise_std"])
        for i in sorted(set(contract["methods"].values())) for s in contract["seeds"] for m in (0,1)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(collect_one,jobs))
    evaluations={};prediction={}
    for name,i in contract["methods"].items():
        episodes=[load_bundle(a.output/f"physical/c{i}/s{s}_m{m}") for s in contract["seeds"] for m in (0,1)]
        rows=[meta for _,meta in episodes]
        evaluations[name]=dict(successes=sum(r["success"] for r in rows),attempted=len(rows),mode_matches=sum(r["observed_mode"]==r["mode"] for r in rows))
        prediction[name]={}
        for steps in (1,3):
            initial=[];commands=[];actual=[]
            for d,_ in episodes:
                valid=d["valid"][0].reshape(-1,10).all(-1)
                for start in range(len(valid)-steps+1):
                    if valid[start:start+steps].all():
                        initial.append(d["carry"][start]);commands.append(d["raw_command"][start:start+steps]);actual.append(d["carry"][start+steps-1]+d["delta"][start+steps-1])
            current=np.asarray(initial,np.float32);commands=np.asarray(commands,np.float32);actual=np.asarray(actual,np.float32)
            for k in range(steps):
                x=np.c_[current,commands[:,k].reshape(len(current),-1)]
                current=project(current+np.asarray(model.predict(x)).mean(0))
            metrics=regression_metrics(current,actual);threshold=contract["prediction_thresholds"][str(steps)]
            prediction[name][str(steps)]=dict(metrics=metrics,windows=len(current),supported=all(metrics[k]<=v for k,v in threshold.items()))
    physical=evaluations["calibrated"]
    passed=physical["successes"]==physical["attempted"] and physical["mode_matches"]==physical["attempted"] and all(row["supported"] for values in prediction.values() for row in values.values())
    publish(a.output/"report.json",dict(complete=True,contract=contract,evaluations=evaluations,prediction=prediction,passed=passed,
        distinct_target_evaluation_episodes=len(jobs),target_task_training_interactions=0,
        scope="actual task-feedback noise and .2/.6 s conditional command-window response; no autonomous whole-episode neural rollout or full sensor-noise claim"))
    print(dict(evaluations=evaluations,prediction=prediction,passed=passed),flush=True)


if __name__=="__main__":main()
