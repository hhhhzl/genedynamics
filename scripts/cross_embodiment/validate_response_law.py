"""Frozen finite joint law, new reset tests, and actual selected-map deployment.

Point-response MSE failures remain failures. This separately evaluates the
distributional object in the theory, using frozen thresholds on untouched
reset groups. Sampled response vectors retain success/force/pose correlations.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
from genedynamics.learning.cross_embodiment.finite_response import FiniteResponseLaw
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from scripts.cross_embodiment.collect_conditional_response import collect_one
from scripts.cross_embodiment.run_theory_feasibility import publish,costs


def table(root,role,seeds,count):
    values=[]
    for i in range(count):
        for mode in (0,1):
            rows=[]
            for seed in seeds:
                m=load_bundle(root/role/f"c{i}"/f"s{seed}_m{mode}")[1]
                rows.append([*m["response_signature"],costs(m,np.zeros(24))["task"],float(not m["success"]),m["seconds"]/15.,float(m["observed_mode"]!=mode)])
            values.append(rows)
    return np.asarray(values,np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--study",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--test-data",type=Path);p.add_argument("--workers",type=int,default=4)
    a=p.parse_args();dc=json.loads((a.dataset/"contract.json").read_text());count=len(dc["candidates"])
    contract=dict(dataset_sha256=hashlib.sha256((a.dataset/"contract.json").read_bytes()).hexdigest(),
        fit_resets=dc["split"]["train"],test_resets=list(range(40000,40008)),physical_resets=list(range(41000,41008)),
        thresholds=dict(mean_signature_marginal_w1=.05,mean_task_cost_w1=.15,mean_failure_probability_error=.1),
        physical_thresholds=dict(success_fraction=.95,mode_fraction=1.),
        criterion="separate finite joint response law; point-model cost RMSE .209 failure is retained",
        source_prior_sha256=dc["source_sha256"],target_task_training_interactions=0)
    if (a.output/"contract.json").exists():
        if json.loads((a.output/"contract.json").read_text())!=contract:raise ValueError("frozen law contract mismatch")
    else:publish(a.output/"contract.json",contract)
    train=table(a.dataset,"train",dc["split"]["train"],count)
    law_path=a.output/"law"
    if not (law_path/"metadata.json").exists():
        law=FiniteResponseLaw.fit(train,backend="native",metadata=dict(candidates=dc["candidates"],estimated=dc["estimated"],
            config=dc["config"],source_prior_sha256=dc["source_sha256"],training_data_contract=contract["dataset_sha256"]))
        law.save(law_path)
    # Expected task cost from the frozen empirical joint law. Task-only and
    # anchored objectives use the same table and budget, as in the twin study.
    source={m:np.mean([load_bundle(path)[1]["response_signature"] for path in (a.study/"source").glob(f"s*_m{m}")],axis=0) for m in (0,1)}
    task=train[:,:,24].mean(1).reshape(count,2).mean(1)
    response=np.asarray([np.mean((train[i,:,:24]-source[i%2])**2)/4. for i in range(count*2)]).reshape(count,2).mean(1)
    joint=.6*task+.25*response+.1*train[:,:,-1].mean(1).reshape(count,2).mean(1)+.05*train[:,:,-2].mean(1).reshape(count,2).mean(1)
    selected=int(np.argmin(joint));task_selected=int(np.argmin(task))
    freeze=dict(selected_candidate=selected,task_only_candidate=task_selected,parameters=dc["candidates"][selected],
        law_sha256=load_bundle(law_path)[1]["tensor_sha256"],expected_task_costs=task.tolist(),expected_joint_costs=joint.tolist())
    if (a.output/"frozen.json").exists() and json.loads((a.output/"frozen.json").read_text())!=freeze:raise ValueError("frozen selection mismatch")
    publish(a.output/"frozen.json",freeze)
    if a.test_data is None:
        print(dict(frozen=freeze,stage="fit_complete_test_not_started"),flush=True);return
    td=json.loads((a.test_data/"contract.json").read_text())
    if td["split"]!={"test":contract["test_resets"]}:raise ValueError("untouched reset split mismatch")
    test=table(a.test_data,"test",contract["test_resets"],count)
    distance=np.asarray([[wasserstein_distance(x[:,j],y[:,j]) for j in range(train.shape[-1])] for x,y in zip(train,test)])
    metrics=dict(mean_signature_marginal_w1=float(distance[:,:24].mean()),mean_task_cost_w1=float(distance[:,24].mean()),
        mean_failure_probability_error=float(np.abs(train[:,:,25].mean(1)-test[:,:,25].mean(1)).mean()))
    supported=all(metrics[k]<=v for k,v in contract["thresholds"].items())
    # A dependence diagnostic supplements the marginals; this does not claim
    # that marginal W1 certifies a 28-dimensional uniform joint kernel bound.
    actual_cov=np.asarray([np.cov(x.T) for x in test]);model_cov=np.asarray([np.cov(x.T) for x in train])
    dependence=float(np.sqrt(np.mean((actual_cov-model_cov)**2)))
    result=dict(complete=False,contract=contract,frozen=freeze,distribution_validation=metrics,distribution_supported=supported,
        response_covariance_rmse=dependence,all_case_distances=distance.tolist(),
        scope="empirical finite response distribution integrated over reset/source randomness; marginal tests and covariance diagnostic do not certify arbitrary conditional kernels")
    publish(a.output/"report.json",result)
    if not supported:
        publish(a.output/"report.json",dict(result,complete=True,status="unsupported_no_deployment"));return
    study=json.loads((a.study/"report.json").read_text());truth=study["contract"]["diagnostic_truth"]
    prior="tmp/cross_embodiment/mechanism_revision/strict_phase_prior"
    methods=dict(direct=0,learned_law=selected,task_only=task_selected,
        nominal=dc["candidates"].index(study["methods"]["nominal_response_transport"]))
    jobs=[(str(a.output/f"physical/c{i}/s{s}_m{m}"),dc["config"],truth,prior,i,dc["candidates"][i],s,m,None)
        for i in sorted(set(methods.values())) for s in contract["physical_resets"] for m in (0,1)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(collect_one,jobs))
    evaluations={}
    for name,i in methods.items():
        rows=[load_bundle(a.output/f"physical/c{i}/s{s}_m{m}")[1] for s in contract["physical_resets"] for m in (0,1)]
        evaluations[name]=dict(successes=sum(r["success"] for r in rows),attempted=len(rows),
            mode_matches=sum(r["observed_mode"]==r["mode"] for r in rows),episodes=rows)
    row=evaluations["learned_law"]
    publish(a.output/"report.json",dict(result,complete=True,status="physical_evaluation_complete",evaluations=evaluations,
        physical_supported=row["successes"]>=.95*row["attempted"] and row["mode_matches"]==row["attempted"],
        distinct_target_evaluation_episodes=len(jobs),task_only_equals_anchored=selected==task_selected,
        target_task_training_interactions=0))
    print(dict(distribution=metrics,selected=selected,evaluation={k:{x:v[x] for x in ("successes","attempted","mode_matches")} for k,v in evaluations.items()}),flush=True)


if __name__=="__main__":main()
