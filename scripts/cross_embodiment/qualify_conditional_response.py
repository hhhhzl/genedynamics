"""Independent episode and short-window neural response qualification.

Fitting, error validation and interval calibration use disjoint reset groups.
The finite candidate episode head is not an autoregressive dynamics model.
The short-window head predicts full controller/encoder carry and is qualified
separately. Neither inherits the structured twin's results.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle,save_bundle
from genedynamics.learning.cross_embodiment.learned import fit_regressor,FrozenRegressor
from genedynamics.learning.cross_embodiment.models import ModelSpec
from scripts.cross_embodiment.run_theory_feasibility import publish,costs
from scripts.cross_embodiment.audit_episode_surrogate import errors


def episode_input(carry,index,mode,count):
    # Candidate identity is categorical because this model's declared support
    # is a finite bank. The physical initial state is observed, not a seed ID.
    return np.r_[np.eye(count*2,dtype=np.float32)[index*2+mode],carry[80:98]]


def regression_metrics(prediction,actual):
    error=prediction-actual
    return dict(position_rmse=float(np.sqrt(np.mean(error[:,80:83]**2))),
        rotation_matrix_rmse=float(np.sqrt(np.mean(error[:,83:92]**2))),
        force_rmse=float(np.sqrt(np.mean(error[:,193]**2))),
        normalized_carry_rmse=float(np.sqrt(np.mean(error**2))))


def conformal_radius(errors,alpha):
    """Finite-sample split-conformal quantile, including infinity if needed."""
    errors=np.asarray(errors);rank=int(np.ceil((len(errors)+1)*(1-alpha)))
    if rank>len(errors):raise ValueError("more independent calibration clusters required")
    return float(np.sort(errors)[rank-1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--study",type=Path,required=True);p.add_argument("--updates",type=int,default=5000)
    a=p.parse_args(); dc=json.loads((a.dataset/"contract.json").read_text())
    if not (a.dataset/"report.json").exists():raise ValueError("collection must finish before qualification")
    contract=dict(dataset_contract=dc,updates=a.updates,seed=901,episode_hidden=[64,64],short_hidden=[128,128],ensemble=3,
        episode_thresholds=dict(signature_rmse=.05,task_cost_rmse=.15,failure_classification_error=.1),
        short_thresholds=dict(position_rmse=.002,rotation_matrix_rmse=.03,force_rmse=1.),
        interval_alpha=.2,interval_scope="maximum error over all candidate/mode episodes within each independent reset cluster",
        data_permissions="only post-calibration model rollouts; no physical target evaluation data")
    if (a.output/"contract.json").exists():
        if json.loads((a.output/"contract.json").read_text())!=contract:raise ValueError("qualification contract mismatch")
    else:publish(a.output/"contract.json",contract)
    source={}
    for mode in (0,1):
        source[mode]=np.mean([load_bundle(path)[1]["response_signature"] for path in (a.study/"source").glob(f"s*_m{mode}")],axis=0)
    ex=[];ey=[];sx=[];sy=[];splits=[];short_splits=[];groups=[]
    for role,seeds in dc["split"].items():
        for seed in seeds:
            for index in range(len(dc["candidates"])):
                for mode in (0,1):
                    d,m=load_bundle(a.dataset/role/f"c{index}"/f"s{seed}_m{mode}")
                    ex.append(episode_input(d["carry"][0],index,mode,len(dc["candidates"])))
                    ey.append([*m["response_signature"],costs(m,source[mode])["task"],float(not m["success"])])
                    splits.append(role);groups.append(seed)
                    valid=d["valid"][0].reshape(-1,10).all(-1)
                    x=np.c_[d["carry"],d["raw_command"].reshape(len(d["carry"]),-1)]
                    sx.extend(x[valid]);sy.extend(d["delta"][valid]);short_splits.extend([role]*sum(valid))
    ex,ey,sx,sy=map(lambda v:np.asarray(v,np.float32),(ex,ey,sx,sy));splits=np.asarray(splits);short_splits=np.asarray(short_splits);groups=np.asarray(groups)
    models={};fits={}
    for name,x,y,split,hidden in (("episode",ex,ey,splits,(64,64)),("short",sx,sy,short_splits,(128,128))):
        path=a.output/name
        if (path/"metadata.json").exists():
            models[name]=FrozenRegressor.load(path);fits[name]=models[name].metadata["fit_report"]
        else:
            model,_,fit=fit_regressor(x,y,split=="train",split=="validation",
                spec=ModelSpec(x.shape[-1],y.shape[-1],hidden=hidden,ensemble=3),seed=901,
                updates=a.updates,batch_size=64,checkpoint_path=a.output/(name+"_resume"),
                metadata=dict(learned_role="conditional_"+name+"_response",dataset=str(a.dataset),
                    layout=dict(carry_dim=197,command_dim=130,context_dim=0),
                    scope="fixed probe-calibrated body/context; response-decision validation only"))
            model.save(path,fit_report=fit);models[name]=model;fits[name]=fit
        print(dict(fitted=name,train_examples=fits[name]["train_examples"]),flush=True)
    v=splits=="validation";sv=short_splits=="validation"
    ep=np.asarray(models["episode"].predict(ex[v])).mean(0)
    sp=np.asarray(models["short"].predict(sx[sv])).mean(0)
    em=errors(ep,ey[v]);sm=regression_metrics(sp,sy[sv])
    calibration=splits=="interval_calibration"
    cp=np.asarray(models["episode"].predict(ex[calibration])).mean(0)
    ce=np.abs(cp-ey[calibration]);cg=groups[calibration]
    grouped=np.stack([ce[cg==g].max(0) for g in np.unique(cg)])
    radii=np.asarray([conformal_radius(grouped[:,j],contract["interval_alpha"]) for j in range(ey.shape[-1])])
    covered=np.abs(ep-ey[v])<=radii
    # Marginal per-coordinate cluster coverage, not simultaneous 26D coverage.
    coverage=np.stack([covered[groups[v]==g].all(0) for g in np.unique(groups[v])]).mean(0)
    selected=int(np.argmin(np.mean(np.asarray(models["episode"].predict(ex[splits=="train"])).mean(0)[:,24].reshape(-1,len(dc["candidates"]),2),axis=(0,2))))
    result=dict(complete=True,contract=contract,episode_validation=em,short_validation=sm,
        episode_supported=all(em[k]<=t for k,t in contract["episode_thresholds"].items()),
        short_supported=all(sm[k]<=t for k,t in contract["short_thresholds"].items()),
        calibrated_intervals=dict(alpha=contract["interval_alpha"],clusters=len(grouped),radii=radii.tolist(),
            independent_validation_cluster_coverage_per_coordinate=coverage.tolist(),
            scope="marginal coordinate coverage for exchangeable reset clusters conditional on this frozen calibrated model; not hidden-target coverage"),
        learned_selected_candidate=selected,learned_selected_parameters=dc["candidates"][selected],fit=fits,
        source_prior_sha256=dc["source_sha256"],target_task_training_interactions=0,
        scope="finite episode model and .2s full-carry response; multi-step, physical occupancy and transport deployment require separate gates")
    save_bundle(a.output/"validation_predictions",dict(episode_inputs=ex[v],episode_prediction=ep,episode_actual=ey[v],
        short_inputs=sx[sv],short_prediction=sp,short_actual=sy[sv]),dict(artifact_kind="conditional_response_validation"))
    publish(a.output/"report.json",result)
    print(dict(episode=em,short=sm,episode_supported=result["episode_supported"],short_supported=result["short_supported"],selected=selected),flush=True)


if __name__=="__main__":main()
