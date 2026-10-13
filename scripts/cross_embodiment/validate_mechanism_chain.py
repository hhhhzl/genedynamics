"""Source→probes→calibrated twin→amortized transport→sealed physical evaluation.

This controlled study qualifies the restricted structured-twin mechanism. It
does not certify the neural response ensemble, general contact uncertainty,
stochastic-transport necessity or superior performance over system ID.
"""
import argparse
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from scipy.optimize import minimize_scalar
from genedynamics.learning.cross_embodiment import MechanismSpec
from genedynamics.learning.cross_embodiment.contact import ContactConfig,research_contact_config
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
from genedynamics.learning.cross_embodiment.reference_transport import ReferenceTransport
from genedynamics.learning.cross_embodiment.structured_response import fit_twin_context,validate_twin
from scripts.cross_embodiment.calibrate_contact_twin import collect_probes
from scripts.cross_embodiment.cpu_mechanism_validation import rollout


def tune_transport(config,context,prior,output,source_force):
    """All optimization uses the calibrated/public twin, never target evaluation."""
    history=[]
    def objective(scale,bias):
        deployed=ReferenceTransport.from_parameters(scale,bias)
        reports=[rollout(config,context,seed=seed,mode=m,prior=prior,transport=deployed) for seed in (130,131) for m in (0,1)]
        costs=[]
        for r in reports:
            goal=np.asarray(config.goal); error=np.linalg.norm(np.asarray(r["final_position"])-goal)/config.position_tolerance
            task=error**2+(r["final_force"]-config.desired_force)**2+(r["orientation_error"]/config.orientation_tolerance)**2
            source=.2*(r["final_force"]-source_force)**2
            failure=10.*(not r["success"])+2.*(r["observed_mode"]!=r["mode"])
            risk=max(r["force_peak"]/config.force_limit-1.,0.)**2
            costs.append(task+source+failure+10.*risk+.01*(scale-1.)**2+.01*np.sum((np.asarray(bias)/.02)**2))
        loss=float(np.mean(costs)); row={"scale":float(scale),"bias":list(map(float,bias)),"loss":loss,"rollouts":reports};history.append(row)
        print({"twin_gain":context.servo_gain,**{k:v for k,v in row.items() if k!="rollouts"}},flush=True)
        return row
    # Fixed, matched compute protocol. Coarse force search followed by two
    # response-based tangential updates; no final target evaluation is used.
    for scale in (1.,1.5,2.,2.5): objective(scale,(0.,0.))
    for _ in range(2):
        best=min(history,key=lambda r:r["loss"])
        errors=np.mean([np.asarray(r["final_position"])[:2]-np.asarray(config.goal)[:2] for r in best["rollouts"]],axis=0)
        bias=np.clip(np.asarray(best["bias"])-4.*errors,-.02,.02)
        for scale in (1.,1.5,2.,2.5): objective(scale,bias)
    best=min(history,key=lambda r:r["loss"]);scale=best["scale"]
    transport=ReferenceTransport.from_parameters(scale,best["bias"],metadata={"calibrated_context":asdict(context),"optimization":"fixed coarse force search and response tangential correction; 48 twin rollouts",
        "source_prior_sha256":prior.regressor.metadata["tensor_sha256"]})
    transport.save(output/"transport")
    (output/"tuning.json").write_text(json.dumps({"history":history,"scale":scale,"bias":best["bias"],"optimizer_success":"not globally certified","twin_rollouts":sum(len(r["rollouts"]) for r in history)},indent=2)+"\n")
    return scale


def summarize(rows):
    return {"attempted":len(rows),"successes":sum(r["success"] for r in rows),
        "mode_matches":sum(r["observed_mode"]==r["mode"] for r in rows),
        "final_force_rmse":float(np.sqrt(np.mean([r["final_force_rmse"]**2 for r in rows]))),
        "force_peak_max":max(r["force_peak"] for r in rows),"mean_seconds":float(np.mean([r["seconds"] for r in rows]))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);p.add_argument("--prior",type=Path,required=True)
    p.add_argument("--robot",default="xarm7");p.add_argument("--gain",type=float,default=.65);p.add_argument("--delay",type=int,default=3)
    p.add_argument("--seeds",type=int,nargs="+",default=[200,201,202]);p.add_argument("--calibration-seed",type=int,default=600)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    prior=PhaseGaussianPrior(FrozenRegressor.load(a.prior));config=research_contact_config(a.robot)
    truth=MechanismSpec(a.gain,a.delay,.5)
    records,ledger=collect_probes(config,truth,a.output/"calibration",seed=a.calibration_seed)
    context,fit=fit_twin_context(config,records[:3]);validation=validate_twin(config,context,records[3:]);nominal=validate_twin(config,MechanismSpec(),records[3:])
    if not validation["supported"]:
        (a.output/"report.json").write_text(json.dumps({"status":"unsupported_response","estimated":asdict(context),"validation":validation,"budget":ledger,"fit":fit},indent=2)+"\n")
        raise RuntimeError("independent response validation rejected calibrated twin; no target task training")
    source_force=np.mean([rollout(research_contact_config(),MechanismSpec(),seed=130,mode=m,prior=prior)["final_force"] for m in (0,1)])
    scale=tune_transport(config,context,prior,a.output,source_force)
    methods={"direct":1.,"calibrated_transport":scale,"system_id_inverse":float(np.clip(1./context.servo_gain,.5,2.5)),
             "wrong_context_transport":float(np.clip(context.servo_gain,.5,2.5))}
    evaluations={}
    for name,force_scale in methods.items():
        deployed=ReferenceTransport.load(a.output/"transport") if name=="calibrated_transport" else (ReferenceTransport.from_parameters(force_scale) if name!="direct" else None)
        rows=[]
        for seed in a.seeds:
            for mode in (0,1):
                r=rollout(config,truth,seed=seed,mode=mode,prior=prior,force_scale=1. if deployed is not None else force_scale,transport=deployed,output=a.output/name/f"s{seed}_m{mode}")
                rows.append(r);print({"method":name,**r},flush=True)
        evaluations[name]={"summary":summarize(rows),"episodes":rows}
    report={"source_prior_sha256":prior.regressor.metadata["tensor_sha256"],"estimated":asdict(context),"fit":fit,
        "validation":validation,"nominal_validation":nominal,"budget":ledger,"transport_scale":scale,"evaluations":evaluations,
        "diagnostic_truth":asdict(truth),"config":asdict(config),"test_seeds":a.seeds,
        "scope":"sealed after five probes; structured twin gain/delay family, known mass; development evidence, not universal superiority"}
    (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print({"estimated":asdict(context),"validation":validation,"evaluations":{k:v["summary"] for k,v in evaluations.items()}},flush=True)


if __name__=="__main__":main()
