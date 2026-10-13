"""Probe identifiability counterexample and closed-loop occupancy diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment.contact import ContactConfig,ContactMechanismSpec
from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.structured_response import PhysicsTwinResponse,response_features
from genedynamics.learning.cross_embodiment.data import code_fingerprint
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from scripts.cross_embodiment.calibrate_contact_twin import collect_probes
from scripts.cross_embodiment.cpu_mechanism_validation import rollout
from scripts.cross_embodiment.run_theory_feasibility import publish,transport,costs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration",type=Path,required=True);p.add_argument("--reference-study",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prior",type=Path,default=Path("tmp/cross_embodiment/mechanism_revision/strict_phase_prior"))
    p.add_argument("--seeds",type=int,nargs="+",default=[34000,34001])
    a=p.parse_args()
    case=json.loads((a.calibration/"contract.json").read_text())
    calibration=json.loads((a.calibration/"calibration.json").read_text())
    reference=json.loads((a.reference_study/"report.json").read_text())
    config=ContactConfig(**case["config"]);truth=ContactMechanismSpec(**case["diagnostic_truth"])
    estimate=ContactMechanismSpec(**calibration["estimated"])
    if config.robot!=reference["contract"]["config"]["robot"]:raise ValueError("frozen body map is not a cross-body adapter")
    contract=dict(calibration_artifact=str(a.calibration),calibration=calibration,seeds=a.seeds,
        map_parameters=reference["methods"]["calibrated_transport"],code_fingerprint=code_fingerprint(),
        thresholds=dict(signature_rmse=.05,task_cost_mae=.15,status_agreement=1.),
        scope="closed-loop response diagnostic for frozen reference map; readout noise only; no target-task adaptation or certified global support")
    if (a.output/"contract.json").exists() and json.loads((a.output/"contract.json").read_text())!=contract:raise ValueError("coverage audit resume mismatch")
    publish(a.output/"contract.json",contract)
    if not calibration["validation"]["supported"]:
        publish(a.output/"report.json",dict(complete=True,status="unsupported_probe_validation",attempted=len(a.seeds)*2,successes=0,calibration=calibration))
        return
    # Reconstruct already-charged probes only. collect_probes rejects an absent
    # record for a charged attempt and cannot silently collect another budget.
    ledger=json.loads((a.calibration/"probes/ledger.json").read_text())
    if len(ledger["attempts"])!=5:raise ValueError("complete counted buffer required")
    records,budget=collect_probes(config,truth,a.calibration/"probes",seed=case["calibration_seed"],tangential=True)
    contrasts=[]
    low=ContactMechanismSpec(estimate.servo_gain,estimate.delay_steps,.5,.15,.02)
    high=ContactMechanismSpec(estimate.servo_gain,estimate.delay_steps,.5,.7,.02)
    models=[PhysicsTwinResponse(config,m) for m in (low,high)]
    for i in (0,1,2):
        pair=[response_features(model.predict_probe(records[i])) for model in models]
        error=(pair[0]-pair[1])[records[i].result.trace.valid]
        contrasts.append(dict(probe=i,purpose=records[i].purpose,kind="free" if i==0 else "contact",
            position_difference_rmse_m=float(np.sqrt(np.mean(error[:,:3]**2))),
            force_difference_rmse_N=float(np.sqrt(np.mean(error[:,12]**2)))))
    prior=PhaseGaussianPrior(FrozenRegressor.load(a.prior));mapping=transport(contract["map_parameters"])
    mechanisms=dict(actual=truth,fitted=estimate,nominal_contact=ContactMechanismSpec(estimate.servo_gain,estimate.delay_steps,.5,.4,.02))
    tables={}
    for name,mechanism in mechanisms.items():
        rows=[]
        for seed in a.seeds:
            for mode in (0,1):
                path=a.output/name/f"s{seed}_m{mode}"
                row=load_bundle(path)[1] if (path/"metadata.json").exists() else rollout(config,mechanism,seed=seed,mode=mode,prior=prior,transport=mapping,output=path)
                rows.append(dict(**row,task_cost=costs(row,np.asarray(row["response_signature"]))["task"]))
        tables[name]=rows
        print(dict(model=name,successes=sum(r["success"] for r in rows),attempted=len(rows)),flush=True)
    metrics={}
    for name in ("fitted","nominal_contact"):
        actual=tables["actual"];prediction=tables[name]
        metrics[name]=dict(signature_rmse=float(np.sqrt(np.mean([(np.asarray(a["response_signature"])-p["response_signature"])**2 for a,p in zip(actual,prediction)]))),
            task_cost_mae=float(np.mean([abs(a["task_cost"]-p["task_cost"]) for a,p in zip(actual,prediction)])),
            status_agreement=float(np.mean([a["status"]==p["status"] for a,p in zip(actual,prediction)])))
    fitted=metrics["fitted"];limits=contract["thresholds"]
    supported=fitted["signature_rmse"]<=limits["signature_rmse"] and fitted["task_cost_mae"]<=limits["task_cost_mae"] and fitted["status_agreement"]>=limits["status_agreement"]
    publish(a.output/"report.json",dict(complete=True,contract=contract,probe_contrasts=contrasts,unchanged_probe_attempts=len(budget["attempts"]),
        task_occupancy_metrics=metrics,occupancy_supported=supported,tables=tables,
        scope="finite development diagnostic on new resets; task validation never used to refit frozen map or contact model"))


if __name__=="__main__":main()
