"""Fresh paired reset-cluster cost audit, with a fixed sample size and policies.

This tests the real cost improvement directly. It cannot certify the response
model, an infinite transport class, or variability across independent fits.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment.contact import ContactConfig,ContactMechanismSpec
from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.research_audit import ResearchAudit
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from genedynamics.learning.cross_embodiment.data import code_fingerprint
from scripts.cross_embodiment.cpu_mechanism_validation import rollout
from scripts.cross_embodiment.run_theory_feasibility import publish,transport,costs
from scripts.cross_embodiment.validate_mechanism_chain import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prior",type=Path,default=Path("tmp/cross_embodiment/mechanism_revision/strict_phase_prior"))
    p.add_argument("--clusters",type=int,default=25);p.add_argument("--seed",type=int,default=42000)
    a=p.parse_args()
    if a.clusters<1:raise ValueError("positive independent cluster count required")
    study=json.loads((a.study/"contract.json").read_text());report=json.loads((a.study/"report.json").read_text())
    if not report["complete"]:raise ValueError("completed frozen development study required")
    seeds=np.random.default_rng(a.seed).integers(1000000,10000000,size=a.clusters).tolist()
    if len(set(seeds))!=len(seeds):raise ValueError("sampled seed collision; predeclare a new seed before collecting")
    parameters=report["methods"]["calibrated_transport"]
    contract=dict(reset_seeds=seeds,parameters=parameters,alpha=.05,methods=["direct","calibrated_transport"],
        cost="task=.5*failure+.25*min(position_error/2mm,1)+.25*min(last15_force_rmse/5N,1); modes averaged per reset",
        code_fingerprint=code_fingerprint(),sample_size_design="25 new clusters selected from development pilot effect before these resets; no optional stopping",
        scope="conditional on one calibration and declared pseudorandom independent reset distribution; not cross-calibration confidence")
    path=a.output/"contract.json"
    if path.exists() and json.loads(path.read_text())!=contract:raise ValueError("paired audit resume mismatch")
    publish(path,contract)
    config=ContactConfig(**study["config"]);truth=ContactMechanismSpec(**study["diagnostic_truth"])
    prior=PhaseGaussianPrior(FrozenRegressor.load(a.prior));rows={"direct":[],"calibrated_transport":[]}
    for seed in seeds:
        for name in rows:
            for mode in (0,1):
                path=a.output/name/f"s{seed}_m{mode}"
                row=load_bundle(path)[1] if (path/"metadata.json").exists() else rollout(config,truth,seed=seed,mode=mode,prior=prior,
                    transport=None if name=="direct" else transport(parameters),output=path)
                rows[name].append(dict(**row,task_cost=costs(row,np.asarray(row["response_signature"]))["task"]))
        print(dict(reset_seed=seed,completed_clusters=len(rows["direct"])//2),flush=True)
        publish(a.output/"report.json",dict(complete=False,contract=contract,episodes=rows))
    table=[[float(np.mean([r["task_cost"] for r in rows[name] if r["seed"]==seed])) for name in rows] for seed in seeds]
    publish(a.output/"report.json",dict(complete=True,contract=contract,episodes=rows,
        summaries={name:summarize(value) for name,value in rows.items()},
        observed_task_cost_certificate=ResearchAudit().paired_improvement(table),
        task_only_search_relation="identical frozen parameters in development study; these episodes also describe its behavior, not an additional independent sample"))


if __name__=="__main__":main()
