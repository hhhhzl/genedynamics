"""Frozen short-horizon task-only MPC using the same calibrated candidate bank."""
import argparse
import json
from pathlib import Path
from genedynamics.learning.cross_embodiment.contact import ContactConfig,ContactMechanismSpec
from genedynamics.learning.cross_embodiment.contact_planner import ContactPlanner
from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from genedynamics.learning.cross_embodiment.data import code_fingerprint
from scripts.cross_embodiment.cpu_mechanism_validation import policy_step,rollout
from scripts.cross_embodiment.run_theory_feasibility import publish,transport
from scripts.cross_embodiment.validate_mechanism_chain import summarize


class MPCSelector:
    def __init__(self,config,estimate,bank,horizon,replan):
        self.bank=bank
        self.planner=ContactPlanner(config,estimate,bank,horizon_chunks=horizon)
        self.replan=replan
        self.calls=0
        self.current=0
        self.history=[]

    def __call__(self,runner,state,memory,prior,seed):
        if self.calls%self.replan==0:
            policy=lambda model,observed,carry:policy_step(model,observed,carry,prior,seed)
            self.current,detail=self.planner.select(state,memory,policy)
            self.history.append(dict(step=int(state.steps[0]),candidate=self.current,**detail))
        self.calls+=1
        return self.bank[self.current]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prior",type=Path,default=Path("tmp/cross_embodiment/mechanism_revision/strict_phase_prior"))
    p.add_argument("--seeds",type=int,nargs="+",default=[26000,26001])
    p.add_argument("--horizon",type=int,default=3);p.add_argument("--replan",type=int,default=5)
    a=p.parse_args()
    if a.replan<1:raise ValueError("positive replanning interval required")
    study=json.loads((a.study/"contract.json").read_text())
    calibration=json.loads((a.study/"calibration.json").read_text())
    if not calibration["validation"]["supported"]:raise ValueError("calibrated model unsupported")
    fitted=json.loads((a.study/"calibrated_search/search.json").read_text())
    parameters=[row["parameters"] for row in fitted["history"]]
    config=ContactConfig(**study["config"])
    estimate=ContactMechanismSpec(**calibration["estimated"])
    truth=ContactMechanismSpec(**study["diagnostic_truth"])
    prior=PhaseGaussianPrior(FrozenRegressor.load(a.prior))
    contract=dict(config=study["config"],calibrated_mechanism=calibration["estimated"],parameters=parameters,
        seeds=a.seeds,horizon_chunks=a.horizon,replan_chunks=a.replan,source_sha256=prior.regressor.metadata["tensor_sha256"],
        code_fingerprint=code_fingerprint(),scope="frozen task-only short-horizon candidate MPC; same calibrated model/bank; extra online model work explicitly counted")
    path=a.output/"contract.json"
    if path.exists() and json.loads(path.read_text())!=contract:raise ValueError("MPC resume contract mismatch")
    publish(path,contract)
    rows=[]
    for seed in a.seeds:
        for mode in (0,1):
            path=a.output/f"s{seed}_m{mode}"
            if (path/"metadata.json").exists():
                row=load_bundle(path)[1]
                details=json.loads((path/"mpc.json").read_text())
            else:
                selector=MPCSelector(config,estimate,[transport(x) for x in parameters],a.horizon,a.replan)
                row=rollout(config,truth,seed=seed,mode=mode,prior=prior,command_selector=selector,output=path)
                details=dict(statistics=selector.planner.statistics(),history=selector.history)
                publish(path/"mpc.json",details)
            rows.append(dict(**row,mpc=details["statistics"]))
            publish(a.output/"report.json",dict(complete=False,contract=contract,episodes=rows,summary=summarize(rows)))
            print(dict(seed=seed,mode=mode,success=row["success"],status=row["status"],mpc=details["statistics"]),flush=True)
    publish(a.output/"report.json",dict(complete=True,contract=contract,episodes=rows,summary=summarize(rows),
        model_control_steps=sum(r["mpc"]["valid_model_control_steps"] for r in rows),
        planner_wall_seconds=sum(r["mpc"]["planner_wall_seconds"] for r in rows)))


if __name__=="__main__":main()
