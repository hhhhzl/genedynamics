"""Frozen Panda event prior on the same matched fixture and moving bodies."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment.contact import mating_contact_config,ContactMechanismSpec
from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from scripts.cross_embodiment.cpu_mechanism_validation import rollout
from scripts.cross_embodiment.run_theory_feasibility import publish
from scripts.cross_embodiment.collect_conditional_response import make_transport


def body_intent(runner,state,memory,command):
    value=np.zeros((*command.shape[:2],3),np.float32)
    if state.steps[0]*runner.config.control_dt<4.:value[...,0]=.005
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--robot",choices=("xarm7","go2_z1"),default="go2_z1")
    p.add_argument("--seed",type=int,default=36000);p.add_argument("--gain",type=float,default=1.)
    p.add_argument("--stiffness",type=float,default=0.);p.add_argument("--force-scale",type=float,default=1.)
    p.add_argument("--approach-speed",type=float,default=.01);p.add_argument("--approach-limit",type=float,default=.03)
    p.add_argument("--all-phase",action="store_true")
    a=p.parse_args()
    config=replace(mating_contact_config("xarm7"),robot=a.robot,environment="mobile_keyed_insertion_recovery",mobile=True,reset_task_noise=0.,
        approach_speed=a.approach_speed,approach_integral_limit=a.approach_limit)
    prior=PhaseGaussianPrior(FrozenRegressor.load("tmp/cross_embodiment/mechanism_revision/strict_phase_prior",backend="native"))
    publish(a.output/"contract.json",dict(config=config.__dict__,source_sha256=prior.regressor.metadata["tensor_sha256"],
        source_frozen=True,mechanism=dict(servo_gain=a.gain),transport=dict(scale=a.force_scale,stiffness=a.stiffness,bias=[0.,0.]),
        body_intent="world vx .005 m/s for first 4 s; stance feet anchors fixed; no walking",role="development pilot"))
    rows=[]
    for mode in (0,1):
        path=a.output/f"s{a.seed}_m{mode}"
        mapping=make_transport(dict(scale=a.force_scale,stiffness=a.stiffness,bias=[0.,0.],style="impedance" if a.all_phase else "stage"))
        row=load_bundle(path)[1] if (path/"metadata.json").exists() else rollout(config,ContactMechanismSpec(a.gain),seed=a.seed,mode=mode,
            prior=prior,output=path,record_carry=True,body_selector=body_intent,transport=mapping)
        rows.append(row);print(row,flush=True)
    publish(a.output/"report.json",dict(complete=True,episodes=rows,successes=sum(r["success"] for r in rows)))


if __name__=="__main__":main()
