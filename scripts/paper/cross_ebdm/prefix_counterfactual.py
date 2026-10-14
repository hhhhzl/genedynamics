"""Paired model-only test of prefix invariance on an already observed failure."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import numpy as np

from genedynamics.experiments.artifacts import POLICY_ENV,atomic_json
from genedynamics.learning.cross_embodiment.contact import mating_contact_config
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from genedynamics.learning.cross_embodiment.data import code_fingerprint
from genedynamics.learning.cross_embodiment.finite_transport import refinement_candidates
from scripts.cross_embodiment.run_pilot import prepare_environment
from scripts.cross_embodiment.run_refinement import collect,immutable,read,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--feasibility',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();prepare_environment();root=args.output.resolve()
    os.environ[POLICY_ENV]=str(root/'artifact_policy.json')
    prior=args.pilot.resolve()/'_source/prior';_,prior_meta=load_bundle(prior)
    cal=args.feasibility.resolve()/'level_xarm7_matched/seed_8100/calibration.json'
    estimate=read(cal)['estimated'];code=code_fingerprint()['aggregate']
    hybrid=refinement_candidates('recovery_v2')[6:]
    control=[dict(style='stage',scale=s,stiffness=k,bias=[0.,0.]) for s in (1.5,1.75,2.) for k in (0.,1.)]
    bank=control+hybrid;jobs=[]
    for index,parameters in enumerate(bank):
        path=root/'prefix_model'/f'c{index}'/'s17100001_m1'
        jobs.append(dict(path=str(path),prior=str(prior),prior_sha256=prior_meta['tensor_sha256'],code=code,
                         config=asdict(mating_contact_config('xarm7')),mechanism=estimate,seed=17100001,mode=1,
                         parameters=parameters,role='prefix_diagnostic_model',extra_noise=False,
                         protocol='prefix_model_counterfactual_v1'))
    immutable(root/'prefix-model-contract.json',dict(jobs=jobs,candidates=bank,
              calibration_reference=dict(path=str(cal),sha256=digest(cal)),model_episode_budget=24,
              new_target_interactions=0,new_probes=0,
              scope='model-only paired diagnosis of an already observed development failure; never selection or target oracle'))
    rows=collect(jobs,8,'prefix_counterfactual_model',root)
    traces=[load_bundle(j['path'])[0] for j in jobs]
    ref=traces[0];valid=ref['valid'][0].astype(bool)
    control_identical=all(np.array_equal(t['tcp_position'][0,valid],ref['tcp_position'][0,valid])
                          and np.array_equal(t['command'][0,valid],ref['command'][0,valid]) for t in traces[:6])
    ordered=[rows[j['path']] for j in jobs]
    report=dict(complete=True,model_episodes=24,new_target_interactions=0,new_probes=0,
                controls_identical_before_failure=control_identical,
                zero_recovery_statuses=[r['status'] for r in ordered[:6]],
                zero_recovery_seconds=[r['seconds'] for r in ordered[:6]],
                zero_recovery_successes=sum(r['success'] for r in ordered[:6]),
                recovery_successes=sum(r['success'] for r in ordered[6:]),
                records=ordered,scope='single known reset model counterfactual; no population or real-system causal guarantee')
    atomic_json(root/'prefix-model-report.json',report)
    print({k:v for k,v in report.items() if k!='records'})


if __name__=='__main__':main()
