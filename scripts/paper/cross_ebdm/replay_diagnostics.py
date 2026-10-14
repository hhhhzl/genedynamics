"""Four declared, charged development replays with controller event state."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from genedynamics.experiments.artifacts import ArtifactPolicy, POLICY_ENV, atomic_json
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from genedynamics.learning.cross_embodiment.data import code_fingerprint
from scripts.cross_embodiment.run_pilot import prepare_environment, digest
from scripts.cross_embodiment.run_refinement import collect, immutable, read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();prepare_environment()
    original=args.root.resolve();output=args.output.resolve()
    policy=output/'artifact_policy.json'
    atomic_json(policy,asdict(ArtifactPolicy(str(output),10*1024**3,20*1024**3)),enforce_policy=False)
    os.environ[POLICY_ENV]=str(policy)
    prior=original/'_source/prior';_,prior_meta=load_bundle(prior)
    cases=[('xarm7_matched',4100,241001,0),('xarm7_matched',4100,241001,1),
           ('xarm7_matched',5100,251000,0),('xarm7_derated',4100,141002,1)]
    jobs=[];references={};fingerprint=code_fingerprint()
    for suite,calibration_seed,seed,mode in cases:
        study=original/('level_'+suite)/f'seed_{calibration_seed}'
        source=study/'evaluation/calibrated_transport'/f's{seed}_m{mode}'
        _,meta=load_bundle(source)
        parameters=read(study/'frozen_evaluation.json')['methods']['calibrated_transport']
        path=output/('level_'+suite)/f'seed_{calibration_seed}'/'replays'/f's{seed}_m{mode}'
        job=dict(path=str(path),prior=str(prior),prior_sha256=prior_meta['tensor_sha256'],code=fingerprint['aggregate'],
                 config=meta['config'],mechanism=meta['diagnostic_mechanism'],seed=seed,mode=mode,
                 parameters=dict(style='stage',**parameters),role='diagnostic_target',extra_noise=False)
        jobs.append(job);references[str(path)]=dict(original=str(source),sha256=meta['tensor_sha256'],status=meta['status'])
    immutable(output/'replay-contract.json',dict(jobs=jobs,references=references,
              source_physics_file_sha256=digest('genedynamics/envs/domains/manipulation/cross_embodiment/backends/contact.py'),
              diagnostic_target_episode_budget=4,new_probes=0,
              scope='already observed development reset replays, charged separately, never final-test data'))
    rows=collect(jobs,4,'replays',output)
    checks=[]
    for path,row in rows.items():
        data,meta=load_bundle(path)
        checks.append(dict(path=path,original_status=references[path]['status'],status=row['status'],
                           status_reproduced=row['status']==references[path]['status'],
                           event_fields=sorted(n for n in data if n.startswith('event.')),
                           valid_ticks=int(data['valid'].sum())))
    atomic_json(output/'replay-report.json',dict(complete=True,diagnostic_target_episodes=len(rows),new_probes=0,
                all_statuses_reproduced=all(r['status_reproduced'] for r in checks),records=checks,
                collection_code=fingerprint['aggregate']))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
