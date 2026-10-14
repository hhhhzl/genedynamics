"""Restricted model-search admission after failed CPU physics equivalence.

The original qualification thresholds/results stay unchanged. Re-evaluate
only retained checkpoints, with no new episodes or probes. Wrench/witness
definitions must agree on MJX's own contact set; CPU/MJX contact-set equality
is an additional physical-equivalence requirement, not assumed here.
"""
import argparse
import hashlib
from pathlib import Path


def admission(root):
    from scripts.cross_embodiment.run_p2 import prepare,read,digest
    prepare(True)
    import jax
    import numpy as np
    from genedynamics.experiments.artifacts import atomic_json
    from genedynamics.learning.cross_embodiment.data import code_fingerprint
    from genedynamics.learning.cross_embodiment.contact import ContactRunner,ContactMechanismSpec,mating_contact_config
    from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle,save_bundle
    contract=read(root/'workflow_contract.json');original=read(root/'gpu_report.json')
    current=code_fingerprint()['files'];numeric=[n for n in current if n.startswith('genedynamics/')]
    if not original['complete'] or any(current[n]!=contract['code']['files'].get(n) for n in numeric):
        raise ValueError('GPU qualification incomplete or numerical providers changed')
    cpu=Path(contract['config']['method_params'].get('cpu_fixture_root',root));groups={};rows=[]
    for case in contract['cases']:groups.setdefault((case['robot'],case['gain'],case['delay']),[]).append(case)
    with jax.experimental.enable_x64():
        for (robot,gain,delay),cases in groups.items():
            runner=ContactRunner(config=mating_contact_config(robot),mechanism=ContactMechanismSpec(gain,delay),backend='jax',device='gpu:0')
            impl=runner._impl;check=jax.jit(jax.vmap(impl._axial_support),device=impl.device)
            forbidden=jax.jit(jax.vmap(lambda d:impl._measure(d)[3]),device=impl.device)
            for case in cases:
                data,meta=load_bundle(cpu/'cpu_fixtures'/case['name'])
                for k in sorted(set([0,min(2,len(data['snapshot.steps'])-1),len(data['snapshot.steps'])-1])):
                    snapshot={n.removeprefix('snapshot.'):v[k] for n,v in data.items() if n.startswith('snapshot.')}
                    state=impl.import_snapshot(snapshot);runner.synchronize(state)
                    axial=bool(np.asarray(check(state.physics))[0]);bad=bool(np.asarray(forbidden(state.physics))[0])
                    contact=jax.device_get(state.physics._impl.contact)
                    pairs=np.asarray(contact.geom[0]);dist=np.asarray(contact.dist[0]);normal=np.asarray(contact.frame[0,:,0])
                    allowed=np.array([min(pair)>=0 and tuple(sorted(map(int,pair))) in impl.reference.allowed for pair in pairs])
                    valid=np.min(pairs,axis=1)>=0
                    expected=bool(np.any(valid&allowed&(dist<=0)&(np.abs(normal[:,2])>.7)))
                    expected_bad=bool(np.any(valid&~allowed&(dist < -1e-4)))
                    row=dict(name=case['name'],window=k,axial=axial,expected_axial=expected,forbidden=bad,
                             expected_forbidden=expected_bad,passed=axial==expected and bad==expected_bad)
                    target=root/'_reports/witness_checkpoints'/f'{case["name"]}_k{k}'
                    save_bundle(target,dict(geom=pairs,dist=dist,normal=normal,allowed=allowed),
                        dict(**row,cpu_snapshot_tensor_sha256=meta['tensor_sha256'],
                             original_qualification_contract_sha256=digest(root/'workflow_contract.json')))
                    rows.append(row)
    qualified=bool(rows) and all(r['passed'] for r in rows) and original['batch_execution_qualified'] and all(r['wrench_semantics_passed'] for r in original['operator_checks'])
    report=dict(complete=True,model_search_qualified=qualified,required_cpu_model_validation=True,
        target_reference_substitution_qualified=False,original_qualification_status='failed_cpu_physics_equivalence',
        original_qualification_report_sha256=digest(root/'gpu_report.json'),
        source_contract_sha256=digest(root/'workflow_contract.json'),witness_checks=rows,
        new_target_episodes=0,new_model_episodes=0,new_probes=0,checkpoint_forward_queries=len(rows),
        analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='restricted approximate model search; original CPU force/contact-set/long-task gates unchanged; independent CPU validation and CPU target confirmation mandatory')
    atomic_json(root/'model_search_admission.json',report)
    if not qualified:raise RuntimeError('restricted GPU model-search admission failed')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();result=admission(args.root);print({k:result[k] for k in ('model_search_qualified','target_reference_substitution_qualified','checkpoint_forward_queries')})


if __name__=='__main__':main()
