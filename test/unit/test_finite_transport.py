"""Scientific guards: abstention, same-class selection and trace diagnostics."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest

from genedynamics.learning.cross_embodiment.finite_transport import (
    select_finite, conditional_selection, response_cost, supported_weights,
    selection_stability_interval)


def entry(task, response, successes=(True, True)):
    return {'episodes': [dict(mode=m, observed_mode=m, success=successes[m],
                              costs=dict(task=task, response=response)) for m in (0, 1)]}


def test_all_failed_candidates_abstain_instead_of_selecting_least_bad():
    history=[entry(.1,.1,(False,False)),entry(.05,.0,(True,False))]
    result=select_finite(history)
    assert result['status']=='no_feasible_candidate'
    assert result['full'] is None and result['task_only'] is None
    modewise=conditional_selection(history)
    assert modewise['supported']==[True,False]
    assert modewise['full'][1] is None


def test_response_objective_has_explicit_task_noninferiority_budget():
    history=[entry(.1,.8),entry(.11,.2),entry(.2,0.),entry(.01,0.,(False,False))]
    result=select_finite(history,task_slack=.02)
    assert result['task_only']==0 and result['full']==1
    assert result['admissible']==[0,1]
    assert result['empirical_task_difference']<=.02
    assert select_finite(history,task_slack=0)['full']==0


def test_missing_mode_cannot_pass_global_feasibility():
    history=[{'episodes':[entry(.1,.2)['episodes'][0]]}]
    assert select_finite(history)['full'] is None


def test_response_units_and_clipping_are_bounded():
    source=np.zeros(24);changed=source.copy();changed[2]=.1 # 2mm
    assert response_cost(changed,source)==pytest.approx(1/24)
    changed=np.full(24,100.)
    assert response_cost(changed,source)==1.
    assert response_cost(source,source)==0.


def test_reweight_cannot_restore_an_unsupported_mode():
    assert supported_weights([0.,0.],[False,False])==[0.,0.]
    assert supported_weights([0.,1.],[False,True])==[0.,1.]
    assert sum(supported_weights([.2,.3],[True,True]))==pytest.approx(1.)


def test_scalarized_selection_interval_matches_pairwise_inequalities():
    history=[entry(.1,.8),entry(.11,.2)]
    result=selection_stability_interval(history,0)
    assert result['lambda_min']==0
    assert result['lambda_max']==pytest.approx(.01/.6)
    result=selection_stability_interval(history,1)
    assert result['lambda_min']==pytest.approx(.01/.6)
    assert result['lambda_max'] is None


def test_reset_roles_are_global_and_disjoint_across_conditions():
    from scripts.cross_embodiment.run_refinement import splits
    params=dict(training_reset_groups=4,model_validation_reset_groups=4,evaluation_reset_groups=5,law_reset_groups=6)
    values=splits(7100,params)
    roles={s:name for name,seeds in values.items() for s in seeds}
    assert len(roles)==sum(map(len,values.values()))
    assert not set(roles).intersection(s for seeds in splits(8100,params).values() for s in seeds)


def test_refinement_config_preserves_task_output_layout():
    from genedynamics.experiments.framework.config import ExperimentConfig
    cfg=ExperimentConfig.from_yaml(Path('configs/arm/keyed_insertion_recovery/validation/cross_ebdm_distribution.yaml'))
    assert not cfg.validate()
    assert cfg.metadata['workflow']=='cross_embodiment_refinement'
    assert cfg.method_params['stage']=='1.7'
    assert str(cfg.output_dir).endswith('results/arm/keyed_insertion_recovery/validation/cross_ebdm_distribution')


def test_optional_event_recording_does_not_change_physics():
    from genedynamics.learning.cross_embodiment.contact import ContactRunner,mating_contact_config,ContactMechanismSpec
    from genedynamics.learning.cross_embodiment.contracts import RandomRequest
    runner=ContactRunner(config=mating_contact_config('xarm7'),mechanism=ContactMechanismSpec())
    state=runner.reset_batch(RandomRequest(4501,(4501,)),initial='jammed')
    commands=np.zeros((1,10,13),np.float32);commands[...,12]=5.
    original=runner.collect_chunk(state,commands)
    runner._impl.record_events=True
    recorded=runner.collect_chunk(state,commands)
    for name in original.trace._fields:
        np.testing.assert_array_equal(getattr(original.trace,name),getattr(recorded.trace,name))
    events=runner._impl.last_event_trace
    assert events['hold_count'].shape==(1,10)
    np.testing.assert_array_equal(events['hold_count'][:,-1],recorded.state.hold_count)
    np.testing.assert_array_equal(events['contact_seen'][:,-1],recorded.state.contact_seen)


def test_cached_episode_cannot_change_request_or_replay_pending(tmp_path):
    from scripts.cross_embodiment import run_refinement as workflow
    from genedynamics.learning.cross_embodiment.core.artifacts import save_bundle
    path=tmp_path/'episode'
    job={'path':str(path),'role':'evaluation','code':'collection'}
    path.mkdir();(path/'attempt.json').write_text('{}')
    with pytest.raises(RuntimeError,match='no automatic free replay'):
        workflow.episode(job)
    save_bundle(path,{'x':np.zeros(1)},dict(artifact_context={'request_sha256':'wrong'},code_fingerprint={'aggregate':'collection'}))
    with pytest.raises(ValueError,match='provenance mismatch'):
        workflow.episode(job)


def test_complete_refinement_and_law_workflows_resume_without_new_interactions(tmp_path,monkeypatch):
    """Exercise real orchestration/artifacts with counted, cheap physics."""
    from concurrent.futures import Future
    from genedynamics.learning.cross_embodiment.core.artifacts import save_bundle
    from scripts.cross_embodiment import run_refinement as wf,cpu_mechanism_validation as physical
    calls={'rollouts':0,'calibrations':0}

    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def map(self,function,jobs):return map(function,jobs)
        def submit(self,function,job):
            future=Future()
            try:future.set_result(function(job))
            except BaseException as error:future.set_exception(error)
            return future

    def calibrate(job):
        path=Path(job['path']);target=path/'calibration.json'
        if target.exists():return json.loads(target.read_text())
        calls['calibrations']+=1
        result=dict(estimated=job['truth'],validation={'supported':True},
                    budget={'attempts':[{'purpose':'fit','outcome':'success'}]*5})
        wf.atomic_json(target,result)
        return result

    def simulate(config,mechanism,*,seed,mode,output,artifact_context,**kwargs):
        calls['rollouts']+=1
        n=25;p=np.tile([.44,-.2,.580],(1,n,1));p[0,0]=[.45,-.2,.596]
        p[0,1:3]=[.45,-.206 if mode else -.2,.614]
        data=dict(valid=np.ones((1,n),bool),tcp_position=p,
                  tcp_rotation=np.broadcast_to(np.diag([1.,-1.,-1.]),(1,n,3,3)).copy(),
                  normal_force=np.full((1,n),5.),force_peak=np.full((1,n),5.),
                  lateral_force_peak=np.zeros((1,n)),forbidden_collision=np.zeros((1,n),bool),
                  contact_lost=np.zeros((1,n),bool),saturation=np.zeros((1,n)),
                  force_integral=np.zeros((1,n)),allowed_contact=np.ones((1,n),bool),
                  command=np.zeros((1,n,13)))
        data['command'][...,12]=5.
        row=dict(status=1,success=True,seed=seed,mode=mode,observed_mode=mode,
                 force_peak=5.,lateral_force_peak=0.,final_position=[.44,-.2,.580],
                 final_force=5.,orientation_error=0.,seconds=.5,final_force_rmse=0.,
                 response_signature=[0.]*24,wall_seconds=.001,
                 config=config.__dict__,code_fingerprint={'aggregate':'test'},artifact_context=artifact_context)
        save_bundle(output,data,row)

    monkeypatch.setattr(wf,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(wf,'calibration',calibrate)
    monkeypatch.setattr(wf,'source_prior',lambda _:None)
    monkeypatch.setattr(physical,'rollout',simulate)
    params=dict(training_reset_groups=1,model_validation_reset_groups=1,
                evaluation_reset_groups=1,law_reset_groups=1,workers=1,task_slack=.02,stage='1.6')
    first=tmp_path/'feasibility'
    contract=dict(config=dict(output_dir=str(first),seeds=[7100],
                suites=[dict(name='matched',env_params=dict(robot='xarm7',gain=1.,delay=0))],method_params=params),
                prior=dict(path=str(tmp_path/'prior'),sha256='prior'),code={'aggregate':'test'},
                candidates=[dict(style='stage',scale=s,stiffness=0.,bias=[0.,0.]) for s in (1.,2.5)],protocol='finite_refinement_v1')
    report=wf.feasibility(contract)
    assert report['artifact_consistency_passed'] and report['all_full_cells_qualified']
    assert report['new_probes']==5
    before=dict(calls);wf.feasibility(contract)
    assert calls==before
    second=tmp_path/'distribution'
    law_contract=dict(contract,config=dict(contract['config'],output_dir=str(second),
                      method_params=dict(params,stage='1.7',feasibility_root=str(first),include_full_reweighted=True)))
    report=wf.distribution(law_contract)
    assert report['artifact_consistency_passed'] and report['new_probes']==0
    assert report['transport_incremental_value']=='not_demonstrated_same_maps'
    assert report['totals']['shuffled_coupling']['successes']==2
    assert report['totals']['shuffled_coupling']['mode_matches']==0
    assert report['totals']['full_reweighted']['successes']==2
    before=dict(calls);wf.distribution(law_contract)
    assert calls==before


@pytest.mark.parametrize('backend',['native','jax','torch'])
def test_recovery_impedance_is_replaced_at_alignment_and_insertion(backend,tmp_path):
    from genedynamics.learning.cross_embodiment.reference_transport import ReferenceTransport
    recovery=(0.,0.,0.,.5,.5,.5);aligned=(.25,.25,0.,1.,1.,1.)
    mapping=ReferenceTransport.from_stage_parameters(1.75,recovery_stiffness=recovery,
             alignment_stiffness=aligned,insertion_stiffness=(0.,)*6,backend=backend)
    commands=np.zeros((4,2,13),np.float32);commands[...,12]=5.
    value=mapping.apply(commands,phase=np.arange(4,dtype=np.int32))
    if backend=='torch':value=value.numpy()
    value=np.asarray(value)
    np.testing.assert_allclose(value[:2,:,6:12],np.broadcast_to(recovery,(2,2,6)),atol=1e-7)
    np.testing.assert_allclose(value[2,:,6:12],np.broadcast_to(aligned,(2,6)),atol=1e-7)
    np.testing.assert_allclose(value[3,:,6:12],0.,atol=1e-7)
    np.testing.assert_allclose(value[:3,:,12],5.)
    np.testing.assert_allclose(value[3,:,12],8.75)
    mapping.save(tmp_path/backend)
    restored=ReferenceTransport.load(tmp_path/backend,backend='native')
    np.testing.assert_allclose(restored.apply(commands,phase=np.arange(4)),value,atol=1e-7)


def test_reusing_calibration_preserves_ledger_without_collecting_probes(tmp_path,monkeypatch):
    from scripts.cross_embodiment import run_refinement as wf,calibrate_contact_twin as probes
    original=tmp_path/'original';(original/'probes').mkdir(parents=True)
    job=dict(path=str(tmp_path/'next'),config={'environment':'test'},truth={'gain':1.},seed=7100,code='new')
    old_job=dict(job,path=str(original),code='old')
    wf.atomic_json(original/'calibration-contract.json',old_job)
    ledger=dict(schema_version=1,limit=5,attempts=[dict(purpose='fit' if i<3 else 'validation',outcome='success') for i in range(5)])
    wf.atomic_json(original/'probes/ledger.json',ledger)
    wf.atomic_json(original/'calibration.json',dict(estimated={'gain':1.},validation={'supported':True},budget=ledger))
    job.update(reuse=str(original/'calibration.json'),reuse_sha256=wf.digest(original/'calibration.json'))
    monkeypatch.setattr(probes,'collect_probes',lambda *a,**k:pytest.fail('must not recollect reused probes'))
    result=wf.calibration(job)
    assert result['new_probes']==0 and result['budget']==ledger
    assert wf.calibration(job)==result
    ledger['attempts'][0]['outcome']='pending'
    wf.atomic_json(original/'probes/ledger.json',ledger)
    with pytest.raises(ValueError,match='ledger is incomplete'):
        wf.calibration(job)
