"""Numerical boundaries of the current finite GPU contact policy.

Real CUDA physics qualification is a separately persisted research workflow.
"""
import numpy as np
import pytest


def test_replayed_prior_noise_is_independent_of_batch_order_and_partition():
    from genedynamics.learning.cross_embodiment.backends.jax.phase_contact import replay_noise
    seeds=[116100000,117100000,118100000]
    draws=replay_noise(seeds,7)
    np.testing.assert_array_equal(draws[:,[2,0,1]],replay_noise([seeds[i] for i in (2,0,1)],7))
    np.testing.assert_array_equal(draws,np.concatenate([replay_noise(seeds[:1],7),replay_noise(seeds[1:],7)],axis=1))
    for k in range(7):
        for i,seed in enumerate(seeds):
            expected=np.random.default_rng(np.random.SeedSequence([seed,k*10])).normal(size=(1,5)).astype(np.float32)
            np.testing.assert_array_equal(draws[k,i],expected[0])


def test_canonical_affine_bank_preserves_every_declared_phase_and_candidate(tmp_path):
    from genedynamics.learning.cross_embodiment.backends.jax.phase_contact import affine_bank
    from genedynamics.learning.cross_embodiment.finite_transport import refinement_candidates,make_mapping
    from genedynamics.learning.cross_embodiment.reference_transport import ReferenceTransport
    maps=[None,ReferenceTransport.from_parameters(1.7,(-.001,.002),backend='native')]
    maps+=[make_mapping(p) for p in refinement_candidates('recovery_v2')]
    paths=[]
    for i,mapping in enumerate(maps):
        path=tmp_path/str(i)
        if mapping is not None:mapping.save(path)
        paths.append(None if mapping is None else path)
    weights,bias=affine_bank(paths)
    raw=np.random.default_rng(72).normal(size=(len(maps),10,13)).astype(np.float32)
    for phase in range(4):
        mask=np.full((*raw.shape[:2],1),phase==3,np.float32)
        align=np.full_like(mask,phase==2)
        features=np.concatenate((raw,raw[...,12:13]*mask,align,mask),axis=-1)
        predicted=np.einsum('bli,bio->blo',features,weights)+bias[:,None]
        for i,mapping in enumerate(maps):
            expected=raw[i:i+1] if mapping is None else mapping.apply(raw[i:i+1],phase=np.array([phase]))
            np.testing.assert_allclose(predicted[i:i+1],expected,atol=2e-6,rtol=2e-6)


def test_new_gpu_protocol_admission_keeps_unimplemented_bodies_explicit():
    from dataclasses import replace
    from genedynamics.learning.cross_embodiment.contact import ContactRunner,mating_contact_config,spatial_guide_config
    for config in (mating_contact_config('ur5e'),replace(mating_contact_config(),mobile=True),spatial_guide_config()):
        with pytest.raises(NotImplementedError,match='fixed Panda/xArm7'):
            ContactRunner(config=config,backend='jax')


def test_law_rejections_contribute_cost_without_a_physical_mode_field(tmp_path):
    import json
    from scripts.paper.cross_ebdm.report_p2 import common_execution
    shared=dict(seed=72,source_mode=1,mode=1,costs=dict(task=.3,response=.2),path='same_frozen_trace')
    full=[dict(seed=72,source_mode=0,unsupported=True,success=False),shared]
    task=[dict(seed=72,source_mode=0,mode=0,costs=dict(task=.2,response=.1),path='task_only_trace'),shared]
    report=dict(replicates=[dict(suite='matched',seed=7100,results=dict(full=dict(episodes=full),task_only=dict(episodes=task)))])
    (tmp_path/'scientific_report.json').write_text(json.dumps(report))
    result=common_execution(tmp_path)
    assert result['requests']==2 and result['common_identical_traces']==1
    assert result['additional_full_rejections']==1
    assert result['full_minus_task_cost_all_requests']==pytest.approx(.4)
    assert result['empirical_stopping_bound']==pytest.approx(.5)
    assert result['common_execution_response_mean_gap']==0.
