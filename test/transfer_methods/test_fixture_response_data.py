"""Moving-fixture records remain causal, versioned, and separate from truth."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from genedynamics.learning.datasets.transfer.target import (
    FIXTURE_FIELDS, FIXTURE_SCHEMA, FIXTURE_VERSION, FOUR_TASK_VERSION,
    PublicTargetTrajectory,
)
from genedynamics.solvers.single.cross_ebdm.learning.response import (
    PublicResponseEnsemble, PublicResponseSpec, fixture_response_windows,
    prepare_fixture_response_data, response_windows,
)
from test.transfer_methods.test_public_observed_response import record


def fixture_record(n=12):
    public, requests = record(n)
    public['joint_position'] = np.zeros((n+1,7),np.float32)
    public['joint_velocity'] = np.zeros((n+1,7),np.float32)
    public['previous_request'][1:] = requests
    public['fixture_position'] = np.tile([.44,-.2,0.],(n+1,1))
    public['fixture_rotation'] = np.tile(np.eye(3),(n+1,1,1))
    public['fixture_twist'] = np.zeros((n+1,6),np.float32)
    metadata = dict(
        version=FIXTURE_VERSION, permission='B_log', robot='xarm7', task='T3',
        environment='compliant_keyed_insertion_recovery',
        observation_schema=FIXTURE_SCHEMA, mode=0, seed=717, purpose='fit',
        chart='shared13', controls=n, recording_end='budget_stop',
    )
    return public, requests, metadata


def test_fixture_trajectory_roundtrip_and_robot_encoder_boundary(tmp_path):
    public, requests, metadata = fixture_record()
    trajectory = PublicTargetTrajectory(metadata,public,requests)
    trajectory.save(tmp_path/'episode')
    loaded = PublicTargetTrajectory.load(tmp_path/'episode')
    assert loaded.digest == trajectory.digest
    assert loaded.prefix(5).public['fixture_position'].shape == (6,3)
    contaminated = deepcopy(public)
    contaminated['joint_position'] = np.zeros((13,8))
    with pytest.raises(ValueError,match='observation'):
        PublicTargetTrajectory(metadata,contaminated,requests)
    with pytest.raises(ValueError,match='sensor schema'):
        PublicTargetTrajectory(metadata,dict(public,fixture_stiffness_Npm=np.ones(13)),requests)


def test_static_schema_does_not_silently_accept_fixture_data():
    public, requests, metadata = fixture_record()
    legacy = {k:v for k,v in metadata.items() if k not in {'environment','observation_schema'}}
    legacy['version'] = FOUR_TASK_VERSION
    with pytest.raises(ValueError,match='sensor schema'):
        PublicTargetTrajectory(legacy,public,requests)
    static_public = {k:v for k,v in public.items() if k not in FIXTURE_FIELDS}
    assert PublicTargetTrajectory(legacy,static_public,requests).metadata['version'] == FOUR_TASK_VERSION
    reflected = deepcopy(public)
    reflected['fixture_rotation'][:,0,0] = -1.
    with pytest.raises(ValueError,match='proper'):
        PublicTargetTrajectory(metadata,reflected,requests)


@pytest.mark.parametrize('violation', ['lateral_position', 'rotation', 'lateral_velocity'])
def test_fixture_data_and_windows_enforce_the_execution_chart(violation):
    public, requests, metadata = fixture_record()
    if violation == 'lateral_position':
        public['fixture_position'][:,1] += .01
    elif violation == 'rotation':
        public['fixture_rotation'][:] = [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
    else:
        public['fixture_twist'][:,1] = .01
    with pytest.raises(ValueError,match='passive_x_v1'):
        PublicTargetTrajectory(metadata,public,requests)
    with pytest.raises(ValueError,match='passive_x_v1'):
        fixture_response_windows(public,requests,body='xarm7',
            calibration_context=[0,0,0],spec=PublicResponseSpec())


def test_segmented_response_observes_transient_that_endpoint_summary_misses():
    public, requests, _ = fixture_record(10)
    spec = PublicResponseSpec(history_len=4,horizon=10,maximum_joints=7,hidden=(8,),members=1)
    changed = deepcopy(public)
    changed['tcp_position'][3:7,0] += .004
    changed['fixture_position'][3:7,0] += .003
    changed['fixture_twist'][3,0] = .02
    kwargs = dict(body='xarm7',calibration_context=[0,0,0],spec=spec)
    a = fixture_response_windows(public,requests,**kwargs)
    b = fixture_response_windows(changed,requests,**kwargs)
    assert a['segment_targets'].shape == (1,3,27)
    assert a['features'].shape == (1,spec.feature_dim+18*spec.history_len)
    np.testing.assert_array_equal(a['features'],b['features'])
    np.testing.assert_array_equal(a['segment_targets'][:,-1],b['segment_targets'][:,-1])
    assert not np.array_equal(a['segment_targets'],b['segment_targets'])
    old_a = response_windows({k:v for k,v in public.items() if k not in FIXTURE_FIELDS},requests,task='T3',**kwargs)
    old_b = response_windows({k:v for k,v in changed.items() if k not in FIXTURE_FIELDS},requests,task='T3',**kwargs)
    np.testing.assert_array_equal(old_a['targets'],old_b['targets'])
    old_model = PublicResponseEnsemble(spec)
    with pytest.raises(ValueError,match='schema mismatch'):
        old_model(torch.from_numpy(a['features']),torch.from_numpy(a['commands']))


def test_fixture_future_is_label_only_and_hidden_physics_is_rejected():
    public, requests, _ = fixture_record()
    spec = PublicResponseSpec()
    kwargs = dict(body='xarm7',calibration_context=[0,0,0],spec=spec)
    a = fixture_response_windows(public,requests,**kwargs)
    changed = deepcopy(public)
    changed['fixture_position'][2:,0] += .01
    b = fixture_response_windows(changed,requests,**kwargs)
    np.testing.assert_array_equal(a['features'][:2],b['features'][:2])
    assert not np.array_equal(a['segment_targets'][0],b['segment_targets'][0])
    with pytest.raises(ValueError,match='sensor fields'):
        fixture_response_windows(dict(public,fixture_stiffness_Npm=np.ones(13)),requests,**kwargs)
    with pytest.raises(ValueError,match='segment steps'):
        fixture_response_windows(public,requests,segment_steps=(3,6,9),**kwargs)


def test_short_episode_produces_no_padded_response_windows():
    public, requests, _ = fixture_record(3)
    values = fixture_response_windows(public,requests,body='xarm7',
        calibration_context=[0,0,0],spec=PublicResponseSpec())
    assert values['features'].shape[0] == 0
    assert values['segment_targets'].shape == (0,3,27)


def test_dataset_keeps_whole_episode_splits_and_rejects_duplicate_resets(tmp_path):
    from genedynamics.solvers.single.cross_ebdm.learning.artifact_bundle import load_bundle

    public, requests, metadata = fixture_record()
    entries=[]
    for i,purpose in enumerate(('fit','validation')):
        episode=PublicTargetTrajectory(dict(metadata,seed=717+i,purpose=purpose),public,requests)
        path=tmp_path/purpose
        episode.save(path)
        entries.append(dict(path=path,public_sha256=episode.digest))
    report=prepare_fixture_response_data(entries,tmp_path/'response')
    data,_=load_bundle(tmp_path/'response')
    assert report['fit_windows'] == report['validation_windows'] == 3
    assert set(data['episode_id'][data['split']==0]) == {0}
    assert set(data['episode_id'][data['split']==1]) == {1}
    assert not report['success_filtering'] and not report['model_trained']
    with pytest.raises(ValueError,match='duplicate reset'):
        prepare_fixture_response_data([entries[0],entries[0]],tmp_path/'duplicate')
    assert not (tmp_path/'duplicate'/'metadata.json').exists()
    changed=[dict(entries[0],public_sha256='wrong'),entries[1]]
    with pytest.raises(ValueError,match='identity changed'):
        prepare_fixture_response_data(changed,tmp_path/'changed')
