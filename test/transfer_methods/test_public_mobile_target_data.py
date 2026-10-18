"""Mobile weak buffers preserve only the declared causal sensor/action schema."""
import json
import numpy as np
import pytest
from genedynamics.learning.datasets.transfer.mobile_target import PublicMobileTargetTrajectory,VERSION
def test_mobile_public_schema_prefix_and_tamper(tmp_path):
    n=20
    dims=dict(joint_position=(25,),joint_velocity=(24,),tcp_position=(3,),tcp_rotation=(3,3),tcp_twist=(6,),measured_wrench=(6,),goal_position=(3,),goal_rotation=(3,3),previous_action=(13,),previous_request=(13,),reference_position=(3,),reference_rotation=(3,3),elapsed_seconds=(1,),body_position=(3,),body_twist=(6,),controller_feedback=(2,))
    public={k:np.zeros((n+1,*v),np.float32) for k,v in dims.items()};public['elapsed_seconds'][:,0]=np.arange(n+1)*.02
    requests=np.zeros((n,13),np.float32);requests[:,0]=np.arange(n)*.001;public['previous_request'][1:]=requests
    meta=dict(version=VERSION,permission='B_log',robot='go2_z1',task='T4',mode=0,seed=0,purpose='fit',chart='shared13',controls=n,recording_end='terminal')
    row=PublicMobileTargetTrajectory(meta,public,requests);part=row.prefix(13);part.save(tmp_path/'buffer')
    assert part.metadata['recording_end']=='budget_stop'
    assert PublicMobileTargetTrajectory.load(tmp_path/'buffer').digest==part.digest
    assert not part.requests.flags.writeable
    for bad in (dict(meta,servo_gain=.65),dict(meta,task='T3'),dict(meta,permission='C_supervised')):
        with pytest.raises(ValueError):PublicMobileTargetTrajectory(bad,public,requests)
    with pytest.raises(ValueError):PublicMobileTargetTrajectory(meta,dict(public,axial_support=np.zeros(n+1)),requests)
    with pytest.raises(ValueError):PublicMobileTargetTrajectory(meta,dict(public,joint_position=np.zeros((n+1,7))),requests)
    shifted={k:v.copy() for k,v in public.items()};shifted['elapsed_seconds']+=.02
    with pytest.raises(ValueError):PublicMobileTargetTrajectory(meta,shifted,requests)
    m=json.loads((tmp_path/'buffer/public.json').read_text());m['mode']=1;(tmp_path/'buffer/public.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='SHA'):PublicMobileTargetTrajectory.load(tmp_path/'buffer')
