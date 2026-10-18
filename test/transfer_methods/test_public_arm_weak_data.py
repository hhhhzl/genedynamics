"""Task-bound arm weak logs retain complete T2 prefixes and reject task mixing."""
import numpy as np
import pytest
from genedynamics.learning.datasets.transfer.weak_arm import PublicArmWeakTrajectory,VERSION
def test_deadline_schema_cold_prefix_and_explicit_task_binding(tmp_path):
    n=900;dims=dict(joint_position=(7,),joint_velocity=(7,),tcp_position=(3,),tcp_rotation=(3,3),tcp_twist=(6,),measured_wrench=(6,),goal_position=(3,),goal_rotation=(3,3),previous_action=(13,),previous_request=(13,),reference_position=(3,),reference_rotation=(3,3),elapsed_seconds=(1,),body_position=(3,),body_twist=(6,),controller_feedback=(2,))
    p={k:np.zeros((n+1,*v),np.float32) for k,v in dims.items()};p['elapsed_seconds'][:,0]=np.arange(n+1)*.02;a=np.zeros((n,13),np.float32);m=dict(version=VERSION,permission='B_log',task='T2',robot='xarm7',mode=0,seed=0,purpose='fit',chart='shared13',controls=n,recording_end='terminal')
    row=PublicArmWeakTrajectory(m,p,a);row.save(tmp_path/'buffer');assert PublicArmWeakTrajectory.load(tmp_path/'buffer').digest==row.digest;assert row.prefix(750).metadata['recording_end']=='budget_stop'
    for bad in (dict(m,task='T3'),dict(m,task='T1'),dict(m,task='T4'),dict(m,servo_gain=.65),dict(m,permission='C_supervised')):
        with pytest.raises(ValueError):PublicArmWeakTrajectory(bad,p,a)
    with pytest.raises(ValueError):PublicArmWeakTrajectory(m,dict(p,body_qpos=np.zeros(n+1)),a)
