"""Nominal task-space Coulomb arithmetic and exact zero-friction behavior."""
import numpy as np
import pytest
from genedynamics.tasks.transfer.controllers.guide_friction_feedforward import friction_wrench,PublicGuideFrictionFeedforwardPolicy
from genedynamics.tasks.transfer.controllers.guide_feedback import PublicGuideFeedbackPolicy
from genedynamics.tasks.transfer.specs import task_spec
from test.transfer_methods.test_public_observed_response import record


def test_coulomb_feedforward_units_direction_and_unactuated_nullspace():
    j=np.c_[np.eye(6)*.25,np.zeros(6)];v=np.asarray([.01,-.01,0,0,.02,0]);mu=np.ones(7);w=friction_wrench(j,v,mu);np.testing.assert_allclose(w,[4,-4,0,0,4,0],rtol=2e-5);np.testing.assert_array_equal(friction_wrench(j,v,np.zeros(7)),0.);np.testing.assert_array_equal(friction_wrench(j,np.zeros(6),mu),0.)
    with pytest.raises(ValueError):friction_wrench(j,v,-mu)


def test_panda_feedforward_is_exact_existing_public_controller_and_checkpointable():
    spec=task_spec('T2',robot='panda');a=PublicGuideFeedbackPolicy(spec,modes=[0],lead_xy_m=.008,position_admittance=True);b=PublicGuideFrictionFeedforwardPolicy(spec,mode=0);p,_=record()
    for i in range(12):
        public={k:v[i:i+1].copy() for k,v in p.items()};public['elapsed_seconds'][:]=i*.02;s=b.get_state();x=b.act(public);np.testing.assert_array_equal(x,a.act(public));b.restore(s);np.testing.assert_array_equal(b.act(public),x)
    assert b.irreversible_model_forwards==0
