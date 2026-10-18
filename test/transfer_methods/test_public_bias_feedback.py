"""Decoder compensation survives source-memory phase resets, with exact state."""
import numpy as np
import pytest

pytest.importorskip("mujoco")
from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackPolicy


def test_pose_compensation_is_preserved_across_alignment_insertion_only():
    task=instantiate_task("T3",robot="panda");public=task.reset((32000,),seed=32000)
    actor=PublicBiasFeedbackPolicy(task.spec)
    actor.observer.memory=actor.observer.memory._replace(phase=np.array([2]));public["elapsed_seconds"][:]=.02
    actor.compensation[0]=[-.01,.005,.003,.001,-.002,.003];actor.act(public)
    previous=actor.compensation.copy();state=actor.get_state()
    actor.observer.memory=actor.observer.memory._replace(phase=np.array([3]),tracking_integral=np.zeros((1,6)))
    public["elapsed_seconds"][:]=.04;actor.act(public)
    assert abs(actor.compensation[0,0]-previous[0,0])<.001 and actor.compensation[0,2]==0
    actor.restore(state);public["elapsed_seconds"][:]=.02;expected=actor.act(public);actor.restore(state)
    np.testing.assert_array_equal(actor.act(public),expected)
    assert not any(k in vars(actor) for k in ("task","mechanism","physics"))
