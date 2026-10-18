import numpy as np
import pytest

pytest.importorskip("mujoco")

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.experiments.plugins.methods.transfer.acp import ACPSharedControllerAdapter, Native19Demonstrator


class Teacher:
    def __init__(self, action): self.action=np.asarray(action,dtype=np.float32)
    def act(self, public): return self.action.copy()


@pytest.mark.parametrize("task,robot", [("T1","xarm7"),("T2","xarm7"),("T3","xarm7")])
def test_native19_teacher_executes_through_same_public_adapter(task,robot):
    physical=instantiate_task(task,robot=robot);public=physical.reset((72000,),seed=72000);physical.set_modes([0])
    action=np.zeros(13,np.float32);action[0]=.01;action[6:12]=.2
    if task!="T1":action[12]=2.
    actor=Native19Demonstrator(physical,Teacher(action))
    adapted=actor.act(public)
    assert adapted.shape==(1,13) and len(actor.requests)==1 and actor.requests[0].shape==(19,)
    np.testing.assert_allclose(adapted[0,:6],action[:6],atol=1e-5)
    assert adapted[0,12]==pytest.approx(action[12],abs=1e-5)
    step=physical.step(adapted)
    assert step.observation["elapsed_seconds"][0,0]==pytest.approx(.02)


def test_adapter_rejects_nonpositive_predicted_stiffness():
    task=instantiate_task("T3",robot="xarm7");public=task.reset((72001,),seed=72001)
    adapter=ACPSharedControllerAdapter(task.spec);request,prediction=adapter.encode_teacher_request(np.zeros((1,13),np.float32),public)
    prediction["stiffness_tool"][0,0,0,0,0]=-1.
    with pytest.raises(ValueError,match="positive"):adapter.decode(prediction,public)
