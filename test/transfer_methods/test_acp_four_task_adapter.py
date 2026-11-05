import numpy as np
import pytest

pytest.importorskip("mujoco")

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.tasks.transfer.controllers.acp import (
    ACPFourTaskActor, ACPSharedControllerAdapter, Native19Demonstrator,
)


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
    prediction["stiffness_scalar"][0,0,0]=-1.
    with pytest.raises(ValueError,match="positive"):adapter.decode(prediction,public)


class _FakeSolver:
    execution_horizon = 3

    def __init__(self):
        self.calls = 0

    def act(self, request):
        self.calls += 1
        return {"call": np.asarray(self.calls)}


class _FakeMemory:
    def __init__(self):
        self.ticks = []

    def update(self, public):
        self.ticks.append(int(round(float(public["elapsed_seconds"][0, 0]) / .02)))
        return np.zeros((1, 1), np.float32)


class _FakeAdapter:
    def __init__(self):
        self.waypoints = []

    def decode(self, prediction, public, *, waypoint=0):
        self.waypoints.append(waypoint)
        return np.asarray([[float(prediction["call"]), float(waypoint)]], np.float32)


def _chunk_actor():
    actor = ACPFourTaskActor.__new__(ACPFourTaskActor)
    actor.spec = type("Spec", (), {"config": type("Config", (), {"control_dt": .02})()})()
    actor.solver = _FakeSolver()
    actor.memory = _FakeMemory()
    actor.adapter = _FakeAdapter()
    actor._history_features = lambda history, length, controller_context=None: {}
    actor._pose = lambda position, rotation: np.eye(4)[None]
    actor.history = []
    actor.length = 1
    actor.context = None
    actor.inference_seconds = []
    actor.irreversible_network_calls = 0
    actor.last_tick = -1
    actor.cached_action = None
    actor.cached_prediction = None
    actor.chunk_control_step = 0
    actor.chunk_waypoint = 0.
    actor.execution_horizon = actor.solver.execution_horizon
    actor.action_down_sample_steps = 2
    actor.execution_control_steps = 6
    return actor


def test_acp_actor_executes_sparse_chunk_before_replanning(monkeypatch):
    import torch

    actor = _chunk_actor()
    actor.solver.device = torch.device("cpu")
    actor._rng = torch.get_rng_state()
    monkeypatch.setattr(
        actor, "_cache_prediction", lambda prediction: dict(prediction),
    )
    public = {
        "elapsed_seconds": np.zeros((1, 1)),
        "tcp_position": np.zeros((1, 3)),
        "tcp_rotation": np.eye(3)[None],
    }
    values = []
    for tick in range(8):
        public["elapsed_seconds"][:] = tick * .02
        values.append(actor.act(public)[0].tolist())
    assert values == [
        [1., 0.], [1., .5], [1., 1.], [1., 1.5], [1., 2.], [1., 2.5],
        [2., 0.], [2., .5],
    ]
    assert actor.solver.calls == actor.irreversible_network_calls == 2
    assert actor.memory.ticks == list(range(8))
    assert actor.adapter.waypoints == [0., .5, 1., 1.5, 2., 2.5, 0., .5]
    np.testing.assert_array_equal(actor.act(public), np.asarray([[2., .5]], np.float32))
    assert actor.solver.calls == 2 and actor.memory.ticks == list(range(8))


def test_acp_adapter_interpolates_sparse_pose_and_scalar():
    task = instantiate_task("T3", robot="xarm7")
    public = task.reset((72002,), seed=72002)
    adapter = ACPSharedControllerAdapter(task.spec)
    first_request, first = adapter.encode_teacher_request(
        np.zeros((1, 13), np.float32), public, scalar_stiffness=400.,
    )
    action = np.zeros((1, 13), np.float32)
    action[:, 0] = .1
    second_request, second = adapter.encode_teacher_request(
        action, public, scalar_stiffness=800.,
    )
    prediction = {
        key: np.concatenate((first[key], second[key]), axis=1)
        for key in first
    }
    interpolated = adapter.decode(prediction, public, waypoint=.5)
    assert interpolated[0, 0] == pytest.approx(.05, abs=1e-5)
    assert np.isfinite(interpolated).all()
    np.testing.assert_allclose(
        prediction["stiffness_scalar"].reshape(1, 2), [[400., 800.]],
    )
