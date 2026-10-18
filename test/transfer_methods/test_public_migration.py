"""Information boundaries and physical semantics of the v2 public adaptation."""
from copy import deepcopy
import numpy as np
import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.tasks.transfer.observation import PublicControllerMemory, context_sequence
from genedynamics.tasks.transfer.protocol import load_frozen_contract, current_contract, canonical
from genedynamics.solvers.single.acp.learning.task_training import response_chart
from genedynamics.solvers.single.acp.backends.compliance import decode_stiffness_tool
from genedynamics.solvers.single.acp.backends import spatial


@pytest.mark.parametrize("name", ["T1", "T3"])
def test_memory_is_causal_idempotent_and_checkpointable_on_real_public_prefix(name):
    task = instantiate_task(name, robot="panda")
    public = task.reset((19001, 19002), seed=19001)
    observer = PublicControllerMemory(task.spec, [0, 1])
    rows = [public]
    first = observer.update(public)
    np.testing.assert_array_equal(first, observer.update(deepcopy(public)))
    assert first.shape == (2, 53)
    for _ in range(12):
        public = task.step(np.zeros((2, 13))).observation
        rows.append(public)
        expected = observer.update(public)
    saved = observer.get_state()
    other = PublicControllerMemory(task.spec, [1, 0]); other.restore(saved)
    np.testing.assert_array_equal(expected, other.update(public))
    other.reorder([1, 0])
    np.testing.assert_array_equal(expected[[1, 0]], other.update({key: value[[1, 0]] for key, value in public.items()}))
    sequence = {key: np.stack([row[key][0] for row in rows]) for key in rows[0]}
    offline = context_sequence(sequence, task.spec, 0)
    np.testing.assert_allclose(offline[-1], expected[0], atol=1e-7)
    changed = deepcopy(sequence)
    changed["reference_position"][-1] += 1.
    np.testing.assert_array_equal(context_sequence(changed, task.spec, 0)[:-1], offline[:-1])
    # No executor or plant truth belongs to the observer state/interface.
    assert not any(key in vars(observer) for key in ("runner", "task", "provider", "mechanism"))
    _, digest = load_frozen_contract(name)
    assert len(digest) == 64
    assert canonical(current_contract(name)) == canonical(load_frozen_contract(name)[0])


def test_response_chart_roundtrip_preserves_native_anisotropic_stiffness_and_force():
    rng = np.random.default_rng(19000)
    ref = np.tile(np.eye(4), (8, 1, 1)); vt = ref.copy()
    ref[:, :3, 3] = rng.normal(0, .02, (8, 3))
    vt[:, :3, 3] = ref[:, :3, 3] + rng.normal(0, .01, (8, 3))
    k = np.array([200., 5000.] * 4)
    native = np.c_[spatial.SE3_to_pose9(ref), spatial.SE3_to_pose9(vt), k]
    decoded = response_chart(response_chart(native), inverse=True)
    np.testing.assert_allclose(decoded, native, atol=1e-12)
    before = decode_stiffness_tool(ref, vt, k, ref)
    after = decode_stiffness_tool(spatial.pose9_to_SE3(decoded[:, :9]), spatial.pose9_to_SE3(decoded[:, 9:18]), decoded[:, 18], ref)
    np.testing.assert_allclose(before, after, atol=1e-9)
    tensor = torch.tensor(native, dtype=torch.float64)
    torch.testing.assert_close(response_chart(response_chart(tensor), inverse=True), tensor)
    with pytest.raises(ValueError): response_chart(np.zeros((1, 19)))


def test_ours_actor_ignores_hidden_delayed_reference_and_bridge_reset_preserves_mode():
    from types import SimpleNamespace
    from genedynamics.tasks.transfer.behavior import PhaseGaussianPrior, behavior_reset, teacher_reference
    from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.source_transport import PublicCrossEBDMPolicy
    # A deterministic reference model isolates the actor/observation boundary;
    # genuine frozen bundles are exercised by the real batch1/4 workload runner.
    class ReferenceModel:
        model=SimpleNamespace(backend="native")
        metadata=dict(learned_role="phase_source_prior",reference_std=[0.]*5)
        def predict(self, x):
            indices=x.argmax(-1)
            memory=behavior_reset(len(x),indices%2)._replace(phase=indices//2)
            return teacher_reference(memory,goal_z=.580)[None]
    task=instantiate_task("T3",robot="panda")
    public=task.reset((19020,),seed=19020)
    policy=PublicCrossEBDMPolicy(task.spec,PhaseGaussianPrior(ReferenceModel()),modes=[1],seed=19020)
    before=policy.get_state(); expected=policy.act(public)
    task.state.reference_position[:] += .03
    task.state.reference_rotation[:] = np.eye(3)
    for key in public:
        np.testing.assert_array_equal(task.observe()[key],public[key])
    policy.restore(before)
    np.testing.assert_array_equal(policy.act(task.observe()),expected)
    policy.reset()
    assert policy.observer.modes.tolist()==[1]
    np.testing.assert_array_equal(policy.act(public),expected)
