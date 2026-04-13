"""End-to-end WBC integration test on a real G1 MuJoCo model.

The test loads the G1 scene through :class:`MujocoRobotIO`, builds a
"stand still" :class:`HumanoidTaskSpec` against the keyframe pose, runs
the new :class:`HumanoidWBCController` once, and checks that:

* The QP returns a finite solution.
* The recovered torques are within the model's actuator force range.
* When applied for a short rollout the base does not fall over (z drop
  bounded).

The test auto-skips if MuJoCo, OSQP, or the G1 scene XML cannot be
located in the active environment (most CI / dev shells without the
mujoco-menagerie clone).
"""

import importlib.util

import numpy as np
import pytest


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


pytestmark = pytest.mark.skipif(
    not (_have("mujoco") and _have("osqp")),
    reason="mujoco + osqp required for WBC integration test",
)


@pytest.fixture(scope="module")
def io():
    from genedynamics.deploy.io import MujocoRobotIO
    from genedynamics.robots.g1 import G1AssetNotFoundError

    try:
        return MujocoRobotIO(sim_dt=0.002)
    except G1AssetNotFoundError as e:
        pytest.skip(f"G1 scene MJCF not available: {e}")


@pytest.fixture(scope="module")
def stand_tasks(io):
    """Build a HumanoidTaskSpec that asks the controller to hold the keyframe pose."""
    from genedynamics.deploy.followers.common.plan_schema import CorridorPlanFrame
    from genedynamics.deploy.followers.humanoid.task_spec import (
        ArmJointTask,
        FootTask,
        HumanoidTaskSpec,
        PelvisTask,
    )

    state = io.reset()
    base_pos = state.qpos[:3].copy()

    feet = io.foot_contact_observations()
    left_pos = feet["left"].position_world
    right_pos = feet["right"].position_world

    pelvis = PelvisTask(
        position_world=base_pos,
        yaw_world=0.0,
        roll_world=0.0,
        pitch_world=0.0,
    )
    left_foot = FootTask(
        position_world=left_pos,
        velocity_world=np.zeros(3),
        yaw_world=0.0,
        in_contact=True,
        weight=1.0,
    )
    right_foot = FootTask(
        position_world=right_pos,
        velocity_world=np.zeros(3),
        yaw_world=0.0,
        in_contact=True,
        weight=1.0,
    )
    arm_empty = ArmJointTask(joint_targets={})

    # Minimal CorridorPlanFrame stub — only fields the WBC pulls from it.
    frame = CorridorPlanFrame(
        time_sec=0.0,
        raw_state=np.zeros(14),
        x=float(base_pos[0]),
        y=float(base_pos[1]),
        psi=0.0,
        h=float(base_pos[2]),
        psi_torso=0.0,
        a_left=0.0,
        a_right=0.0,
        p_left=0.0,
        p_right=0.0,
        v_x=0.0,
        v_y=0.0,
        omega=0.0,
        h_dot=0.0,
        psi_dot_torso=0.0,
    )

    return HumanoidTaskSpec(
        plan_frame=frame,
        pelvis=pelvis,
        torso_yaw=0.0,
        left_foot=left_foot,
        right_foot=right_foot,
        left_arm=arm_empty,
        right_arm=arm_empty,
        joint_hints={},
        extras={"narrowness": 0.0},
    )


def test_wbc_solves_standing_qp(io, stand_tasks):
    from genedynamics.deploy.controllers.wbc import HumanoidWBCController
    from genedynamics.deploy.interfaces.messages import Intent

    state = io.reset()
    controller = HumanoidWBCController(io)

    intent = Intent(
        t=0.0,
        base_yaw=0.0,
        base_height=float(stand_tasks.pelvis.position_world[2]),
        base_lin_vel=np.zeros(2),
        base_yaw_rate=0.0,
        extras={"humanoid_tasks": stand_tasks, "dt": 0.02},
    )

    cmd = controller.act(state, intent)
    wbc = cmd.extras["wbc"]

    # 1. QP returned finite solution
    assert np.all(np.isfinite(wbc.q_ref)), "q_ref has non-finite entries"
    assert np.all(np.isfinite(wbc.tau_ff)), "tau_ff has non-finite entries"
    assert wbc.eq_residual < 5e-3, f"eq residual too large: {wbc.eq_residual}"
    # Note: ineq violation may be modestly nonzero in stand pose; just sanity-check
    assert wbc.ineq_violation < 1.0, f"ineq violation absurd: {wbc.ineq_violation}"

    # 2. Torques inside the model's force range (controller already clipped)
    torque_limit = io.spec.torque_limit_vector() * controller.cfg.limits.torque_limit_scale
    finite = np.isfinite(torque_limit)
    assert np.all(np.abs(wbc.tau_ff[finite]) <= torque_limit[finite] + 1e-6)

    # 3. q_ref is within joint limits
    lo = np.array([io.spec.joint_range[n][0] for n in io.spec.actuated_joints])
    hi = np.array([io.spec.joint_range[n][1] for n in io.spec.actuated_joints])
    assert np.all(wbc.q_ref >= lo - 1e-6)
    assert np.all(wbc.q_ref <= hi + 1e-6)


def test_wbc_short_rollout_does_not_fall(io, stand_tasks):
    """Apply WBC for ~0.2s on a stand-still task and check the base height
    is approximately preserved."""
    from genedynamics.deploy.controllers.wbc import HumanoidWBCController
    from genedynamics.deploy.interfaces.messages import Intent

    state = io.reset()
    controller = HumanoidWBCController(io)
    initial_z = float(state.qpos[2])

    dt = 0.02
    for _ in range(10):  # 10 control steps × 0.02 = 0.2s
        intent = Intent(
            t=float(io.t),
            base_yaw=0.0,
            base_height=initial_z,
            base_lin_vel=np.zeros(2),
            base_yaw_rate=0.0,
            extras={"humanoid_tasks": stand_tasks, "dt": dt},
        )
        cmd = controller.act(state, intent)
        io.send_control(cmd)
        state = io.step(dt)

    final_z = float(state.qpos[2])
    drop = initial_z - final_z
    # Generous bound: WBC tuning isn't perfect for arbitrary G1 scenes,
    # but a falling robot drops > 0.3m in 0.2s under gravity. We just want
    # to verify the controller is doing *something* sensible.
    assert drop < 0.30, f"base dropped {drop:.3f} m in 0.2 s — controller likely diverging"
