"""P0--P4 acceptance gates for the single-arm MuJoCo/MJX PegInsert task.

This is the only probe entry point for insertion.  It deliberately uses no
probe YAMLs: task physics, impedance, MGA hooks, hidden execution mismatch,
metrics, and the canonical algorithm configs are validated in one place.
"""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from genedynamics.envs.factories import make_env
from genedynamics.experiments.plugins.metrics.extractors import peg_insert_metrics_plugin
from genedynamics.solvers.common.receding_horizon import RecedingHorizonResult
from genedynamics.experiments.plugins.methods.contact_receding import (
    INSERT_TASK,
    make_controller,
    make_mga,
)
from genedynamics.experiments.framework.config import ExperimentConfig


SMALL = dict(
    Hsample=4,
    Hnode=2,
    Nsample=8,
    Ndiffuse_init=2,
    Ndiffuse=1,
    temp_sample=0.1,
    action_limit=1.0,
    dt=0.02,
    ctrl_dt=0.02,
    seed=0,
)


def test_reward_does_not_discount_unfinished_insertion_when_misaligned():
    from genedynamics.envs.domains.manipulation.peg_insert_brax import (
        PegInsertConfig, PegInsertEnv,
    )

    # Exercise the actual task reward without allocating a physics pipeline.
    cfg = PegInsertConfig(w_lateral=0.0, w_orientation=0.0, w_force=0.0)
    env = SimpleNamespace(_config=cfg, spec=SimpleNamespace(s_slice=slice(6, 12)))
    info = {"jammed": jnp.float32(0.0), "success": jnp.float32(0.0)}
    wrench = dict(axial_force=0.0, lateral_force=0.0, bending_torque=0.0)
    reward = lambda x, depth, angle: PegInsertEnv._reward(
        env, info, jnp.array([x, 0.0, depth]), jnp.array([angle, 0.0, 0.0]),
        wrench, jnp.zeros(13),
    )
    for depth in (-cfg.approach_gap, 0.0, 0.015):
        assert float(reward(0.0, depth, 0.0)) > float(reward(0.004, depth, 0.0))
        assert float(reward(0.0, depth, 0.0)) > float(reward(0.0, depth, 0.1))
    assert float(reward(0.0, 0.025, 0.0)) > float(reward(0.0, 0.015, 0.0))


def _force_raw(force, env):
    cfg = env._config
    lo, hi = cfg.f_min - cfg.f_cmd_pad, cfg.f_max + cfg.f_cmd_pad
    return 2.0 * (force - lo) / (hi - lo) - 1.0


def _action(env, *, x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0, force=12.0):
    return (
        jnp.zeros(env.action_size)
        .at[:6]
        .set(jnp.asarray([x, y, z, rx, ry, rz]))
        .at[-1]
        .set(_force_raw(force, env))
    )


def _roll(env, state, actions):
    step = jax.jit(env.step)
    states = [state]
    for action in actions:
        state = step(state, action)
        states.append(state)
    jax.block_until_ready(state.obs)
    return state, states


def test_atacom_viability_constraint_has_contact_action_authority():
    """PegInsert must expose the dynamics Jacobian required by ATACOM.

    The ordinary reporting residual is intentionally a measurement of the
    current physical wrench and therefore has zero action Jacobian.  ATACOM's
    task-owned viability chart must instead forecast that wrench through the
    nominal transition, as required by its controllable-state construction.
    """
    env = make_env(INSERT_TASK, level="wide", atacom_viability_gain=0.02)
    state = env.reset(jax.random.PRNGKey(91))
    insertion = _action(env, z=1.0, force=12.0)
    _, states = _roll(env, state, [insertion] * 25)
    state = next(
        candidate for candidate in states
        if int(candidate.info["contact_count"]) > 0
    )
    zero = jnp.zeros((env.action_size,), jnp.float32)
    static_jacobian = jax.jacfwd(
        lambda u: env.constraint_residual(state, u)[1]
    )(zero)
    viability_jacobian = jax.jacfwd(
        lambda u: env.atacom_constraint_residual(state, u)[1]
    )(zero)
    jax.block_until_ready(viability_jacobian)
    np.testing.assert_allclose(static_jacobian, 0.0, atol=1.0e-8)
    assert np.all(np.isfinite(np.asarray(viability_jacobian)))
    assert float(jnp.linalg.norm(viability_jacobian)) > 1.0e-4


def test_atacom_peg_exploration_rollout_remains_finite():
    from genedynamics.solvers.single.atacom.wrapper import AtacomEnvWrapper

    env = make_env(INSERT_TASK, level="wide", atacom_viability_gain=0.02)
    wrapped = AtacomEnvWrapper(env, Kc=1.0, action_limit=1.0)
    state = wrapped.reset(jax.random.PRNGKey(92))
    step = jax.jit(wrapped.step)
    key = jax.random.PRNGKey(93)
    for index in range(64):
        key, sample = jax.random.split(key)
        alpha = jax.random.uniform(
            sample, (wrapped.action_size,), minval=-1.0, maxval=1.0
        )
        state = step(state, alpha)
        jax.block_until_ready(state.obs)
        finite = (
            bool(jnp.all(jnp.isfinite(state.obs)))
            and bool(jnp.isfinite(state.reward))
            and bool(jnp.all(jnp.isfinite(state.info["atacom_u"])))
            and bool(jnp.all(jnp.isfinite(state.info["atacom_s"])))
        )
        assert finite, f"ATACOM became non-finite at exploration step {index}"


def test_atacom_terminal_manifold_unloads_after_success():
    from genedynamics.solvers.single.atacom.backends.atacom_jax import (
        atacom_null_dim,
        init_slack,
        make_atacom_transform,
    )

    env = make_env(INSERT_TASK, level="wide", atacom_viability_gain=0.02)
    state = env.reset(jax.random.PRNGKey(94))
    state = state.replace(info={
        **state.info,
        "success": jnp.asarray(1.0, jnp.float32),
    })
    transform = jax.jit(make_atacom_transform(
        env, Kc=1.0, time_step=env.dt, action_limit=1.0,
    ))
    control, _ = transform(
        state,
        jnp.zeros((atacom_null_dim(env),), jnp.float32),
        init_slack(env, state),
    )
    jax.block_until_ready(control)
    assert float(control[2]) < -0.20
    assert float(env._force_cmd(control[env.spec.nu_slice][0])) < 1.0


def _p0_physics():
    env = make_env(INSERT_TASK, level="wide")
    state = env.reset(jax.random.PRNGKey(0))
    socket_types = np.asarray(env._mj_model.geom_type)[np.asarray(env._socket_geoms)]
    contract = (
        env.action_size == 13
        and state.obs.shape == (env.observation_size,)
        and len(env._socket_geoms) >= 9
        and np.all(socket_types == mujoco.mjtGeom.mjGEOM_BOX)
        and int(state.info["contact_count"]) == 0
    )

    rim = make_env(
        INSERT_TASK,
        level="wide",
        approach_gap=0.0,
        hole_position_offset=(0.009, 0.0),
    )
    mj_state = rim.reset(jax.random.PRNGKey(0))
    mjx_wrench, mjx_count, _ = rim._contact_wrench(mj_state.pipeline_state)
    native = mujoco.MjData(rim._mj_model)
    q = np.asarray(rim._home_qpos, np.float64)
    native.qpos[: q.size] = q
    mujoco.mj_forward(rim._mj_model, native)
    native_normal = 0.0
    for index in range(native.ncon):
        wrench = np.zeros(6, np.float64)
        mujoco.mj_contactForce(rim._mj_model, native, index, wrench)
        native_normal += abs(float(wrench[0]))
    contact_parity = (
        int(mjx_count) > 0
        and native.ncon > 0
        and native_normal > 1.0e-4
        and float(jnp.linalg.norm(mjx_wrench[:3])) > 1.0e-4
    )

    traces = []
    for timestep in (0.002, 0.001):
        probe = make_env(INSERT_TASK, level="wide", timestep=timestep)
        x0 = probe.reset(jax.random.PRNGKey(1))
        action = _action(probe, z=1.0, force=12.0)
        final, states = _roll(probe, x0, [action] * 25)
        depths = np.asarray([float(s.info["prev_depth"]) for s in states[1:]])
        axial = np.asarray(
            [float(probe.wrench_components(s.info["true_wrench"])["axial_force"]) for s in states[1:]]
        )
        traces.append((float(np.max(depths)), float(np.max(axial))))
    depth_rel = abs(traces[0][0] - traces[1][0]) / max(traces[1][0], 1.0e-6)
    force_rel = abs(traces[0][1] - traces[1][1]) / max(traces[1][1], 1.0)
    convergence = depth_rel < 0.05 and force_rel < 0.10
    ok = contract and contact_parity and convergence
    print(
        f"[P0] contract={contract} native/mjx-contact={contact_parity} "
        f"dt traces={np.round(traces, 4).tolist()} rel=({depth_rel:.3f},{force_rel:.3f}) "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def _p1_impedance_and_recovery():
    env = make_env(INSERT_TASK, level="wide")
    force_steps = {}
    force_steps_ok = True
    for force in (5.0, 10.0, 15.0):
        probe_state = env.reset(jax.random.PRNGKey(int(force)))
        probe_action = _action(env, z=1.0, force=force)
        _, probe_states = _roll(env, probe_state, [probe_action] * 25)
        axial = np.asarray([
            float(env.wrench_components(s.info["true_wrench"])["axial_force"])
            for s in probe_states[1:]
        ])
        lateral = np.asarray([
            float(env.wrench_components(s.info["true_wrench"])["lateral_force"])
            for s in probe_states[1:]
        ])
        force_steps[int(force)] = (float(np.max(axial)), float(np.max(lateral)))
        force_steps_ok &= bool(
            np.all(np.isfinite(axial))
            and np.all(np.isfinite(lateral))
            and np.max(axial) < 2.0 * env._config.f_max
            and np.max(lateral) < 2.0 * env._config.lateral_force_limit
        )

    x0 = env.reset(jax.random.PRNGKey(2))
    centered = _action(env, z=1.0, force=12.0)
    final, states = _roll(env, x0, [centered] * 45)
    max_depth = max(float(s.info["max_depth"]) for s in states)
    centered_ok = max_depth >= env._config.success_depth and bool(final.info["success"] > 0.5)

    # Deliberately hit the rim laterally, retract with zero force, recenter, and
    # insert.  This is an environment-feasibility oracle, not a learned policy.
    jam_env = make_env(
        INSERT_TASK,
        level="tight",
        jam_dwell_steps=2,
        jam_force_fraction=0.20,
        jam_progress_tol=0.0005,
    )
    s0 = jam_env.reset(jax.random.PRNGKey(3))
    hit = _action(jam_env, x=1.0, z=1.0, force=15.0)
    retract = _action(jam_env, x=-1.0, z=-1.0, force=0.0)
    recenter = _action(jam_env, x=-1.0, z=0.0, force=0.0)
    insert = _action(jam_env, z=1.0, force=10.0)
    schedule = [hit] * 12 + [retract] * 8 + [recenter] * 4 + [insert] * 42
    recovered_final, recovery_states = _roll(jam_env, s0, schedule)
    jam_seen = max(float(s.info["jammed_once"]) for s in recovery_states) > 0.5
    recovery_seen = max(float(s.info["recovered"]) for s in recovery_states) > 0.5
    recovery_depth = max(float(s.info["max_depth"]) for s in recovery_states)
    recovery_ok = jam_seen and recovery_seen and recovery_depth >= 0.025
    ok = force_steps_ok and centered_ok and recovery_ok
    print(
        f"[P1] force steps={force_steps} finite/bounded={force_steps_ok}; "
        f"centered success={centered_ok} depth={max_depth:.4f}; "
        f"jam={jam_seen} recovered={recovery_seen} recovery_depth={recovery_depth:.4f} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok, recovery_states


def _p2_mga_geometry(recovery_states):
    env, solver = make_mga(
        INSERT_TASK,
        "mga_controllable_gate",
        level="wide",
        env_overrides={"realization_probe_horizon": 1},
        prior_fallback_mode="receding_incumbent",
        **SMALL,
    )
    state = env.reset(jax.random.PRNGKey(4))
    nodes = jnp.zeros((SMALL["Hnode"] + 1, env.action_size))
    dense = solver._get_backend_impl().spline.node2u(nodes)
    prepared = env.prepare_realization_context(state, dense)
    B = np.asarray(prepared.info["_mga_realization_B"])
    residual = env.manifold_residual_horizon_controllable(prepared, dense, 0.0)
    geometry = solver.geometry_fn(prepared, nodes, 0.0)
    h_soft, g_soft = env.soft_feasibility_residual(state, nodes[0])
    emergency = env.emergency_plan(state, nodes, 0.0)
    shifted_emergency = solver._get_backend_impl().shift(emergency)
    shifted_ordinary = solver._get_backend_impl().shift(nodes)
    emergency_delta, emergency_s, emergency_force = env._unpack(emergency[0])
    contact_state = state.replace(
        info={**state.info, "contact_count": jnp.int32(1)}
    )
    contact_emergency = env.emergency_plan(contact_state, nodes, 0.0)
    contact_delta, _, contact_force = env._unpack(contact_emergency[0])
    stalled_state = contact_state.replace(info={
        **contact_state.info, "stall_steps": jnp.int32(1)
    })
    stalled_emergency = env.emergency_plan(stalled_state, nodes, 0.0)
    stalled_delta, _, _ = env._unpack(stalled_emergency[0])
    unsafe_wrench = state.info["measured_wrench"].at[0].set(
        1.1 * env._config.lateral_force_limit
    )
    unsafe_state = state.replace(
        info={**state.info, "measured_wrench": unsafe_wrench}
    )
    unsafe_emergency = env.emergency_plan(unsafe_state, nodes, 0.0)
    unsafe_delta, _, _ = env._unpack(unsafe_emergency[0])
    loaded_wrench = state.info["measured_wrench"].at[0].set(
        0.9 * env._config.lateral_force_limit
    )
    loaded_state = state.replace(
        info={**state.info, "measured_wrench": loaded_wrench}
    )
    loaded_emergency = env.emergency_plan(loaded_state, nodes, 0.0)
    loaded_delta, _, _ = env._unpack(loaded_emergency[0])
    out_of_range = jnp.full_like(nodes, 2.0)
    stable_projection = env.project_mga_candidate(state, out_of_range)
    dropout_state = state.replace(info={
        **state.info,
        "step": jnp.int32(env._config.realization_probe_horizon),
        "prev_depth": jnp.float32(0.5 * env._config.socket_depth),
        "contact_count": jnp.float32(0.0),
        "contact_volatility": jnp.float32(1.0),
    })
    dropout_projection = env.project_mga_candidate(
        dropout_state, out_of_range
    )
    reentry_state = dropout_state.replace(info={
        **dropout_state.info,
        "contact_count": jnp.float32(4.0),
        "contact_count_delta": jnp.float32(4.0),
    })
    reentry_projection = env.project_mga_candidate(
        reentry_state, out_of_range
    )
    partial_release_state = reentry_state.replace(info={
        **reentry_state.info,
        "contact_count": jnp.float32(4.0),
        "contact_count_delta": jnp.float32(-4.0),
    })
    partial_release_projection = env.project_mga_candidate(
        partial_release_state, out_of_range
    )
    stable_contact_state = reentry_state.replace(info={
        **reentry_state.info, "contact_count_delta": jnp.float32(0.0)
    })
    stable_contact_projection = env.project_mga_candidate(
        stable_contact_state, out_of_range
    )
    shallow_switch_state = dropout_state.replace(info={
        **dropout_state.info,
        "prev_depth": jnp.float32(env._config.chamfer_depth),
    })
    shallow_switch_projection = env.project_mga_candidate(
        shallow_switch_state, out_of_range
    )
    projection_contract = (
        float(jnp.max(jnp.abs(stable_projection))) <= 1.0 + 1.0e-6
        and np.isclose(float(stable_projection[0, 2]), 1.0)
        and np.isclose(float(stable_projection[1, 2]), 1.0)
        and np.isclose(
            float(dropout_projection[0, 2]),
            env._config.contact_switch_axial_action_limit,
        )
        and np.isclose(float(dropout_projection[1, 2]), 1.0)
        and np.isclose(
            float(reentry_projection[0, 2]),
            env._config.contact_switch_axial_action_limit,
        )
        and np.isclose(
            float(partial_release_projection[0, 2]),
            env._config.contact_switch_axial_action_limit,
        )
        and np.isclose(
            float(shallow_switch_projection[0, 2]),
            1.0,
        )
        and np.isclose(float(stable_contact_projection[0, 2]), 1.0)
        and solver.candidate_projection_fn is not None
    )
    emergency_contract = (
        emergency.shape == nodes.shape
        and np.isclose(float(emergency_delta[2]), 0.0, atol=1.0e-7)
        and np.allclose(
            np.asarray(emergency_delta[jnp.asarray([0, 1, 3, 4, 5])]),
            0.0,
            atol=1.0e-7,
        )
        and np.isclose(float(contact_delta[2]), 0.0, atol=1.0e-7)
        and np.allclose(
            np.asarray(jnp.concatenate([
                contact_emergency[:2, :2],
                contact_emergency[:2, 3:6],
            ], axis=-1)),
            0.0,
            atol=1.0e-7,
        )
        and bool(env.emergency_plan_should_override(stalled_state))
        and not bool(env.emergency_plan_should_override(state))
        and float(stalled_delta[2]) < 0.0
        and float(unsafe_delta[2]) < 0.0
        and float(jnp.linalg.norm(
            loaded_delta[jnp.asarray([0, 1, 3, 4, 5])]
        ))
        <= float(jnp.linalg.norm(
            emergency_delta[jnp.asarray([0, 1, 3, 4, 5])]
        ))
        and float(jnp.max(jnp.abs(emergency[:, :6]))) <= 1.0 + 1.0e-6
        and float(jnp.max(jnp.abs(loaded_emergency[:, :6]))) <= 1.0 + 1.0e-6
        and np.isclose(
            float(env._unpack(contact_emergency[-1])[0][2]), 0.0, atol=1.0e-7
        )
        and np.isclose(float(emergency_force), env._config.f_min, atol=1.0e-6)
        and np.isclose(float(contact_force), env._config.f_min, atol=1.0e-6)
        and np.isclose(
            float(np.asarray(emergency_s)[0]),
            env._config.s_ref_diag,
            atol=1.0e-6,
        )
        and np.isclose(
            float(np.asarray(emergency_s)[3]),
            env._config.s_ref_diag,
            atol=1.0e-6,
        )
        and float(np.asarray(emergency_s)[5]) < env._config.s_ref_diag
        and not bool(env.emergency_plan_is_active(shifted_emergency))
        and np.allclose(
            np.asarray(shifted_ordinary),
            np.asarray(solver._get_backend_impl().spline.shift_nodes(nodes)),
        )
    )
    delayed = make_env(INSERT_TASK, level="wide", action_delay_steps=1)
    delayed_state = delayed.reset(jax.random.PRNGKey(41))
    delayed_memory_state = delayed_state.replace(info={
        **delayed_state.info,
        "prev_depth": jnp.float32(0.5 * delayed._config.socket_depth),
        "contact_count": jnp.float32(4.0),
        "contact_count_delta": jnp.float32(0.0),
        "contact_volatility": jnp.float32(0.2),
    })
    # The backend owns hooks from the nominal model env but passes the real
    # execution state.  Verify that delay metadata therefore crosses that
    # model/execution boundary instead of being read from ``env._config``.
    delayed_memory_projection = env.project_mga_candidate(
        delayed_memory_state, out_of_range
    )
    delayed_retract = out_of_range.at[0, 2].set(-0.25)
    delayed_retract_projection = env.project_mga_candidate(
        delayed_memory_state, delayed_retract
    )
    # A node submitted under action delay executes only after the commands
    # already in the queue.  Near the bore, projection must constrain that
    # accumulated command target without freezing axial progress or future
    # receding-horizon nodes.
    queued_capture = delayed_state.info["action_queue"].at[0, 0].set(0.2)
    capture_command = delayed_state.info["command_pose"].at[0].set(4.0e-4)
    capture_command = capture_command.at[2].set(4.0e-3)
    capture_state = delayed_state.replace(info={
        **delayed_state.info,
        "step": jnp.int32(delayed._config.realization_probe_horizon),
        "action_queue": queued_capture,
        "command_pose": capture_command,
    })
    capture_nodes = jnp.ones_like(nodes)
    capture_projection = delayed.project_mga_candidate(
        capture_state, capture_nodes
    )
    capture_pose, _ = delayed._actual_pose(capture_state.pipeline_state)
    capture_pose_scale = jnp.concatenate([
        delayed._translation_step,
        jnp.full((3,), delayed._config.rotation_step),
    ])
    capture_precommand = (
        capture_command
        + queued_capture[0, delayed.spec.r_slice] * capture_pose_scale
    )
    capture_target = (
        capture_precommand
        + capture_projection[0, delayed.spec.r_slice] * capture_pose_scale
    )
    capture_command_radius = jnp.maximum(
        delayed._capture_lateral_radius(capture_target[2])
        - jnp.linalg.norm(capture_pose[:2] - capture_command[:2]),
        0.0,
    )
    contact_state = state.replace(
        info={**state.info, "contact_count": jnp.int32(1)}
    )
    near_contact_state = state.replace(
        info={**state.info, "prev_depth": jnp.float32(1.0e-4)}
    )
    near_contact_emergency = env.emergency_plan(
        near_contact_state, nodes, 0.0
    )
    committed_correction = jnp.asarray([0, 1, 3, 4, 5])
    commit_contract = (
        int(env.risk_commit_horizon(state, SMALL["Hsample"] + 1)) == 1
        and int(delayed.risk_commit_horizon(
            delayed_state, SMALL["Hsample"] + 1
        )) == 2
        and int(env.risk_commit_horizon(
            delayed_state, SMALL["Hsample"] + 1
        )) == 2
        and int(env.risk_commit_horizon(
            delayed_memory_state, SMALL["Hsample"] + 1
        )) == 4
        and int(env.risk_commit_horizon(
            contact_state, SMALL["Hsample"] + 1
        )) == env._config.realization_probe_horizon
        and int(env.risk_commit_horizon(
            near_contact_state, SMALL["Hsample"] + 1
        )) == env._config.realization_probe_horizon
        and np.isclose(
            float(delayed_memory_projection[0, 2]),
            0.0,
        )
        and np.allclose(
            np.asarray(delayed_memory_projection[0, [0, 1, 3, 4, 5]]),
            0.0,
        )
        and np.allclose(
            np.asarray(delayed_memory_projection[1, [0, 1, 3, 4, 5]]),
            1.0,
        )
        and np.isclose(float(delayed_memory_projection[1, 2]), 1.0)
        and np.isclose(float(delayed_retract_projection[0, 2]), -0.25)
        and float(jnp.linalg.norm(capture_target[:2]))
        <= float(capture_command_radius) + 1.0e-7
        and np.isclose(
            float(capture_projection[0, 2]),
            1.0,
        )
        and np.isclose(float(capture_projection[1, 0]), 1.0)
        and float(jnp.linalg.norm(
            near_contact_emergency[0, committed_correction]
        ))
        <= float(jnp.linalg.norm(
            emergency[0, committed_correction]
        )) + 1.0e-7
    )
    rank = np.linalg.matrix_rank(B, tol=1.0e-7)
    hooks = (
        residual.shape == ((SMALL["Hsample"] + 1) * 7,)
        and geometry.shape == nodes.shape
        and rank >= 5
        and h_soft.shape == (0,)
        and g_soft.shape == (4,)
        and np.all(np.isfinite(B))
        and np.all(np.isfinite(np.asarray(geometry)))
        and emergency_contract
        and projection_contract
        and commit_contract
    )

    # Deterministic backend probe: make both performance candidates unsafe in
    # the task-owned risk schema and verify that the generic acceptance layer
    # selects the task-owned executable horizon at both cold and steady replans.
    backend = solver._get_backend_impl()

    def synthetic_score_risk(s, actions, aug_lambda=0.0, aug_rho=0.0):
        del s, aug_lambda, aug_rho
        # The task-owned commit horizon covers the zero-force prefix; a future
        # recovery suffix is revalidated after the next measurement.
        safe = jnp.all(actions[:3, -1] < -0.5)
        violation = (~safe).astype(jnp.float32)
        return jnp.mean(actions[:, 2]), jnp.asarray(
            [violation, violation, violation, 0.0], jnp.float32
        )

    backend.score_risk_fn = synthetic_score_risk
    unsafe_incumbent = jnp.zeros_like(nodes).at[:, 2].set(1.0)
    unsafe_refined = jnp.zeros_like(nodes).at[:, 2].set(0.8)
    selected0, triage0 = backend._accept_refinement_jit(
        state, unsafe_incumbent, unsafe_refined, jnp.float32(0.0), None, emergency
    )
    selected1, triage1 = backend._accept_refinement_jit(
        state, unsafe_incumbent, unsafe_refined, jnp.float32(1.0), None, emergency
    )
    safe_refined = (
        emergency.at[:, env.spec.s_slice].set(0.0).at[:, 2].add(0.1)
    )
    recovered, recovery_triage = backend._accept_refinement_jit(
        state, emergency, safe_refined, jnp.float32(2.0), None, emergency
    )
    held, hold_triage = backend._accept_refinement_jit(
        state, emergency, unsafe_refined, jnp.float32(2.0), None, emergency
    )
    jam_override_state = state.replace(info={
        **state.info,
        "jammed": jnp.float32(1.0),
        "jammed_once": jnp.float32(1.0),
    })
    forced, forced_triage = backend._accept_refinement_jit(
        jam_override_state, emergency, safe_refined,
        jnp.float32(2.0), None, emergency,
    )

    jax.block_until_ready(
        (selected0, selected1, recovered, held, forced, triage0, triage1,
         recovery_triage, hold_triage, forced_triage)
    )
    fallback_triage = (
        np.allclose(np.asarray(selected0), np.asarray(emergency))
        and np.allclose(np.asarray(selected1), np.asarray(emergency))
        and float(triage0["emergency_selected"]) == 1.0
        and float(triage1["emergency_selected"]) == 1.0
        and float(triage1["incumbent_revalidated_safe"]) == 0.0
        and float(triage1["refined_revalidated_safe"]) == 0.0
        and float(triage1["emergency_revalidated_safe"]) == 1.0
        and float(triage1["selected_revalidated_safe"]) == 1.0
        and float(triage1["emergency_unrecoverable"]) == 0.0
        and np.allclose(np.asarray(recovered), np.asarray(safe_refined))
        and float(recovery_triage["prior_accepted"]) == 1.0
        and float(recovery_triage["emergency_incumbent_active"]) == 0.0
        and np.allclose(np.asarray(held), np.asarray(emergency))
        and float(hold_triage["emergency_selected"]) == 0.0
        and np.allclose(np.asarray(forced), np.asarray(emergency))
        and float(forced_triage["emergency_selected"]) == 1.0
        and float(forced_triage["emergency_task_override"]) == 1.0
    )

    clear_gate = float(env.geometry_reliability(state)["scalar"])
    jam_state = max(
        recovery_states,
        key=lambda s: float(s.info["stall_steps"]),
    )
    # The recovery trajectory belongs to a shape-compatible task instance; use
    # its task-owned reliability to avoid mixing static socket frames.
    jam_env = make_env(
        INSERT_TASK, level="tight", jam_dwell_steps=2,
        jam_force_fraction=0.20, jam_progress_tol=0.0005,
    )
    jam_gate = float(jam_env.geometry_reliability(jam_state)["scalar"])
    gate_active = jam_gate < clear_gate - 1.0e-3
    ok = hooks and gate_active and fallback_triage
    print(
        f"[P2] B-rank={rank}/6 residual={residual.shape} geom={geometry.shape}; "
        f"gate clear={clear_gate:.3f} jam={jam_gate:.3f}; "
        f"emergency={emergency_contract} projection={projection_contract} "
        f"triage={fallback_triage} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def _p3_hidden_execution():
    nominal = make_env(INSERT_TASK, level="wide", friction=0.6)
    execution = make_env(
        INSERT_TASK,
        level="wide",
        friction=0.25,
        action_delay_steps=1,
        sensor_delay_steps=1,
        force_sensor_bias=(1.5, -1.0, 2.0),
        torque_sensor_bias=(0.05, -0.05, 0.02),
    )
    sn = nominal.reset(jax.random.PRNGKey(5))
    se = execution.reset(jax.random.PRNGKey(5))
    no_parameter_leak = np.allclose(np.asarray(sn.obs), np.asarray(se.obs), atol=1.0e-6)
    neutral_delay_queue = all(
        np.isclose(
            float(execution._unpack(queued)[2]),
            execution._config.f_min,
            atol=1.0e-6,
        )
        for queued in se.info["action_queue"]
    )
    action = _action(nominal, z=1.0, force=12.0)
    sn1 = jax.jit(nominal.step)(sn, action)
    se1 = jax.jit(execution.step)(se, action)
    delayed = float(se1.info["command_pose"][2]) < float(sn1.info["command_pose"][2]) - 1.0e-5

    model_task, solver = make_mga(
        INSERT_TASK,
        "mga_controllable_gate",
        level="wide",
        env_overrides={"realization_probe_horizon": 1},
        execution_env_overrides={
            "level": "ood",
            "action_delay_steps": 1,
            "sensor_delay_steps": 1,
        },
        **SMALL,
    )
    paired = solver.execution_env is not None and solver._step_fn is not None
    backend = solver._get_backend_impl()
    execution_task = solver.execution_env
    ood_capture_profile = (
        float(execution_task._capture_lateral_radius(-execution_task._config.approach_gap))
        > float(execution_task._capture_lateral_radius(execution_task._config.chamfer_depth))
        and np.isclose(
            float(execution_task._capture_lateral_radius(execution_task._config.chamfer_depth)),
            execution_task._config.ood_clearance_range[0],
            atol=1.0e-7,
        )
    )
    task_contract = (
        backend._task_contract_env is model_task
        and getattr(solver.candidate_projection_fn, "__self__", None)
        is model_task
        and getattr(backend.reliability_sequence_feature_fn, "__self__", None)
        is model_task
        and ood_capture_profile
    )
    topology = (
        solver.execution_env.action_size == nominal.action_size
        and solver.execution_env.sys.q_size() == nominal.sys.q_size()
    )
    ok = (
        no_parameter_leak
        and neutral_delay_queue
        and delayed
        and paired
        and task_contract
        and topology
    )
    print(
        f"[P3] no-oracle-leak={no_parameter_leak} delay-active={delayed} "
        f"neutral-delay-queue={neutral_delay_queue} "
        f"ood-capture-profile={ood_capture_profile} "
        f"dual-step={paired} task-contract={task_contract} topology={topology} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def _p4_algorithms_metrics():
    root = Path("configs/arm/peg_insert")
    paths = [
        root / "main/mga.yaml",
        root / "ablation/no_rl_prior.yaml",
        root / "ablation/no_retraction.yaml",
        root / "baseline/issa.yaml",
        root / "baseline/atacom.yaml",
        root / "baseline/mppi.yaml",
        root / "baseline/dial.yaml",
        root / "baseline/pegasusflow.yaml",
    ]
    configs = [ExperimentConfig.from_yaml(path) for path in paths]
    names = {cfg.name for cfg in configs}
    suites = {suite["name"] for suite in configs[0].suites}
    config_ok = names == {
        "dial", "mppi", "pegasusflow", "issa", "atacom",
        "no_rl_prior", "no_retraction", "mga",
    } and suites == {
        "id_wide", "ood_pose", "ood_sensing"
    } and len(configs) == 8
    for cfg in configs:
        uses_policy = cfg.name in {
            "issa", "atacom", "mga", "no_retraction",
        }
        config_ok = config_ok and (
            bool(cfg.method_params.get("policy_ckpt")) == uses_policy
        )
    full = next(cfg for cfg in configs if cfg.name == "mga")
    config_ok = config_ok and "atacom_policy_ckpt" not in full.method_params
    config_ok = config_ok and full.method_params["prior_stochastic_samples"] == 8
    config_ok = config_ok and full.method_params["prior_atacom_samples"] == 0
    config_ok = config_ok and "reliability_ckpt" in full.method_params
    config_ok = config_ok and full.metadata["training"]["rl"]["num_timesteps"] == 200000

    plans = {}
    for method in ("dial", "mppi", "mga_controllable_gate"):
        env, solver = make_controller(
            INSERT_TASK,
            method,
            level="wide",
            env_overrides={"realization_probe_horizon": 1},
            **SMALL,
        )
        state = env.reset(jax.random.PRNGKey(6))
        backend = (
            solver._get_brax_backend_impl()
            if method == "mppi"
            else solver._get_backend_impl()
        )
        plan = backend.replan(
            state,
            backend.init_plan_var(),
            backend.make_schedule(1),
            jax.random.PRNGKey(7),
        )
        plans[method] = np.asarray(plan)
    planner_ok = all(
        value.ndim == 2
        and value.shape[1] == 13
        and value.shape[0] in {SMALL["Hnode"] + 1, SMALL["Hsample"] + 1}
        and np.all(np.isfinite(value))
        for value in plans.values()
    )

    env = make_env(INSERT_TASK, level="wide")
    x0 = env.reset(jax.random.PRNGKey(8))
    action = _action(env, z=1.0, force=12.0)
    final, states = _roll(env, x0, [action] * 12)
    result = RecedingHorizonResult(states=states, actions=[action] * 12, plan_vars=[], infos=[])
    metrics = peg_insert_metrics_plugin().compute(
        result, env, None, None, x0=x0, planning_time=0.1
    )
    required = {
        "insertion_success", "safe_insertion_success",
        "prefix_safe_insertion_success", "max_insertion_depth", "completion_time",
        "peak_lateral_force", "peak_axial_force", "peak_bending_torque",
        "force_torque_violation_rate", "jam_rate", "recovery_success",
        "cumulative_safety_cost",
    }
    metrics_ok = required.issubset(metrics) and all(
        np.isfinite(metrics[k]) for k in required - {"recovery_success"}
    )
    ok = config_ok and planner_ok and metrics_ok
    print(
        f"[P4] algorithms={sorted(names)} suites={len(suites)} planners={planner_ok} "
        f"metrics={metrics_ok} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def main():
    test_reward_does_not_discount_unfinished_insertion_when_misaligned()
    p0 = _p0_physics()
    p1, recovery_states = _p1_impedance_and_recovery()
    p2 = _p2_mga_geometry(recovery_states)
    p3 = _p3_hidden_execution()
    p4 = _p4_algorithms_metrics()
    ok = p0 and p1 and p2 and p3 and p4
    print("RESULT:", "PEG INSERT P0-P4 OK" if ok else "PEG INSERT FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
