"""Arm surface-scan env (idea.txt Exp I, S1-S4) — docker, real brax.

Validates the rebuilt 10D arm primitive + NURBS surface families:
  * action_size == 10 (Δξ,Δη,Δψ + svec(3×3)=6 + F_n);
  * all families {plane, cylinder, s2(convex NURBS), s3(bumpy), s4(unseen+DR)}
    construct, reset, step finite;
  * constraint_residual -> h(6) [h_surf;h_normal] + g(2) [g_force], and g_force is
    NON-vacuous (nu=+1 commands a force above f_max -> g[0]>0);
  * manifold_geometry returns (Hnode+1, nu);
  * S4 domain randomization (surface stiffness / friction) varies across seeds;
  * MDAC ablations (no_softfeas / no_stiffness / no_tangent) still change the plan.

Invoke (from repo root):
  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
              python test/integration/test_arm_surface_docker.py"
"""

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.envs.factories import make_env
from genedynamics.envs.domains.manipulation.panda_brax import (
    PandaSurfaceScanDomainEnv,
)
from genedynamics.solvers.single.mdac.experiment import make_mdac, ARM_TASK

FAMILIES = ["plane", "cylinder", "convex", "bumpy", "unseen"]
CFG = dict(Hsample=8, Hnode=4, Nsample=64, Ndiffuse_init=3, Ndiffuse=2,
           temp_sample=0.1, action_limit=1.0, dt=0.02, ctrl_dt=0.02, seed=0)


def _check_family(fam):
    env = make_env(ARM_TASK, level=fam, surface_seed=1)
    x0 = env.reset(jax.random.PRNGKey(0))
    _, n0, p0, _ = env._desired_pose(env._xi0, env._eta0)
    ee0 = x0.pipeline_state.site_xpos[env._ee_site]
    reset_error = ee0 - p0
    reset_tangent = reset_error - jnp.dot(reset_error, n0) * n0
    reset_tangent_mm = float(jnp.linalg.norm(reset_tangent) * 1e3)
    a = jax.random.uniform(jax.random.PRNGKey(1), (env.action_size,), minval=-1.0, maxval=1.0)
    s1 = env.step(x0, a)
    finite = bool(jnp.all(jnp.isfinite(s1.obs)) and jnp.isfinite(s1.reward))
    h, g = env.constraint_residual(x0, a)
    ag = env.manifold_geometry(x0, jnp.zeros((CFG["Hnode"] + 1, env.action_size)), 0.0)
    ok = (env.action_size == 10 and finite and h.shape == (6,) and g.shape == (2,)
          and ag.shape == (CFG["Hnode"] + 1, 10)
          and reset_tangent_mm < 0.1)
    print(f"  [{fam:8s}] action={env.action_size} finite={finite} h={h.shape} g={g.shape} "
          f"geom={tuple(ag.shape)} k_surf={float(env._k_surf):.0f} mu={float(env._mu):.2f} "
          f"reset_tangent={reset_tangent_mm:.3f}mm "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


def _g_force_nonvacuous():
    env = make_env(ARM_TASK, level="plane")
    x0 = env.reset(jax.random.PRNGKey(0))
    a = jnp.zeros(env.action_size).at[-1].set(1.0)        # nu=+1 -> force above f_max
    _, g = env.constraint_residual(x0, a)
    print(f"  g_force at nu=+1: {np.round(np.asarray(g), 3)} (g[0]>0 == active constraint)")
    return float(g[0]) > 0.0


def _s4_dr_varies():
    e1 = make_env(ARM_TASK, level="unseen", surface_seed=1)
    e2 = make_env(ARM_TASK, level="unseen", surface_seed=2)
    d = abs(float(e1._k_surf) - float(e2._k_surf)) > 1.0 or abs(float(e1._mu) - float(e2._mu)) > 1e-3
    print(f"  S4 DR: seed1 (k={float(e1._k_surf):.0f},mu={float(e1._mu):.2f}) vs "
          f"seed2 (k={float(e2._k_surf):.0f},mu={float(e2._mu):.2f}) -> differs={d}")
    return d


def _soft_s4_compliance_varies():
    e1 = make_env(ARM_TASK, level="unseen", medium="soft", surface_seed=1)
    e2 = make_env(ARM_TASK, level="unseen", medium="soft", surface_seed=2)
    d = abs(e1._contact_stiffness_draw - e2._contact_stiffness_draw) > 1.0
    print(f"  soft S4 compliance: seed1 k={e1._contact_stiffness_draw:.0f} "
          f"seed2 k={e2._contact_stiffness_draw:.0f} -> differs={d}")
    return d


def _hybrid_map_crosses_finite_scan_segment():
    common = dict(
        level="plane",
        medium="hybrid",
        stiffness_map_xi_origin=0.1,
        stiffness_map_xi_span=0.0495,
        stiffness_transition_width=0.08,
        k_hard=8000.0,
        k_soft=2000.0,
    )
    stripes = make_env(ARM_TASK, stiffness_map="stripes", **common)
    striped = np.asarray([
        stripes._k_surf_fn(jnp.float32(x), jnp.float32(0.5))
        for x in (0.105, 0.117, 0.130, 0.142)
    ])
    center_hard = make_env(ARM_TASK, stiffness_map="center_hard", **common)
    centered = np.asarray([
        center_hard._k_surf_fn(jnp.float32(x), jnp.float32(0.5))
        for x in (0.105, 0.125, 0.149)
    ])
    state = stripes.reset(jax.random.PRNGKey(23))
    fallback = jnp.zeros((CFG["Hnode"] + 1, stripes.action_size))
    emergency = stripes.emergency_plan(state, fallback, 0.0)
    _, _, emergency_force = stripes._unpack(emergency[0])
    _, _, recovery_force = stripes._unpack(emergency[1])
    recovery_scan = float(emergency[1, stripes.spec.r_slice.start])
    emergency_stiffness_diag = np.asarray(
        emergency[0, stripes.spec.s_slice]
    )[[0, 3, 5]]
    no_stiff = make_env(
        ARM_TASK, stiffness_mode="none", stiffness_map="stripes", **common
    )
    _, s0, _ = no_stiff._unpack(jnp.zeros(no_stiff.action_size))
    _, s1, _ = no_stiff._unpack(jnp.ones(no_stiff.action_size))
    euclid = make_env(
        ARM_TASK, stiffness_mode="euclid", stiffness_map="stripes", **common
    )
    log_spd = make_env(
        ARM_TASK, stiffness_mode="log_spd", stiffness_map="stripes", **common
    )
    _, s_ref_e, _ = euclid._unpack(jnp.zeros(euclid.action_size))
    _, s_ref_l, _ = log_spd._unpack(jnp.zeros(log_spd.action_size))
    k_ref_e = np.asarray(euclid._stiffness(s_ref_e))
    k_ref_l = np.asarray(log_spd._stiffness(s_ref_l))
    ok = (
        striped[0] > 5000.0 and striped[1] < 5000.0
        and striped[2] > 5000.0 and striped[3] < 5000.0
        and centered[0] < 5000.0 and centered[1] > 5000.0
        and centered[2] < 5000.0
        and float(emergency_force) == 0.0
        and float(recovery_force) == 0.0
        and np.isclose(
            recovery_scan,
            stripes._config.scan_rate / stripes._config.coord_scale,
        )
        and np.array_equal(emergency_stiffness_diag, [1.0, 1.0, 1.0])
        and bool(stripes.emergency_plan_is_active(emergency))
        and bool(stripes.emergency_plan_is_active(emergency[1:]))
        and np.array_equal(no_stiff._stiffness(s0), no_stiff._stiffness(s1))
        and np.allclose(k_ref_e, k_ref_l, rtol=1.0e-5, atol=1.0e-4)
    )
    print(
        f"  finite-scan hybrid stripes={striped.tolist()} "
        f"center_hard={centered.tolist()} emergency_F={float(emergency_force):.1f} "
        f"recovery_F={float(recovery_force):.1f} "
        f"recovery_scan={recovery_scan:.3f} "
        f"-> {'OK' if ok else 'FAIL'}"
    )
    return ok


def _contact_friction_reaches_both_geoms():
    env = make_env(ARM_TASK, level="cylinder", friction=0.3)
    surf = int(env._surf_geom)
    probe = int(env._probe_geom)
    geom_friction = np.asarray(env.sys.geom_friction)
    ok = (
        abs(float(geom_friction[surf, 0]) - 0.3) < 1e-5
        and abs(float(geom_friction[probe, 0]) - 0.3) < 1e-5
    )
    print(
        f"  contact mu: probe={geom_friction[probe, 0]:.3f} "
        f"surface={geom_friction[surf, 0]:.3f} -> {'OK' if ok else 'FAIL'}"
    )
    return ok


def _robot_swap_contract():
    """One task/action contract must bind to two robot-only MJCFs."""
    results = []
    for robot in ("panda", "xarm7"):
        env = make_env(ARM_TASK, robot=robot, level="plane")
        state = env.reset(jax.random.PRNGKey(13))
        action = jnp.zeros(env.action_size)
        next_state = env.step(state, action)
        h, g = env.constraint_residual(state, action)
        valid = (
            env.action_size == 10
            and h.shape == (6,)
            and g.shape == (2,)
            and len(env._robot_binding.actuator_indices)
            == env._robot_profile.num_actuated
            and bool(jnp.isfinite(next_state.reward))
        )
        print(
            f"  robot swap {robot}: joints={env._robot_profile.num_actuated} "
            f"action={env.action_size} h={h.shape} g={g.shape} finite={valid}"
        )
        results.append(valid)
    return all(results)


def _rl_observation_risk_and_randomized_wrapper():
    common = dict(
        level="cylinder",
        observation_mode="rl_realized",
        scan_rate=0.0005,
        reward_path_scale=0.0495,
        reward_force_scale=5.0,
        deformation_scale=0.001,
    )
    e_lo = make_env(ARM_TASK, friction=0.05, medium="rigid", **common)
    e_hi = make_env(ARM_TASK, friction=0.20, medium="soft", **common)
    key = jax.random.PRNGKey(4)
    s_lo, s_hi = e_lo.reset(key), e_hi.reset(key)
    # Exact material parameters are not present in rl_realized observation:
    # before interaction, the two domains expose the same measurable state.
    no_oracle_leak = np.allclose(
        np.asarray(s_lo.obs), np.asarray(s_hi.obs), atol=1e-6
    )
    action = jnp.linspace(-0.2, 0.2, e_lo.action_size)
    s1 = e_lo.step(s_lo, action)
    previous_action_visible = np.allclose(
        np.asarray(s1.obs[-e_lo.action_size:]), np.asarray(action), atol=1e-6
    )
    risk = e_lo.sequence_risk(
        s_lo, jnp.broadcast_to(action, (CFG["Hsample"] + 1, e_lo.action_size))
    )

    mixed = PandaSurfaceScanDomainEnv([e_lo, e_hi])
    reset_jit = jax.jit(mixed.reset)
    step_jit = jax.jit(mixed.step)
    sm = reset_jit(jax.random.PRNGKey(9))
    sm1 = step_jit(sm, jnp.zeros(mixed.action_size))
    domain_index = int(sm.info["_rl_domain_index"])
    mixed_ok = (
        0 <= domain_index < 2
        and mixed.action_size == 10
        and bool(jnp.all(jnp.isfinite(sm1.obs)))
        and bool(jnp.isfinite(sm1.reward))
    )
    ok = (
        no_oracle_leak
        and previous_action_visible
        and risk.shape == (4,)
        and np.all(np.isfinite(np.asarray(risk)))
        and mixed_ok
    )
    print(
        f"  RL obs={s_lo.obs.shape} no_oracle_leak={no_oracle_leak} "
        f"prev_action={previous_action_visible} risk={np.round(np.asarray(risk), 3)} "
        f"mixed_domain={domain_index} -> {'OK' if ok else 'FAIL'}"
    )
    return ok


def _geometry_factors_are_independent():
    e_t, no_tangent = make_mdac(ARM_TASK, "mdac_no_tangent", level="plane", **CFG)
    e_r, no_retraction = make_mdac(ARM_TASK, "mdac_no_retraction", level="plane", **CFG)
    ok = (no_tangent.geometry_fn is None and no_tangent.retraction is not None
          and no_retraction.geometry_fn is not None and no_retraction.retraction is None)
    print(f"  no_tangent: geometry={no_tangent.geometry_fn is not None} "
          f"retraction={no_tangent.retraction is not None}; "
          f"no_retraction: geometry={no_retraction.geometry_fn is not None} "
          f"retraction={no_retraction.retraction is not None} -> {'OK' if ok else 'FAIL'}")
    return ok


def _position_only_manifold():
    env, _ = make_mdac(ARM_TASK, "mdac_position_only", level="plane", **CFG)
    x0 = env.reset(jax.random.PRNGKey(0))
    nodes = jnp.zeros((CFG["Hnode"] + 1, env.action_size))
    c = env.manifold_residual(x0, nodes)
    ok = c.shape == (2 * (CFG["Hnode"] + 1),)
    print(f"  position-only residual shape={c.shape} -> {'OK' if ok else 'FAIL'}")
    return ok


def _staged_geometry_routes():
    env_h, horizon = make_mdac(ARM_TASK, "mdac_horizon", level="plane", **CFG)
    env_s, scalar = make_mdac(ARM_TASK, "mdac_scalar_gate", level="plane", **CFG)
    env_c, component = make_mdac(
        ARM_TASK, "mdac_component_gate", level="plane", **CFG
    )
    x0 = env_c.reset(jax.random.PRNGKey(0))
    nodes = jnp.zeros((CFG["Hnode"] + 1, env_c.action_size))
    dense = component._get_backend_impl().spline.node2u(nodes)
    residual = env_c.manifold_residual_horizon(x0, dense, 0.0)
    gh = horizon.geometry_fn(x0, nodes, 0.0)
    gs = scalar.geometry_gate_fn(x0, nodes, 0.0)
    gc = component.geometry_gate_fn(x0, nodes, 0.0)
    expected_residual = (CFG["Hsample"] + 1) * 3  # xi, eta, force
    scalar_uniform = np.ptp(np.asarray(gs["action"])) < 1e-6
    component_nonuniform = np.ptp(np.asarray(gc["action"])) > 1e-6
    ok = (
        residual.shape == (expected_residual,)
        and gh.shape == nodes.shape
        and scalar.retraction is not None
        and component.retraction is not None
        and scalar_uniform
        and component_nonuniform
        and np.all((np.asarray(gc["action"]) >= 0.0) & (np.asarray(gc["action"]) <= 1.0))
    )
    print(
        f"  horizon residual={residual.shape} geom={gh.shape}; "
        f"scalar_uniform={scalar_uniform} component_nonuniform={component_nonuniform} "
        f"-> {'OK' if ok else 'FAIL'}"
    )
    return ok


def _realization_geometry_routes():
    env, solver = make_mdac(
        ARM_TASK, "mdac_realization", level="cylinder", **CFG
    )
    state = env.reset(jax.random.PRNGKey(0))
    # Acquire contact before measuring the task-owned frozen coordinate map.
    for _ in range(5):
        state = env.step(state, jnp.zeros(env.action_size))
    nodes = jnp.zeros((CFG["Hnode"] + 1, env.action_size))
    dense = solver._get_backend_impl().spline.node2u(nodes)
    offset = env.realization_coordinate_offset(state)
    Jq = env._surface_coordinate_jacobian(state.info["xi"], state.info["eta"])
    clean = env.manifold_residual_horizon(state, dense, 5.0)
    realized = env.manifold_residual_horizon_realized(state, dense, 5.0)
    geometry = solver.geometry_fn(state, nodes, 5.0)
    singular = np.linalg.svd(np.asarray(Jq), compute_uv=False)
    ok = (
        offset.shape == (2,)
        and Jq.shape == (3, 2)
        and np.all(np.isfinite(np.asarray(offset)))
        and singular[-1] > 1e-4
        and clean.shape == realized.shape
        and not np.allclose(np.asarray(clean), np.asarray(realized))
        and geometry.shape == nodes.shape
        and np.all(np.isfinite(np.asarray(geometry)))
    )
    print(
        f"  realization offset={np.round(np.asarray(offset), 4)} "
        f"surface-J singular={np.round(singular, 4)} "
        f"residual_changed={not np.allclose(np.asarray(clean), np.asarray(realized))} "
        f"-> {'OK' if ok else 'FAIL'}"
    )
    return ok


def _controllability_geometry_routes():
    env, solver = make_mdac(
        ARM_TASK,
        "mdac_controllable",
        level="cylinder",
        env_overrides={
            "realization_probe_horizon": 5,
            "realization_probe_eps": 0.02,
        },
        **CFG,
    )
    state = env.reset(jax.random.PRNGKey(0))
    nominal = jnp.zeros(env.action_size).at[0].set(0.01).at[-1].set(0.375)
    for _ in range(10):
        state = env.step(state, nominal)
    nodes = jnp.broadcast_to(
        nominal, (CFG["Hnode"] + 1, env.action_size)
    )
    dense = solver._get_backend_impl().spline.node2u(nodes)
    B = env.realization_control_jacobian(state, dense)
    prepared = env.prepare_realization_context(state, dense)
    residual = env.manifold_residual_horizon_controllable(
        prepared, dense, 10.0
    )
    reliability = env.geometry_reliability(state)
    gate_zero = prepared.replace(info={
        **prepared.info,
        "_mdac_realization_gate": jnp.zeros((2,), dtype=dense.dtype),
    })
    gated_residual = env.manifold_residual_horizon_controllable(
        gate_zero, dense, 10.0
    )
    geometry = solver.geometry_fn(prepared, nodes, 10.0)
    singular = np.linalg.svd(np.asarray(B), compute_uv=False)
    ok = (
        B.shape == (2, 2)
        and np.all(np.isfinite(np.asarray(B)))
        and singular[-1] > 1e-5
        # With reset/reference alignment repaired, increasing the xi command
        # must move the physical probe in increasing surface coordinates.
        and float(B[0, 0]) > 0.0
        and float(B[1, 1]) > 0.0
        and residual.shape == ((CFG["Hsample"] + 1) * 3,)
        and np.array_equal(
            np.asarray(reliability["clean_action"][:2]),
            np.ones((2,), dtype=np.float32),
        )
        and reliability["clean_action"].shape == (env.action_size,)
        and not np.allclose(np.asarray(residual), np.asarray(gated_residual))
        and geometry.shape == nodes.shape
        and np.all(np.isfinite(np.asarray(geometry)))
    )
    print(
        f"  controllability B={np.round(np.asarray(B), 5).tolist()} "
        f"singular={np.round(singular, 5)} -> {'OK' if ok else 'FAIL'}"
    )
    return ok


def _plan(env, sol, rng):
    b = sol._get_backend_impl()
    x0 = env.reset(jax.random.PRNGKey(1))
    return np.asarray(b.replan(x0, b.init_plan_var(), b.make_schedule(b.Ndiffuse_init), rng))


def _ablations_active():
    rng = jax.random.PRNGKey(0)
    env0, full = make_mdac(ARM_TASK, "mdac", level="bumpy", **CFG)
    base = _plan(env0, full, rng)
    ok = True
    for m in ("mdac_no_softfeas", "mdac_no_stiffness", "mdac_no_tangent"):
        e, s = make_mdac(ARM_TASK, m, level="bumpy", **CFG)
        d = float(np.max(np.abs(_plan(e, s, rng) - base)))
        ok &= d > 1e-4
        print(f"  mdac vs {m}: max|Δplan|={d:.3e} -> {'ACTIVE' if d > 1e-4 else 'NO-OP!'}")
    return ok


def main():
    print("== families ==")
    a = all(_check_family(f) for f in FAMILIES)
    print("== robot swap contract =="); b = _robot_swap_contract()
    print("== g_force non-vacuous =="); c = _g_force_nonvacuous()
    print("== unseen domain randomization =="); d = (
        _s4_dr_varies()
        and _soft_s4_compliance_varies()
        and _contact_friction_reaches_both_geoms()
    )
    print("== finite-scan hybrid map =="); d1 = (
        _hybrid_map_crosses_finite_scan_segment()
    )
    print("== RL observation/risk/shared domains =="); d2 = (
        _rl_observation_risk_and_randomized_wrapper()
    )
    print("== independent geometry factors =="); e = _geometry_factors_are_independent()
    print("== position-only manifold =="); f = _position_only_manifold()
    print("== staged horizon/gated geometry =="); g = _staged_geometry_routes()
    print("== frozen realization geometry =="); h = _realization_geometry_routes()
    print("== true-dynamics controllability geometry =="); i = _controllability_geometry_routes()
    print("== ablations active (arm, bumpy) =="); j = _ablations_active()
    ok = a and b and c and d and d1 and d2 and e and f and g and h and i and j
    print("RESULT:", "ARM OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
