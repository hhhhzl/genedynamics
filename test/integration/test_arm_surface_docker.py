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
    print("== g_force non-vacuous =="); b = _g_force_nonvacuous()
    print("== unseen domain randomization =="); c = (
        _s4_dr_varies()
        and _soft_s4_compliance_varies()
        and _contact_friction_reaches_both_geoms()
    )
    print("== independent geometry factors =="); d = _geometry_factors_are_independent()
    print("== position-only manifold =="); e = _position_only_manifold()
    print("== staged horizon/gated geometry =="); f = _staged_geometry_routes()
    print("== frozen realization geometry =="); g = _realization_geometry_routes()
    print("== true-dynamics controllability geometry =="); h = _controllability_geometry_routes()
    print("== ablations active (arm, bumpy) =="); i = _ablations_active()
    ok = a and b and c and d and e and f and g and h and i
    print("RESULT:", "ARM OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
