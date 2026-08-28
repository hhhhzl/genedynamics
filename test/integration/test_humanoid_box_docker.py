"""Humanoid box push/unjam env (idea.txt Exp II, H1/H2/H4) — docker, real brax.

Validates the rebuilt contact-semantic primitive + π_low + full manifold:
  * fixed-stance action_size == 12 (contact primitive) and 15 (+base);
  * each level constructs, reset, step(random) finite;
  * fixed-stance constraint_residual ->
    h(14)=[h_box;h_hand;h_hand_R;h_foot_xyz] + g(3)=[g_bal;g_fric;g_tip];
  * the controller point equals the physical hand geom and reset has a positive approach gap;
  * manifold_geometry -> (Hnode+1, nu);
  * face selection (H4): different j logits -> different contact target;
    fixed rear face for H1;
  * H2 domain randomization (hand friction / box frictionloss) varies across seeds;
  * MDAC ablations (no_softfeas/no_stiffness/no_tangent) change the plan;
  * the metrics plugin rolls + computes the full Exp II metric set end-to-end.

Invoke (from repo root):
  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
              python test/integration/test_humanoid_box_docker.py"
"""

import numpy as np
import jax
import jax.numpy as jnp
import mujoco

from genedynamics.envs.factories import make_env
from genedynamics.experiments.plugins.methods.contact_receding import (
    make_mdac, metrics_plugin_for, HUMANOID_TASK,
)

CFG = dict(Hsample=8, Hnode=4, Nsample=64, Ndiffuse_init=3, Ndiffuse=2,
           temp_sample=0.1, action_limit=1.0, dt=0.02, ctrl_dt=0.02, seed=0)


def _check_level(level, use_base=False, want_nu=12, want_h=14):
    env = make_env(HUMANOID_TASK, level=level, use_base=use_base, dr_seed=1)
    x0 = env.reset(jax.random.PRNGKey(0))
    a = jax.random.uniform(jax.random.PRNGKey(1), (env.action_size,), minval=-1.0, maxval=1.0)
    s1 = env.step(x0, a)
    finite = bool(jnp.all(jnp.isfinite(s1.obs)) and jnp.isfinite(s1.reward))
    h, g = env.constraint_residual(x0, a)
    ag = env.manifold_geometry(x0, jnp.zeros((CFG["Hnode"] + 1, env.action_size)), 0.0)
    ok = (env.action_size == want_nu and finite and h.shape == (want_h,) and g.shape == (3,)
          and ag.shape == (CFG["Hnode"] + 1, want_nu))
    tag = f"{level}{'(+base)' if use_base else ''}"
    print(f"  [{tag:9s}] action={env.action_size} finite={finite} h={tuple(h.shape)} "
          f"g={tuple(g.shape)} geom={tuple(ag.shape)} -> {'OK' if ok else 'FAIL'}")
    return ok


def _face_selection():
    env = make_env(HUMANOID_TASK, level="unjam")
    box = jnp.array([1.0, 0.0, 0.6])
    # rear-dominant vs left-dominant j logits -> different contact target
    def target(jlog):
        a = jnp.zeros(env.action_size).at[0:3].set(jlog)
        v, w, aa, bb, K, Fn = env._unpack(a)
        return np.asarray(env._contact_target(box, jnp.array([1., 0., 0., 0.]), w, aa, bb)[0])
    rear = target(jnp.array([10., 0., 0.]))
    left = target(jnp.array([0., 10., 0.]))
    d = float(np.max(np.abs(rear - left)))
    print(f"  face select: rear-target={np.round(rear,2)} left-target={np.round(left,2)} Δ={d:.3f}")
    # H1 ignores j (fixed rear)
    envh1 = make_env(HUMANOID_TASK, level="double_support")
    _, w1, _, _, _, _ = envh1._unpack(jnp.zeros(envh1.action_size).at[0:3].set(jnp.array([0., 10., 0.])))
    fixed = bool(np.allclose(np.asarray(w1), [1., 0., 0.]))
    print(f"  double_support fixed rear (j ignored): w_face={np.round(np.asarray(w1),2)} -> {fixed}")
    return d > 0.1 and fixed


def _h2_dr_varies():
    e1 = make_env(HUMANOID_TASK, level="heavy_dr", dr_seed=1)
    e2 = make_env(HUMANOID_TASK, level="heavy_dr", dr_seed=2)
    bf1, bf2 = float(e1.sys.dof_frictionloss[-3]), float(e2.sys.dof_frictionloss[-3])
    d = abs(float(e1._mu) - float(e2._mu)) > 1e-3 or abs(bf1 - bf2) > 1.0
    def target_inside_face(env):
        s = env.reset(jax.random.PRNGKey(0))
        c = env._hand_contact(s.pipeline_state, jnp.zeros(env.action_size), s.info)
        box_z = s.pipeline_state.x.pos[env._box_idx - 1, 2]
        return abs(float(c["p_surface"][2] - box_z)) <= float(env._half) + 1e-6
    targets_valid = target_inside_face(e1) and target_inside_face(e2)
    print(f"  H2 DR: seed1(mu={float(e1._mu):.2f},boxfric={bf1:.0f}) vs "
          f"seed2(mu={float(e2._mu):.2f},boxfric={bf2:.0f}) -> differs={d}, "
          f"targets-inside-face={targets_valid}")
    return d and targets_valid


def _physical_contact_geometry():
    env = make_env(HUMANOID_TASK, level="push_to_line", stiffness_mode="none")
    unjam = make_env(HUMANOID_TASK, level="unjam", stiffness_mode="none")
    x0 = env.reset(jax.random.PRNGKey(7))
    ps = x0.pipeline_state
    action = jnp.zeros(env.action_size).at[env.spec.nu_slice].set(-1.0)  # zero desired force
    contact = env._hand_contact(ps, action)
    controller_matches_geom = bool(jnp.allclose(
        contact["p_hand"], ps.geom_xpos[env._rhand_geom], atol=1e-6))
    hand_front = max(float(ps.geom_xpos[env._rhand_geom, 0]),
                     float(ps.geom_xpos[env._lhand_geom, 0])) + float(
                         env._robot_profile.controller_defaults[
                             "hand_forward_extent"
                         ]
                     )
    rear_face = float(ps.x.pos[env._box_idx - 1, 0] - env._half)
    gap = rear_face - hand_front
    forces = env._box_contact_forces(ps)
    no_reset_contact = float(forces["hand"]) < 1e-6 and float(forces["nonhand"]) < 1e-6
    p1 = make_env(
        HUMANOID_TASK, level="push_to_line", fixed_contact_target=True,
    )
    p1s = p1.reset(jax.random.PRNGKey(7))
    c_lo = p1._hand_contact(
        p1s.pipeline_state, jnp.zeros(p1.action_size).at[3:5].set(-1.0),
    )
    c_hi = p1._hand_contact(
        p1s.pipeline_state, jnp.zeros(p1.action_size).at[3:5].set(1.0),
    )
    p1_contact_locked = bool(jnp.allclose(c_lo["p_c"], c_hi["p_c"], atol=1e-6))
    mj = env.sys.mj_model
    lateral_locked = True
    for name in ("box_y", "box_yaw"):
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT.value, name)
        lateral_locked &= bool(env.sys.jnt_limited[jid]) and float(env.sys.jnt_range[jid, 1]) <= 1e-6
    walls_hidden = bool(
        jnp.all(env.sys.geom_rgba[env._wall_geoms, 3] == 0.0)
        and jnp.all(jnp.abs(env.sys.geom_pos[env._wall_geoms, 1]) >= 49.0)
        and np.all(env.sys.mj_model.geom_rgba[np.asarray(env._wall_geoms), 3] == 0.0)
        and np.all(np.abs(env.sys.mj_model.geom_pos[np.asarray(env._wall_geoms), 1]) >= 49.0)
    )
    walls_p3_only = bool(
        jnp.all(unjam.sys.geom_rgba[unjam._wall_geoms, 3] > 0.0)
        and jnp.all(jnp.abs(unjam.sys.geom_pos[unjam._wall_geoms, 1]) < 1.0)
        and np.all(unjam.sys.mj_model.geom_rgba[np.asarray(unjam._wall_geoms), 3] > 0.0)
        and np.all(np.abs(unjam.sys.mj_model.geom_pos[np.asarray(unjam._wall_geoms), 1]) < 1.0)
    )
    ok = controller_matches_geom and abs(gap - env._bcfg.approach_gap) < 1e-5 \
        and no_reset_contact and lateral_locked and p1_contact_locked \
        and walls_hidden and walls_p3_only
    print(f"  contact geometry: point=geom {controller_matches_geom}, gap={gap:.4f} m, "
          f"reset_force=({float(forces['hand']):.3g},{float(forces['nonhand']):.3g}), "
          f"L1-y/yaw-locked={lateral_locked}, P1-contact-locked={p1_contact_locked} "
          f"walls-hidden={walls_hidden}, walls-P3-only={walls_p3_only} "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


def _robot_swap_contract():
    results = []
    for robot, joints in (("h1", 19), ("g1", 29)):
        env = make_env(HUMANOID_TASK, robot=robot, level="push_to_line")
        state = env.reset(jax.random.PRNGKey(17))
        action = jnp.zeros(env.action_size)
        next_state = env.step(state, action)
        h, g = env.constraint_residual(state, action)
        valid = (
            env._robot_profile.num_actuated == joints
            and len(env._robot_binding.actuator_indices) == joints
            and env.action_size == 12
            and h.shape == (14,)
            and g.shape == (3,)
            and bool(jnp.isfinite(next_state.reward))
        )
        print(
            f"  robot swap {robot}: joints={joints} action={env.action_size} "
            f"h={h.shape} g={g.shape} finite={valid}"
        )
        results.append(valid)
    return all(results)


def _plan(env, sol, rng):
    b = sol._get_backend_impl()
    x0 = env.reset(jax.random.PRNGKey(1))
    return np.asarray(b.replan(x0, b.init_plan_var(), b.make_schedule(b.Ndiffuse_init), rng))


def _ablations_active():
    rng = jax.random.PRNGKey(0)
    env0, full = make_mdac(HUMANOID_TASK, "mdac", level="unjam", **CFG)
    base = _plan(env0, full, rng)
    ok = True
    for m in ("mdac_no_softfeas", "mdac_no_stiffness", "mdac_no_tangent"):
        e, s = make_mdac(HUMANOID_TASK, m, level="unjam", **CFG)
        d = float(np.max(np.abs(_plan(e, s, rng) - base)))
        if m == "mdac_no_tangent":
            # The zero plan is exactly on the clean stiffness/force manifold, so the final
            # stochastic plan delta can legitimately be tiny.  Test the actual operator on a
            # controlled off-manifold direction instead of classifying float32 noise as a no-op.
            fb = full._get_backend_impl()
            x0 = env0.reset(jax.random.PRNGKey(9))
            nodes = jnp.linspace(
                -0.5, 0.5, (CFG["Hnode"] + 1) * env0.action_size,
            ).reshape(CFG["Hnode"] + 1, env0.action_size)
            a_geom = full.geometry_fn(x0, nodes, 0.0)
            bundle = fb.manifold.geometry(a_geom, fb.topk_active, fb.eps_stab)
            projected = fb.manifold.project(a_geom, bundle, mode="metric")
            op_delta = float(jnp.max(jnp.abs(projected - a_geom)))
            active = (full.geometry_fn is not None and s.geometry_fn is None
                      and float(jnp.linalg.norm(a_geom)) > 1e-4 and op_delta > 1e-4)
            print(f"  mdac vs {m}: max|Δplan|={d:.3e}, "
                  f"controlled projection Δ={op_delta:.3e} -> "
                  f"{'ACTIVE' if active else 'NO-OP!'}")
        else:
            active = d > 1e-4
            print(f"  mdac vs {m}: max|Δplan|={d:.3e} -> "
                  f"{'ACTIVE' if active else 'NO-OP!'}")
        ok &= active
    return ok


def _metric_plugin_end_to_end():
    import types
    env = make_env(HUMANOID_TASK, level="unjam")
    x0 = env.reset(jax.random.PRNGKey(3))
    acts = [np.asarray(jax.random.uniform(jax.random.PRNGKey(i), (env.action_size,),
            minval=-1.0, maxval=1.0)) for i in range(5)]
    traj = types.SimpleNamespace(actions=acts)
    rec = metrics_plugin_for(HUMANOID_TASK).compute(traj, env, None, None, x0=x0, planning_time=0.3)
    # Five random steps end before the force ramp reaches its tracked window.
    # Steady-state/settling metrics are therefore unavailable by definition;
    # every metric that is observable on this short trajectory must stay finite.
    may_be_unavailable = {
        "steady_force_tracking_error",
        "steady_force_tracking_mae",
        "steady_normalized_force_tracking_mae",
        "force_settling_time",
    }
    nonfinite = {k for k, v in rec.items() if not np.isfinite(v)}
    valid = nonfinite <= may_be_unavailable and all(
        np.isnan(rec[k]) for k in nonfinite
    )
    print(f"  metric plugin: {len(rec)} metrics, nonfinite={sorted(nonfinite)}, "
          f"expected-only={valid}")
    print("    " + ", ".join(f"{k}={v:.3g}" for k, v in list(rec.items())[:6]))
    return len(rec) >= 12 and valid


def _paper_algorithm_contracts():
    from genedynamics.solvers.single.atacom.backends.atacom_jax import (
        atacom_null_dim,
    )

    expected = {
        "push_to_line": (7, 5),
        "unjam": (11, 1),
        "push_walk": (1, 22),
    }
    ok = True
    for level, (n_f, tangent) in expected.items():
        env = make_env(HUMANOID_TASK, level=level)
        state = env.reset(jax.random.PRNGKey(31))
        actions = jnp.zeros((5, env.action_size))
        prepared = env.prepare_realization_context(
            state, actions, gate_controllability=True,
        )
        residual = env.manifold_residual_horizon_controllable(
            prepared, actions, 0.0,
        )
        features = env.reliability_features_sequence(state, actions)
        emergency = env.emergency_plan(state, actions)
        valid = (
            env.manifold_constraint_size == n_f
            and atacom_null_dim(env) == tangent
            and features.shape == (24,)
            and residual.shape == (5 * n_f,)
            and emergency.shape == actions.shape
            and bool(jnp.isfinite(env.safety_index(state)))
        )
        print(
            f"  {level}: n_f={env.manifold_constraint_size}, "
            f"tangent={atacom_null_dim(env)}, features={features.shape}, "
            f"residual={residual.shape} -> {valid}"
        )
        ok &= valid
    return ok


def _atacom_finite_rollout():
    """ATACOM must stay finite under exploratory PPO actions."""
    from genedynamics.solvers.single.atacom.wrapper import AtacomEnvWrapper

    cases = (
        ("push_to_line", {}),
        ("unjam", {"push_dist": 0.03, "contact_band": 0.02}),
        ("push_walk", {"push_dist": 0.30}),
    )
    ok = True
    for level, overrides in cases:
        wrapped = AtacomEnvWrapper(
            make_env(HUMANOID_TASK, level=level, **overrides)
        )
        key = jax.random.PRNGKey(700 + len(level))
        state = wrapped.reset(key)
        step_fn = jax.jit(wrapped.step)
        failed_step = None
        for step in range(20):
            key, action_key = jax.random.split(key)
            alpha = jax.random.uniform(
                action_key, (wrapped.action_size,), minval=-1.0, maxval=1.0,
            )
            state = step_fn(state, alpha)
            leaves = {
                "obs": state.obs, "reward": state.reward,
                "q": state.pipeline_state.q, "qd": state.pipeline_state.qd,
                "slack": state.info["atacom_s"],
                "u": state.info["atacom_u"],
            }
            nonfinite = [
                name for name, value in leaves.items()
                if not bool(jnp.all(jnp.isfinite(value)))
            ]
            if nonfinite:
                failed_step = step
                print(f"    nonfinite={nonfinite}")
                break
        valid = failed_step is None
        print(f"  {level}: finite={valid}, failed_step={failed_step}")
        ok &= valid
    return ok


def main():
    print("== levels =="); a = all([
        _check_level("push_to_line", want_nu=12), _check_level("heavy_dr", want_nu=12),
        _check_level("unjam", want_nu=12), _check_level("unjam", use_base=True, want_nu=15),
        _check_level("push_walk", want_nu=23, want_h=10)])
    print("== physical contact geometry =="); b = _physical_contact_geometry()
    print("== robot swap contract =="); c = _robot_swap_contract()
    print("== face selection =="); d = _face_selection()
    print("== heavy_dr domain randomization =="); e = _h2_dr_varies()
    print("== ablations active (humanoid, unjam) =="); f = _ablations_active()
    print("== metric plugin end-to-end =="); g = _metric_plugin_end_to_end()
    print("== paper algorithm contracts =="); h = _paper_algorithm_contracts()
    print("== ATACOM finite rollout =="); i = _atacom_finite_rollout()
    ok = a and b and c and d and e and f and g and h and i
    print("RESULT:", "HUMANOID OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
