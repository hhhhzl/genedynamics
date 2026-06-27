"""Humanoid box push/unjam env (idea.txt Exp II, H1/H2/H4) — docker, real brax.

Validates the rebuilt contact-semantic primitive + π_low + full manifold:
  * action_size == 12 (H1/H2/H4-A) and 15 (H4-B, use_base);
  * each level constructs, reset, step(random) finite;
  * constraint_residual -> h(11)=[h_box;h_hand;h_hand_R;h_foot] + g(3)=[g_bal;g_fric;g_tip];
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

from genedynamics.envs.factories import make_env
from genedynamics.solvers.single.mdac.experiment import (
    make_mdac, metrics_plugin_for, HUMANOID_TASK,
)

CFG = dict(Hsample=8, Hnode=4, Nsample=64, Ndiffuse_init=3, Ndiffuse=2,
           temp_sample=0.1, action_limit=1.0, dt=0.02, ctrl_dt=0.02, seed=0)


def _check_level(level, use_base=False, want_nu=12):
    env = make_env(HUMANOID_TASK, level=level, use_base=use_base, dr_seed=1)
    x0 = env.reset(jax.random.PRNGKey(0))
    a = jax.random.uniform(jax.random.PRNGKey(1), (env.action_size,), minval=-1.0, maxval=1.0)
    s1 = env.step(x0, a)
    finite = bool(jnp.all(jnp.isfinite(s1.obs)) and jnp.isfinite(s1.reward))
    h, g = env.constraint_residual(x0, a)
    ag = env.manifold_geometry(x0, jnp.zeros((CFG["Hnode"] + 1, env.action_size)), 0.0)
    ok = (env.action_size == want_nu and finite and h.shape == (11,) and g.shape == (3,)
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
        return np.asarray(env._contact_target(box, w, aa, bb)[0])
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
    bf1, bf2 = float(e1.sys.dof_frictionloss[-1]), float(e2.sys.dof_frictionloss[-1])
    d = abs(float(e1._mu) - float(e2._mu)) > 1e-3 or abs(bf1 - bf2) > 1.0
    print(f"  H2 DR: seed1(mu={float(e1._mu):.2f},boxfric={bf1:.0f}) vs "
          f"seed2(mu={float(e2._mu):.2f},boxfric={bf2:.0f}) -> differs={d}")
    return d


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
        ok &= d > 1e-4
        print(f"  mdac vs {m}: max|Δplan|={d:.3e} -> {'ACTIVE' if d > 1e-4 else 'NO-OP!'}")
    return ok


def _metric_plugin_end_to_end():
    import types
    env = make_env(HUMANOID_TASK, level="unjam")
    x0 = env.reset(jax.random.PRNGKey(3))
    acts = [np.asarray(jax.random.uniform(jax.random.PRNGKey(i), (env.action_size,),
            minval=-1.0, maxval=1.0)) for i in range(5)]
    traj = types.SimpleNamespace(actions=acts)
    rec = metrics_plugin_for(HUMANOID_TASK).compute(traj, env, None, None, x0=x0, planning_time=0.3)
    finite = all(np.isfinite(v) for v in rec.values())
    print(f"  metric plugin: {len(rec)} metrics, all finite={finite}")
    print("    " + ", ".join(f"{k}={v:.3g}" for k, v in list(rec.items())[:6]))
    return len(rec) >= 12 and finite


def main():
    print("== levels =="); a = all([
        _check_level("double_support", want_nu=12), _check_level("heavy_dr", want_nu=12),
        _check_level("unjam", want_nu=12), _check_level("unjam", use_base=True, want_nu=15)])
    print("== face selection =="); b = _face_selection()
    print("== heavy_dr domain randomization =="); c = _h2_dr_varies()
    print("== ablations active (humanoid, unjam) =="); d = _ablations_active()
    print("== metric plugin end-to-end =="); e = _metric_plugin_end_to_end()
    ok = a and b and c and d and e
    print("RESULT:", "HUMANOID OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
