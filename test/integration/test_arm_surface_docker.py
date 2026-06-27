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
    a = jax.random.uniform(jax.random.PRNGKey(1), (env.action_size,), minval=-1.0, maxval=1.0)
    s1 = env.step(x0, a)
    finite = bool(jnp.all(jnp.isfinite(s1.obs)) and jnp.isfinite(s1.reward))
    h, g = env.constraint_residual(x0, a)
    ag = env.manifold_geometry(x0, jnp.zeros((CFG["Hnode"] + 1, env.action_size)), 0.0)
    ok = (env.action_size == 10 and finite and h.shape == (6,) and g.shape == (2,)
          and ag.shape == (CFG["Hnode"] + 1, 10))
    print(f"  [{fam:8s}] action={env.action_size} finite={finite} h={h.shape} g={g.shape} "
          f"geom={tuple(ag.shape)} k_surf={float(env._k_surf):.0f} mu={float(env._mu):.2f} "
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
    print("== unseen domain randomization =="); c = _s4_dr_varies()
    print("== ablations active (arm, bumpy) =="); d = _ablations_active()
    ok = a and b and c and d
    print("RESULT:", "ARM OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
