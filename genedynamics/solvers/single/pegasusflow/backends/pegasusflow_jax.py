"""JAX backend for the PegasusFlow / WBFO baseline — FAITHFUL reproduction.

Re-implements `baselines/PegasusFlow/traj_sampling` (torch + IsaacGym, arXiv 2509.08435) on
the brax/jax stack, matching the vendored algorithm element-for-element:

  * Control = a CATMULL-ROM cubic spline over ``Hnode+1`` knots interpolated to the dense
    ``Hsample+1`` rollout (`spline.py::compute_cubic_spline_basis_piecewise`). The SAME basis
    matrix Φ (Hsample+1 × Hnode+1) is used for node→dense interpolation AND the WBFO reward
    projection.
  * Update = Weighted Basis Function Optimization (`optimizer.py::WeightedBasisFunctionOptimizer`):
        W = S · Φ                      (per-DENSE-step rewards S → per-NODE weights, Hsample+1→Hnode+1)
        W = (W − mean_s W) / (std_s W) (per-node z-score over samples)
        W = softmax(W / τ, axis=samples)   (per-NODE softmax — NOT one softmax over total return)
        nodes ← Σ_s W[s] ⊙ node_samples[s]    (per-node weighted mean)
    optional AVWBFO node-cumulative discount γ (`ActionValueWBFO`).
  * Sampling (`trajopt_policy.py::_eval_traj_grad_batch`): N+1 samples (sample 0 = the
    unperturbed mean); the FIRST NODE is held fixed (current action); per-node-per-dim noise.
  * Rolling-denoising noise schedule (`noise_scheduler.py::S3NoiseScheduler`):
        σ(node, it) = DimScale · Shape(node_t) · Decay(it)
    Shape default = sine (sin(π·t) — zeroes the endpoints, fix-endpoint optimization);
    Decay default = geometric (decay_rate^it). NO learned score network (model-free).

Implements ``WarmStartPlanner`` so it runs through the shared ``RecedingHorizonController``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.solvers.common.env_rollout import (
    is_brax_env, build_brax_rollout, build_brax_step,
)


def _catmull_rom_basis(n_eval: int, n_knots: int) -> np.ndarray:
    """Catmull-Rom cubic basis Φ (n_eval × n_knots), an exact port of
    ``spline.py::compute_cubic_spline_basis_piecewise`` (knots/eval on ``linspace(0,1,·)``)."""
    if n_knots < 4:
        raise ValueError("Catmull-Rom basis needs >= 4 knots (Hnode+1)")
    t = np.linspace(0.0, 1.0, n_eval)
    knots = np.linspace(0.0, 1.0, n_knots)
    basis = np.zeros((n_eval, n_knots), np.float32)
    for i, tv in enumerate(t):
        seg = int(np.searchsorted(knots[1:], tv, side="left"))
        seg = min(max(seg, 0), n_knots - 2)
        t0, t1 = knots[seg], knots[seg + 1]
        u = (tv - t0) / (t1 - t0) if (t1 - t0) > 1e-10 else 0.0
        pm1, p0, p1 = max(0, seg - 1), seg, seg + 1
        p2 = min(n_knots - 1, seg + 2)
        if seg == 0:
            pm1 = p0
        if seg >= n_knots - 2:
            p2 = p1
        u2, u3 = u * u, u * u * u
        wm1 = (-u3 + 2 * u2 - u) / 2
        w0 = (3 * u3 - 5 * u2 + 2) / 2
        w1 = (-3 * u3 + 4 * u2 + u) / 2
        w2 = (u3 - u2) / 2
        if seg == 0 and pm1 == p0:
            basis[i, p0] += wm1 + w0
        else:
            basis[i, pm1] += wm1
            basis[i, p0] += w0
        basis[i, p1] += w1
        if seg >= n_knots - 2 and p2 == p1:
            basis[i, p1] += w2
        else:
            basis[i, p2] += w2
    return basis


class PegasusFlowBackendJax:
    """WarmStartPlanner + single-shot ``plan()`` — faithful WBFO over a Catmull-Rom basis."""

    def __init__(self, solver: Any = None, *, nu: Optional[int] = None,
                 rollout_fn: Optional[Callable] = None, step_fn: Optional[Callable] = None,
                 Hsample: int = 16, Hnode: int = 4, Nsample: int = 2048, noise_sigma: float = 0.3,
                 decay_rate: float = 0.9, temp_tau: float = 0.1, gamma: float = 0.0,
                 action_limit: float = 1.0, Ndiffuse: int = 2, Ndiffuse_init: int = 10,
                 seed: int = 0) -> None:
        if solver is not None:
            cfg = solver.config
            nu = nu if nu is not None else int(solver.nu)
            Hsample = int(cfg.get("Hsample", Hsample))
            Hnode = int(cfg.get("Hnode", Hnode))
            Nsample = int(cfg.get("Nsample", Nsample))
            noise_sigma = float(cfg.get("noise_sigma", noise_sigma))
            decay_rate = float(cfg.get("decay_rate", cfg.get("sigma_decay", decay_rate)))
            temp_tau = float(cfg.get("temp_sample", cfg.get("temp_tau", temp_tau)))
            gamma = float(cfg.get("wbfo_gamma", gamma))
            action_limit = float(cfg.get("action_limit", action_limit))
            Ndiffuse = int(cfg.get("Ndiffuse", Ndiffuse))
            Ndiffuse_init = int(cfg.get("Ndiffuse_init", Ndiffuse_init))
            seed = int(getattr(solver, "seed", seed))
            rollout_fn = rollout_fn or getattr(solver, "_rollout_fn", None)
            step_fn = step_fn or getattr(solver, "_step_fn", None)
        if nu is None:
            raise ValueError("PegasusFlowBackendJax requires nu (action dim).")

        self.nu, self.Hsample, self.Hnode = int(nu), int(Hsample), int(Hnode)
        self.Nsample = int(Nsample)
        self.base_sigma, self.decay, self.temp = float(noise_sigma), float(decay_rate), float(temp_tau)
        self.gamma = float(gamma)
        self.action_limit = float(action_limit)
        self.Ndiffuse, self.Ndiffuse_init, self.seed = int(Ndiffuse), int(Ndiffuse_init), int(seed)

        # Catmull-Rom basis Φ (Hsample+1 × Hnode+1): node→dense interp AND WBFO reward projection.
        phi = _catmull_rom_basis(self.Hsample + 1, self.Hnode + 1)
        self._phi = jnp.asarray(phi)
        self._phi_pinv = jnp.asarray(np.linalg.pinv(phi))             # dense→node (for shift)
        # S3 Shape(node_t) = sine (zeroes the endpoints — fix-endpoint optimization).
        self._shape = jnp.asarray(np.sin(np.pi * np.linspace(0.0, 1.0, self.Hnode + 1)), jnp.float32)

        env = getattr(solver, "dynamics", None)
        self._rollout_fn = rollout_fn or (build_brax_rollout(env) if is_brax_env(env) else None)
        self._step_fn = step_fn or (build_brax_step(env) if is_brax_env(env) else None)
        if self._rollout_fn is None:
            raise ValueError("PegasusFlowBackendJax needs a brax env (or an injected rollout_fn).")
        self._wbfo = jax.jit(self._wbfo_step)

    def _node2dense(self, V: jnp.ndarray) -> jnp.ndarray:          # (..,Hnode+1,nu) -> (..,Hsample+1,nu)
        return jnp.einsum("hk,...ka->...ha", self._phi, V)

    # one WBFO rolling-denoising refine step at noise scale σ_it
    def _wbfo_step(self, state, mean, sigma_it, rng):
        N, K, nu = self.Nsample, self.Hnode + 1, self.nu
        eps = jax.random.normal(rng, (N + 1, K, nu))
        samples = jnp.clip(eps * sigma_it[None] + mean[None], -self.action_limit, self.action_limit)
        samples = samples.at[:, 0].set(mean[0])                    # first node fixed (current action)
        samples = samples.at[0].set(mean)                          # sample 0 = unperturbed mean
        rews = self._rollout_fn(state, self._node2dense(samples), 0.0)   # (N+1, Hsample+1)
        W = rews @ self._phi                                       # (N+1, Hnode+1)  WBFO: W = S·Φ
        if self.gamma > 0.0:                                       # AVWBFO node-cumulative discount
            def back(carry, w_k):
                carry = w_k + self.gamma * carry
                return carry, carry
            _, W = jax.lax.scan(back, jnp.zeros((N + 1,), W.dtype), W.T[::-1])
            W = W[::-1].T
        W = (W - W.mean(0, keepdims=True)) / (W.std(0, keepdims=True) + 1e-8)
        W = jax.nn.softmax(W / self.temp, axis=0)                  # per-NODE softmax over samples
        return jnp.einsum("sn,sna->na", W, samples)               # per-node weighted mean

    # --- WarmStartPlanner protocol ---
    def init_plan_var(self) -> jnp.ndarray:
        return jnp.zeros((self.Hnode + 1, self.nu), jnp.float32)

    def make_schedule(self, n_diffuse: int) -> jnp.ndarray:
        return jnp.arange(int(n_diffuse))

    def replan(self, state, warm_start, schedule, rng, t0=0.0) -> jnp.ndarray:
        mean = warm_start
        for k in range(int(schedule.shape[0])):                    # σ decays over the diffusion steps
            rng, sub = jax.random.split(rng)
            sigma_it = self._shape[:, None] * (self.base_sigma * (self.decay ** k))
            mean = self._wbfo(state, mean, sigma_it, sub)
        return jnp.clip(mean, -self.action_limit, self.action_limit)

    def first_action(self, plan_var) -> jnp.ndarray:
        return self._node2dense(plan_var)[0]

    def shift(self, plan_var) -> jnp.ndarray:
        dense = self._node2dense(plan_var)                         # (Hsample+1, nu)
        dense = jnp.concatenate([dense[1:], jnp.zeros((1, self.nu), dense.dtype)], 0)
        return self._phi_pinv @ dense

    # --- single-shot plan ---
    def plan(self, x0: Any, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        Y = self.replan(x0, self.init_plan_var(), self.make_schedule(self.Ndiffuse_init), rng_key)
        return {"actions": np.asarray(self._node2dense(Y), np.float32)}

    def plan_batch(self, x0: Any, keys: Any) -> List[Dict[str, Any]]:
        return [self.plan(x0, rng_key=k) for k in keys]


__all__ = ["PegasusFlowBackendJax"]
