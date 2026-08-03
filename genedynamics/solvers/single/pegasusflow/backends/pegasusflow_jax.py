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
  * Rolling-denoising noise schedule (`noise_scheduler.py::S2NoiseScheduler`):
        σ(node, it) = base_scale · Shape(node_t) · Decay(it)
    The vendored trajectory-optimization config uses a linear shape and exponential
    decay.  NO learned score network (model-free).

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


_NOISE_SHAPES = {"constant", "sine", "linear", "quadratic", "exponential"}
_NOISE_DECAYS = {"constant", "linear", "exponential", "cosine"}


def _noise_shape(name: str, n_nodes: int) -> jnp.ndarray:
    """Upstream ``noise_scheduler.py`` temporal shape functions."""
    if name not in _NOISE_SHAPES:
        raise ValueError(f"unsupported PegasusFlow noise shape '{name}'")
    t = jnp.linspace(0.0, 1.0, int(n_nodes), dtype=jnp.float32)
    if name == "constant":
        return jnp.ones_like(t)
    if name == "sine":
        return jnp.sin(jnp.pi * t)
    if name == "linear":
        return t
    if name == "quadratic":
        return t ** 2
    return jnp.exp(t)


def _noise_decay(name: str, iteration: int, max_iterations: int, *,
                 decay_rate: float, final_ratio: float) -> jnp.ndarray:
    """Upstream scalar decay functions, including their one-iteration edge case."""
    if name not in _NOISE_DECAYS:
        raise ValueError(f"unsupported PegasusFlow noise decay '{name}'")
    if name == "constant" or int(max_iterations) <= 1:
        return jnp.asarray(1.0, jnp.float32)
    progress = jnp.asarray(iteration / float(max_iterations - 1), jnp.float32)
    if name == "linear":
        return 1.0 + progress * (float(final_ratio) - 1.0)
    if name == "cosine":
        return float(final_ratio) + (1.0 - float(final_ratio)) * 0.5 * (
            1.0 + jnp.cos(jnp.pi * progress)
        )
    return jnp.asarray(float(decay_rate) ** int(iteration), jnp.float32)


def _weighted_basis_update(samples: jnp.ndarray, rewards: jnp.ndarray,
                           phi: jnp.ndarray, *, temp: float,
                           update_method: str, gamma: float) -> jnp.ndarray:
    """WBFO/AVWBFO update matching the vendored PyTorch implementation.

    PyTorch ``Tensor.std`` defaults to Bessel correction.  Using ``ddof=1`` is
    important for the relatively small sample counts used by CPU smoke tests.
    """
    W = rewards @ phi
    if update_method == "avwbfo":
        def back(carry, w_k):
            carry = w_k + float(gamma) * carry
            return carry, carry

        _, rev = jax.lax.scan(
            back, jnp.zeros((W.shape[0],), W.dtype), W.T[::-1]
        )
        W = rev[::-1].T
    elif update_method != "wbfo":
        raise ValueError(
            "PegasusFlow JAX supports update_method in {'wbfo', 'avwbfo'}"
        )
    W = (W - W.mean(0, keepdims=True)) / (
        W.std(0, keepdims=True, ddof=1) + 1e-8
    )
    W = jax.nn.softmax(W / float(temp), axis=0)
    return jnp.einsum("sn,sna->na", W, samples)


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
                 Hsample: int = 16, Hnode: int = 4, Nsample: int = 2048, noise_sigma: float = 1.0,
                 decay_rate: float = 0.9, temp_tau: float = 0.1, gamma: float = 1.0,
                 update_method: str = "avwbfo", noise_scheduler_type: str = "s2",
                 noise_shape_fn: str = "linear", noise_decay_fn: str = "exponential",
                 noise_final_ratio: float = 0.1,
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
            update_method = str(cfg.get("update_method", update_method)).lower()
            noise_scheduler_type = str(cfg.get("noise_scheduler_type", noise_scheduler_type)).lower()
            noise_shape_fn = str(cfg.get("noise_shape_fn", noise_shape_fn)).lower()
            noise_decay_fn = str(cfg.get("noise_decay_fn", noise_decay_fn)).lower()
            noise_final_ratio = float(cfg.get("noise_final_ratio", noise_final_ratio))
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
        self.update_method = str(update_method).lower()
        if self.update_method not in ("wbfo", "avwbfo"):
            raise ValueError("PegasusFlow update_method must be 'wbfo' or 'avwbfo'")
        if str(noise_scheduler_type).lower() != "s2":
            raise ValueError("the CPU PegasusFlow reproduction supports the upstream S2 scheduler")
        self.noise_shape_fn = str(noise_shape_fn).lower()
        self.noise_decay_fn = str(noise_decay_fn).lower()
        self.noise_final_ratio = float(noise_final_ratio)
        self.action_limit = float(action_limit)
        self.Ndiffuse, self.Ndiffuse_init, self.seed = int(Ndiffuse), int(Ndiffuse_init), int(seed)

        # Catmull-Rom basis Φ (Hsample+1 × Hnode+1): node→dense interp AND WBFO reward projection.
        phi = _catmull_rom_basis(self.Hsample + 1, self.Hnode + 1)
        self._phi = jnp.asarray(phi)
        self._phi_pinv = jnp.asarray(np.linalg.pinv(phi))             # dense→node (for shift)
        self._shape = _noise_shape(self.noise_shape_fn, self.Hnode + 1)

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
        return _weighted_basis_update(
            samples, rews, self._phi, temp=self.temp,
            update_method=self.update_method, gamma=self.gamma,
        )

    # --- WarmStartPlanner protocol ---
    def init_plan_var(self) -> jnp.ndarray:
        return jnp.zeros((self.Hnode + 1, self.nu), jnp.float32)

    def make_schedule(self, n_diffuse: int) -> jnp.ndarray:
        return jnp.arange(int(n_diffuse))

    def replan(self, state, warm_start, schedule, rng, t0=0.0) -> jnp.ndarray:
        mean = warm_start
        n_iter = int(schedule.shape[0])
        for k in range(n_iter):                                   # σ decays over the diffusion steps
            rng, sub = jax.random.split(rng)
            decay = _noise_decay(
                self.noise_decay_fn, k, n_iter,
                decay_rate=self.decay, final_ratio=self.noise_final_ratio,
            )
            sigma_it = self._shape[:, None] * self.base_sigma * decay
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


__all__ = [
    "PegasusFlowBackendJax", "_catmull_rom_basis", "_noise_shape",
    "_noise_decay", "_weighted_basis_update",
]
