"""DiffuseBotBaseline — Table 1 "DiffuseBot-style" / "first-order ours" baseline.

In-framework (JAX MPM), GRADIENT-based co-design: Adam ascent on θ=(x, φ) through
the differentiable MPM rollout (jax.grad of reward). This is the gradient-guided
analogue of DiffuseBot's physics-augmented diffusion, and simultaneously the
"first-order ours" ablation — the apples-to-apples counterpoint to the
gradient-FREE MBD that substantiates contribution 1 ("why gradient-free").

Single mode + single (fine) fidelity, matching DiffuseBot's mode-blind,
fixed-fidelity setting. Morphology is initialized from `method_params.morphology`
(an n_voxels occupancy, e.g. derived from a 3D-prior body via
`morphology.robotize_point_cloud`); absent that, from a uniform x_mean body.
An optional `prior_guidance_weight` pulls x toward x_mean each step — a cheap
proxy for DiffuseBot's embedding/classifier guidance (the faithful neural-SDF
guidance is a separate GPU path).

NOTE: run() differentiates through the MPM rollout, so it requires a JAX-MPM
evaluator and is verified on GPU; the class/registration/non-rollout logic is
CPU-importable.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..baseline import BaselineConfig, BaselineResult, BaselineProtocol


class DiffuseBotBaseline(BaselineProtocol):
    """Gradient-based (x, φ) co-design through the differentiable JAX MPM."""

    @property
    def name(self) -> str:
        return "diffusebot"

    def run(
        self,
        config: BaselineConfig,
        evaluator: Any,
        task_spec: Any,
        *,
        x_dim: int,
        phi_dim: int,
        **kwargs: Any,
    ) -> BaselineResult:
        import jax
        import jax.numpy as jnp
        from genedynamics.envs.external.jax_mpm.scene import rollout_return

        extra = config.extra
        if not hasattr(evaluator, "_scene") or not hasattr(evaluator, "_mpm_cfg"):
            raise RuntimeError(
                "DiffuseBotBaseline requires a JAX-MPM evaluator with `_scene` "
                "and `_mpm_cfg` (gradient guidance differentiates the rollout)."
            )
        scene = evaluator._scene
        mpm_cfg = evaluator._mpm_cfg

        # Single mode (DiffuseBot is mode-blind): first regime's friction.
        friction = float(extra.get("friction", 0.5))
        if getattr(evaluator, "_regime_bank", None):
            friction = float(evaluator._regime_bank[0].friction)
        elif getattr(evaluator, "_mode_friction", None):
            friction = float(evaluator._mode_friction[0])

        x_lo, x_hi = float(extra.get("x_lo", 0.2)), float(extra.get("x_hi", 1.0))
        phi_lo, phi_hi = float(extra.get("phi_lo", -0.5)), float(extra.get("phi_hi", 0.5))
        x_mean = float(extra.get("x_mean", 0.6))
        n_iters = int(extra.get("n_iters", 200))
        lr = float(extra.get("lr", 1e-2))
        num_env_steps = int(extra.get("num_env_steps", getattr(mpm_cfg, "env_horizon", 200)))
        prior_guidance_weight = float(extra.get("prior_guidance_weight", 0.0))

        # Morphology init: explicit occupancy (e.g. from a 3D-prior body) or uniform.
        morph = extra.get("morphology", None)
        if morph is not None:
            x0 = np.clip(np.asarray(morph, np.float32).reshape(-1), x_lo, x_hi)
        else:
            x0 = np.full(x_dim, x_mean, dtype=np.float32)
        phi0 = np.asarray(extra.get("phi_init", np.zeros(phi_dim)), np.float32).reshape(-1)
        theta = jnp.asarray(np.concatenate([x0, phi0]), dtype=jnp.float32)
        x_mean_vec = jnp.full((x_dim,), x_mean, dtype=jnp.float32)

        def neg_reward(th):
            x = jnp.clip(th[:x_dim], x_lo, x_hi)
            phi = jnp.clip(th[x_dim:], phi_lo, phi_hi)
            r, _, _ = rollout_return(x, phi, jnp.asarray(friction, jnp.float32),
                                     scene, mpm_cfg, num_env_steps)
            reg = prior_guidance_weight * jnp.sum((x - x_mean_vec) ** 2)
            return -r + reg

        value_and_grad = jax.jit(jax.value_and_grad(neg_reward))

        # Hand-rolled Adam (no optax in the target env).
        m = jnp.zeros_like(theta)
        v = jnp.zeros_like(theta)
        b1, b2, eps = 0.9, 0.999, 1e-8
        lo = jnp.concatenate([jnp.full((x_dim,), x_lo), jnp.full((phi_dim,), phi_lo)])
        hi = jnp.concatenate([jnp.full((x_dim,), x_hi), jnp.full((phi_dim,), phi_hi)])

        best_theta = theta
        best_reward = -np.inf
        t0 = time.perf_counter()
        for t in range(1, n_iters + 1):
            neg_r, g = value_and_grad(theta)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            theta = jnp.clip(theta - lr * mh / (jnp.sqrt(vh) + eps), lo, hi)
            r = float(-neg_r)
            if r > best_reward:
                best_reward = r
                best_theta = theta
        wall = time.perf_counter() - t0

        bt = np.asarray(best_theta, dtype=np.float32)
        x_star, phi_star = bt[:x_dim], bt[x_dim:]
        return BaselineResult(
            theta=bt,
            x=x_star,
            phi=phi_star,
            return_=float(best_reward),
            success=bool(best_reward > 0.0),
            num_evaluations=int(n_iters),
            wall_time=float(wall),
            metadata={
                "method": "gradient_codesign",
                "friction": friction,
                "n_iters": n_iters,
                "lr": lr,
                "prior_guidance_weight": prior_guidance_weight,
            },
        )
