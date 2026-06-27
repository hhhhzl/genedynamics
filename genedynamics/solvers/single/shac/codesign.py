"""SHACBaseline — writeup §13.1 #9 ("SHAC-only") + Q3 controller-only experiment.

Wraps the SHAC solver behind the BaselineProtocol so it slots into the same
experiment platform as MRMFMBD / CMA-ES / CEM. The morphology is fixed (the
baseline does not optimize x); only φ is learned.

When the YAML supplies ``method_params.morphology`` (a list of n_voxels
floats), that fixes the body. Otherwise the body is the all-ones occupancy
on the evaluator's default scene — fine for crawling_ground.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from genedynamics.experiments.framework.baseline import BaselineConfig, BaselineResult, BaselineProtocol


class SHACBaseline(BaselineProtocol):
    """SHAC over fixed morphology, optimizing the controller phi."""

    @property
    def name(self) -> str:
        return "shac"

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
        from genedynamics.solvers.single.shac import SHACConfig, SHACSolver

        extra = config.extra
        # SHAC needs the JAX-direct evaluator (jax_mpm) to access scene/mpm_cfg.
        if not hasattr(evaluator, "_scene") or not hasattr(evaluator, "_mpm_cfg"):
            raise RuntimeError(
                "SHACBaseline requires a JAX-MPM evaluator with `_scene` and "
                "`_mpm_cfg` attributes (subclass of JaxMpmRolloutEvaluator)."
            )
        scene = evaluator._scene
        mpm_cfg = evaluator._mpm_cfg

        # Fixed morphology (Q3 controller-only): default = all-ones (full body).
        morph = extra.get("morphology", None)
        if morph is not None:
            morph = np.asarray(morph, dtype=np.float32).reshape(-1)
        # Friction: pull from the first regime if a regime bank was set,
        # otherwise from mode_friction[0], else 0.5.
        friction = float(extra.get("friction", 0.5))
        if hasattr(evaluator, "_regime_bank") and evaluator._regime_bank:
            friction = float(evaluator._regime_bank[0].friction)
        elif hasattr(evaluator, "_mode_friction") and evaluator._mode_friction:
            friction = float(evaluator._mode_friction[0])

        # Optional terrain (Phase 2 hook). Use first regime's terrain when
        # the evaluator carries one; else None.
        terrain_height = None
        if hasattr(evaluator, "_regime_bank") and evaluator._regime_bank:
            from genedynamics.envs.external.jax_mpm.terrain import to_grid
            terrain_height = to_grid(evaluator._regime_bank[0].terrain, mpm_cfg.n_grid)

        # Build SHACConfig from method_params (yaml `extra`).
        cfg = SHACConfig(
            h=int(extra.get("h", 32)),
            n_envs=int(extra.get("n_envs", 32)),
            n_episodes=int(extra.get("n_episodes", 500)),
            discount=float(extra.get("discount", 0.99)),
            td_lambda=float(extra.get("td_lambda", 0.95)),
            env_horizon=int(extra.get("env_horizon", 200)),
            reset_every=int(extra.get("reset_every", 1)),
            phi_dim=int(phi_dim),
            explore_sigma=float(extra.get("explore_sigma", 0.1)),
            deterministic=bool(extra.get("deterministic", False)),
            phi_lo=float(extra.get("phi_lo", -0.5)),
            phi_hi=float(extra.get("phi_hi", 0.5)),
            critic_hidden=tuple(extra.get("critic_hidden", (64, 64))),
            target_alpha=float(extra.get("target_alpha", 0.995)),
            n_critic_iters=int(extra.get("n_critic_iters", 16)),
            n_critic_minibatches=int(extra.get("n_critic_minibatches", 4)),
            actor_lr=float(extra.get("actor_lr", 2.0e-3)),
            critic_lr=float(extra.get("critic_lr", 5.0e-4)),
            actor_grad_clip=float(extra.get("actor_grad_clip", 1.0)),
            critic_grad_clip=float(extra.get("critic_grad_clip", 1.0)),
            n_actuators=int(extra.get("n_actuators", 10)),
            morphology=morph,
            friction=friction,
            seed=int(config.seed),
            show_tqdm=bool(extra.get("show_tqdm", False)),
            log_every=int(extra.get("log_every", 10)),
        )

        solver = SHACSolver(
            scene, mpm_cfg, cfg,
            backend=str(extra.get("backend", "jax")),
            morphology=morph,
            friction=friction,
            terrain_height=terrain_height,
        )

        t0 = time.perf_counter()
        result = solver.train()
        wall = time.perf_counter() - t0

        # Build the BaselineResult. x is fixed (or all-ones); phi is optimized.
        x_full = morph if morph is not None else np.ones(x_dim, dtype=np.float32)
        phi_star = np.asarray(result["phi"], dtype=np.float32)

        # SHAC reports its own training-time mean episode return; the
        # `final_return` is from a clean evaluation rollout post-training.
        return BaselineResult(
            theta=np.concatenate([x_full, phi_star]),
            x=x_full.astype(np.float32),
            phi=phi_star,
            return_=float(result["final_return"]),
            success=bool(result["final_return"] > 0.0),
            num_evaluations=int(cfg.n_envs * cfg.n_episodes),
            wall_time=float(result["wall_clock"]),
            metadata={
                "training_history": [
                    {
                        "episode": h.episode,
                        "mean_episode_return": h.mean_episode_return,
                        "actor_loss": h.actor_loss,
                        "critic_loss": h.critic_loss,
                        "actor_grad_norm": h.actor_grad_norm,
                        "wall_time": h.wall_time,
                    }
                    for h in result["history"]
                ],
                "config": {
                    "h": cfg.h, "n_envs": cfg.n_envs, "n_episodes": cfg.n_episodes,
                    "explore_sigma": cfg.explore_sigma, "actor_lr": cfg.actor_lr,
                },
            },
        )
