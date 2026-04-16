"""JAX-MPM rollout evaluator.

Implements the same protocol as SoftZooRolloutEvaluator:
    evaluator.evaluate_batch(RolloutBatchRequest, parallel=True, use_cache=False)
        -> RolloutBatchResult(returns, successes, ...)

Unlike softzoo we do NOT need a subprocess pool — everything is pure JAX, so
a single `jax.vmap` call processes the entire batch inside one GPU kernel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .protocols import (
    RolloutBatchRequest,
    RolloutBatchResult,
    RolloutResult,
)


@dataclass
class JaxMpmEvaluatorConfig:
    """Config for JaxMpmRolloutEvaluator.

    We deliberately mirror SoftZooEvaluatorConfig's knobs (max_workers,
    reward_shaping_weight) where they make sense, but JAX-MPM ignores most
    of them (no pool, shaping is baked into the reward).
    """

    max_workers: int = 1       # ignored (JAX handles batching internally)
    cache_size: int = 0         # ignored
    reward_shaping_weight: float = 100.0  # picked up by scene.MPMConfig
    n_grid: int = 64
    voxel_dims: Optional[tuple] = None  # (vx, vy, vz); None → MPMConfig default (3,3,3)
    extra: Dict[str, Any] = field(default_factory=dict)


class JaxMpmRolloutEvaluator:
    """JAX-MPM implementation of the RolloutEvaluator protocol."""

    def __init__(
        self,
        *,
        config: Optional[JaxMpmEvaluatorConfig] = None,
        project_root: Optional[str] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        from genedynamics.envs.external.jax_mpm.scene import MPMConfig, build_scene

        self.config = config or JaxMpmEvaluatorConfig(**kwargs)
        self._project_root = project_root
        self._runtime_config = dict(runtime_config or {})

        mpm_kwargs = {
            "n_grid": int(self.config.n_grid),
            "shaping_weight": float(self.config.reward_shaping_weight),
        }
        if self.config.voxel_dims is not None:
            mpm_kwargs["voxel_dims"] = tuple(int(v) for v in self.config.voxel_dims)
        # Let runtime_config override anything in MPMConfig (e.g., dt, gravity).
        for k in ("dt", "gravity", "scale", "p_vol", "friction_coeff",
                  "actuation_strength_scale", "act_strength_base"):
            if k in self._runtime_config:
                mpm_kwargs[k] = float(self._runtime_config[k])

        self._mpm_cfg = MPMConfig(**{k: v for k, v in mpm_kwargs.items()
                                     if k in MPMConfig.__dataclass_fields__})
        self._scene = build_scene(self._mpm_cfg)

        # Mode → friction table. Mirrors softzoo task_registry.DEFAULT_MODES.
        self._mode_friction = [0.3, 0.4, 0.5, 0.6]

    # -- protocol ------------------------------------------------------------

    def evaluate_batch(
        self,
        request: RolloutBatchRequest,
        *,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> RolloutBatchResult:
        from genedynamics.envs.external.jax_mpm.adapters import evaluate_batch_request

        t0 = time.perf_counter()
        returns, disps = evaluate_batch_request(
            request, self._scene, self._mpm_cfg, self._mode_friction
        )
        wall = time.perf_counter() - t0

        results = [
            RolloutResult(
                return_=float(r),
                success=bool(r > 0 and d > 0),
                num_steps=0,
                mean_return=float(r),
                std_return=0.0,
                failure_code=None,
                wall_time=wall / max(len(returns), 1),
                trajectory=None,
            )
            for r, d in zip(returns, disps)
        ]
        return RolloutBatchResult(
            results=results,
            returns=returns.astype(np.float32),
            successes=np.array([r.success for r in results], dtype=bool),
            mean_return=float(np.mean(returns)) if returns.size else 0.0,
            std_return=float(np.std(returns)) if returns.size > 1 else 0.0,
            wall_time_total=wall,
            failure_codes=[None] * len(results),
        )

    def evaluate_single(
        self,
        morphology_params: np.ndarray,
        controller_params: np.ndarray,
        *,
        task_id: str = "crawling_ground",
        mode_id: int = 0,
        fidelity_level: int = 2,
        seed: int = 0,
        num_repeats: int = 1,
        record: bool = False,
        **kwargs: Any,
    ) -> RolloutResult:
        from genedynamics.envs.external.jax_mpm.adapters import evaluate_rollout_request

        fr = self._mode_friction[int(mode_id) % len(self._mode_friction)]
        r, d = evaluate_rollout_request(
            morphology_params, controller_params, mode_id, fidelity_level,
            fr, self._scene, self._mpm_cfg,
        )
        return RolloutResult(
            return_=float(r),
            success=bool(r > 0 and d > 0),
            num_steps=0,
            mean_return=float(r),
            std_return=0.0,
            failure_code=None,
            wall_time=0.0,
            trajectory=None,
        )

    def close(self) -> None:
        # No resources to release: scene lives on GPU, garbage-collected
        # when this evaluator goes out of scope.
        pass
