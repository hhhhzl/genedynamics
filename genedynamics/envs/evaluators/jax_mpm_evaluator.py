"""JAX-MPM rollout evaluator.

Implements the RolloutEvaluator protocol:
    evaluator.evaluate_batch(RolloutBatchRequest, parallel=True, use_cache=False)
        -> RolloutBatchResult(returns, successes, ...)

No subprocess pool needed — everything is pure JAX, so a single `jax.vmap`
call processes the entire batch inside one GPU kernel.
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

    max_workers and cache_size are accepted for protocol compatibility but
    ignored — JAX-MPM batches inside one vmap kernel.

    Phase 2.2 fields
    ----------------
    task : str
        Selects rollout flavor when regime_bank is set: "locomotion" routes
        through rollout_return_batch with terrain; "push" uses
        rollout_return_push_batch with terrain + manipuland. Default
        "crawling_ground" preserves the legacy mode_friction code path.
    regime_bank_kind : Optional[str]
        When non-None, build the regime bank at evaluator-init time:
        "train_locomotion" / "test_locomotion" / "train_push" / "test_push".
        mode_id then indexes that bank instead of mode_friction_table.
    push_goal_x, push_weights
        Static knobs forwarded to rollout_return_push_batch.
    """

    max_workers: int = 1       # ignored (JAX handles batching internally)
    cache_size: int = 0         # ignored
    reward_shaping_weight: float = 100.0  # picked up by scene.MPMConfig
    n_grid: int = 64
    voxel_dims: Optional[tuple] = None  # (vx, vy, vz); None → MPMConfig default (3,3,3)
    task: str = "crawling_ground"
    regime_bank_kind: Optional[str] = None  # "train_locomotion" | "test_locomotion" | "train_push" | "test_push"
    push_goal_x: float = 0.85
    push_weights: tuple = (1.0, 5.0, 0.01, 50.0)
    # Phase 3: optional pre-robotized SoftBodySpec on disk. When set, the
    # evaluator skips build_scene(cfg) and instead deserializes the spec via
    # load_spec_npz, then calls build_scene_from_spec. Used by configs that
    # consume mesh-derived bodies (configs/soft_robot/main_v2/crawling_from_mesh.yaml).
    softbody_spec_path: Optional[str] = None
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
                  "actuation_strength_scale", "act_strength_base",
                  "backward_penalty_weight"):
            if k in self._runtime_config:
                mpm_kwargs[k] = float(self._runtime_config[k])

        self._mpm_cfg = MPMConfig(**{k: v for k, v in mpm_kwargs.items()
                                     if k in MPMConfig.__dataclass_fields__})

        # Phase 3: when softbody_spec_path is set, swap the default body for
        # the cached mesh-derived spec. We also surface the spec's voxel_dims
        # back into mpm_cfg so downstream code (occupancy mass field, x_dim
        # derivation in the baseline) sees a single source of truth.
        if self.config.softbody_spec_path:
            from genedynamics.morphology import load_spec_npz
            from genedynamics.envs.external.jax_mpm.scene import build_scene_from_spec
            spec = load_spec_npz(self.config.softbody_spec_path)
            if tuple(spec.voxel_dims) != tuple(self._mpm_cfg.voxel_dims):
                # Rebuild MPMConfig with the spec's voxel_dims so the simulator
                # and the baseline's x_dim calculation agree.
                fresh = dict(mpm_kwargs)
                fresh["voxel_dims"] = tuple(int(v) for v in spec.voxel_dims)
                self._mpm_cfg = MPMConfig(**{
                    k: v for k, v in fresh.items() if k in MPMConfig.__dataclass_fields__
                })
            self._scene = build_scene_from_spec(spec, self._mpm_cfg)
        else:
            self._scene = build_scene(self._mpm_cfg)

        # Mode → friction table. Override via runtime_config["mode_friction"]
        # (list of floats) so single-mode ablations can pick a specific
        # coefficient (e.g., the mean 0.45 instead of the default-first 0.3).
        default_mode_friction = [0.3, 0.4, 0.5, 0.6]
        override = self._runtime_config.get("mode_friction")
        if override is not None:
            self._mode_friction = [float(f) for f in override]
        else:
            self._mode_friction = default_mode_friction

        # Phase 2.2: optional regime bank. When set, mode_id indexes a list of
        # RegimeSpec (terrain + friction + manipuland) instead of a friction
        # scalar. Defaults to None → legacy crawling_ground behavior.
        self._regime_bank: Optional[List[Any]] = None
        if self.config.regime_bank_kind is not None:
            from genedynamics.envs.external.jax_mpm.tasks import (
                make_train_bank, make_test_bank,
            )
            kind = str(self.config.regime_bank_kind)
            if kind == "train_locomotion":
                self._regime_bank = make_train_bank(self._mpm_cfg.n_grid, task="locomotion")
            elif kind == "test_locomotion":
                self._regime_bank = make_test_bank(self._mpm_cfg.n_grid, task="locomotion")
            elif kind == "train_push":
                self._regime_bank = make_train_bank(self._mpm_cfg.n_grid, task="push")
            elif kind == "test_push":
                self._regime_bank = make_test_bank(self._mpm_cfg.n_grid, task="push")
            else:
                raise ValueError(f"Unknown regime_bank_kind: {kind!r}")

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
            request, self._scene, self._mpm_cfg, self._mode_friction,
            regime_bank=self._regime_bank,
            task=self.config.task,
            push_goal_x=self.config.push_goal_x,
            push_weights=tuple(self.config.push_weights),
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
