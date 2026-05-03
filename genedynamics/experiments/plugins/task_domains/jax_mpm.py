"""JAX-MPM task domain provider.

Provides task_spec resolution and evaluator creation for the JAX MPM
co-design pipeline. Self-contained — no external softzoo dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class _JaxMpmTaskSpec:
    """Minimal task spec the baseline platform consumes.

    The mrmfmbd / cmaes baselines read x_dim, phi_dim, and (optionally)
    num_modes off this object. Everything else is supplied via yaml extras.
    """
    task_id: str
    x_dim: int
    phi_dim: int
    num_modes: int = 4
    max_steps: int = 200


# Tasks supported by the jax_mpm backend. Add entries here as new ones come online.
# Phase 2.2 added "locomotion" (terrain bank) and "push" (terrain + manipuland).
_TASKS = ("crawling_ground", "locomotion", "push")


class JaxMpmTaskDomainProvider:
    """Provider for JAX-MPM-backed soft-robot tasks."""

    @property
    def domain_name(self) -> str:
        return "jax_mpm"

    def get_task_spec(self, task_id: str) -> Any:
        if task_id not in _TASKS:
            raise KeyError(f"Unknown jax_mpm task: {task_id!r}. Known: {list(_TASKS)}")
        from genedynamics.envs.external.jax_mpm.scene import MPMConfig
        cfg = MPMConfig()
        # Controller layout: W(n_act*K) + b(n_act) + g(n_act) + a(n_act) + c(n_act)
        phi_dim = cfg.n_actuators * cfg.n_sin_waves + 4 * cfg.n_actuators
        vx, vy, vz = cfg.voxel_dims
        x_dim = vx * vy * vz
        # num_modes scales with regime bank size for the new tasks; mrmfmbd
        # plumbing reads .num_modes for the marginalizer, which then folds the
        # full bank into the importance weight.
        if task_id == "locomotion":
            num_modes = 12   # 4 terrains × 3 frictions (writeup §10.3 train)
        elif task_id == "push":
            num_modes = 36   # 4 terrains × 3 frictions × 3 masses
        else:
            num_modes = 4
        return _JaxMpmTaskSpec(
            task_id=task_id, x_dim=int(x_dim), phi_dim=int(phi_dim),
            num_modes=int(num_modes),
        )

    def create_evaluator(
        self,
        project_root: str,
        *,
        max_workers: int = 0,
        cache_size: int = 64,
        **kwargs: Any,
    ) -> Any:
        from genedynamics.envs.evaluators import (
            JaxMpmRolloutEvaluator,
            JaxMpmEvaluatorConfig,
        )
        runtime_keys = ("dt", "gravity", "scale", "p_vol", "friction_coeff",
                        "actuation_strength_scale", "act_strength_base",
                        "mode_friction", "backward_penalty_weight")
        runtime_config = {k: kwargs[k] for k in runtime_keys if k in kwargs}
        voxel_dims = kwargs.get("voxel_dims")
        if voxel_dims is not None:
            voxel_dims = tuple(int(v) for v in voxel_dims)
        # Phase 2.2 evaluator-side config (default values keep crawling_ground
        # behavior identical when these keys are absent from yaml).
        cfg_kwargs = dict(
            max_workers=max_workers,
            cache_size=cache_size,
            reward_shaping_weight=float(kwargs.get("reward_shaping_weight", 100.0) or 100.0),
            n_grid=int(kwargs.get("n_grid", 64)),
            voxel_dims=voxel_dims,
            task=str(kwargs.get("task", "crawling_ground")),
            regime_bank_kind=kwargs.get("regime_bank_kind"),
            push_goal_x=float(kwargs.get("push_goal_x", 0.85)),
            # Phase 3: pre-robotized SoftBodySpec on disk.
            softbody_spec_path=kwargs.get("softbody_spec_path"),
        )
        if "push_weights" in kwargs and kwargs["push_weights"] is not None:
            cfg_kwargs["push_weights"] = tuple(float(w) for w in kwargs["push_weights"])
        cfg = JaxMpmEvaluatorConfig(**cfg_kwargs)
        return JaxMpmRolloutEvaluator(
            config=cfg,
            project_root=project_root,
            runtime_config=runtime_config,
        )

    def list_tasks(self) -> List[str]:
        return list(_TASKS)


def register() -> None:
    from genedynamics.experiments.framework.task_domain_provider import (
        register_task_domain_provider,
    )
    register_task_domain_provider(JaxMpmTaskDomainProvider())


register()
