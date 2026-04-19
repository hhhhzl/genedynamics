"""JAX-MPM task domain provider.

Serves the same `crawling_ground` task as softzoo but backed by our JAX MPM
simulator. Reuses the softzoo task_spec schema so the rest of the pipeline
(baseline_platform, MRMFMBD, CMA-ES, reporters) doesn't need to care.
"""

from __future__ import annotations

from typing import Any, List


class JaxMpmTaskDomainProvider:
    """Provider for JAX-MPM-backed soft-robot tasks."""

    @property
    def domain_name(self) -> str:
        return "jax_mpm"

    def get_task_spec(self, task_id: str) -> Any:
        # Reuse softzoo's task registry — modes/fidelity semantics are identical;
        # JaxMpmRolloutEvaluator reads `modes[*].friction` and maps fidelity_level
        # through its own FIDELITY_STEPS table. SoftZooTaskSpec is frozen and has
        # no phi_dim/x_dim fields, so we proxy it and expose those attributes for
        # the controller: W(n_act*K) + b(n_act) + g(n_act) + a(n_act) + c(n_act) = 80.
        from genedynamics.envs.external.jax_mpm.scene import MPMConfig
        from genedynamics.envs.external.softzoo.task_registry import get_task_spec
        spec = get_task_spec(task_id)
        cfg = MPMConfig()
        phi_dim = cfg.n_actuators * cfg.n_sin_waves + 4 * cfg.n_actuators
        vx, vy, vz = cfg.voxel_dims
        x_dim = vx * vy * vz   # voxel occupancy = morphology

        class _JaxMpmSpecProxy:
            def __init__(self, inner, x_dim, phi_dim):
                object.__setattr__(self, "_inner", inner)
                object.__setattr__(self, "phi_dim", int(phi_dim))
                object.__setattr__(self, "x_dim", int(x_dim))
            def __getattr__(self, name):
                return getattr(self._inner, name)

        return _JaxMpmSpecProxy(spec, x_dim, phi_dim)

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
                        "actuation_strength_scale", "act_strength_base")
        runtime_config = {k: kwargs[k] for k in runtime_keys if k in kwargs}
        voxel_dims = kwargs.get("voxel_dims")
        if voxel_dims is not None:
            voxel_dims = tuple(int(v) for v in voxel_dims)
        cfg = JaxMpmEvaluatorConfig(
            max_workers=max_workers,
            cache_size=cache_size,
            reward_shaping_weight=float(kwargs.get("reward_shaping_weight", 100.0) or 100.0),
            n_grid=int(kwargs.get("n_grid", 64)),
            voxel_dims=voxel_dims,
        )
        return JaxMpmRolloutEvaluator(
            config=cfg,
            project_root=project_root,
            runtime_config=runtime_config,
        )

    def list_tasks(self) -> List[str]:
        from genedynamics.envs.external.softzoo.task_registry import list_tasks
        return list_tasks()


def register() -> None:
    from genedynamics.experiments.framework.task_domain_provider import (
        register_task_domain_provider,
    )
    register_task_domain_provider(JaxMpmTaskDomainProvider())


register()
