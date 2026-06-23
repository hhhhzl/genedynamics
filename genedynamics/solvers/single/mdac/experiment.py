"""MDAC experiment harness: build a fairly-configured (env, solver) for a method.

Lives in the MDAC solver package (NOT the general ``experiments/`` framework —
that stays algorithm-agnostic). A method name (``mdac`` / ``dial`` / ``mppi`` /
``mdac_no_*``) resolves to :class:`MethodFlags`; this harness routes those flags
to BOTH the env and the solver so every component is actually toggled (not a
no-op):

  use_soft_feasibility   -> backend uses the AL-augmented rollout (aug_rho)
  use_stiffness / log_spd_stiffness -> env ``stiffness_mode``
                           (none / log_spd / euclid / fixed)
  use_tangent_projection -> solver gets the env's ``mdac_geometry_fn`` (else None)
  use_retraction         -> solver gets the CFS retraction (genemetry
                           ``CfsRetraction`` + ``env.mdac_constraint`` filter,
                           wired like 2GO); lives in the geometry path
  use_adaptive_schedule  -> backend couples rho_k / kappa_k up across reverse
                           steps (genemetry ``ScheduleOverlay``)
  use_rl_prior           -> solver gets the ``prior`` (else None)

All methods share one CFG so the sample budget ``(Nsample, Hsample, Ndiffuse)``
is identical (``assert_fair``). All six of the above are docker-validated active
(``test_mdac_ablation_docker``). ``use_mb_rollout`` is inert by design — the
parallel kernel IS the model-based rollout (weighted-mean over rolled rewards).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from genedynamics.core import get_backend
from genedynamics.envs.factories import make_env
from genedynamics.solvers.single.mdac.mdac import MDACSolver
from genedynamics.solvers.single.mdac.core.method_registry import resolve_method

ARM_TASK = "manipulator_surface_scan"
HUMANOID_TASK = "humanoid_box_push"


def stiffness_mode_for(method: str, flags) -> str:
    """Map flags (+ method name) to the env stiffness chart."""
    if not flags.use_stiffness:
        return "none"
    if flags.log_spd_stiffness:
        return "log_spd"
    return "fixed" if "fixed" in method else "euclid"


def make_mdac(task: str, method: str = "mdac", *, level: Optional[str] = None,
              prior: Any = None, aug_lambda: float = 2.0, aug_rho: float = 200.0,
              backend: Any = None, **cfg: Any) -> Tuple[Any, MDACSolver]:
    """Build (env, MDACSolver) for ``method``, routing its flags to env + solver."""
    flags = resolve_method(method)
    backend = backend or get_backend("jax")
    env_kw = {"stiffness_mode": stiffness_mode_for(method, flags)}
    if level is not None and task == ARM_TASK:
        env_kw["level"] = level
    env = make_env(task, **env_kw)
    geometry_fn = env.mdac_geometry_fn if flags.use_tangent_projection else None
    # CFS retraction (genemetry CfsRetraction + MDAC filter_fn, wired like 2GO);
    # lives inside the geometry path, so it needs the tangent projection on too.
    retraction = None
    if flags.use_tangent_projection and flags.use_retraction:
        from genedynamics.solvers.single.mdac.core.retraction import make_mdac_retraction
        retraction = make_mdac_retraction(env)
    solver = MDACSolver(
        env, None, backend, method=method,
        geometry_fn=geometry_fn, retraction=retraction,
        prior=(prior if flags.use_rl_prior else None),
        aug_lambda=aug_lambda, aug_rho=aug_rho, **cfg,
    )
    return env, solver


def metrics_plugin_for(task: str):
    """The general metrics plugin for a task (shared library + task extractor)."""
    from genedynamics.experiments.plugins.metrics.extractors import (
        arm_surface_scan_metrics_plugin, humanoid_box_push_metrics_plugin,
    )
    return (arm_surface_scan_metrics_plugin() if task == ARM_TASK
            else humanoid_box_push_metrics_plugin())


__all__ = ["ARM_TASK", "HUMANOID_TASK", "stiffness_mode_for", "make_mdac", "metrics_plugin_for"]
