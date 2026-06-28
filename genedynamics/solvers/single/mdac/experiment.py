"""MDAC experiment harness: build a fairly-configured (env, solver) for a method.

Lives in the MDAC solver package (NOT the general ``experiments/`` framework —
that stays algorithm-agnostic). A method name (``mdac`` / ``dial`` / ``mppi`` /
``mdac_no_*``) resolves to :class:`MethodFlags`; this harness routes those flags
to BOTH the env and the solver so every component is actually toggled (not a
no-op):

  use_soft_feasibility   -> backend uses the AL-augmented rollout (aug_rho)
  use_stiffness / log_spd_stiffness -> env ``stiffness_mode``
                           (none / log_spd / euclid / fixed)
  use_tangent_projection -> solver gets the env's ``manifold_geometry`` (else None)
  use_retraction         -> solver gets the CFS retraction (genemetry
                           ``CfsRetraction`` + ``env.manifold_residual`` filter,
                           wired like 2GO); lives in the geometry path
  use_adaptive_schedule  -> backend reads (margin, rho) off the current iterate's
                           constraint each reverse step and feeds the genemetry
                           ``ScheduleOverlay`` (2GO usage) for a kappa that scales
                           the geometry tracking — NON-monotonic; aug_rho and the
                           sampling sigma are left untouched (as in 2GO)
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
from genedynamics.solvers.single.mdac.core.method_registry import resolve_method, METHOD_TABLE

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
              surface_seed: int = 0, use_base: bool = False, prior: Any = None,
              aug_lambda: float = 2.0, aug_rho: float = 200.0,
              backend: Any = None, env_overrides: Optional[dict] = None,
              **cfg: Any) -> Tuple[Any, MDACSolver]:
    """Build (env, MDACSolver) for ``method``, routing its flags to env + solver.

    ``level`` selects the task family (arm S1-S4 / humanoid H1/H2/H4);
    ``surface_seed`` selects the random NURBS (arm S2-S4) or the humanoid H2 DR
    draw; ``use_base`` opens the humanoid H4-B 15D primitive (+v_base).
    ``env_overrides`` (the yaml ``env_params``) forwards physical/task env config
    to ``make_env``; ``stiffness_mode`` is method-derived and cannot be overridden."""
    flags = resolve_method(method)
    backend = backend or get_backend("jax")
    env_kw = {"stiffness_mode": stiffness_mode_for(method, flags)}
    if task == ARM_TASK:
        env_kw["surface_seed"] = surface_seed              # selects NURBS (S2-S4) + DR
        if level is not None:
            env_kw["level"] = level
    elif task == HUMANOID_TASK:
        env_kw["use_base"] = use_base                      # H4-B (+v_base)
        env_kw["dr_seed"] = surface_seed                   # H2 domain-randomization draw
        if level is not None:
            env_kw["level"] = level
    if env_overrides:                                      # yaml env_params (method owns stiffness_mode)
        env_kw.update({k: v for k, v in env_overrides.items() if k != "stiffness_mode"})
    env = make_env(task, **env_kw)
    geometry_fn = env.manifold_geometry if flags.use_tangent_projection else None
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


# --- baseline dispatcher: method -> (env, runner) on the SAME brax env + budget ---------
def _build_env(task, method, level, surface_seed, use_base, env_overrides):
    """Build the (medium-aware) env for a NON-MDAC baseline. Baselines SHARE the
    position-stiffness primitive (idea.txt) -> stiffness_mode='log_spd'."""
    env_kw = {"stiffness_mode": "log_spd"}
    if task == ARM_TASK:
        env_kw["surface_seed"] = surface_seed
        if level is not None:
            env_kw["level"] = level
    elif task == HUMANOID_TASK:
        env_kw["use_base"] = use_base
        env_kw["dr_seed"] = surface_seed
        if level is not None:
            env_kw["level"] = level
    if env_overrides:
        env_kw.update({k: v for k, v in env_overrides.items() if k != "stiffness_mode"})
    return make_env(task, **env_kw)


def _build_baseline_solver(method, env, backend, **cfg):
    """Construct the NON-MDAC baseline solver (its OWN registered solver: mppi/pegasusflow/
    atacom/issa) on the brax env. Each exposes ``run_receding(x0, n_steps, rng)`` natively."""
    Hsample = int(cfg.get("Hsample", 16))
    Hnode = int(cfg.get("Hnode", 4))
    Nsample = int(cfg.get("Nsample", 2048))
    Ndiffuse = int(cfg.get("Ndiffuse", 2))
    Ndiffuse_init = int(cfg.get("Ndiffuse_init", 10))
    action_limit = float(cfg.get("action_limit", 1.0))
    seed = int(cfg.get("seed", 0))
    if method == "mppi":
        from genedynamics.solvers.single.mppi.mppi import MPPISolver
        return MPPISolver(env, None, backend, Hsample=Hsample, Hnode=Hnode, Nsample=Nsample,
                          Ndiffuse=Ndiffuse, Ndiffuse_init=Ndiffuse_init,
                          noise_sigma=float(cfg.get("noise_sigma", 0.3)),
                          lambda_=float(cfg.get("lambda_", 1.0)), action_limit=action_limit, seed=seed)
    if method == "pegasusflow":
        from genedynamics.solvers.single.pegasusflow.pegasusflow import PegasusFlowSolver
        return PegasusFlowSolver(env, Hsample=Hsample, Hnode=Hnode, Nsample=Nsample,
                                 temp_sample=float(cfg.get("temp_sample", 0.1)), Ndiffuse=Ndiffuse,
                                 Ndiffuse_init=Ndiffuse_init, action_limit=action_limit, seed=seed)
    if method in ("atacom", "issa"):
        # RL baselines (trained brax policy, engine in learning/): ISSA = raw-action policy +
        # AdamBA safe-set projection at deploy; ATACOM = a manifold-resident tangent-space policy
        # (its ckpt is trained on the AtacomEnvWrapper, action_size = nu - n_f, NOT shared w/ ISSA).
        from genedynamics.learning.train_rl_policy import load_policy, build_policy_act
        ckpt = cfg.get("policy_ckpt")
        if ckpt is None:
            raise ValueError(f"'{method}' needs a trained 'policy_ckpt' "
                             f"(run scripts/tasks/robot/arm/train_rl_baseline.py "
                             f"{'--atacom ' if method == 'atacom' else ''}first)")
        act_fn = build_policy_act(*load_policy(ckpt))
        if method == "atacom":
            from genedynamics.solvers.single.atacom.atacom import AtacomSolver
            return AtacomSolver(env, act_fn, action_limit=action_limit, seed=seed)
        from genedynamics.solvers.single.issa.issa import IssaSolver
        return IssaSolver(env, act_fn, action_limit=action_limit, seed=seed)
    raise NotImplementedError(f"baseline method '{method}' not recognized")


def make_controller(task, method="mdac", *, level=None, surface_seed=0, use_base=False,
                    prior=None, aug_lambda=2.0, aug_rho=200.0, backend=None,
                    env_overrides=None, **cfg):
    """Build ``(env, runner)`` for ANY method on the SAME brax env at the SAME budget:
    an MDAC variant (``MDACSolver`` via ``make_mdac``), a sampling baseline (``mppi`` /
    ``pegasusflow``), or an RL baseline (``atacom`` / ``issa``). All expose
    ``runner.run_receding(x0, n_steps, rng) -> RecedingHorizonResult``."""
    if method in METHOD_TABLE:                       # MDAC variant (incl. dial/mbd/anchors)
        return make_mdac(task, method, level=level, surface_seed=surface_seed, use_base=use_base,
                         prior=prior, aug_lambda=aug_lambda, aug_rho=aug_rho, backend=backend,
                         env_overrides=env_overrides, **cfg)
    env = _build_env(task, method, level, surface_seed, use_base, env_overrides)
    backend = backend or get_backend("jax")
    return env, _build_baseline_solver(method, env, backend, **cfg)


def metrics_plugin_for(task: str):
    """The general metrics plugin for a task (shared library + task extractor)."""
    from genedynamics.experiments.plugins.metrics.extractors import (
        arm_surface_scan_metrics_plugin, humanoid_box_push_metrics_plugin,
    )
    return (arm_surface_scan_metrics_plugin() if task == ARM_TASK
            else humanoid_box_push_metrics_plugin())


__all__ = ["ARM_TASK", "HUMANOID_TASK", "stiffness_mode_for", "make_mdac",
           "make_controller", "metrics_plugin_for"]
