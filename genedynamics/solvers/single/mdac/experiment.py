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
                           wired like 2GO), independently of tangent projection
  use_force_manifold     -> arm CLEAN manifold includes commanded-force equality
                           (off = position-only reliable geometry diagnostic)
    use_horizon_geometry   -> map existing node spline to dense incremental controls
                           and use the env's cumulative, time-indexed residual
  use_realization_compensation -> use the task-owned frozen real-EE coordinate
                           bias in that horizon residual
  use_controllability_geometry -> estimate a stop-gradient true-dynamics local
                           response map and lift its correction into the residual
  use_geometry_gate      -> route env realization reliability into the backend's
                           raw/projected and raw/retracted blend
  component_geometry_gate -> action-block gate (else scalar-gate ablation)
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
              policy_ckpt: Optional[str] = None,
              atacom_policy_ckpt: Optional[str] = None,
              reliability_ckpt: Optional[str] = None,
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
        env_kw["clean_manifold_force"] = flags.use_force_manifold
        if level is not None:
            env_kw["level"] = level
    elif task == HUMANOID_TASK:
        env_kw["use_base"] = use_base                      # H4-B (+v_base)
        env_kw["dr_seed"] = surface_seed                   # H2 domain-randomization draw
        if level is not None:
            env_kw["level"] = level
    if env_overrides:                                      # method owns chart/manifold composition
        env_kw.update({
            k: v for k, v in env_overrides.items()
            if k not in ("stiffness_mode", "clean_manifold_force")
        })
    env = make_env(task, **env_kw)
    if prior is None and policy_ckpt is not None:
        from genedynamics.learning.train_rl_policy import (
            build_policy_prior,
            load_policy,
        )

        params, policy_config = load_policy(policy_ckpt)
        if int(policy_config["action_size"]) != int(env.action_size):
            raise ValueError(
                "policy/environment action mismatch: "
                f"{policy_config['action_size']} != {env.action_size}"
            )
        if int(policy_config["observation_size"]) != int(env.observation_size):
            raise ValueError(
                "policy/environment observation mismatch: "
                f"{policy_config['observation_size']} != {env.observation_size}"
            )
        prior = build_policy_prior(
            params,
            policy_config,
            env=env,
            Hsample=int(cfg.get("Hsample", 16)),
            Hnode=int(cfg.get("Hnode", 4)),
            ctrl_dt=float(cfg.get("ctrl_dt", 0.02)),
        )
    atacom_prior = None
    if atacom_policy_ckpt is not None:
        from genedynamics.learning.priors.rl import RLPrior
        from genedynamics.learning.train_rl_policy import load_policy
        from genedynamics.solvers.single.atacom.backends.atacom_jax import (
            atacom_null_dim,
        )
        from genedynamics.solvers.single.atacom.prior import (
            AtacomHorizonPrior,
        )

        atacom_params, atacom_config = load_policy(atacom_policy_ckpt)
        expected = atacom_null_dim(env)
        if int(atacom_config.get("action_size", -1)) != expected:
            raise ValueError(
                "ATACOM prior action mismatch: checkpoint has "
                f"{atacom_config.get('action_size')}, expected {expected}"
            )
        if int(atacom_config.get("observation_size", -1)) != int(env.observation_size):
            raise ValueError("ATACOM prior observation mismatch")
        if "atacom" not in str(atacom_config.get("protocol", "")):
            raise ValueError("ATACOM prior requires a tangent-policy checkpoint")
        tangent_prior = RLPrior(
            backend="jax",
            observation_size=atacom_config["observation_size"],
            action_size=atacom_config["action_size"],
            params=atacom_params,
            normalize_observations=atacom_config["normalize_observations"],
            policy_hidden_layer_sizes=atacom_config["policy_hidden_layer_sizes"],
            deterministic=True,
        )
        atacom_prior = AtacomHorizonPrior(
            env, tangent_prior,
            Hsample=int(cfg.get("Hsample", 16)),
            Hnode=int(cfg.get("Hnode", 4)),
            ctrl_dt=float(cfg.get("ctrl_dt", 0.02)),
            Kc=float(cfg.get("atacom_Kc", 1.0)),
            action_limit=float(cfg.get("action_limit", 1.0)),
        )
    reliability_model = None
    if reliability_ckpt is not None:
        if not hasattr(env, "reliability_features"):
            raise ValueError(
                f"method '{method}' requires env.reliability_features"
            )
        from genedynamics.learning.reliability import LinearReliabilityModel

        reliability_model = LinearReliabilityModel.load(reliability_ckpt)
    residual_fn = None
    if flags.use_horizon_geometry:
        if flags.use_controllability_geometry:
            residual_hook_name = "manifold_residual_horizon_controllable"
        elif flags.use_realization_compensation:
            residual_hook_name = "manifold_residual_horizon_realized"
        else:
            residual_hook_name = "manifold_residual_horizon"
        if not hasattr(env, residual_hook_name):
            raise ValueError(
                f"method '{method}' requires env.{residual_hook_name}"
            )
        residual_hook = getattr(env, residual_hook_name)
        from genedynamics.solvers.single.dial.spline import NodeSpline
        spline = NodeSpline.build(
            int(cfg.get("Hnode", 4)),
            int(cfg.get("Hsample", 16)),
            float(cfg.get("ctrl_dt", 0.02)),
        )

        def residual_fn(state, nodes, t0):
            return residual_hook(state, spline.node2u(nodes), t0)

        if flags.use_tangent_projection:
            import jax

            def geometry_fn(state, nodes, t0):
                objective = lambda y: 0.5 * jax.numpy.mean(
                    residual_fn(state, y, t0) ** 2
                )
                return jax.grad(objective)(nodes)
        else:
            geometry_fn = None
    else:
        geometry_fn = env.manifold_geometry if flags.use_tangent_projection else None

    geometry_gate_fn = None
    # A controllability gate attenuates only the additional response-map target
    # lead inside the task-owned residual (see prepare_realization_context).
    # The clean path/force projection and retraction must remain active so an
    # unreliable/contact-loss state can recover.  Legacy scalar/component gate
    # methods without a controllability lift keep the backend blend below.
    if flags.use_geometry_gate and not flags.use_controllability_geometry:
        if not hasattr(env, "geometry_reliability"):
            raise ValueError(f"method '{method}' requires env.geometry_reliability")
        import jax.numpy as jnp

        def geometry_gate_fn(state, nodes, t0):
            diag = env.geometry_reliability(state)
            action = (
                diag["action"]
                if flags.component_geometry_gate
                else jnp.ones_like(diag["action"]) * diag["scalar"]
            )
            return {**diag, "action": action}

    prepare_state_fn = None
    if flags.use_controllability_geometry:
        def prepare_state_fn(state, nodes, t0):
            return env.prepare_realization_context(
                state, spline.node2u(nodes)
            )

    # CFS retraction is independent of tangent shaping so each mechanism has a
    # genuine single-factor ablation.
    retraction = None
    if flags.use_retraction:
        from genedynamics.solvers.single.mdac.core.retraction import make_mdac_retraction
        retraction = make_mdac_retraction(env, residual_fn=residual_fn)
    # Acceptance remains solver-generic: the task owns the semantics and
    # normalization of force/contact/deformation risk.
    risk_fn = getattr(env, "sequence_risk", None)
    solver = MDACSolver(
        env, None, backend, method=method,
        geometry_fn=geometry_fn, retraction=retraction,
        geometry_gate_fn=geometry_gate_fn,
        prepare_state_fn=prepare_state_fn,
        prior=(prior if flags.use_rl_prior else None),
        atacom_prior=(atacom_prior if flags.use_rl_prior else None),
        risk_fn=risk_fn,
        reliability_model=reliability_model,
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
                                 Ndiffuse_init=Ndiffuse_init,
                                 noise_sigma=float(cfg.get("noise_sigma", 1.0)),
                                 sigma_decay=float(cfg.get("sigma_decay", 0.9)),
                                 update_method=str(cfg.get("update_method", "avwbfo")),
                                 wbfo_gamma=float(cfg.get("wbfo_gamma", 1.0)),
                                 noise_scheduler_type=str(cfg.get("noise_scheduler_type", "s2")),
                                 noise_shape_fn=str(cfg.get("noise_shape_fn", "linear")),
                                 noise_decay_fn=str(cfg.get("noise_decay_fn", "exponential")),
                                 noise_final_ratio=float(cfg.get("noise_final_ratio", 0.1)),
                                 action_limit=action_limit, seed=seed)
    if method in ("rl", "atacom", "issa"):
        # RL baselines (trained brax policy, engine in learning/): ISSA = raw-action policy +
        # AdamBA safe-set projection at deploy; ATACOM = a manifold-resident tangent-space policy
        # (its ckpt is trained on the AtacomEnvWrapper, action_size = nu - n_f, NOT shared w/ ISSA).
        from genedynamics.learning.train_rl_policy import load_policy, build_policy_act
        ckpt = cfg.get("policy_ckpt")
        if ckpt is None:
            raise ValueError(f"'{method}' needs a trained 'policy_ckpt' "
                             f"(run scripts/tasks/robot/arm/train_rl_baseline.py "
                             f"{'--atacom ' if method == 'atacom' else ''}first)")
        params, policy_config = load_policy(ckpt)
        expected_action_size = int(env.action_size)
        if method == "atacom":
            from genedynamics.solvers.single.atacom.backends.atacom_jax import (
                atacom_null_dim,
            )
            expected_action_size = atacom_null_dim(env)
            protocol = str(policy_config.get("protocol", ""))
            if "atacom" not in protocol:
                raise ValueError(
                    "ATACOM requires a dedicated tangent-policy checkpoint; "
                    f"got protocol={protocol!r}"
                )
        if int(policy_config.get("action_size", -1)) != expected_action_size:
            raise ValueError(
                f"{method} policy action mismatch: checkpoint has "
                f"{policy_config.get('action_size')}, expected {expected_action_size}"
            )
        if int(policy_config.get("observation_size", -1)) != int(env.observation_size):
            raise ValueError(
                f"{method} policy observation mismatch: checkpoint has "
                f"{policy_config.get('observation_size')}, expected {env.observation_size}"
            )
        act_fn = build_policy_act(params, policy_config)
        if method == "rl":
            from genedynamics.solvers.common.rl_policy_controller import (
                RLPolicyController,
            )
            return RLPolicyController(env, act_fn, seed=seed)
        if method == "atacom":
            from genedynamics.solvers.single.atacom.atacom import AtacomSolver
            return AtacomSolver(
                env, act_fn, Kc=float(cfg.get("Kc", 1.0)),
                action_limit=action_limit, seed=seed,
            )
        from genedynamics.solvers.single.issa.issa import IssaSolver
        return IssaSolver(
            env, act_fn,
            n_dirs=int(cfg.get("n_dirs", 20)),
            n_iters=int(cfg.get("n_iters", 50)),
            bound=float(cfg.get("adamba_bound", 1e-4)),
            threshold=float(cfg.get("safety_threshold", 0.0)),
            enforce_absolute=bool(cfg.get("enforce_absolute", True)),
            action_limit=action_limit, seed=seed,
        )
    raise NotImplementedError(f"baseline method '{method}' not recognized")


def make_controller(task, method="mdac", *, level=None, surface_seed=0, use_base=False,
                    prior=None, policy_ckpt=None, atacom_policy_ckpt=None,
                    aug_lambda=2.0, aug_rho=200.0, backend=None,
                    env_overrides=None, **cfg):
    """Build ``(env, runner)`` for ANY method on the SAME brax env at the SAME budget:
    an MDAC variant (``MDACSolver`` via ``make_mdac``), a sampling baseline (``mppi`` /
    ``pegasusflow``), or an RL baseline (raw ``rl`` / ``atacom`` / ``issa``). All expose
    ``runner.run_receding(x0, n_steps, rng) -> RecedingHorizonResult``."""
    if method in METHOD_TABLE:                       # MDAC variant (incl. dial/mbd/anchors)
        return make_mdac(task, method, level=level, surface_seed=surface_seed, use_base=use_base,
                         prior=prior, policy_ckpt=policy_ckpt,
                         atacom_policy_ckpt=atacom_policy_ckpt,
                         aug_lambda=aug_lambda, aug_rho=aug_rho, backend=backend,
                         env_overrides=env_overrides, **cfg)
    env = _build_env(task, method, level, surface_seed, use_base, env_overrides)
    backend = backend or get_backend("jax")
    return env, _build_baseline_solver(
        method, env, backend, policy_ckpt=policy_ckpt, **cfg
    )


def metrics_plugin_for(task: str):
    """The general metrics plugin for a task (shared library + task extractor)."""
    from genedynamics.experiments.plugins.metrics.extractors import (
        arm_surface_scan_metrics_plugin, humanoid_box_push_metrics_plugin,
    )
    return (arm_surface_scan_metrics_plugin() if task == ARM_TASK
            else humanoid_box_push_metrics_plugin())


__all__ = ["ARM_TASK", "HUMANOID_TASK", "stiffness_mode_for", "make_mdac",
           "make_controller", "metrics_plugin_for"]
