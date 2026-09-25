"""Independent unified-runner adapters for receding contact controllers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from genedynamics.core import get_backend
from genedynamics.core.types import ExecutionRejected, Trajectory
from genedynamics.envs.factories import make_env
from genedynamics.experiments.framework.base import MethodPlugin
from genedynamics.experiments.plugins.environments._contact_task import (
    ARM_TASK,
    HUMANOID_TASK,
    INSERT_TASK,
    stiffness_mode_for,
)
from genedynamics.solvers.single.mga.core.method_registry import (
    METHOD_TABLE,
    canonical_method_name,
    resolve_method,
)
from genedynamics.solvers.single.mga.mga import MGASolver


def _validate_policy_interface(
    env, policy_config, *, label="policy", atacom=None, execution_env=None,
):
    """Honor a task-owned semantic contract when shape checks are insufficient."""
    expected = getattr(env, "policy_interface", None)
    if execution_env is not None:
        actual = getattr(execution_env, "policy_interface", None)
        if expected != actual and (expected is not None or actual is not None):
            raise ValueError(f"{label} model/execution policy interface mismatch")
    require_atacom_contract = atacom is not None and (
        expected is not None
        or callable(getattr(env, "atacom_constraint_residual", None))
    )
    if require_atacom_contract:
        contract = dict(atacom)
        recorded_transform = policy_config.get("atacom_transform")
        if (
            not callable(getattr(env, "atacom_constraint_residual", None))
            and isinstance(recorded_transform, dict)
            and "alpha_limit" not in recorded_transform
        ):
            # Preserve already-frozen Humanoid checkpoints created before the
            # tangent/output limit names were separated for PegInsert.
            contract.pop("alpha_limit", None)
        transform = {
            **contract,
            "time_step": float(getattr(getattr(env, "_config", None), "dt", 0.02)),
        }
        if recorded_transform != transform:
            raise ValueError(f"{label} ATACOM training/deployment transform mismatch")
    if expected is None:
        return  # Preserve legacy H1 and other tasks' checkpoint behavior.
    if policy_config.get("policy_interface") != expected:
        raise ValueError(
            f"{label} policy interface mismatch: a checkpoint trained for the "
            "active task/action semantics is required (matching dimensions are insufficient)"
        )
    expected_action_transform = getattr(env, "policy_action_transform", None)
    if expected_action_transform is not None and atacom is None:
        if policy_config.get("policy_action_transform") != expected_action_transform:
            raise ValueError(
                f"{label} policy action transform mismatch: the checkpoint must "
                "use the task-owned residual chart for the active absolute action semantics"
            )
        for key in ("action_bias", "action_scale"):
            if policy_config.get(key) != expected_action_transform[key]:
                raise ValueError(f"{label} {key} does not match its policy action transform")


def _task_env_kwargs(
    task: str,
    *,
    method: str,
    level: Optional[str],
    seed: int,
    use_base: bool,
    overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve task-owned construction arguments for direct probe callers."""
    method = canonical_method_name(method)
    if method in METHOD_TABLE:
        flags = resolve_method(method)
        kwargs: Dict[str, Any] = {
            "stiffness_mode": stiffness_mode_for(method, flags),
        }
        if task == ARM_TASK:
            kwargs["clean_manifold_force"] = flags.use_force_manifold
    else:
        kwargs = {"stiffness_mode": "log_spd"}
    if task == ARM_TASK:
        kwargs["surface_seed"] = seed
    elif task == HUMANOID_TASK:
        kwargs.update(use_base=use_base, dr_seed=seed)
    elif task == INSERT_TASK:
        kwargs["domain_seed"] = seed
    else:
        raise ValueError(f"unsupported contact task: {task!r}")
    if level is not None:
        kwargs["level"] = level
    if overrides:
        protected = {"stiffness_mode", "clean_manifold_force"}
        kwargs.update({k: v for k, v in overrides.items() if k not in protected})
    return kwargs


def make_mga(
    task: str,
    method: str = "mga_base",
    *,
    level: Optional[str] = None,
    surface_seed: int = 0,
    use_base: bool = False,
    prior: Any = None,
    policy_ckpt: Optional[str] = None,
    atacom_policy_ckpt: Optional[str] = None,
    reliability_ckpt: Optional[str] = None,
    aug_lambda: float = 2.0,
    aug_rho: float = 200.0,
    backend: Any = None,
    env_overrides: Optional[dict] = None,
    execution_env_overrides: Optional[dict] = None,
    model_env: Any = None,
    execution_env: Any = None,
    **cfg: Any,
) -> Tuple[Any, MGASolver]:
    """Compose one MGA controller for a contact-task experiment adapter.

    The public name ``mga`` resolves to ``mga_controllable_gate``; direct
    regression probes can request ``mga_base`` explicitly.
    """
    method = canonical_method_name(method)
    prior_rollout_mode = str(
        cfg.pop("prior_rollout_mode", "closed_loop")
    ).lower()
    if prior_rollout_mode not in {"closed_loop", "current_observation"}:
        raise ValueError(
            "prior_rollout_mode must be 'closed_loop' or "
            "'current_observation'"
        )
    if (
        prior_rollout_mode == "current_observation"
        and int(cfg.get("prior_stochastic_samples", 0)) > 0
    ):
        raise ValueError(
            "current_observation prior does not support stochastic closed-loop "
            "horizon samples"
        )
    flags = resolve_method(method)
    backend = backend or get_backend("jax")
    env_kwargs = _task_env_kwargs(
        task,
        method=method,
        level=level,
        seed=surface_seed,
        use_base=use_base,
        overrides=env_overrides,
    )
    env = model_env if model_env is not None else make_env(task, **env_kwargs)
    if execution_env is None and execution_env_overrides:
        execution_kwargs = dict(env_kwargs)
        execution_kwargs.update(execution_env_overrides)
        execution_env = make_env(task, **execution_kwargs)

    if prior is None and policy_ckpt is not None:
        from genedynamics.learning.train_rl_policy import (
            build_policy_prior,
            load_policy,
        )

        params, policy_config = load_policy(policy_ckpt)
        _validate_policy_interface(
            env, policy_config, label="MGA prior", execution_env=execution_env,
        )
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
            rollout_mode=prior_rollout_mode,
        )

    atacom_prior = None
    if atacom_policy_ckpt is not None:
        from genedynamics.learning.priors.rl import RLPrior
        from genedynamics.learning.train_rl_policy import load_policy
        from genedynamics.solvers.single.atacom.backends.atacom_jax import (
            atacom_null_dim,
        )
        from genedynamics.solvers.single.atacom.prior import AtacomHorizonPrior

        atacom_params, atacom_config = load_policy(atacom_policy_ckpt)
        _validate_policy_interface(env, atacom_config, label="ATACOM prior", execution_env=execution_env, atacom={
            "Kc": float(cfg.get("atacom_Kc", 1.0)),
            "action_limit": float(cfg.get("action_limit", 1.0)),
            "alpha_limit": float(cfg.get(
                "atacom_alpha_limit", cfg.get("action_limit", 1.0)
            )),
        })
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
            env,
            tangent_prior,
            Hsample=int(cfg.get("Hsample", 16)),
            Hnode=int(cfg.get("Hnode", 4)),
            ctrl_dt=float(cfg.get("ctrl_dt", 0.02)),
            Kc=float(cfg.get("atacom_Kc", 1.0)),
            action_limit=float(cfg.get("action_limit", 1.0)),
            alpha_limit=float(cfg.get(
                "atacom_alpha_limit", cfg.get("action_limit", 1.0)
            )),
        )

    reliability_model = None
    if reliability_ckpt is not None:
        if not (
            hasattr(env, "reliability_features")
            or hasattr(env, "reliability_features_sequence")
        ):
            raise ValueError(f"method '{method}' requires a reliability feature hook")
        from genedynamics.learning.reliability import LinearReliabilityModel

        validator = getattr(env, "validate_reliability_checkpoint", None)
        if validator is not None:
            import json
            from pathlib import Path

            horizon_steps = int(cfg.get("Hsample", 16)) + 1
            payload = json.loads(Path(reliability_ckpt).read_text())
            # An unvalidated checkpoint may be loaded only as a diagnostic
            # under the explicit model-based abstention policy.  It must not
            # veto candidates (the backend checks promotion metadata); a
            # promoted/authoritative gate still requires an exact active-task
            # contract match.
            allow_abstaining_ood = bool(
                cfg.get("reliability_ood_policy") == "model_based"
                and not bool((payload.get("metadata") or {}).get(
                    "performance_validated", False
                ))
                and not bool((payload.get("metadata") or {}).get(
                    "promotion_eligible", False
                ))
            )
            validator(
                payload,
                horizon_steps=horizon_steps,
                allow_abstaining_ood=allow_abstaining_ood,
            )
            if execution_env is not None:
                execution_validator = getattr(execution_env, "validate_reliability_checkpoint", None)
                if execution_validator is None:
                    raise ValueError("reliability model/execution task contract mismatch")
                execution_validator(
                    payload,
                    horizon_steps=horizon_steps,
                    allow_abstaining_ood=allow_abstaining_ood,
                )
        reliability_model = LinearReliabilityModel.load(reliability_ckpt)
        expected_features = getattr(env, "reliability_feature_size", None)
        if (
            expected_features is not None
            and len(reliability_model.feature_names) != int(expected_features)
        ):
            raise ValueError(
                "reliability checkpoint/environment feature mismatch: "
                f"{len(reliability_model.feature_names)} != {expected_features}"
            )

    residual_fn = None
    spline = None
    if flags.use_horizon_geometry:
        if flags.use_controllability_geometry:
            hook_name = "manifold_residual_horizon_controllable"
        elif flags.use_realization_compensation:
            hook_name = "manifold_residual_horizon_realized"
        else:
            hook_name = "manifold_residual_horizon"
        if not hasattr(env, hook_name):
            raise ValueError(f"method '{method}' requires env.{hook_name}")
        residual_hook = getattr(env, hook_name)
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
    if flags.use_geometry_gate:
        if not hasattr(env, "geometry_reliability"):
            raise ValueError(f"method '{method}' requires env.geometry_reliability")
        import jax.numpy as jnp

        def geometry_gate_fn(state, nodes, t0):
            del nodes, t0
            diagnostics = env.geometry_reliability(state)
            action = (
                diagnostics["action"]
                if flags.component_geometry_gate
                else jnp.ones_like(diagnostics["action"]) * diagnostics["scalar"]
            )
            return {**diagnostics, "action": action}

    prepare_state_fn = None
    if flags.use_controllability_geometry:
        if spline is None:
            raise ValueError("controllability geometry requires horizon geometry")

        def prepare_state_fn(state, nodes, t0):
            del t0
            return env.prepare_realization_context(
                state,
                spline.node2u(nodes),
                gate_controllability=flags.use_geometry_gate,
            )

    # Projection is part of the controller's nominal model.  The execution
    # environment may contain hidden hole pose/clearance/friction and must only
    # advance the physical state; observed execution signals still arrive in
    # ``state.info`` and are therefore available to the nominal task hook.
    candidate_projection_fn = (
        getattr(env, "project_mga_candidate", None)
        if flags.use_geometry_gate
        else None
    )
    retraction = None
    if flags.use_retraction:
        from genedynamics.solvers.single.mga.core.retraction import make_mga_retraction

        retraction = make_mga_retraction(env, residual_fn=residual_fn)
    solver = MGASolver(
        env,
        None,
        backend,
        method=method,
        step_fn=(execution_env.step if execution_env is not None else None),
        geometry_fn=geometry_fn,
        retraction=retraction,
        geometry_gate_fn=geometry_gate_fn,
        prepare_state_fn=prepare_state_fn,
        candidate_projection_fn=candidate_projection_fn,
        prior=(prior if flags.use_rl_prior else None),
        atacom_prior=(atacom_prior if flags.use_rl_prior else None),
        risk_fn=getattr(env, "sequence_risk", None),
        reliability_model=reliability_model,
        aug_lambda=aug_lambda,
        aug_rho=aug_rho,
        **cfg,
    )
    solver.execution_env = execution_env
    return env, solver


def _build_baseline_solver(method, env, backend, *, execution_env=None, **cfg):
    """Construct an independent baseline controller at the shared budget."""
    Hsample = int(cfg.get("Hsample", 16))
    Hnode = int(cfg.get("Hnode", 4))
    Nsample = int(cfg.get("Nsample", 2048))
    Ndiffuse = int(cfg.get("Ndiffuse", 2))
    Ndiffuse_init = int(cfg.get("Ndiffuse_init", 10))
    action_limit = float(cfg.get("action_limit", 1.0))
    seed = int(cfg.get("seed", 0))
    if method == "mppi":
        from genedynamics.solvers.single.mppi.mppi import MPPISolver

        return MPPISolver(
            env, None, backend, Hsample=Hsample, Hnode=Hnode,
            Nsample=Nsample, Ndiffuse=Ndiffuse, Ndiffuse_init=Ndiffuse_init,
            noise_sigma=float(cfg.get("noise_sigma", 0.3)),
            lambda_=float(cfg.get("lambda_", 1.0)),
            action_limit=action_limit,
            step_fn=(execution_env.step if execution_env is not None else None),
            seed=seed,
        )
    if method == "pegasusflow":
        from genedynamics.solvers.single.pegasusflow.pegasusflow import PegasusFlowSolver

        return PegasusFlowSolver(
            env, Hsample=Hsample, Hnode=Hnode, Nsample=Nsample,
            temp_sample=float(cfg.get("temp_sample", 0.1)),
            Ndiffuse=Ndiffuse, Ndiffuse_init=Ndiffuse_init,
            noise_sigma=float(cfg.get("noise_sigma", 1.0)),
            sigma_decay=float(cfg.get("sigma_decay", 0.9)),
            update_method=str(cfg.get("update_method", "avwbfo")),
            wbfo_gamma=float(cfg.get("wbfo_gamma", 1.0)),
            noise_scheduler_type=str(cfg.get("noise_scheduler_type", "s2")),
            noise_shape_fn=str(cfg.get("noise_shape_fn", "linear")),
            noise_decay_fn=str(cfg.get("noise_decay_fn", "exponential")),
            noise_final_ratio=float(cfg.get("noise_final_ratio", 0.1)),
            action_limit=action_limit,
            step_fn=(execution_env.step if execution_env is not None else None),
            seed=seed,
        )
    if method in {"rl", "atacom", "issa"}:
        from genedynamics.learning.train_rl_policy import build_policy_act, load_policy

        checkpoint = cfg.get("policy_ckpt")
        if checkpoint is None:
            raise ValueError(f"'{method}' needs a trained 'policy_ckpt'")
        params, policy_config = load_policy(checkpoint)
        _validate_policy_interface(
            env, policy_config, label=method,
            execution_env=execution_env,
            atacom={
                "Kc": float(cfg.get("Kc", 1.0)),
                "action_limit": action_limit,
                "alpha_limit": float(cfg.get("alpha_limit", action_limit)),
            }
            if method == "atacom" else None,
        )
        expected_action_size = int(env.action_size)
        if method == "atacom":
            from genedynamics.solvers.single.atacom.backends.atacom_jax import atacom_null_dim

            expected_action_size = atacom_null_dim(env)
            if "atacom" not in str(policy_config.get("protocol", "")):
                raise ValueError("ATACOM requires a dedicated tangent-policy checkpoint")
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
            from genedynamics.solvers.common.rl_policy_controller import RLPolicyController

            return RLPolicyController(execution_env or env, act_fn, seed=seed)
        if method == "atacom":
            from genedynamics.solvers.single.atacom.atacom import AtacomSolver

            return AtacomSolver(
                env, act_fn, Kc=float(cfg.get("Kc", 1.0)),
                action_limit=action_limit,
                alpha_limit=float(cfg.get("alpha_limit", action_limit)),
                seed=seed,
                step_env=execution_env,
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
            step_env=execution_env,
        )
    raise NotImplementedError(f"baseline method '{method}' not recognized")


def make_controller(
    task,
    method="mga_controllable_gate",
    *,
    level=None,
    surface_seed=0,
    use_base=False,
    prior=None,
    policy_ckpt=None,
    atacom_policy_ckpt=None,
    reliability_ckpt=None,
    aug_lambda=2.0,
    aug_rho=200.0,
    backend=None,
    env_overrides=None,
    execution_env_overrides=None,
    model_env=None,
    execution_env=None,
    **cfg,
):
    """Build an algorithm-level contact controller for the unified adapter."""
    method = canonical_method_name(method)
    if method in METHOD_TABLE:
        return make_mga(
            task, method, level=level, surface_seed=surface_seed,
            use_base=use_base, prior=prior, policy_ckpt=policy_ckpt,
            atacom_policy_ckpt=atacom_policy_ckpt,
            reliability_ckpt=reliability_ckpt,
            aug_lambda=aug_lambda, aug_rho=aug_rho, backend=backend,
            env_overrides=env_overrides,
            execution_env_overrides=execution_env_overrides,
            model_env=model_env, execution_env=execution_env, **cfg,
        )
    env = model_env
    if env is None:
        env = make_env(task, **_task_env_kwargs(
            task, method=method, level=level, seed=surface_seed,
            use_base=use_base, overrides=env_overrides,
        ))
    if execution_env is None and execution_env_overrides:
        execution_env = make_env(task, **_task_env_kwargs(
            task, method=method, level=level, seed=surface_seed,
            use_base=use_base,
            overrides={**(env_overrides or {}), **execution_env_overrides},
        ))
    backend = backend or get_backend("jax")
    solver = _build_baseline_solver(
        method, env, backend, execution_env=execution_env,
        policy_ckpt=policy_ckpt, **cfg,
    )
    solver.execution_env = execution_env
    return env, solver


_FACTORY_KEYS = {
    "aug_lambda",
    "aug_rho",
    "policy_ckpt",
    "atacom_policy_ckpt",
    "reliability_ckpt",
}
_CHECKPOINT_KEYS = ("policy_ckpt", "atacom_policy_ckpt", "reliability_ckpt")
_PLUGIN_KEYS = {
    "learned_reliability",
    "experiment_run_class",
    "experiment_formal_experiment",
    *{f"{key}_by_suite" for key in _CHECKPOINT_KEYS},
}


def _resolve_suite_checkpoints(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve shape-specific learned artifacts without task special cases."""
    resolved = dict(config)
    suite = str(config.get("suite", ""))
    for key in _CHECKPOINT_KEYS:
        mapping = config.get(f"{key}_by_suite")
        if not mapping:
            continue
        if not isinstance(mapping, dict):
            raise ValueError(f"{key}_by_suite must be a mapping")
        value = mapping.get(suite, mapping.get("*"))
        if not value:
            raise ValueError(
                f"no {key} binding for suite {suite!r}; "
                f"available={sorted(mapping)}"
            )
        resolved[key] = value
    return resolved


@dataclass
class _ContactPlanner:
    solver: Any
    env: Any
    execution_env: Any
    task: str
    controller_method: str
    n_steps: int
    seed: int
    component_contract: Dict[str, Any]


class RecedingContactMethodPlugin(MethodPlugin):
    """Adapter that delegates algorithm semantics to the existing factory."""

    def __init__(self, plugin_name: str, default_controller_method: str) -> None:
        self._name = str(plugin_name)
        self._default_controller_method = str(default_controller_method)

    @property
    def name(self) -> str:
        return self._name

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> Any:
        del energy
        config = _resolve_suite_checkpoints(config)
        task = str(config.get("task") or getattr(env, "_experiment_task", ""))
        if not task:
            raise ValueError(f"method '{self.name}' requires a contact task")
        method = canonical_method_name(
            config.get("controller_method", self._default_controller_method)
        )
        validation_authoritative = bool(
            config.get("reliability_validation_authoritative", False)
        )
        if validation_authoritative:
            if (
                self.name != "mga"
                or task != INSERT_TASK
                or config.get("experiment_run_class") != "development"
                or bool(config.get("experiment_formal_experiment", True))
            ):
                raise ValueError(
                    "reliability_validation_authoritative is restricted to "
                    "development-only PegInsert MGA validation"
                )
            if (
                not bool(config.get("learned_reliability", True))
                or not config.get("reliability_ckpt")
                or str(config.get("reliability_ood_policy")) != "model_based"
            ):
                raise ValueError(
                    "authoritative reliability validation requires the learned "
                    "candidate under the final model_based OOD policy"
                )
        if self.name == "mga":
            if method not in METHOD_TABLE:
                raise ValueError("mga must resolve to an MGA method contract")
            flags = resolve_method(method)
            learned_reliability = bool(config.get("learned_reliability", True))
            missing = []
            if flags.use_rl_prior and not config.get("policy_ckpt"):
                missing.append("policy_ckpt")
            if learned_reliability and not config.get("reliability_ckpt"):
                missing.append("reliability_ckpt")
            if missing:
                raise ValueError(
                    "mga requires frozen learned components: "
                    + ", ".join(missing)
                )
            if not flags.use_rl_prior and config.get("policy_ckpt"):
                raise ValueError(
                    "no-RL-prior ablation must not load a policy checkpoint"
                )
            if not learned_reliability and config.get("reliability_ckpt"):
                raise ValueError(
                    "no-learned-reliability ablation must not load a "
                    "reliability checkpoint"
                )
        elif method in METHOD_TABLE and self.name in {
            "mppi", "pegasusflow", "issa", "atacom", "standalone_rl"
        }:
            raise ValueError(
                f"baseline plugin '{self.name}' cannot route through MGA method '{method}'"
            )

        factory_kwargs = {
            key: config[key] for key in _FACTORY_KEYS if key in config
        }
        solver_cfg = {
            key: value for key, value in config.items()
            if key not in _FACTORY_KEYS
            and key not in _PLUGIN_KEYS
            and key not in {
                "constraint_pipeline", "scheduler", "obstacles",
                "obstacle_config", "env_plugin", "env_name", "task",
                "n_steps", "controller_method", "execution_env_params",
                "env_params", "suite", "task_level", "np_random_seed",
            }
        }
        # ``np_random_seed`` is the runner's resolved per-run seed.  A seed in
        # the shared method YAML is only a construction default; retaining it
        # here would make every evaluation seed reuse the same stochastic
        # planner trajectory while randomizing only the environment.  Route
        # the resolved run seed into DIAL/MGA/MPPI/PegasusFlow as well as the
        # environment so a formal seed denotes one complete experiment.
        if "np_random_seed" in config:
            solver_cfg["seed"] = int(config["np_random_seed"])
        execution_env = getattr(env, "_experiment_execution_env", None)
        model_env, solver = make_controller(
            task,
            method,
            level=config.get("task_level"),
            surface_seed=int(config.get("np_random_seed", 0)),
            prior=None,
            model_env=env,
            execution_env=execution_env,
            **factory_kwargs,
            **solver_cfg,
        )
        if model_env is not env:
            raise RuntimeError("contact controller silently replaced the model environment")
        if validation_authoritative:
            candidate_metadata = dict(
                getattr(getattr(solver, "reliability_model", None), "metadata", {})
                or {}
            )
            if (
                candidate_metadata.get("task") != INSERT_TASK
                or candidate_metadata.get("fit_scope") != "development_fit_only"
                or bool(candidate_metadata.get("performance_validated", False))
                or bool(candidate_metadata.get("promotion_eligible", False))
            ):
                raise ValueError(
                    "authoritative development validation requires an unpromoted "
                    "PegInsert development-fit reliability candidate"
                )
        contract: Dict[str, Any] = {
            "plugin": self.name,
            "controller_method": method,
            "policy_ckpt": config.get("policy_ckpt"),
            "atacom_policy_ckpt": config.get("atacom_policy_ckpt"),
            "reliability_ckpt": config.get("reliability_ckpt"),
            "rl_prior": False,
            "model_based_rollout": method in METHOD_TABLE,
            "learned_reliability": False,
        }
        if method in METHOD_TABLE:
            flags = resolve_method(method)
            contract["mga_flags"] = asdict(flags)
            contract["rl_prior"] = bool(flags.use_rl_prior and config.get("policy_ckpt"))
            contract["learned_reliability"] = bool(
                self.name == "mga"
                and config.get("learned_reliability", True)
                and config.get("reliability_ckpt")
            )
            if contract["learned_reliability"] and config.get("reliability_ckpt"):
                metadata = getattr(solver, "reliability_model", None)
                metadata = getattr(metadata, "metadata", {}) or {}
                validation_gate = bool(
                    getattr(
                        solver, "reliability_validation_authoritative", False
                    )
                )
                contract["reliability_gate_authoritative"] = bool(
                    str(config.get("reliability_ood_policy", "veto")) == "veto"
                    or validation_gate
                    or (
                        metadata.get("performance_validated", False)
                        and metadata.get("promotion_eligible", False)
                    )
                )
                contract["reliability_validation_authoritative"] = validation_gate
                contract["reliability_promotion_eligible"] = bool(
                    metadata.get("performance_validated", False)
                    and metadata.get("promotion_eligible", False)
                )
        elif method in {"rl", "issa"}:
            contract["policy"] = bool(config.get("policy_ckpt"))
            if method == "issa":
                contract["execution_environment_semantics"] = (
                    "nominal_projection_execution_step"
                )
        elif method == "atacom":
            contract["policy"] = bool(config.get("policy_ckpt"))
            contract["tangent_policy"] = True
            contract["execution_environment_semantics"] = (
                "nominal_manifold_execution_step"
            )
        return _ContactPlanner(
            solver=solver,
            env=env,
            execution_env=execution_env or env,
            task=task,
            controller_method=method,
            n_steps=int(config.get("n_steps", 1)),
            seed=int(config.get("np_random_seed", 0)),
            component_contract=contract,
        )

    def plan(self, planner: _ContactPlanner, initial_state: Any, rng: Any) -> Dict[str, Any]:
        del rng
        import jax

        controller_kwargs = dict(collect_states=True, synchronize_steps=True)
        initialize_plan = getattr(planner.env, "plan_initializer", None)
        if callable(initialize_plan) and callable(
            getattr(planner.solver, "make_controller", None)
        ):
            # Task coordinates can have a nonzero neutral action. Only MPC
            # controllers own a cold horizon; raw/tangent RL do not use one.
            controller_kwargs["initialize_plan"] = initialize_plan
        if isinstance(planner.solver, MGASolver):
            # Use the same task execution context as direct MGA deployment.
            execution_step = getattr(
                planner.solver._get_backend_impl(), "execution_step", None
            )
            if execution_step is not None:
                controller_kwargs["execution_step"] = execution_step
        execution_status = None
        try:
            result = planner.solver.run_receding(
                initial_state,
                planner.n_steps,
                jax.random.PRNGKey(1000 + planner.seed),
                **controller_kwargs,
            )
        except ExecutionRejected as exc:
            # Only a task's explicit pre-physics rejection may become an
            # aborted attempt. Generic simulator/planner errors still raise.
            if not getattr(exc, "partial_states_complete", False):
                raise
            result = exc.partial_result
            execution_status = {
                "state": "aborted_unrecoverable",
                "reason": exc.reason,
                "details": exc.details,
                "requested_steps": int(exc.requested_steps),
                "executed_steps": len(result.actions),
                "rejected_step": int(exc.rejected_step),
                "rejected_action": exc.rejected_action,
                "rejected_info": exc.rejected_info,
                "metrics_scope": "actual_execution_prefix_only",
            }
        states = list(result.states)
        actions = list(result.actions)
        infos = list(getattr(result, "infos", ()))
        costs = list(getattr(result, "costs", ()))
        trajectory = Trajectory(states=states, actions=actions, info={
            "infos": infos,
            "costs": costs,
            **({"execution_status": execution_status} if execution_status else {}),
        })
        return {
            "trajectory": trajectory,
            "states": states,
            "actions": actions,
            "infos": infos,
            "costs": costs,
            "initial_state": initial_state,
            "execution_env": planner.execution_env,
            "receding_result": result,
            "component_contract": planner.component_contract,
            **({"execution_status": execution_status} if execution_status else {}),
        }


class MGAMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("mga", "mga_controllable_gate")


class ModelBasedOnlyMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("model_based_only", "mga_controllable")


class DIALContactMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("dial", "dial")


class MPPIContactMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("mppi", "mppi")


class PegasusFlowContactMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("pegasusflow", "pegasusflow")


class ISSAContactMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("issa", "issa")


class ATACOMContactMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("atacom", "atacom")


class StandaloneRLMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("standalone_rl", "rl")


__all__ = [
    "ARM_TASK",
    "HUMANOID_TASK",
    "INSERT_TASK",
    "make_mga",
    "make_controller",
    "RecedingContactMethodPlugin",
    "MGAMethodPlugin",
    "ModelBasedOnlyMethodPlugin",
    "DIALContactMethodPlugin",
    "MPPIContactMethodPlugin",
    "PegasusFlowContactMethodPlugin",
    "ISSAContactMethodPlugin",
    "ATACOMContactMethodPlugin",
    "StandaloneRLMethodPlugin",
]
