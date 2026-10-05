"""Train the frozen H1 box-push policies used by the paper matrix.

This is a task-owned CLI wrapper around :mod:`genedynamics.learning`; policy
implementation and serialization remain in the shared learning package.
New training excludes the evaluation seeds declared by the canonical protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import gc
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper, VmapWrapper

from genedynamics.envs.factories import make_env
from genedynamics.envs.domains.humanoid.box_push_brax import (
    HumanoidBoxPushConfig,
    HumanoidBoxPushDomainEnv,
    HumanoidBoxPushResidualActionEnv,
)
from genedynamics.experiments.framework.config import ExperimentConfig
from genedynamics.learning.train_rl_policy import (
    save_policy,
    scalar_metrics,
    train_rl_policy,
)


TASK = "humanoid_box_push"
SCHEMAS = ("fixed", "walk", "atacom_force_push", "atacom_unjamming", "atacom_walk_and_push")
CANONICAL_CONFIG = Path(__file__).resolve().parents[4] / "configs/humanoid/push_to_line/_base.yaml"
_FORCE_PUSH_SUITES = ("force_regulation_15n", "force_regulation_30n", "fixed_stance_push_nominal", "fixed_stance_push_ood")
SCHEMA_SUITES = {
    "fixed": (*_FORCE_PUSH_SUITES, "unjamming"),
    "walk": ("walk_and_push",),
    "atacom_force_push": _FORCE_PUSH_SUITES,
    "atacom_unjamming": ("unjamming",),
    "atacom_walk_and_push": ("walk_and_push",),
}
_H1_TRANSITION_METRICS = (
    "h1_active_steps", "h1_success_padding_steps",
    "h1_goal_completions", "h1_failures",
)
_H1_TERMINATION_CONTRACT = {
    "schema_version": 1,
    "objective": "continuing_discounted_success_absorbing",
    "success": "retain_absorbing_goal_until_episode_time_limit",
    "failure": "immediate_termination_with_priority_over_goal",
    "time_limit": "truncation_with_existing_brax_bootstrap",
    "padding_reward": "task_owned_active_reward_upper_bound",
    "completed_transition_reward": "unchanged_task_reward",
    "environment_step_budget": "includes_success_padding",
    "domain_coverage": "active_transitions_excluding_success_padding",
    "transition_counts": "logged_episode_metrics_not_global_totals",
}


class _StratifiedH1Batch(VmapWrapper):
    """Keep every stage domain represented even with a small CPU batch."""

    def reset(self, rng):
        count = len(self.env.domains)
        if rng.shape[0] < count:
            raise ValueError("Each H1 device batch must cover every training domain")
        indices = jnp.arange(rng.shape[0], dtype=jnp.int32) % count
        branches = tuple(
            (lambda key, domain=domain: domain.reset(key))
            for domain in self.env.domains
        )

        def reset_one(key, index):
            state = jax.lax.switch(index, branches, key)
            missing = {"task_success", "task_fallen", "success_padding"} - state.info.keys()
            if missing:
                raise ValueError(f"H1 training termination contract missing: {sorted(missing)}")
            return state.replace(
                info={**state.info, "_rl_domain_index": index},
                metrics={
                    **state.metrics,
                    **{f"h1_domain_{i}_steps": jnp.float32(0.0) for i in range(count)},
                    **{key: jnp.float32(0.0) for key in _H1_TRANSITION_METRICS},
                },
            )

        return jax.vmap(reset_one)(rng, indices)

    def step(self, state, action):
        advanced = super().step(state, action)
        indices = state.info["_rl_domain_index"]
        fallen = advanced.info["task_fallen"] > 0.5
        goal = (advanced.info["task_success"] > 0.5) & ~fallen
        padding = advanced.info["success_padding"].astype(bool)
        active = ~padding
        # This sits INSIDE EpisodeWrapper: retain the real goal state until
        # the sampling time limit, where Brax still records a truncation.
        # Suppressing done after AutoReset would be too late to preserve it.
        done = jnp.where(fallen, jnp.ones_like(advanced.done),
                         jnp.where(goal, jnp.zeros_like(advanced.done), advanced.done))
        return advanced.replace(done=done, metrics={
            **advanced.metrics,
            **{f"h1_domain_{i}_steps": ((indices == i) & active).astype(jnp.float32)
               for i in range(len(self.env.domains))},
            "h1_active_steps": active.astype(jnp.float32),
            "h1_success_padding_steps": padding.astype(jnp.float32),
            "h1_goal_completions": (goal & (state.info["task_success"] < 0.5)).astype(jnp.float32),
            "h1_failures": (fallen & (state.info["task_fallen"] < 0.5)).astype(jnp.float32),
        })


class _CompleteH1AutoReset(AutoResetWrapper):
    """Reset task memory together with Brax's cached physics and observation.

    Brax retains terminal info for episode logging.  Restore the initial info
    at the next step, before H1 can see a stale absorbing task_success latch.
    """

    def reset(self, rng):
        state = super().reset(rng)
        initial_info = {
            key: value for key, value in state.info.items()
            if key not in {"first_pipeline_state", "first_obs"}
        }
        initial_info = jax.tree_util.tree_map(lambda value: value, initial_info)
        return state.replace(info={**state.info, "_h1_initial_info": initial_info})

    def step(self, state, action):
        def reset_done(initial, current):
            mask = state.done.astype(bool)
            mask = mask.reshape(mask.shape + (1,) * (current.ndim - mask.ndim))
            return jnp.where(mask, initial, current)

        restored = {
            key: jax.tree_util.tree_map(reset_done, initial, state.info[key])
            for key, initial in state.info["_h1_initial_info"].items()
        }
        state = state.replace(info={**state.info, **restored})
        return super().step(state, action)


class _BehaviorAnchoredH1Env:
    """Add an explicit frozen-teacher auxiliary reward during PPO only.

    The wrapped task, absolute action transform and deployment reward remain
    unchanged.  This training-only term keeps a residual policy near a
    violation-free model-based behavior while PPO improves it with the task
    return.  It is opt-in and never used by canonical/legacy training unless
    the CLI declares a positive anchor weight.
    """

    def __init__(self, env, teacher_action_fn, weight):
        self.env = env
        self.teacher_action_fn = teacher_action_fn
        self.weight = jnp.float32(weight)

    @property
    def domains(self):
        return self.env.domains

    @property
    def action_size(self):
        return self.env.action_size

    @property
    def observation_size(self):
        return self.env.observation_size

    @property
    def backend(self):
        return self.env.backend

    @property
    def dt(self):
        return self.env.dt

    def reset(self, rng):
        return self.env.reset(rng)

    def step(self, state, residual_action):
        teacher = jax.lax.stop_gradient(self.teacher_action_fn(state.obs))
        residual_action = jnp.asarray(residual_action, jnp.float32)
        penalty = self.weight * jnp.mean((residual_action - teacher) ** 2)
        active = ~jnp.asarray(
            state.info.get("success_padding", False), jnp.bool_
        )
        advanced = self.env.step(state, residual_action)
        return advanced.replace(
            reward=advanced.reward - jnp.where(active, penalty, 0.0)
        )


def wrap_h1_training(env, episode_length, action_repeat=1, randomization_fn=None):
    """H1-only replacement for the generic PPO training wrapper."""
    if randomization_fn is not None:
        raise ValueError("H1 training domains own randomization")
    return _CompleteH1AutoReset(
        EpisodeWrapper(_StratifiedH1Batch(env), episode_length, action_repeat)
    )


def _check_training_metrics(metrics, max_policy_kl):
    values = scalar_metrics(metrics)
    for name, value in values.items():
        if not np.isfinite(value):
            raise RuntimeError(f"Refusing divergent H1 training: {name}={value}")
        if name.endswith("kl_mean") and value > max_policy_kl:
            raise RuntimeError(f"Refusing divergent H1 training: {name}={value}")


def _validate_stage_for_export(params, config, history, stage, domain_count,
                               normalizer_std_eps, max_policy_kl):
    """Require observed domain coverage and a numerically usable checkpoint."""
    final_metrics = config.get("final_metrics", {})
    if not any(key.endswith("kl_mean") for key in final_metrics):
        raise RuntimeError("H1 checkpoint has no final PPO KL diagnostic")
    _check_training_metrics(final_metrics, max_policy_kl)
    if not all(np.all(np.isfinite(np.asarray(x)))
               for x in jax.tree_util.tree_leaves(params)):
        raise RuntimeError("Refusing H1 checkpoint with nonfinite parameters")
    normalizer_active = bool(config.get("normalize_observations", True))
    if normalizer_active and np.min(np.asarray(params[0].std)) < normalizer_std_eps * 0.99:
        raise RuntimeError("H1 checkpoint violates its normalizer std floor")
    coverage = {}
    for index in range(domain_count):
        key = f"episode/h1_domain_{index}_steps"
        observed = [float(row[key]) for row in history
                    if row.get("stage") == stage and key in row]
        if not observed or max(observed) <= 0.0:
            raise RuntimeError(f"H1 stage {stage} domain {index} has no observed episode coverage")
        coverage[str(index)] = max(observed)
    return {
        "stage": stage, "max_logged_episode_mean_domain_steps": coverage,
        "domain_coverage_unit": "active_transitions_excluding_success_padding",
        "max_logged_episode_mean_transition_counts": {
            key: max((float(row[f"episode/{key}"]) for row in history
                      if row.get("stage") == stage and f"episode/{key}" in row), default=None)
            for key in _H1_TRANSITION_METRICS
        },
        "normalizer_min_std": float(np.min(np.asarray(params[0].std))),
        "policy_observation_normalization": "running_statistics" if normalizer_active else "disabled",
        "policy_normalizer_floor_checked": normalizer_active,
        "final_metrics": scalar_metrics(final_metrics),
    }


def _observed_training_steps(history):
    """Brax starts env_steps at zero for each stage, including restore."""
    counts = {}
    for row in history:
        stage = str(row["stage"])
        counts[stage] = max(counts.get(stage, 0), int(row["num_steps"]))
    return counts


def _write_training_status(path, payload):
    """Keep one auditable run artifact even when health checks stop export."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def _training_source_hashes(root, *, atacom=False):
    paths = [
        "scripts/tasks/robot/humanoid/train_box_push_rl.py",
        "genedynamics/learning/train_rl_policy.py",
        "genedynamics/envs/domains/humanoid/box_push_brax.py",
        "genedynamics/envs/legged_brax_base.py",
        "genedynamics/core/control/humanoid_contact.py",
        "genedynamics/core/control/bipedal_gait.py",
        "genedynamics/core/control/stiffness.py",
    ]
    if atacom:
        paths.extend([
            "genedynamics/solvers/single/atacom/wrapper.py",
            "genedynamics/solvers/single/atacom/backends/atacom_jax.py",
        ])
    return {path: hashlib.sha256((Path(root) / path).read_bytes()).hexdigest()
            for path in paths}


def _ppo_training_kwargs(*, schema, atacom, num_envs, cpu_low_memory,
                         normalizer_std_eps):
    kwargs = {
        "batch_size": 8,
        "num_minibatches": 2,
        # Keep contact onset and its short delayed force response in the same
        # advantage window, including the low-memory CPU training path.
        "unroll_length": 10,
        "discounting": 0.99,
        # With one minibatch and one pass, pre-gradient KL would compare the
        # identical policy to itself.  The second pass measures the first
        # update using the same observations and sampled data.
        "num_updates_per_batch": 2,
        "reward_scaling": 1e-4 if atacom else 0.01,
        "max_grad_norm": 1.0,
        "policy_hidden_layer_sizes": (32, 32, 32),
        "learning_rate": 1e-4,
        # H1's observable coordinates already include task-force/support
        # ratios.  Using them directly keeps behavior and PPO loss inputs
        # consistent across curriculum stages without a moving normalizer or
        # Brax's unbounded-for-this-task adaptive learning-rate increase.
        "learning_rate_schedule": "NONE",
        "normalize_observations": False,
        "run_evals": False,
        "wrap_env_fn": wrap_h1_training,
        "normalize_observations_std_eps": float(normalizer_std_eps),
        "log_training_metrics": True,
        "training_metrics_steps": 1000,
    }
    if cpu_low_memory:
        kwargs.update(
            batch_size=max(1, num_envs), num_minibatches=1,
            policy_hidden_layer_sizes=(16, 16),
        )
    elif schema == "walk":
        # Walk-and-Push exposes the full DIAL-derived 11-joint target chart.  Learning a
        # dynamic alternating gait from the 91D state needs temporal credit
        # and state normalization; the tiny fixed-stance CPU network is not a
        # meaningful locomotion architecture.  This remains Walk-and-Push-only and does
        # not alter Force Regulation, Fixed-Stance Push, and Unjamming, Surface, or Peg checkpoints.
        kwargs.update(
            batch_size=16,
            num_minibatches=4,
            unroll_length=20,
            discounting=0.995,
            policy_hidden_layer_sizes=(64, 64),
            learning_rate=1e-4,
            normalize_observations=True,
            init_noise_std=0.25,
        )
    if schema == "walk":
        # The verified DIAL sequence is the policy centre, so Walk-and-Push learns local
        # loaded-contact corrections rather than rediscovering locomotion.
        # A smaller fixed step and bounded initial variance prevent the first
        # PPO update from erasing the BC initialization (observed as a large
        # post-update KL), including in the CPU-low-memory architecture.
        kwargs.update(learning_rate=2e-5, init_noise_std=0.25)
    return kwargs


def _training_suites(schema: str, config=None):
    config = config or ExperimentConfig.from_yaml(CANONICAL_CONFIG)
    if schema not in SCHEMA_SUITES:
        raise ValueError(schema)
    errors = config.validate()
    if errors:
        raise ValueError("Invalid canonical H1 protocol: " + "; ".join(errors))
    available = {suite["name"]: suite for suite in config.suites}
    return [config.for_suite(available[name]) for name in SCHEMA_SUITES[schema]]


def _domain_specs(schema: str, config=None):
    specs = [dict(resolved.env_params) for resolved in _training_suites(schema, config)]
    for spec in specs:
        if spec.get("level") == "heavy_dr":
            spec["dr_seed"] = 101
    return specs


def _development_env_specs(schema, specs, raw_overrides):
    """Apply explicit development Walk-and-Push overrides without changing canonical YAML."""
    if raw_overrides is None:
        return specs, {}
    if schema not in {"walk", "atacom_walk_and_push"}:
        raise ValueError("env-overrides are restricted to walk and atacom_walk_and_push development training")
    overrides = json.loads(raw_overrides)
    if not isinstance(overrides, dict):
        raise ValueError("env-overrides must be a JSON object")
    json.dumps(overrides, allow_nan=False)
    unknown = set(overrides) - {field.name for field in fields(HumanoidBoxPushConfig)}
    if unknown:
        raise ValueError(f"Unknown HumanoidBoxPushConfig fields: {sorted(unknown)}")
    resolved = [{**spec, **overrides} for spec in specs]
    for spec in resolved:
        if (spec.get("level") != "push_walk"
                or spec.get("robot", "h1") != "h1"
                or spec.get("use_base", False)
                or spec.get("walk_success_mode", "legacy") != "locomotion"
                or spec.get("walk_leg_control", "legacy") != "joint_target"):
            raise ValueError("Walk-and-Push development overrides require H1 push_walk, locomotion and joint_target")
    return resolved, overrides


def _atacom_training_options(schema, config=None):
    if not schema.startswith("atacom_"):
        return None
    config = config or ExperimentConfig.from_yaml(
        CANONICAL_CONFIG.parent / "baseline" / "atacom.yaml"
    )
    options = [{
        "Kc": float(suite.method_params.get("Kc", 1.0)),
        "action_limit": float(suite.method_params.get("action_limit", 1.0)),
    } for suite in _training_suites(schema, config)]
    if any(option != options[0] for option in options):
        raise ValueError("ATACOM training domains must share projection parameters")
    return options[0]


def _environment_training_contract(domains, atacom_options=None):
    """Read actual task defaults/semantics before a tangent wrapper hides them."""
    interfaces = [getattr(domain, "policy_interface", None) for domain in domains]
    if any(interface != interfaces[0] for interface in interfaces):
        raise ValueError("H1 training domains must share policy interface semantics")
    # Config defaults include JAX arrays (kp/kd); store portable JSON values.
    resolved = json.loads(json.dumps(
        [asdict(domain._bcfg) for domain in domains],
        default=lambda value: np.asarray(value).tolist(), allow_nan=False,
    ))
    transform = None
    if atacom_options is not None:
        times = {float(getattr(domain._config, "dt", 0.02)) for domain in domains}
        if len(times) != 1:
            raise ValueError("ATACOM training domains must share control dt")
        transform = {**atacom_options, "time_step": times.pop()}
    return interfaces[0], resolved, transform


def _environment_policy_action_transform(domains):
    """Resolve one task-owned residual chart across the training domains."""
    transforms = [getattr(domain, "policy_action_transform", None)
                  for domain in domains]
    if any(transform != transforms[0] for transform in transforms):
        raise ValueError("H1 training domains must share policy action transform")
    return transforms[0]


def _walk_curriculum_groups(schema: str, specs, enabled=False):
    """Optional Walk-and-Push-only low-load-to-task curriculum with one action contract.

    DIAL can optimize a fresh joint sequence online, whereas PPO must first
    discover a supported gait through long-horizon exploration.  The warm-up
    stage keeps the same H1 scene, observation layout, action chart and goal;
    it only removes commanded contact load and task penalties that are
    irrelevant before the feet can exchange support.  The final stage is the
    exact requested Walk-and-Push domain and is the only interface exported.
    """
    if not enabled:
        return [specs[:-1], specs[-1:]] if schema == "fixed" else [specs]
    if schema != "walk" or len(specs) != 1:
        raise ValueError("walk curriculum is defined only for the single-domain walk schema")
    final = dict(specs[0])
    warmup = {
        **final,
        # Learn supported locomotion before loaded contact.  Keep the same
        # scene, observation/action shapes and Cartesian impedance controller,
        # but make that impedance deliberately weak and remove feed-forward
        # force.  This is the PPO analogue of the separately validated
        # unloaded DIAL feasibility screen, without moving the box/goal out of
        # the policy's state distribution.
        "f_min": 0.0,
        "f_max": 1.0,
        "f_target": 0.0,
        "fixed_force_target": True,
        # The training-only unloaded stage lowers the physical force ceiling
        # to 1 N.  Keep the task-owned retract inside that same contract;
        # otherwise a valid final Walk-and-Push emergency magnitude makes construction of
        # the deliberately low-load warm-up environment fail validation.
        "emergency_retract_force": 0.0,
        "s_ref_diag": 0.0,
        "s_scale": 0.0,
        "w_contact": 0.0,
        "w_force": 0.0,
        "w_force_limit": 0.0,
        "w_nonhand": 0.0,
        "w_bal": 0.0,
        "w_stiffness_nominal": 0.0,
    }
    return [[warmup], [final]]


def _load_walk_expert(path, *, observation_size, action_size, action_transform,
                      policy_interface=None, expert_target="reference"):
    """Load one terminal-clean DIAL trajectory as PPO initialization data.

    The teacher remains an initialization only: PPO subsequently optimizes the
    shared task reward and all standalone/projected/MGA methods consume the
    same frozen policy.  Primitive coordinates are deliberately centered at
    zero because they were irrelevant in the unloaded DIAL screen.  Without
    a task-owned trajectory center, only the verified leg/waist behavior is
    supervised; with that center, the policy learns a zero residual around
    the exact hash-locked reference and then PPO learns loaded corrections.
    """
    if expert_target not in {"reference", "residual"}:
        raise ValueError("walk expert target must be 'reference' or 'residual'")
    path = Path(path)
    payload = json.loads(path.read_text())
    actions = np.asarray(payload.get("actions"), np.float32)
    states = payload.get("states") or []
    observations = np.asarray([state.get("obs") for state in states[:-1]], np.float32)
    if (actions.ndim != 2 or actions.shape[1] != int(action_size)
            or observations.shape != (len(actions), int(observation_size))):
        raise ValueError("DIAL expert trajectory action/observation interface mismatch")
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(observations)):
        raise ValueError("DIAL expert trajectory must be finite")
    signals = payload.get("task_signals") or {}
    fallen = np.asarray(signals.get("task_fallen", ()), dtype=float)
    left = np.asarray(signals.get("walk_left_steps", ()), dtype=float)
    right = np.asarray(signals.get("walk_right_steps", ()), dtype=float)
    if (fallen.shape != (len(actions),) or np.any(fallen > 0.5)
            or left.shape != (len(actions),) or right.shape != (len(actions),)
            or np.max(left, initial=0.0) < 1.0 or np.max(right, initial=0.0) < 1.0):
        raise ValueError("DIAL expert must contain a fall-free bilateral supported walk")
    if expert_target == "residual":
        # A model-based prefix is useful supervision even when its next
        # receding decision had no certified continuation.  It must never be
        # treated as a safety label for that rejected action: admit only the
        # physically executed prefix, with complete fast-loop coverage and no
        # observed safety-margin violation.  Exact task-interface equality
        # prevents an old contact/retraction realization from being silently
        # relabelled as current data.
        recorded_interface = (
            ((signals.get("task_metadata") or {}).get("reliability_contract") or {})
            .get("policy_interface")
        )
        if policy_interface is None or recorded_interface != policy_interface:
            raise ValueError(
                "model-based residual expert policy interface mismatch"
            )
        execution = payload.get("execution_status") or {}
        if (execution.get("executed_steps") != len(actions)
                or execution.get("metrics_scope") != "actual_execution_prefix_only"):
            raise ValueError(
                "model-based residual expert must identify its actual executed prefix"
            )
        valid = np.asarray(signals.get("physics_samples_valid", ())).reshape(-1)
        margins = np.asarray(signals.get("physics_safety_margins", ()), np.float64)
        if (valid.shape != (len(actions),) or not np.all(valid > 0.5)
                or margins.ndim != 3 or margins.shape[0] != len(actions)
                or margins.shape[-1] != 4 or not np.all(np.isfinite(margins))
                or np.any(margins > 0.0)):
            raise ValueError(
                "model-based residual expert requires complete violation-free "
                "fast-loop evidence"
            )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    bias = np.asarray(action_transform["action_bias"], np.float32)
    scale = np.asarray(action_transform["action_scale"], np.float32)
    if bias.shape != (action_size,) or scale.shape != (action_size,) or np.any(scale <= 0.0):
        raise ValueError("DIAL expert requires the task-owned PPO action transform")
    primitive_width = int(action_size) - 11
    if (expert_target == "reference"
            and action_transform.get("center") == "time_indexed_dial_reference"):
        layout = (policy_interface or {}).get("action_layout", {})
        if layout.get("planner_reference_sha256") != digest:
            raise ValueError(
                "DIAL expert must exactly match the task-owned time-indexed planner reference"
            )
        # The environment already adds this exact joint sequence at each
        # control step.  Supervise a zero residual around it; fitting the
        # absolute teacher action here would add the gait twice at execution.
        residual = np.zeros_like(actions, dtype=np.float32)
        target_semantics = "zero_residual_around_verified_DIAL_planner_reference"
        leg_rmse = 0.0
    else:
        residual = np.clip((actions - bias) / scale, -0.999, 0.999)
        residual[:, :primitive_width] = 0.0
        reconstructed = np.clip(bias + scale * residual, -1.0, 1.0)
        target_semantics = (
            "zero_contact_primitive_plus_model_based_leg_residual"
            if expert_target == "residual" else
            "zero_contact_primitive_plus_verified_DIAL_leg_residual"
        )
        leg_rmse = float(np.sqrt(np.mean(
            (reconstructed[:, primitive_width:] - actions[:, primitive_width:]) ** 2
        )))
    return observations, residual.astype(np.float32), {
        "path": str(path),
        "sha256": digest,
        "transitions": int(len(actions)),
        "verified_left_steps": int(np.max(left)),
        "verified_right_steps": int(np.max(right)),
        "absolute_leg_action_rmse": leg_rmse,
        "target_semantics": target_semantics,
    }


def _load_walk_expert_set(paths, **kwargs):
    """Load an ordered, hash-identified set of verified walk trajectories."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    else:
        paths = list(paths)
    if not paths:
        raise ValueError("walk expert set must contain at least one trajectory")

    loaded = [_load_walk_expert(path, **kwargs) for path in paths]
    semantics = {report["target_semantics"] for _, _, report in loaded}
    if len(semantics) != 1:
        raise ValueError("walk expert trajectories use different target semantics")
    observations = np.concatenate([item[0] for item in loaded], axis=0)
    targets = np.concatenate([item[1] for item in loaded], axis=0)
    reports = [item[2] for item in loaded]
    digests = [report["sha256"] for report in reports]
    collection_digest = hashlib.sha256(
        json.dumps(digests, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    transitions = np.asarray(
        [report["transitions"] for report in reports], np.float64
    )
    squared_error = np.asarray([
        report["absolute_leg_action_rmse"] ** 2 for report in reports
    ], np.float64)
    report = {
        "path": reports[0]["path"] if len(reports) == 1 else None,
        "paths": [item["path"] for item in reports],
        "sha256": digests[0] if len(digests) == 1 else collection_digest,
        "trajectory_sha256": digests,
        "trajectory_count": len(reports),
        "transitions": int(np.sum(transitions)),
        "verified_left_steps": int(min(
            item["verified_left_steps"] for item in reports
        )),
        "verified_right_steps": int(min(
            item["verified_right_steps"] for item in reports
        )),
        "absolute_leg_action_rmse": float(np.sqrt(
            np.sum(transitions * squared_error) / np.sum(transitions)
        )),
        "target_semantics": semantics.pop(),
    }
    return observations, targets, report


def _pretrain_walk_policy_from_expert(env, action_transform, policy_interface, path, *, seed,
                                      hidden_sizes, normalize_observations,
                                      normalizer_std_eps, init_noise_std, steps,
                                      expert_target="reference"):
    """Fit the PPO policy mean to a verified DIAL gait, then return restore params."""
    if steps <= 0:
        raise ValueError("expert-pretrain-steps must be positive")
    import optax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    observations, targets, report = _load_walk_expert_set(
        path, observation_size=env.observation_size, action_size=env.action_size,
        action_transform=action_transform, policy_interface=policy_interface,
        expert_target=expert_target,
    )
    obs = jnp.asarray(observations)
    target = jnp.asarray(targets)
    normalizer = running_statistics.init_state(
        jnp.zeros((env.observation_size,), jnp.float32),
        std_eps=float(normalizer_std_eps),
    )
    if normalize_observations:
        normalizer = running_statistics.update(
            normalizer, obs, std_min_value=float(normalizer_std_eps),
        )
    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=(
            running_statistics.normalize if normalize_observations else (lambda x, y: x)
        ),
        policy_hidden_layer_sizes=tuple(hidden_sizes),
        init_noise_std=float(init_noise_std),
    )
    policy_key, value_key = jax.random.split(jax.random.PRNGKey(int(seed) + 9137))
    policy_params = networks.policy_network.init(policy_key)
    value_params = networks.value_network.init(value_key)

    # Brax's current tanh-normal policy path builds a plain 2A-output MLP and
    # does not consume ``init_noise_std`` (the argument is used only by its
    # unsquashed normal policy module).  Leaving the scale half random made a
    # BC-initialized H1 policy start PPO with roughly unit, state-dependent
    # exploration and produced a catastrophic first-update KL.  Make the
    # intended exploration contract explicit in the restored parameter tree:
    # zero its state-dependent scale kernel and set the softplus bias so the
    # actual pre-tanh standard deviation equals ``init_noise_std``.
    min_std = 1.0e-3
    if not np.isfinite(init_noise_std) or init_noise_std <= min_std:
        raise ValueError("expert PPO init_noise_std must exceed 0.001")
    network_params = dict(policy_params["params"])
    output_name = f"hidden_{len(tuple(hidden_sizes))}"
    if output_name not in network_params:
        raise ValueError("Cannot locate PPO output layer for expert initialization")
    output_params = dict(network_params[output_name])
    kernel = output_params["kernel"]
    bias = output_params["bias"]
    if kernel.shape[-1] != 2 * int(env.action_size) or bias.shape != (
            2 * int(env.action_size),):
        raise ValueError("Unexpected PPO distribution head shape")
    scale_logit = np.log(np.expm1(float(init_noise_std) - min_std))
    output_params["kernel"] = kernel.at[:, env.action_size:].set(0.0)
    output_params["bias"] = bias.at[env.action_size:].set(scale_logit)
    network_params[output_name] = output_params
    policy_params = {**policy_params, "params": network_params}

    # When the environment already adds the verified time-indexed DIAL joint
    # reference, the exact teacher is the zero residual for *every* state, not
    # only for the observations stored in the unloaded demonstration.  A
    # regression fit on those observations can have tiny demonstration MSE
    # while producing arbitrary residuals as soon as the loaded box changes
    # the observation.  Zero only the mean half of the final distribution
    # head, retaining its initialized exploration scale for PPO.
    analytic_zero_center = (
        report["target_semantics"]
        == "zero_residual_around_verified_DIAL_planner_reference"
    )
    if analytic_zero_center:
        network_params = dict(policy_params["params"])
        output_params = dict(network_params[output_name])
        kernel = output_params["kernel"]
        bias = output_params["bias"]
        output_params["kernel"] = kernel.at[:, :env.action_size].set(0.0)
        output_params["bias"] = bias.at[:env.action_size].set(0.0)
        network_params[output_name] = output_params
        policy_params = {**policy_params, "params": network_params}

    optimizer = optax.adam(1e-3)
    optimizer_state = optimizer.init(policy_params)

    def loss(params):
        logits = networks.policy_network.apply(normalizer, params, obs)
        predicted = networks.parametric_action_distribution.mode(logits)
        return jnp.mean((predicted - target) ** 2)

    initial_loss = float(loss(policy_params))

    if analytic_zero_center:
        if initial_loss > 1e-12:
            raise RuntimeError(
                f"Analytic DIAL residual initialization is not zero: {initial_loss}"
            )
        report.update({
            "pretrain_steps": 0,
            "requested_pretrain_steps": int(steps),
            "initial_action_mse": initial_loss,
            "final_action_mse": initial_loss,
            "minimum_logged_action_mse": initial_loss,
            "target": report["target_semantics"],
            "initialization": "analytic_global_zero_policy_mean",
            "initial_exploration_std": float(init_noise_std),
            "exploration_initialization": "constant_tanh_normal_scale_head",
        })
        return (normalizer, policy_params, value_params), report

    @jax.jit
    def update(carry, _):
        params, state = carry
        value, grads = jax.value_and_grad(loss)(params)
        updates, state = optimizer.update(grads, state, params)
        return (optax.apply_updates(params, updates), state), value

    (policy_params, _), losses = jax.lax.scan(
        update, (policy_params, optimizer_state), None, length=int(steps)
    )
    final_loss = float(loss(policy_params))
    if not np.isfinite(final_loss) or final_loss >= initial_loss:
        raise RuntimeError(
            f"DIAL expert initialization did not converge: {initial_loss} -> {final_loss}"
        )
    report.update({
        "pretrain_steps": int(steps),
        "initial_action_mse": initial_loss,
        "final_action_mse": final_loss,
        "minimum_logged_action_mse": float(jnp.min(losses)),
        "target": report["target_semantics"],
        "initial_exploration_std": float(init_noise_std),
        "exploration_initialization": "constant_tanh_normal_scale_head",
    })
    return (normalizer, policy_params, value_params), report


def _walk_behavior_anchor_action(params, env, *, hidden_sizes,
                                 normalize_observations, init_noise_std):
    """Rebuild the deterministic frozen BC action used by the PPO anchor."""
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    normalizer, policy_params, _ = params
    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=(
            running_statistics.normalize
            if normalize_observations else (lambda x, y: x)
        ),
        policy_hidden_layer_sizes=tuple(hidden_sizes),
        init_noise_std=float(init_noise_std),
    )

    def action(obs):
        logits = networks.policy_network.apply(normalizer, policy_params, obs)
        return networks.parametric_action_distribution.mode(logits)

    return action


def _stage_episode_lengths(schema: str, resolved_suites, override=None, *, stage_count=None):
    if override is not None and (isinstance(override, bool) or not isinstance(override, int) or override <= 0):
        raise ValueError("episode-length must be a positive integer")
    groups = [resolved_suites[:-1], resolved_suites[-1:]] if schema == "fixed" else [resolved_suites]
    lengths = []
    for group in groups:
        canonical = {suite.n_steps for suite in group}
        if len(canonical) != 1:
            raise ValueError("H1 domains within one training stage require the same episode length")
        lengths.append(int(override) if override is not None else canonical.pop())
    if stage_count is not None and len(lengths) == 1 and stage_count > 1:
        lengths = lengths * int(stage_count)
    if stage_count is not None and len(lengths) != int(stage_count):
        raise ValueError("training stages and episode lengths disagree")
    return lengths


def _default_output(schema: str, seed: int) -> Path:
    return Path("results/humanoid/box_push/_policies") / f"{schema}_ppo_seed{seed}.pkl"


def _validate_training_seed(seed: int, config) -> None:
    formal_seeds = {int(value) for value in config.seeds}
    if int(seed) in formal_seeds:
        raise ValueError(
            f"training seed {seed} overlaps canonical formal evaluation seeds "
            f"{sorted(formal_seeds)}; use a separate development seed"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True, choices=SCHEMAS)
    parser.add_argument("--num-timesteps", type=int, default=200_000)
    parser.add_argument("--episode-length", type=int, default=None,
                        help="Override the canonical suite episode length for training")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--out")
    parser.add_argument("--env-overrides", default=None,
                        help="JSON HumanoidBoxPushConfig overrides for development walk/atacom_walk_and_push only")
    parser.add_argument("--walk-curriculum", action="store_true",
                        help="Train walk PPO through a low-load gait stage before exact Walk-and-Push")
    parser.add_argument(
        "--expert-trajectory", action="append", default=None,
        help=(
            "Verified fall-free reference or model-based trajectory used only "
            "to initialize walk PPO; repeat the option for multiple verified "
            "state distributions"
        ),
    )
    parser.add_argument(
        "--expert-target", choices=("reference", "residual"), default="reference",
        help=(
            "Interpret the expert as the shared DIAL reference (zero residual) "
            "or as an interface-matched, violation-free model-based residual prefix"
        ),
    )
    parser.add_argument("--expert-pretrain-steps", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu-low-memory", action="store_true")
    parser.add_argument("--normalizer-std-eps", type=float, default=0.05,
                        help="Compatibility statistics setting; unused by the unnormalized H1 policy")
    parser.add_argument("--max-policy-kl", type=float, default=10.0)
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help=(
            "Optional audited PPO learning-rate override; defaults to the "
            "schema-specific locked training value"
        ),
    )
    parser.add_argument(
        "--init-noise-std", type=float, default=None,
        help=(
            "Optional audited pre-tanh PPO exploration standard deviation; "
            "expert initialization writes it explicitly into the tanh-normal "
            "scale head"
        ),
    )
    parser.add_argument(
        "--behavior-anchor-weight", type=float, default=0.0,
        help=(
            "Training-only squared-action penalty to a frozen residual expert; "
            "requires --expert-target residual"
        ),
    )
    args = parser.parse_args()
    if any(not np.isfinite(value) or value <= 0
           for value in (args.normalizer_std_eps, args.max_policy_kl)):
        raise ValueError("normalizer-std-eps and max-policy-kl must be positive")
    output = Path(args.out) if args.out else _default_output(args.schema, args.seed)
    if output.exists():
        raise FileExistsError(f"Frozen checkpoint exists; choose a new --out: {output}")

    canonical_config = ExperimentConfig.from_yaml(CANONICAL_CONFIG)
    if args.expert_trajectory is not None and args.schema != "walk":
        raise ValueError("expert-trajectory is restricted to the walk PPO schema")
    if args.expert_target != "reference" and args.expert_trajectory is None:
        raise ValueError("expert-target residual requires expert-trajectory")
    if (not np.isfinite(args.behavior_anchor_weight)
            or args.behavior_anchor_weight < 0.0):
        raise ValueError("behavior-anchor-weight must be finite and nonnegative")
    if args.behavior_anchor_weight > 0.0 and args.expert_target != "residual":
        raise ValueError(
            "behavior-anchor-weight requires --expert-target residual"
        )
    _validate_training_seed(args.seed, canonical_config)
    canonical_config_sha = hashlib.sha256(CANONICAL_CONFIG.read_bytes()).hexdigest()
    resolved_suites = _training_suites(args.schema, canonical_config)
    specs = _domain_specs(args.schema, canonical_config)
    specs, env_overrides = _development_env_specs(args.schema, specs, args.env_overrides)
    groups = _walk_curriculum_groups(args.schema, specs, args.walk_curriculum)
    if args.env_overrides is not None:
        development_root = CANONICAL_CONFIG.parents[3] / "results" / "_development"
        if not output.resolve().is_relative_to(development_root.resolve()):
            raise ValueError("Walk-and-Push env-overrides require --out under results/_development")
    stage_episode_lengths = _stage_episode_lengths(
        args.schema, resolved_suites, args.episode_length, stage_count=len(groups),
    )
    atacom = args.schema.startswith("atacom_")
    atacom_options = _atacom_training_options(args.schema)
    num_timesteps = 4096 if args.smoke else int(args.num_timesteps)
    num_envs = max(4 if args.schema == "fixed" else len(specs), 2) if args.smoke else int(args.num_envs)
    kwargs = _ppo_training_kwargs(
        schema=args.schema, atacom=atacom, num_envs=num_envs,
        cpu_low_memory=args.cpu_low_memory,
        normalizer_std_eps=args.normalizer_std_eps,
    )
    if args.learning_rate is not None:
        if not np.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
            raise ValueError("learning-rate must be finite and positive")
        kwargs["learning_rate"] = float(args.learning_rate)
    if args.init_noise_std is not None:
        if (not np.isfinite(args.init_noise_std)
                or args.init_noise_std <= 1.0e-3):
            raise ValueError("init-noise-std must exceed 0.001")
        kwargs["init_noise_std"] = float(args.init_noise_std)
    history = []
    started = time.monotonic()
    source_hashes = _training_source_hashes(Path(__file__).resolve().parents[4], atacom=atacom)
    trainer_sha = source_hashes["scripts/tasks/robot/humanoid/train_box_push_rl.py"]
    training_root = output.parent / "_training" / output.stem
    status_path = training_root / "training_status.json"
    if status_path.exists():
        raise FileExistsError(f"Training audit exists; choose a new --out: {status_path}")
    stage_coverage = []
    expert_initialization = None
    policy_interface = None
    policy_action_transform = None
    action_bias = None
    action_scale = None
    resolved_env_params = []
    atacom_transform = None
    behavior_anchor = None

    current_stage = 0

    def progress(step, metrics):
        values = scalar_metrics(metrics)
        row = {
            "stage": int(current_stage),
            "num_steps": int(step),
            "elapsed_seconds": float(time.monotonic() - started),
            **{key: value if np.isfinite(value) else str(value)
               for key, value in values.items()},
        }
        history.append(row)
        write_status("running")
        print(json.dumps(row, sort_keys=True), flush=True)
        _check_training_metrics(metrics, args.max_policy_kl)

    def write_status(status, error=None):
        observed = _observed_training_steps(history)
        _write_training_status(status_path, {
            "status": status,
            "purpose": "pipeline_smoke" if args.smoke else "policy_training",
            "performance_validated": False,
            "schema": args.schema, "training_seed": int(args.seed),
            "evaluation_seeds_excluded": sorted(int(v) for v in canonical_config.seeds),
            "output": str(output), "cli": vars(args),
            "trainer_sha256": trainer_sha, "canonical_config_sha256": canonical_config_sha,
            "startup_source_sha256": source_hashes,
            "env_overrides": env_overrides,
            "resolved_env_params": resolved_env_params,
            "stage_episode_lengths": stage_episode_lengths,
            "policy_interface": policy_interface,
            "policy_action_transform": policy_action_transform,
            "atacom_transform": atacom_transform,
            "expert_initialization": expert_initialization,
            "behavior_anchor": behavior_anchor,
            "training_termination_contract": _H1_TERMINATION_CONTRACT,
            "ppo_update_contract": {
                "learning_rate_schedule": kwargs["learning_rate_schedule"],
                "learning_rate": kwargs["learning_rate"],
                "init_noise_std": kwargs.get("init_noise_std"),
                "unroll_length": kwargs["unroll_length"],
                "discounting": kwargs["discounting"],
                "num_updates_per_batch": kwargs["num_updates_per_batch"],
                "observation_normalization": (
                    "running_statistics" if kwargs["normalize_observations"] else "disabled"
                ),
                "normalizer_std_eps_used_by_policy": bool(kwargs["normalize_observations"]),
            },
            "requested_environment_steps": num_timesteps,
            "requested_stage_steps": stage_budgets,
            "observed_stage_steps": observed,
            "observed_environment_steps": sum(observed.values()),
            "observed_steps_are_lower_bound": status != "completed",
            "stage_health": stage_coverage, "progress_history": history,
            "elapsed_seconds": time.monotonic() - started,
            "failure": error,
        })

    kwargs["progress_fn"] = progress
    # Unjamming's nearby wall changes MJX's static contact-candidate pytree.  It
    # cannot be mixed with open Force Regulation/Fixed-Stance Push through lax.switch, even though policy
    # observation/action shapes match.  Train one shared fixed policy through
    # an explicit two-stage curriculum.  Brax restores normalizer/policy/value
    # parameters; its optimizer and env-step counter restart in each stage.
    stage_budgets = (
        [num_timesteps // 2, num_timesteps - num_timesteps // 2]
        if len(groups) == 2 else [num_timesteps]
    )
    restore = None
    params = config = env = None
    write_status("running")
    try:
        for stage, (stage_specs, stage_budget) in enumerate(zip(groups, stage_budgets)):
            current_stage = stage
            domains = [make_env(TASK, **spec) for spec in stage_specs]
            stage_interface, actual_params, stage_transform = _environment_training_contract(
                domains, atacom_options,
            )
            stage_policy_action_transform = (
                None if atacom else _environment_policy_action_transform(domains)
            )
            if stage and stage_interface != policy_interface:
                # The Walk-and-Push curriculum changes reward/loading only; the final
                # environment owns the exported task interface.  Structural
                # action/observation compatibility is checked explicitly.
                if not args.walk_curriculum:
                    raise ValueError("Cannot restore PPO parameters across different policy interfaces")
                if (stage_interface["action_layout"] != policy_interface["action_layout"]
                        or stage_interface["observation_layout"] != policy_interface["observation_layout"]
                        or stage_interface["control"] != policy_interface["control"]):
                    raise ValueError("Walk curriculum changed policy input/output semantics")
            if stage and stage_policy_action_transform != policy_action_transform:
                raise ValueError("Cannot restore PPO parameters across different policy action transforms")
            if args.env_overrides is not None and stage_interface is None:
                raise ValueError("New Walk-and-Push development environment must expose its policy_interface")
            policy_interface = stage_interface
            policy_action_transform = stage_policy_action_transform
            atacom_transform = stage_transform
            resolved_env_params.extend(actual_params)
            write_status("running")
            if atacom:
                from genedynamics.solvers.single.atacom.wrapper import AtacomEnvWrapper
                domains = [AtacomEnvWrapper(domain, **atacom_options) for domain in domains]
            elif policy_action_transform is not None:
                action_bias = policy_action_transform["action_bias"]
                action_scale = policy_action_transform["action_scale"]
                domains = [HumanoidBoxPushResidualActionEnv(
                    domain, action_bias, action_scale,
                ) for domain in domains]
            env = HumanoidBoxPushDomainEnv(domains)
            stage_kwargs = dict(kwargs)
            checkpoint_root = training_root / f"stage_{stage}"
            stage_kwargs["save_checkpoint_path"] = str(checkpoint_root.resolve())
            if restore is not None:
                stage_kwargs["restore_checkpoint_path"] = str(restore.resolve())
            elif args.expert_trajectory is not None:
                restore_params, expert_initialization = _pretrain_walk_policy_from_expert(
                    env, policy_action_transform, policy_interface, args.expert_trajectory,
                    seed=int(args.seed),
                    hidden_sizes=kwargs["policy_hidden_layer_sizes"],
                    normalize_observations=kwargs["normalize_observations"],
                    normalizer_std_eps=args.normalizer_std_eps,
                    init_noise_std=kwargs.get("init_noise_std", 1.0),
                    steps=int(args.expert_pretrain_steps),
                    expert_target=args.expert_target,
                )
                stage_kwargs["restore_params"] = restore_params
                if args.behavior_anchor_weight > 0.0:
                    teacher_action = _walk_behavior_anchor_action(
                        restore_params, env,
                        hidden_sizes=kwargs["policy_hidden_layer_sizes"],
                        normalize_observations=kwargs["normalize_observations"],
                        init_noise_std=kwargs.get("init_noise_std", 1.0),
                    )
                    env = _BehaviorAnchoredH1Env(
                        env, teacher_action, args.behavior_anchor_weight,
                    )
                    behavior_anchor = {
                        "weight": float(args.behavior_anchor_weight),
                        "space": "normalized_residual_action",
                        "teacher": "frozen_pre_ppo_behavior_cloning_mean",
                        "expert_sha256": expert_initialization["sha256"],
                    }
                write_status("running")
            params, config = train_rl_policy(
                env, algo="ppo", num_timesteps=int(stage_budget),
                episode_length=stage_episode_lengths[stage], num_envs=num_envs,
                seed=int(args.seed), **stage_kwargs,
            )
            jax.effects_barrier()
            stage_coverage.append(_validate_stage_for_export(
                params, config, history, stage, len(stage_specs),
                args.normalizer_std_eps, args.max_policy_kl,
            ))
            write_status("running")
            checkpoints = sorted(
                (path for path in checkpoint_root.iterdir()
                 if path.is_dir() and path.name.isdigit()),
                key=lambda path: int(path.name),
            )
            if stage + 1 < len(groups):
                if not checkpoints:
                    raise RuntimeError(
                        f"PPO stage {stage} produced no restorable checkpoint"
                    )
                restore = checkpoints[-1]
                if args.cpu_low_memory:
                    # The restorable parameter checkpoint and tiny params
                    # arrays remain live; completed Force Regulation/Fixed-Stance Push physics executables
                    # need not coexist with Unjamming's different contact graph.
                    env = None
                    domains = None
                    jax.clear_caches()
                    gc.collect()
                    stage_coverage[-1]["cpu_stage_cache_released"] = True
                    write_status("running")
    except BaseException as error:
        write_status("failed", f"{type(error).__name__}: {error}")
        raise
    actual_timesteps = sum(_observed_training_steps(history).values())
    protocol = f"humanoid_box_push_{args.schema}_ppo_v1"
    config.update({
        "protocol": protocol,
        "task": TASK,
        "schema": args.schema,
        "training_seed": int(args.seed),
        "evaluation_seeds_excluded": sorted(int(value) for value in canonical_config.seeds),
        "training_domains": specs,
        "env_overrides": env_overrides,
        "resolved_env_params": resolved_env_params,
        "policy_interface": policy_interface,
        "policy_action_transform": policy_action_transform,
        "action_bias": action_bias,
        "action_scale": action_scale,
        "atacom_transform": atacom_transform,
        "expert_initialization": expert_initialization,
        "behavior_anchor": behavior_anchor,
        "training_termination_contract": _H1_TERMINATION_CONTRACT,
        "training_protocol_source": {
            "path": "configs/humanoid/push_to_line/_base.yaml",
            "sha256": canonical_config_sha,
        },
        "training_suite_names": list(SCHEMA_SUITES[args.schema]),
        "canonical_suite_episode_lengths": [suite.n_steps for suite in resolved_suites],
        "stage_episode_lengths": stage_episode_lengths,
        "episode_length_override": args.episode_length,
        "requested_training_budget": {"unit": "environment_steps", "value": num_timesteps},
        "training_budget": {"unit": "environment_steps", "value": actual_timesteps},
        "num_timesteps": actual_timesteps,
        "selection_rule": "final_checkpoint_after_fixed_budget",
        "progress_history": history,
        "training_wall_seconds": float(time.monotonic() - started),
        "training_complete": True,
        "training_purpose": "pipeline_smoke" if args.smoke else "policy_training",
        "performance_validated": False,
        "trainer_sha256": trainer_sha,
        "startup_source_sha256": source_hashes,
        "stage_restore_scope": "normalizer_policy_value_parameters_only",
        "ppo_update_contract": {
            "learning_rate_schedule": kwargs["learning_rate_schedule"],
            "learning_rate": kwargs["learning_rate"],
            "init_noise_std": kwargs.get("init_noise_std"),
            "unroll_length": kwargs["unroll_length"],
            "discounting": kwargs["discounting"],
            "num_updates_per_batch": kwargs["num_updates_per_batch"],
            "observation_normalization": (
                "running_statistics"
                if kwargs["normalize_observations"] else "disabled"
            ),
            "normalizer_std_eps_used_by_policy": bool(
                kwargs["normalize_observations"]
            ),
        },
        "observed_stage_steps": _observed_training_steps(history),
        "training_wrapper": (
            "h1_success_absorbing_complete_info_reset_stratified_domains"
            "+task_owned_residual_action"
            if policy_action_transform is not None
            else "h1_success_absorbing_complete_info_reset_stratified_domains"
        ),
        "observed_domain_coverage": stage_coverage,
        "checkpoint_validation": {
            "normalizer_std_eps": float(args.normalizer_std_eps),
            "policy_normalizer_floor_checked": bool(config["normalize_observations"]),
            "max_policy_kl": float(args.max_policy_kl),
            "finite_parameters": True,
            "all_stage_domains_observed": True,
        },
    })
    try:
        save_policy(str(output), params, config)
    except BaseException as error:
        write_status("failed", f"{type(error).__name__}: {error}")
        raise
    write_status("completed")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({"checkpoint": str(output), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
