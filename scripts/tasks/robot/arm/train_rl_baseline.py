"""Train the arm surface-scan RL policy and save one reusable checkpoint.

  * ISSA  -> trains on the RAW env (action_size = nu); deploy adds the AdamBA safe-set
             projection. ckpt: ``_policies/<algo>.pkl``.
  * ATACOM (``--atacom``) -> trains ON the constraint manifold (the tangent-space env wrapper,
             action_size = nu - n_f); the policy is manifold-resident, NOT post-hoc projected,
             so it needs its OWN ckpt: ``_policies/atacom_<algo>.pkl``.

The MGA path loads a formal task base YAML and trains one shared raw-action
PPO across its seen material/geometry domains.  The same checkpoint is deployed
standalone and reconstructed as the MGA horizon prior.

Needs real brax/mjx -> run in docker (genedynamics/dev-cpu:torch):
  docker compose -f docker/compose.cpu.yml run --rm genedynamics-dev-cpu \
    python scripts/tasks/robot/arm/train_rl_baseline.py \
      --config configs/arm/surface_scan/_base.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from genedynamics.envs.factories import make_env
from genedynamics.envs.domains.manipulation.panda_brax import (
    PandaResidualActionEnv,
    PandaSurfaceScanDomainEnv,
)
from genedynamics.learning.train_rl_policy import (
    load_policy,
    save_policy,
    scalar_metrics,
    train_rl_policy,
)
from genedynamics.experiments.framework.config import (
    ExperimentConfig,
    deep_merge,
)

ARM_TASK = "manipulator_surface_scan"
INSERT_TASK = "manipulator_peg_insert"


class _BestEvalSelector:
    """Pair Brax's params-before-eval callbacks and retain the best policy."""

    def __init__(self, metric="eval/episode_reward", checkpoint_fn=None):
        self._snapshots = {}
        self._observations = {}
        self.metric = str(metric)
        self.params = None
        self.step = None
        self.score = -np.inf
        self.metrics = {}
        self.checkpoint_fn = checkpoint_fn

    def capture(self, num_steps, _make_policy, params):
        import jax

        step = int(num_steps)
        self._snapshots[step] = jax.device_get(params)
        metrics = self._observations.pop(step, None)
        if metrics is not None:
            self._consider(step, metrics)

    def observe(self, num_steps, metrics):
        step = int(num_steps)
        if step not in self._snapshots:
            # Brax reports the step-zero evaluation before its params callback.
            # Retain that metric so a useful residual primitive is a genuine
            # lower bound: training must improve it before replacing it.
            self._observations[step] = dict(metrics)
            return
        self._consider(step, metrics)

    def _consider(self, step, metrics):
        snapshot = self._snapshots.pop(step, None)
        self._snapshots = {
            k: value for k, value in self._snapshots.items() if k > step
        }
        self._observations = {
            k: value for k, value in self._observations.items() if k > step
        }
        if snapshot is None:
            return
        score = float(metrics.get(self.metric, np.nan))
        if np.isfinite(score) and score > self.score:
            self.params = snapshot
            self.step = step
            self.score = score
            self.metrics = dict(metrics)
            if self.checkpoint_fn is not None:
                self.checkpoint_fn(self.params, self.step, self.metrics)


def _cpu_low_memory_ppo_kwargs(num_envs: int) -> dict:
    """Return the smallest Brax-valid PPO batch for a vectorized CPU env."""
    vector_width = int(num_envs)
    if vector_width < 1:
        raise ValueError("num_envs must be positive")
    return {
        "batch_size": vector_width,
        "num_minibatches": 1,
        "unroll_length": 1,
        "num_updates_per_batch": 1,
        "policy_hidden_layer_sizes": (16, 16),
    }


def _atacom_training_options(resolved: dict) -> dict:
    """Resolve the projection contract shared by training and deployment."""
    method_params = dict(resolved.get("method_params") or {})
    options = {
        "Kc": float(method_params.get("Kc", 1.0)),
        "action_limit": float(method_params.get("action_limit", 1.0)),
        "alpha_limit": float(method_params.get(
            "alpha_limit", method_params.get("action_limit", 1.0)
        )),
    }
    if not np.isfinite(options["Kc"]) or options["Kc"] <= 0.0:
        raise ValueError("ATACOM Kc must be finite and positive")
    if (
        not np.isfinite(options["action_limit"])
        or options["action_limit"] <= 0.0
        or options["action_limit"] > 1.0
    ):
        raise ValueError("ATACOM action_limit must be finite and in (0, 1]")
    if (
        not np.isfinite(options["alpha_limit"])
        or options["alpha_limit"] <= 0.0
        or options["alpha_limit"] > 1.0
    ):
        raise ValueError("ATACOM alpha_limit must be finite and in (0, 1]")
    return options


def _policy_validation_summary(
    results_root: Path,
    *,
    expected_suites,
    expected_seeds,
    evaluated_sha256: str,
    require_nonzero_safe_success: bool = False,
) -> dict:
    """Validate a disjoint development matrix before freezing a policy."""
    expected = {
        (str(suite), int(seed))
        for suite in expected_suites
        for seed in expected_seeds
    }
    observed = {}
    for path in sorted(results_root.glob("level_*/seed_*/results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (str(result.get("level")), int(result.get("seed", -1)))
        if key not in expected:
            continue
        if key in observed:
            raise ValueError(f"duplicate policy validation result for {key}")
        provenance = result.get("provenance") or {}
        checkpoint = (provenance.get("checkpoints") or {}).get("policy_ckpt") or {}
        if checkpoint.get("sha256") != evaluated_sha256:
            raise ValueError(
                f"validation {key} used policy sha256={checkpoint.get('sha256')!r}, "
                f"expected {evaluated_sha256!r}"
            )
        metrics = (result.get("metrics") or {}).get("peg_insert_metrics") or {}
        required = {
            name: float(metrics[name])
            for name in (
                "insertion_success",
                "safe_insertion_success",
                "max_insertion_depth",
                "force_torque_violation_rate",
                "jam_rate",
                "rho_cvar95",
            )
        }
        if not all(np.isfinite(value) for value in required.values()):
            raise ValueError(f"validation {key} contains non-finite metrics")
        observed[key] = required
    missing = sorted(expected.difference(observed))
    if missing:
        raise ValueError(f"missing policy validation results: {missing}")
    insertion = np.asarray([
        observed[key]["insertion_success"] for key in sorted(expected)
    ])
    if not bool(np.all(insertion >= 1.0)):
        raise ValueError(
            "policy health gate requires insertion success on every "
            f"development run, got {int(np.sum(insertion >= 1.0))}/{len(insertion)}"
        )
    safe_insertion = np.asarray([
        observed[key]["safe_insertion_success"] for key in sorted(expected)
    ])
    if require_nonzero_safe_success and not bool(np.any(safe_insertion >= 1.0)):
        raise ValueError(
            "ATACOM policy health gate requires nonzero whole-window safe "
            "insertion success on the independent development matrix"
        )
    summary = {
        "result_root": str(results_root),
        "suites": [str(item) for item in expected_suites],
        "seeds": [int(item) for item in expected_seeds],
        "run_count": len(observed),
        "insertion_success_rate": float(np.mean(insertion)),
        "safe_insertion_success_rate": float(np.mean(safe_insertion)),
        "mean_max_insertion_depth": float(np.mean([
            observed[key]["max_insertion_depth"] for key in sorted(expected)
        ])),
        "mean_force_torque_violation_rate": float(np.mean([
            observed[key]["force_torque_violation_rate"] for key in sorted(expected)
        ])),
        "mean_jam_rate": float(np.mean([
            observed[key]["jam_rate"] for key in sorted(expected)
        ])),
        "mean_rho_cvar95": float(np.mean([
            observed[key]["rho_cvar95"] for key in sorted(expected)
        ])),
        "evaluated_checkpoint_sha256": evaluated_sha256,
        "health_gate": (
            "all_runs_insert_success_nonzero_safe_success_and_all_required_metrics_finite"
            if require_nonzero_safe_success
            else "all_runs_insert_success_and_all_required_metrics_finite"
        ),
        "health_gate_passed": True,
    }
    return summary


def _freeze_existing_policy(
    path: Path,
    *,
    selected_step: int,
    validation_results: Path,
    validation_suites,
    validation_seeds,
    require_nonzero_safe_success: bool = False,
) -> dict:
    """Attach independent validation provenance to an already-trained policy."""
    payload = path.read_bytes()
    evaluated_sha256 = hashlib.sha256(payload).hexdigest()
    params, config = load_policy(str(path))
    summary = _policy_validation_summary(
        validation_results,
        expected_suites=validation_suites,
        expected_seeds=validation_seeds,
        evaluated_sha256=evaluated_sha256,
        require_nonzero_safe_success=require_nonzero_safe_success,
    )
    selection_metric = (
        "independent_development_safe_insertion_success_rate"
        if require_nonzero_safe_success
        else "independent_development_insertion_success_rate"
    )
    selected_metric_value = (
        summary["safe_insertion_success_rate"]
        if require_nonzero_safe_success
        else summary["insertion_success_rate"]
    )
    config.update({
        "selected_step": int(selected_step),
        "selection_metric": selection_metric,
        "selected_metric_value": selected_metric_value,
        "selected_eval_reward": None,
        "selection_rule": "final_budget_after_independent_health_gate",
        "selection_summary": summary,
        "inline_eval": False,
        "inline_eval_disabled_reason": "cpu_memory_limit_independent_validation_used",
        "training_complete": True,
    })
    save_policy(str(path), params, config)
    return summary


def _shared_domain_specs(
    base_env, rl_cfg, *, task=ARM_TASK, num_domains_override=None
):
    templates = list(rl_cfg.get("domain_templates", ()))
    if not templates:
        return []
    dr = dict(rl_cfg.get("domain_randomization", {}))
    n_domains = int(
        num_domains_override
        if num_domains_override is not None
        else dr.get("num_domains", len(templates))
    )
    if n_domains < 1:
        raise ValueError("num_domains must be positive")
    rng = np.random.default_rng(int(dr.get("seed", 0)))
    friction_range = tuple(dr.get("friction_range", (0.05, 0.2)))
    soft_range = tuple(dr.get("soft_stiffness_range", (800.0, 2000.0)))
    parameter_ranges = dict(dr.get("parameter_ranges", {}))
    domain_seed_start = int(dr.get("domain_seed_start", 100))
    specs = []
    training_env_params = dict(rl_cfg.get("training_env_params", {}))
    for i in range(n_domains):
        spec = deep_merge(
            deep_merge(base_env, training_env_params),
            templates[i % len(templates)],
        )
        if task == ARM_TASK:
            spec["surface_seed"] = int(spec.get("surface_seed", i))
            spec["friction"] = float(rng.uniform(*friction_range))
            if str(spec.get("medium", "rigid")).lower() == "soft":
                spec["soft_stiffness"] = float(rng.uniform(*soft_range))
        elif task == INSERT_TASK:
            spec["domain_seed"] = int(
                spec.get("domain_seed", domain_seed_start + i)
            )
            for name, bounds in parameter_ranges.items():
                if len(bounds) != 2:
                    raise ValueError(
                        f"domain_randomization.parameter_ranges.{name} "
                        "must be [low, high]"
                    )
                spec[name] = float(rng.uniform(float(bounds[0]), float(bounds[1])))
        else:
            raise ValueError(f"shared manipulation RL does not support task={task!r}")
        specs.append(spec)
    return specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="canonical inherited YAML for the shared MGA policy")
    ap.add_argument("--medium", default=None, choices=["rigid", "soft", "hybrid"])
    ap.add_argument("--algo", default=None, choices=["ppo", "sac"],
                    help="arm RL baselines use SAC by default")
    ap.add_argument("--level", default=None, help="training surface (seen). hybrid: a plane map.")
    ap.add_argument("--stiffness-map", default="center_hard", help="hybrid only: the spatial map")
    ap.add_argument("--num-timesteps", type=int, default=None)
    ap.add_argument("--episode-length", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument(
        "--num-domains",
        type=int,
        default=None,
        help="CPU resource override; canonical GPU training keeps all configured domains",
    )
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="SAC only: replay-buffer prefill (store-only, no updates) before learning")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--num-evals", type=int, default=None,
        help="CPU segment-size override; does not change the total environment steps",
    )
    ap.add_argument(
        "--resume-training", action="store_true",
        help="initialize policy/value parameters from the latest Brax checkpoint",
    )
    ap.add_argument("--atacom", action="store_true",
                    help="train the policy ON the constraint manifold (ATACOM tangent-space env "
                         "wrapper, action_size = nu - n_f); produces an ATACOM-specific ckpt")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU integration run: 4096 steps and four vector envs")
    ap.add_argument(
        "--cpu-low-memory",
        action="store_true",
        help=(
            "use a one-step PPO unroll and compact network so MJX gradients fit "
            "inside Docker Desktop's CPU memory budget"
        ),
    )
    ap.add_argument(
        "--skip-inline-eval",
        action="store_true",
        help=(
            "do not compile Brax's full-episode evaluator during CPU training; "
            "select checkpoints later with the task's held-out rollout"
        ),
    )
    ap.add_argument(
        "--freeze-existing-step",
        type=int,
        default=None,
        help="freeze an already-trained deploy policy after independent validation",
    )
    ap.add_argument(
        "--validation-results",
        default=None,
        help="method result directory used by --freeze-existing-step",
    )
    ap.add_argument(
        "--validation-seeds", type=int, nargs="+", default=(104, 105),
    )
    ap.add_argument(
        "--validation-suites", nargs="+",
        default=("id_wide", "ood_pose", "ood_sensing"),
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    resolved = (
        ExperimentConfig.from_yaml(Path(a.config)).to_dict()
        if a.config else {}
    )
    training = dict((resolved.get("metadata") or {}).get("training") or {})
    rl_cfg = dict(training.get("rl") or {})
    base_env = dict(resolved.get("env_params", {}))
    task = str(resolved.get("env_name", ARM_TASK))
    if task not in (ARM_TASK, INSERT_TASK):
        raise ValueError(
            f"arm RL baseline supports {ARM_TASK!r} or {INSERT_TASK!r}, got {task!r}"
        )
    algo = a.algo or rl_cfg.get("algo") or ("sac" if not a.config else "ppo")
    medium = a.medium or base_env.get("medium", "rigid")
    level = a.level or resolved.get("level", "convex")
    num_timesteps = (
        a.num_timesteps
        if a.num_timesteps is not None
        else 4096 if a.smoke
        else int(rl_cfg.get("num_timesteps", 2_000_000))
    )
    episode_length = (
        a.episode_length or int(rl_cfg.get("episode_length", 100 if a.config else 64))
    )
    num_envs = (
        a.num_envs or 4
        if a.smoke
        else a.num_envs or int(rl_cfg.get("num_envs", 64))
    )
    warmup_steps = a.warmup_steps or int(rl_cfg.get("warmup_steps", 5_000))
    seed = a.seed if a.seed is not None else int(rl_cfg.get("seed", 0))
    atacom_options = _atacom_training_options(resolved) if a.atacom else None

    domain_specs = _shared_domain_specs(
        base_env,
        rl_cfg,
        task=task,
        num_domains_override=a.num_domains,
    )
    if a.smoke and len(domain_specs) > 2:
        # Compile one rigid and one soft branch on CPU. The full canonical
        # distribution remains unchanged for formal training.
        soft_index = next(
            (
                i for i, spec in enumerate(domain_specs)
                if str(spec.get("medium", "rigid")).lower() == "soft"
            ),
            1,
        )
        domain_specs = [domain_specs[0], domain_specs[soft_index]]
    if domain_specs:
        domains = [make_env(task, **spec) for spec in domain_specs]
        if a.atacom:
            # Wrap each concrete branch before domain randomization.  The
            # randomized dispatcher then has one shape-compatible 7D action
            # contract and never needs to interpret ATACOM geometry itself.
            from genedynamics.solvers.single.atacom.wrapper import AtacomEnvWrapper
            domains = [
                AtacomEnvWrapper(domain, **atacom_options)
                for domain in domains
            ]
        if task == INSERT_TASK:
            from genedynamics.envs.domains.manipulation.peg_insert_brax import (
                PegInsertDomainEnv,
            )
            env = PegInsertDomainEnv(domains)
        else:
            env = PandaSurfaceScanDomainEnv(domains)
        env_desc = (
            f"{len(domains)} shared seen domains"
            + (" + ATACOM tangent wrappers" if a.atacom else "")
        )
    else:
        if task == INSERT_TASK:
            env_kw = deep_merge(
                base_env,
                {"level": level, "stiffness_mode": "log_spd"},
            )
            env = make_env(task, **env_kw)
            env_desc = f"task=peg_insert level={env_kw['level']}"
        else:
            env_kw = deep_merge(
                base_env,
                {
                    "level": ("plane" if medium == "hybrid" else level),
                    "medium": medium,
                    "stiffness_mode": "log_spd",
                },
            )
            if medium == "hybrid":
                env_kw["stiffness_map"] = a.stiffness_map
            env = make_env(task, **env_kw)
            env_desc = f"medium={medium} level={env_kw['level']}"

    action_bias = None
    action_scale = None
    if domain_specs and not a.atacom and rl_cfg.get("action_bias") is not None:
        action_bias = np.asarray(rl_cfg["action_bias"], dtype=np.float32)
        action_scale = np.asarray(
            rl_cfg.get("action_scale", np.ones_like(action_bias)),
            dtype=np.float32,
        )
        if task == INSERT_TASK:
            from genedynamics.envs.domains.manipulation.peg_insert_brax import (
                PegInsertResidualActionEnv,
            )
            env = PegInsertResidualActionEnv(env, action_bias, action_scale)
        else:
            env = PandaResidualActionEnv(env, action_bias, action_scale)
        env_desc += " + residual action prior"

    expected_action_size = int(rl_cfg.get(
        "expected_action_size", 13 if task == INSERT_TASK else 10
    ))
    if env.action_size != expected_action_size and not a.atacom:
        raise ValueError(
            "shared MGA prior action contract mismatch: "
            f"expected {expected_action_size}, got {env.action_size}"
        )
    expected_observation_mode = rl_cfg.get("expected_observation_mode")
    if expected_observation_mode is not None and domain_specs and any(
        spec.get("observation_mode") != expected_observation_mode
        for spec in domain_specs
    ):
        raise ValueError(
            "all shared policy domains must use observation_mode="
            f"{expected_observation_mode}"
        )

    if a.atacom and not domain_specs:
        # ATACOM remains a comparison policy and never becomes the MGA prior.
        from genedynamics.solvers.single.atacom.wrapper import AtacomEnvWrapper
        env = AtacomEnvWrapper(env, **atacom_options)

    atacom_transform = None
    if a.atacom:
        dt = float(getattr(getattr(env, "_config", None), "dt", base_env.get("dt", 0.02)))
        atacom_transform = {**atacom_options, "time_step": dt}

    if a.out:
        out = a.out
    elif domain_specs:
        template = (
            rl_cfg.get(
                "atacom_checkpoint",
                "results/arm/impedence/_policies/shared_atacom_ppo_seed{seed}.pkl",
            )
            if a.atacom else rl_cfg.get(
                "checkpoint",
                "results/arm/impedence/_policies/shared_ppo_seed{seed}.pkl",
            )
        )
        out = str(Path(template.format(seed=seed)))
    else:
        name = f"atacom_{algo}" if a.atacom else algo
        family = "peg_insert" if task == INSERT_TASK else f"impedence/{medium}"
        out = f"results/arm/{family}/_policies/{name}.pkl"
    protocol_name = (
        f"{task}_shared_atacom_v1" if domain_specs and a.atacom
        else f"{task}_shared_mga_v1" if domain_specs
        else "arm_atacom_v1" if a.atacom
        else "arm_baseline_v1"
    )

    if a.freeze_existing_step is not None:
        if not a.validation_results:
            raise ValueError(
                "--validation-results is required with --freeze-existing-step"
            )
        summary = _freeze_existing_policy(
            Path(out),
            selected_step=a.freeze_existing_step,
            validation_results=Path(a.validation_results),
            validation_suites=a.validation_suites,
            validation_seeds=a.validation_seeds,
            require_nonzero_safe_success=bool(a.atacom),
        )
        print(
            f"froze policy -> {out} "
            f"(validation runs={summary['run_count']}, "
            f"success={summary['insertion_success_rate']:.3f}, "
            f"safe_success={summary['safe_insertion_success_rate']:.3f})",
            flush=True,
        )
        return 0

    train_kwargs = dict(rl_cfg.get("train_kwargs", {}))
    if a.num_evals is not None:
        train_kwargs["num_evals"] = int(a.num_evals)
    if domain_specs and algo == "ppo":
        training_checkpoint_dir = str(
            (Path(out).parent / "_training" / Path(out).stem).resolve()
        )
        train_kwargs.setdefault("save_checkpoint_path", training_checkpoint_dir)
        if a.resume_training and Path(training_checkpoint_dir).exists():
            checkpoints = sorted(
                path for path in Path(training_checkpoint_dir).iterdir()
                if path.is_dir() and path.name.isdigit()
            )
            if not checkpoints:
                raise ValueError(
                    f"no complete Brax checkpoints in {training_checkpoint_dir}"
                )
            train_kwargs.setdefault(
                "restore_checkpoint_path", str(checkpoints[-1].resolve())
            )
    if a.smoke and algo == "ppo":
        train_kwargs.update(
            batch_size=8,
            num_minibatches=2,
            unroll_length=10,
            num_updates_per_batch=1,
        )
    if a.cpu_low_memory and algo == "ppo":
        # Brax requires ``batch_size * num_minibatches`` to be divisible by
        # ``num_envs``.  Keep the smallest legal batch for the canonical
        # vector width instead of hard-coding one (which made PegInsert's
        # four-env CPU path fail before the first update).
        train_kwargs.update(_cpu_low_memory_ppo_kwargs(num_envs))
    if a.skip_inline_eval and algo == "ppo":
        train_kwargs["run_evals"] = False
    progress_history = []
    select_best = (
        bool(rl_cfg.get("select_best_eval", False))
        and algo == "ppo"
        and not a.skip_inline_eval
    )
    selection_metric = str(rl_cfg.get(
        "select_best_eval_metric", "eval/episode_reward"
    ))

    def checkpoint_best(params, selected_step, selected_metrics):
        """Persist each improved eval so a long CPU run remains usable."""
        partial_config = {
            "algo": algo,
            "observation_size": int(env.observation_size),
            "action_size": int(env.action_size),
            "policy_hidden_layer_sizes": tuple(train_kwargs.get(
                "policy_hidden_layer_sizes", (32, 32, 32, 32)
            )),
            "normalize_observations": bool(
                train_kwargs.get("normalize_observations", True)
            ),
            "num_timesteps": int(num_timesteps),
            "episode_length": int(episode_length),
            "num_envs": int(num_envs),
            "learning_rate": float(train_kwargs.get("learning_rate", 3.0e-4)),
            "warmup_steps": int(warmup_steps),
            "seed": int(seed),
            "protocol": protocol_name,
            "task": task,
            "domain_specs": domain_specs,
            "domain_count": len(domain_specs),
            "action_bias": action_bias.tolist() if action_bias is not None else None,
            "action_scale": action_scale.tolist() if action_scale is not None else None,
            "atacom_transform": atacom_transform,
            "progress_history": list(progress_history),
            "selection_metric": selection_metric,
            "selected_step": int(selected_step),
            "selected_metric_value": float(selected_metrics[selection_metric]),
            "selected_eval_reward": selected_metrics.get("eval/episode_reward"),
            "training_complete": False,
            "resumed_training": bool(a.resume_training),
        }
        save_policy(out, params, partial_config)

    selector = (
        _BestEvalSelector(selection_metric, checkpoint_fn=checkpoint_best)
        if select_best else None
    )
    train_started = time.monotonic()

    def progress_fn(num_steps, metrics):
        record = {
            "num_steps": int(num_steps),
            "elapsed_seconds": float(time.monotonic() - train_started),
            **scalar_metrics(metrics),
        }
        progress_history.append(record)
        if selector is not None:
            selector.observe(num_steps, record)
        reward = record.get("eval/episode_reward", float("nan"))
        print(
            f"  progress steps={int(num_steps)} "
            f"reward={reward:.6g} elapsed={record['elapsed_seconds']:.1f}s",
            flush=True,
        )

    train_kwargs.setdefault("progress_fn", progress_fn)
    if selector is not None:
        train_kwargs.setdefault("policy_params_fn", selector.capture)

    print(
        f"training {algo}{' (ATACOM-manifold)' if a.atacom else ''} on "
        f"{env_desc} action_size={env.action_size} "
        f"({num_timesteps} steps, episode={episode_length}, envs={num_envs})...",
        flush=True,
    )
    params, config = train_rl_policy(
        env,
        algo=algo,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        num_envs=num_envs,
        warmup_steps=warmup_steps,
        seed=seed,
        **train_kwargs,
    )
    if selector is not None and selector.params is not None:
        params = selector.params
    config.update({
        "protocol": protocol_name,
        "task": task,
        "episode_length": int(episode_length),
        "training_seed": int(seed),
        "domain_specs": domain_specs,
        "domain_count": len(domain_specs),
        "action_bias": (
            action_bias.tolist() if action_bias is not None else None
        ),
        "action_scale": (
            action_scale.tolist() if action_scale is not None else None
        ),
        "atacom_transform": atacom_transform,
        "progress_history": progress_history,
        "training_wall_seconds": float(time.monotonic() - train_started),
        "selection_metric": selector.metric if selector is not None else None,
        "selected_step": selector.step if selector is not None else None,
        "selected_metric_value": selector.score if selector is not None else None,
        "selected_eval_reward": (
            selector.metrics.get("eval/episode_reward")
            if selector is not None else None
        ),
        "training_complete": True,
    })
    save_policy(out, params, config)
    print("saved policy ->", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
