"""Train the arm surface-scan RL policy and save one reusable checkpoint.

  * ISSA  -> trains on the RAW env (action_size = nu); deploy adds the AdamBA safe-set
             projection. ckpt: ``_policies/<algo>.pkl``.
  * ATACOM (``--atacom``) -> trains ON the constraint manifold (the tangent-space env wrapper,
             action_size = nu - n_f); the policy is manifold-resident, NOT post-hoc projected,
             so it needs its OWN ckpt: ``_policies/atacom_<algo>.pkl``.

The Phase-4 MGA path loads the canonical YAML and trains one shared raw-action
PPO across its seen material/geometry domains.  The same checkpoint is deployed
standalone and reconstructed as the MDAC horizon prior.  The legacy per-medium
CLI remains available for ISSA/ATACOM comparisons.

Needs real brax/mjx -> run in docker (genedynamics/dev-cpu:torch):
  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
              python scripts/tasks/robot/arm/train_rl_baseline.py --medium rigid --algo sac"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from genedynamics.envs.factories import make_env
from genedynamics.envs.domains.manipulation.panda_brax import (
    PandaResidualActionEnv,
    PandaSurfaceScanDomainEnv,
)
from genedynamics.learning.train_rl_policy import (
    save_policy,
    scalar_metrics,
    train_rl_policy,
)
from genedynamics.solvers.single.mdac.config import (
    deep_merge,
    load_experiment_config,
)

ARM_TASK = "manipulator_surface_scan"


class _BestEvalSelector:
    """Pair Brax's params-before-eval callbacks and retain the best policy."""

    def __init__(self, metric="eval/episode_reward"):
        self._snapshots = {}
        self.metric = str(metric)
        self.params = None
        self.step = None
        self.score = -np.inf
        self.metrics = {}

    def capture(self, num_steps, _make_policy, params):
        import jax

        self._snapshots[int(num_steps)] = jax.device_get(params)

    def observe(self, num_steps, metrics):
        step = int(num_steps)
        snapshot = self._snapshots.pop(step, None)
        # The step-zero params callback runs after its progress callback, so its
        # snapshot is discarded at the first trained evaluation.
        self._snapshots = {
            k: value for k, value in self._snapshots.items() if k > step
        }
        if step <= 0 or snapshot is None:
            return
        score = float(metrics.get(self.metric, np.nan))
        if np.isfinite(score) and score > self.score:
            self.params = snapshot
            self.step = step
            self.score = score
            self.metrics = dict(metrics)


def _shared_domain_specs(base_env, rl_cfg, num_domains_override=None):
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
    specs = []
    for i in range(n_domains):
        spec = deep_merge(base_env, templates[i % len(templates)])
        spec["surface_seed"] = int(spec.get("surface_seed", i))
        spec["friction"] = float(rng.uniform(*friction_range))
        if str(spec.get("medium", "rigid")).lower() == "soft":
            spec["soft_stiffness"] = float(rng.uniform(*soft_range))
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
    ap.add_argument("--atacom", action="store_true",
                    help="train the policy ON the constraint manifold (ATACOM tangent-space env "
                         "wrapper, action_size = nu - n_f); produces an ATACOM-specific ckpt")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU integration run: 4096 steps and four vector envs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    resolved = load_experiment_config(a.config) if a.config else {}
    rl_cfg = dict(resolved.get("rl", {}))
    base_env = dict(resolved.get("env_params", {}))
    algo = a.algo or rl_cfg.get("algo") or ("sac" if not a.config else "ppo")
    medium = a.medium or base_env.get("medium", "rigid")
    level = a.level or resolved.get("level", "convex")
    num_timesteps = (
        4096 if a.smoke
        else a.num_timesteps or int(rl_cfg.get("num_timesteps", 2_000_000))
    )
    episode_length = (
        a.episode_length or int(rl_cfg.get("episode_length", 100 if a.config else 64))
    )
    num_envs = 4 if a.smoke else a.num_envs or int(rl_cfg.get("num_envs", 64))
    warmup_steps = a.warmup_steps or int(rl_cfg.get("warmup_steps", 5_000))
    seed = a.seed if a.seed is not None else int(rl_cfg.get("seed", 0))

    domain_specs = _shared_domain_specs(
        base_env, rl_cfg, num_domains_override=a.num_domains
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
        domains = [make_env(ARM_TASK, **spec) for spec in domain_specs]
        if a.atacom:
            # Wrap each concrete branch before domain randomization.  The
            # randomized dispatcher then has one shape-compatible 7D action
            # contract and never needs to interpret ATACOM geometry itself.
            from genedynamics.solvers.single.atacom.wrapper import AtacomEnvWrapper
            domains = [AtacomEnvWrapper(domain) for domain in domains]
        env = PandaSurfaceScanDomainEnv(domains)
        env_desc = (
            f"{len(domains)} shared seen domains"
            + (" + ATACOM tangent wrappers" if a.atacom else "")
        )
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
        env = make_env(ARM_TASK, **env_kw)
        env_desc = f"medium={medium} level={env_kw['level']}"

    action_bias = None
    action_scale = None
    if domain_specs and not a.atacom and rl_cfg.get("action_bias") is not None:
        action_bias = np.asarray(rl_cfg["action_bias"], dtype=np.float32)
        action_scale = np.asarray(
            rl_cfg.get("action_scale", np.ones_like(action_bias)),
            dtype=np.float32,
        )
        env = PandaResidualActionEnv(env, action_bias, action_scale)
        env_desc += " + residual scan bias"

    if env.action_size != 10 and not a.atacom:
        raise ValueError(
            f"shared MGA prior requires the 10D primitive, got {env.action_size}"
        )
    if domain_specs and any(
        spec.get("observation_mode") != "rl_realized" for spec in domain_specs
    ):
        raise ValueError("all shared policy domains must use observation_mode=rl_realized")

    if a.atacom and not domain_specs:
        # ATACOM remains a comparison policy and never becomes the MGA prior.
        from genedynamics.solvers.single.atacom.wrapper import AtacomEnvWrapper
        env = AtacomEnvWrapper(env)

    train_kwargs = dict(rl_cfg.get("train_kwargs", {}))
    if a.smoke and algo == "ppo":
        train_kwargs.update(
            batch_size=8,
            num_minibatches=2,
            unroll_length=10,
            num_updates_per_batch=1,
        )
    progress_history = []
    select_best = bool(rl_cfg.get("select_best_eval", False)) and algo == "ppo"
    selection_metric = str(rl_cfg.get(
        "select_best_eval_metric", "eval/episode_reward"
    ))
    selector = _BestEvalSelector(selection_metric) if select_best else None
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
        "protocol": (
            "arm_shared_atacom_v1" if domain_specs and a.atacom
            else "arm_shared_mga_v1" if domain_specs
            else "arm_atacom_v1" if a.atacom
            else "arm_baseline_v1"
        ),
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
        "progress_history": progress_history,
        "training_wall_seconds": float(time.monotonic() - train_started),
        "selection_metric": selector.metric if selector is not None else None,
        "selected_step": selector.step if selector is not None else None,
        "selected_metric_value": selector.score if selector is not None else None,
        "selected_eval_reward": (
            selector.metrics.get("eval/episode_reward")
            if selector is not None else None
        ),
    })

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
        out = f"results/arm/impedence/{medium}/_policies/{name}.pkl"
    save_policy(out, params, config)
    print("saved policy ->", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
