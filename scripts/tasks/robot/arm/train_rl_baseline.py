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

import numpy as np

from genedynamics.envs.factories import make_env
from genedynamics.envs.domains.manipulation.panda_brax import (
    PandaSurfaceScanDomainEnv,
)
from genedynamics.learning.train_rl_policy import train_rl_policy, save_policy
from genedynamics.solvers.single.mdac.config import (
    deep_merge,
    load_experiment_config,
)

ARM_TASK = "manipulator_surface_scan"


def _shared_domain_specs(base_env, rl_cfg):
    templates = list(rl_cfg.get("domain_templates", ()))
    if not templates:
        return []
    dr = dict(rl_cfg.get("domain_randomization", {}))
    n_domains = int(dr.get("num_domains", len(templates)))
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

    domain_specs = _shared_domain_specs(base_env, rl_cfg)
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
        if a.atacom:
            raise ValueError("shared MGA policy uses the raw 10D action, not ATACOM")
        domains = [make_env(ARM_TASK, **spec) for spec in domain_specs]
        env = PandaSurfaceScanDomainEnv(domains)
        env_desc = f"{len(domains)} shared seen domains"
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

    if env.action_size != 10 and not a.atacom:
        raise ValueError(
            f"shared MGA prior requires the 10D primitive, got {env.action_size}"
        )
    if domain_specs and any(
        spec.get("observation_mode") != "rl_realized" for spec in domain_specs
    ):
        raise ValueError("all shared policy domains must use observation_mode=rl_realized")

    if a.atacom:
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
    config.update({
        "protocol": "arm_shared_mga_v1" if domain_specs else "arm_baseline_v1",
        "episode_length": int(episode_length),
        "training_seed": int(seed),
        "domain_specs": domain_specs,
    })

    if a.out:
        out = a.out
    elif domain_specs:
        template = rl_cfg.get(
            "checkpoint",
            "results/arm/impedence/_policies/shared_ppo_seed{seed}.pkl",
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
