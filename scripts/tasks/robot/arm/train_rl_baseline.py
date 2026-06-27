"""Train an RL baseline policy (PPO / SAC) on the arm surface-scan env, per medium; save a
checkpoint for the ISSA / ATACOM deploy.

  * ISSA  -> trains on the RAW env (action_size = nu); deploy adds the AdamBA safe-set
             projection. ckpt: ``_policies/<algo>.pkl``.
  * ATACOM (``--atacom``) -> trains ON the constraint manifold (the tangent-space env wrapper,
             action_size = nu - n_f); the policy is manifold-resident, NOT post-hoc projected,
             so it needs its OWN ckpt: ``_policies/atacom_<algo>.pkl``.

Per the experiment granularity: ONE policy per (medium, algo[, atacom]), trained on the
medium's seen surface(s), evaluated on seen + unseen.

Needs real brax/mjx -> run in docker (genedynamics/dev-cpu:torch):
  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
              python scripts/tasks/robot/arm/train_rl_baseline.py --medium rigid --algo sac"
"""

from __future__ import annotations

import argparse

from genedynamics.envs.factories import make_env
from genedynamics.learning.train_rl_policy import train_rl_policy, save_policy

ARM_TASK = "manipulator_surface_scan"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--medium", default="rigid", choices=["rigid", "soft", "hybrid"])
    ap.add_argument("--algo", default="sac", choices=["ppo", "sac"],   # arm: SAC (off-policy, sample-efficient)
                    help="arm RL baselines use SAC by default")
    ap.add_argument("--level", default="convex", help="training surface (seen). hybrid: a plane map.")
    ap.add_argument("--stiffness-map", default="center_hard", help="hybrid only: the spatial map")
    ap.add_argument("--num-timesteps", type=int, default=2_000_000)
    ap.add_argument("--episode-length", type=int, default=64)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--warmup-steps", type=int, default=5_000,
                    help="SAC only: replay-buffer prefill (store-only, no updates) before learning")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atacom", action="store_true",
                    help="train the policy ON the constraint manifold (ATACOM tangent-space env "
                         "wrapper, action_size = nu - n_f); produces an ATACOM-specific ckpt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    env_kw = {"level": ("plane" if a.medium == "hybrid" else a.level),
              "medium": a.medium, "stiffness_mode": "log_spd"}
    if a.medium == "hybrid":
        env_kw["stiffness_map"] = a.stiffness_map
    env = make_env(ARM_TASK, **env_kw)
    if a.atacom:                                          # ATACOM: policy resides on the manifold
        from genedynamics.solvers.single.atacom.wrapper import AtacomEnvWrapper
        env = AtacomEnvWrapper(env)

    print(f"training {a.algo}{' (ATACOM-manifold)' if a.atacom else ''} on medium={a.medium} "
          f"level={env_kw['level']} action_size={env.action_size} ({a.num_timesteps} steps)...",
          flush=True)
    params, config = train_rl_policy(
        env, algo=a.algo, num_timesteps=a.num_timesteps,
        episode_length=a.episode_length, num_envs=a.num_envs,
        warmup_steps=a.warmup_steps, seed=a.seed)

    _name = f"atacom_{a.algo}" if a.atacom else a.algo
    out = a.out or f"results/arm/impedence/{a.medium}/_policies/{_name}.pkl"
    save_policy(out, params, config)
    print("saved policy ->", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
