"""Train the frozen H1 box-push policies used by the paper matrix.

This is a task-owned CLI wrapper around :mod:`genedynamics.learning`; policy
implementation and serialization remain in the shared learning package.
Formal evaluation seeds 10--19 are never used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from genedynamics.envs.factories import make_env
from genedynamics.envs.domains.humanoid.box_push_brax import (
    HumanoidBoxPushDomainEnv,
)
from genedynamics.learning.train_rl_policy import (
    save_policy,
    scalar_metrics,
    train_rl_policy,
)


TASK = "humanoid_box_push"
SCHEMAS = ("fixed", "walk", "atacom_p12", "atacom_p3", "atacom_p4")


def _domain_specs(schema: str):
    common = {
        "hand_contact_solref": (0.05, 1.0),
        "wall_contact_solref": (0.10, 1.0),
        "stance_force_ankle_gain": -1.5,
        "stance_com_ankle_gain": 250.0,
        "stance_com_ankle_damping": 20.0,
        "stance_force_hip_gain": 0.007,
        "stance_force_hip_deadband": 15.0,
    }
    p1 = {
        **common, "level": "push_to_line", "f_target": 15.0,
        "fixed_force_target": True, "fixed_contact_target": True,
        "box_frictionloss": 10000.0, "push_dist": 0.5,
        "w_box": 0.0, "w_prog": 0.0, "w_force": 10.0,
    }
    p2 = {**common, "level": "push_to_line", "push_dist": 0.10, "goal_eps": 0.005}
    p2_ood = {
        **common, "level": "heavy_dr", "dr_seed": 101,
        "f_target": 45.0, "push_dist": 0.10, "goal_eps": 0.005,
    }
    p3 = {
        **common, "level": "unjam", "f_target": 45.0,
        "push_dist": 0.03, "goal_eps": 0.005,
        "unjam_yaw_eps": 0.03, "yaw_frictionloss": 2.0,
        "contact_band": 0.02,
    }
    p4 = {
        **common, "level": "push_walk", "push_dist": 0.30,
        "s_scale": 0.25, "w_stiffness_nominal": 2.0,
        "box_frictionloss": 8.0, "w_gait": 30.0, "w_vel": 1.0,
        "target_vx": 0.25, "approach_time": 0.30,
        "force_ramp_time": 0.30, "gait_ramp_time": 0.5,
    }
    if schema == "fixed":
        return [p1, {**p1, "f_target": 30.0}, p2, p2_ood, p3]
    if schema == "walk" or schema == "atacom_p4":
        return [p4]
    if schema == "atacom_p12":
        return [p1, {**p1, "f_target": 30.0}, p2, p2_ood]
    if schema == "atacom_p3":
        return [p3]
    raise ValueError(schema)


def _default_output(schema: str, seed: int) -> Path:
    return Path("results/humanoid/box_push/_policies") / f"{schema}_ppo_seed{seed}.pkl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True, choices=SCHEMAS)
    parser.add_argument("--num-timesteps", type=int, default=200_000)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu-low-memory", action="store_true")
    args = parser.parse_args()
    if args.seed in range(10, 20):
        raise ValueError("formal evaluation seeds 10--19 are forbidden in training")

    specs = _domain_specs(args.schema)
    atacom = args.schema.startswith("atacom_")
    num_timesteps = 4096 if args.smoke else int(args.num_timesteps)
    num_envs = 2 if args.smoke else int(args.num_envs)
    kwargs = {
        "batch_size": 8,
        "num_minibatches": 2,
        "unroll_length": 10,
        "num_updates_per_batch": 1,
        # Contact impulses make the raw H1 reward substantially larger than
        # the arm-task rewards.  Scaling leaves the optimum unchanged while
        # keeping PPO's critic and KL in a numerically useful range.
        # ATACOM exploration can momentarily hit very large contact penalties;
        # a smaller positive scale preserves ordering/optima and prevents the
        # critic from dominating an otherwise finite tangent-policy update.
        "reward_scaling": 1e-4 if atacom else 0.01,
        "max_grad_norm": 1.0,
        "policy_hidden_layer_sizes": (32, 32, 32),
        "learning_rate": 1e-4,
        "run_evals": False,
    }
    if args.cpu_low_memory:
        kwargs.update(
            batch_size=max(1, num_envs), num_minibatches=1, unroll_length=1,
            policy_hidden_layer_sizes=(16, 16),
        )
    history = []
    started = time.monotonic()

    current_stage = 0

    def progress(step, metrics):
        row = {
            "stage": int(current_stage),
            "num_steps": int(step),
            "elapsed_seconds": float(time.monotonic() - started),
            **scalar_metrics(metrics),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    kwargs["progress_fn"] = progress
    # P3's nearby wall changes MJX's static contact-candidate pytree.  It
    # cannot be mixed with open P1/P2 through lax.switch, even though policy
    # observation/action shapes match.  Train one shared fixed policy through
    # an explicit two-stage curriculum and restore the complete PPO state.
    groups = [specs[:-1], specs[-1:]] if args.schema == "fixed" else [specs]
    stage_budgets = (
        [num_timesteps // 2, num_timesteps - num_timesteps // 2]
        if len(groups) == 2 else [num_timesteps]
    )
    output = Path(args.out) if args.out else _default_output(args.schema, args.seed)
    training_root = output.parent / "_training" / output.stem
    restore = None
    params = config = env = None
    for stage, (stage_specs, stage_budget) in enumerate(zip(groups, stage_budgets)):
        current_stage = stage
        domains = [make_env(TASK, **spec) for spec in stage_specs]
        if atacom:
            from genedynamics.solvers.single.atacom.wrapper import AtacomEnvWrapper
            domains = [AtacomEnvWrapper(domain) for domain in domains]
        env = HumanoidBoxPushDomainEnv(domains)
        stage_kwargs = dict(kwargs)
        checkpoint_root = training_root / f"stage_{stage}"
        stage_kwargs["save_checkpoint_path"] = str(checkpoint_root.resolve())
        if restore is not None:
            stage_kwargs["restore_checkpoint_path"] = str(restore.resolve())
        params, config = train_rl_policy(
            env, algo="ppo", num_timesteps=int(stage_budget),
            episode_length=int(args.episode_length), num_envs=num_envs,
            seed=int(args.seed), **stage_kwargs,
        )
        checkpoints = sorted(
            path for path in checkpoint_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
        if stage + 1 < len(groups):
            if not checkpoints:
                raise RuntimeError(
                    f"PPO stage {stage} produced no restorable checkpoint"
                )
            restore = checkpoints[-1]
    protocol = f"humanoid_box_push_{args.schema}_ppo_v1"
    config.update({
        "protocol": protocol,
        "task": TASK,
        "schema": args.schema,
        "training_seed": int(args.seed),
        "training_domains": specs,
        "training_budget": {"unit": "environment_steps", "value": num_timesteps},
        "num_timesteps": num_timesteps,
        "selection_rule": "final_checkpoint_after_fixed_budget",
        "progress_history": history,
        "training_wall_seconds": float(time.monotonic() - started),
        "training_complete": True,
    })
    save_policy(str(output), params, config)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({"checkpoint": str(output), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
