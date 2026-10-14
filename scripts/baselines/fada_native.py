"""Run the real released G1/T1 FADA oracle/student/adaptation training chain.

The mjwarp route retains official robot, oracle and randomization, but is recorded
as a simulator adaptation from the released IsaacSim source-training preset.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "290b3a9c5c0ebe2e9851b61350617544be05e9bc"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            result.update(chunk)
    return result.hexdigest()


def source_setup(source):
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError("FADA native entrypoint requires the locked official source commit")
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "--", "src/holosoma"], check=True)
    sys.path.insert(0, str(source / "src/holosoma"))


def prepare_oracle(args):
    import torch
    from holosoma.config_values import simulator
    from holosoma.config_types.experiment import ExperimentConfig

    original = torch.load(args.oracle_checkpoint, map_location="cpu", weights_only=True)
    cfg = ExperimentConfig(**original["experiment_config"])
    if args.engine != "keep":
        cfg = dataclasses.replace(cfg, simulator=getattr(simulator, args.engine))
    cfg = dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, num_envs=args.num_envs, headless=True, seed=args.seed))
    variant = f"{args.engine}_envs{args.num_envs}_seed{args.seed}"
    derived = args.runtime / "derived_oracle" / variant / args.oracle_checkpoint.parent.name
    derived.mkdir(parents=True, exist_ok=True)
    # Preserve the complete author history required by reward-aware weak-policy data.
    for checkpoint in args.oracle_checkpoint.parent.glob("model_*.pt"):
        destination = derived / checkpoint.name
        if checkpoint.name != args.oracle_checkpoint.name and not destination.exists():
            os.link(checkpoint, destination)
    payload = dict(original)
    payload["experiment_config"] = cfg.to_serializable_dict()
    destination = derived / args.oracle_checkpoint.name
    torch.save(payload, destination)
    receipt = {"original": str(args.oracle_checkpoint), "original_sha256": digest(args.oracle_checkpoint),
               "derived": str(destination), "derived_sha256": digest(destination),
               "changes": {"simulator": args.engine, "num_envs": args.num_envs, "headless": True, "seed": args.seed},
               "actor_critic_optimizer_normalizer_weights": "unchanged from original author checkpoint",
               "ignore_unsupported_randomization": cfg.randomization.ignore_unsupported}
    if cfg.randomization.ignore_unsupported:
        raise RuntimeError("Native qualification must not silently ignore unsupported oracle randomizers")
    derivations = args.output / "oracle_derivations"
    derivations.mkdir(exist_ok=True)
    (derivations / f"{variant}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (args.output / "oracle_derivation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return destination


def build_oracle(args, checkpoint):
    import torch
    from holosoma.fada.planner_idm.config import FADAConfig
    from holosoma.fada.planner_idm.trainer import build_env_and_expert

    cfg = FADAConfig(expert_checkpoint=str(checkpoint), num_envs=args.num_envs, headless=True,
                     device=args.device, seed=args.seed, simulator_override="keep", wandb_mode="disabled")
    env, app, expert, device, experiment = build_env_and_expert(cfg, runtime_log_dir=args.output / "oracle_logs")
    return env, app, expert, torch.device(device), experiment


def physics_fingerprint(env, engine):
    backend = env.simulator.backend
    data = getattr(backend, "mjw_data", None)
    device = str(data.qpos.device) if data is not None else "cpu"
    if engine == "mjwarp" and (type(backend).__name__ != "WarpBackend" or not device.startswith("cuda")):
        raise RuntimeError("Native mjwarp requested but GPU physics backend was not established")
    return {"backend_class": type(backend).__name__, "physics_device": device,
            "control_dt": float(env.dt), "ignore_unsupported_randomization": False}


def oracle_run(args, checkpoint):
    import torch
    from genedynamics.learning.baselines.fada.checkpoint import capture_rng_state, restore_rng_state, to_cpu

    env, app, expert, device, experiment = build_oracle(args, checkpoint)
    obs = env.reset_all()
    resets = 0
    rewards = []
    heights = []
    start = time.monotonic()
    for _ in range(args.steps):
        action = expert.act(obs)
        if not bool(torch.isfinite(action).all()):
            raise RuntimeError("Nonfinite author oracle action")
        obs, reward, done, info = env.step({"actions": action})
        rewards.append(float(reward.mean().item()))
        resets += int(done.sum().item())
        heights.append(float(env.simulator.robot_root_states[:, 2].mean().item()))
    metrics = {"control_steps": args.steps, "num_envs": args.num_envs,
               "simulated_seconds": args.steps * float(env.dt), "wall_seconds": time.monotonic() - start,
               "episode_resets": resets, "mean_reward": sum(rewards) / len(rewards),
               "minimum_mean_base_height": min(heights), "checkpoint_sha256": digest(checkpoint),
               "qualification": "real author oracle closed-loop; simulator adaptation, not paper score",
               "physics": physics_fingerprint(env, args.engine), "reset_seed": args.seed}
    (args.output / "oracle_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    if args.oracle_updates:
        algo = expert.algo
        algo._train_mode()
        before = {name: value.detach().clone() for name, value in algo.actor.named_parameters()}
        observations = env.reset_all()
        losses = []
        for _ in range(args.oracle_updates):
            observations = algo._rollout_step(observations)
            result = algo._training_step()
            losses.append({key: float(value) for key, value in result.items()})
            algo.current_learning_iteration += 1
        changed = any(not torch.equal(value, before[name]) for name, value in algo.actor.named_parameters())
        if not changed:
            raise RuntimeError("Oracle qualification update did not change actor weights")
        resume = Path(algo.log_dir) / "model_native_resume.pt"
        algo.save(str(resume))
        state = capture_rng_state()
        torch.save(state, args.output / "oracle_rng.pt")
        # Load the real learner checkpoint through the source loader; verify optimizer
        # and normalization state survive, then restore the independent RNG snapshot.
        original_optimizer = to_cpu(algo.actor_optimizer.state_dict())
        normalizer = getattr(algo, "actor_obs_normalizer", None)
        original_normalizer = to_cpu(normalizer.state_dict()) if normalizer is not None else None
        original_actor = to_cpu(algo.actor.state_dict())
        with torch.no_grad():
            next(algo.actor.parameters()).add_(0.1)
        algo.load(str(resume))
        def assert_equal(left, right):
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
            elif isinstance(left, dict):
                assert left.keys() == right.keys()
                for key in left: assert_equal(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                assert len(left) == len(right)
                for a, b in zip(left, right): assert_equal(a, b)
            else:
                assert left == right
        assert_equal(original_optimizer, algo.actor_optimizer.state_dict())
        assert_equal(original_actor, algo.actor.state_dict())
        if original_normalizer is not None:
            assert_equal(original_normalizer, normalizer.state_dict())
        restore_rng_state(torch.load(args.output / "oracle_rng.pt", weights_only=True))
        metrics.update(oracle_updates=args.oracle_updates, actor_weights_changed=changed, update_losses=losses,
                       resume_checkpoint=str(resume), resume_sha256=digest(resume),
                       optimizer_normalizer_restore=True, rng_restore=True)
    (args.output / "oracle_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


def dagger_run(args, checkpoint):
    import psutil
    command = [sys.executable, "-m", "scripts.baselines.fada_native_train", "--expert-checkpoint", str(checkpoint),
               "--output-dir", str(args.output / "dagger"), "--run-name", args.dagger_run_name,
               "--device", args.device, "--wandb-mode", "disabled", "--num-envs", str(args.num_envs),
               "--seed", str(args.seed), "--simulator-override", "keep"]
    budget_arguments = (
        ("dagger_batch_size", "--batch-size", 32), ("replay_capacity", "--replay-capacity", 8192),
        ("warmup_episodes", "--warmup-episodes", 3),
        ("warmup_horizon", "--warmup-max-steps-per-episode", 128),
        ("offline_steps", "--warmup-collect-steps", 384),
        ("warmup_updates", "--warmup-train-steps", 2), ("dagger_iterations", "--dagger-iters", 1),
        ("dagger_rollout_steps", "--rollout-steps-per-iter", 64),
        ("dagger_updates", "--train-steps-per-iter", 2),
    )
    for name, flag, small_default in budget_arguments:
        value = getattr(args, name)
        if value is None and args.profile == "qualification":
            value = small_default
        if value is not None:
            command.extend([flag, str(value)])
    if args.profile == "qualification":
        command.extend(["--eval-max-steps", "128", "--show-progress", "false"])
    # Native weak-policy source, oracle-shadow labels, full model and both learner
    # passes stay enabled. Qualification only reduces explicit runtime budgets.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(args.source / "src/holosoma") + os.pathsep + str(ROOT)
    environment["FADA_NATIVE_SOURCE"] = str(args.source)
    command_receipt = {"argv": command, "profile": args.profile,
        "qualification": "training-chain verification; bounded updates are not convergence",
        "compatibility": ["MjwDofStateView.detach materializes [:] for official DAgger snapshot"]}
    (args.output / f"{args.dagger_run_name}.command.json").write_text(json.dumps(command_receipt, indent=2) + "\n")
    profile = dict(peak_owned_gpu_mib=0, peak_owned_rss_bytes=0, max_seconds=args.max_seconds,
                   max_gpu_mib=args.max_gpu_mib, sampled_seconds=0)
    started = time.monotonic()
    log_path = args.output / f"{args.dagger_run_name}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                try:
                    processes = [psutil.Process(process.pid)]
                    processes.extend(processes[0].children(recursive=True))
                except psutil.NoSuchProcess:
                    if process.poll() is not None:
                        break
                    continue
                pids = {item.pid for item in processes}
                rss = 0
                for item in processes:
                    try:
                        rss += item.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                memory = subprocess.check_output([
                    "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
                owned_gpu = 0
                for row in memory.splitlines():
                    pieces = [piece.strip() for piece in row.split(",")]
                    if len(pieces) == 2 and pieces[0].isdigit() and int(pieces[0]) in pids and pieces[1].isdigit():
                        owned_gpu += int(pieces[1])
                profile["peak_owned_gpu_mib"] = max(profile["peak_owned_gpu_mib"], owned_gpu)
                profile["peak_owned_rss_bytes"] = max(profile["peak_owned_rss_bytes"], rss)
                profile["sampled_seconds"] = time.monotonic() - started
                (args.output / f"{args.dagger_run_name}.resources.json").write_text(json.dumps(profile, indent=2) + "\n")
                if profile["sampled_seconds"] > args.max_seconds or owned_gpu > args.max_gpu_mib:
                    raise RuntimeError("Native FADA bounded time/GPU resource budget exceeded")
                time.sleep(1)
        finally:
            if process.poll() is None:
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in reversed(children):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            profile.update(returncode=process.returncode, wall_seconds=time.monotonic() - started,
                           log=str(log_path), source_commit=SOURCE_COMMIT,
                           rss_metric="sum of owned process RSS; shared pages may be counted more than once")
            (args.output / f"{args.dagger_run_name}.resources.json").write_text(json.dumps(profile, indent=2) + "\n")
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


def student_run(args, oracle_checkpoint, *, collect):
    import h5py
    import numpy as np
    import torch
    from holosoma.fada.common.compact_obs import extract_compact_obs
    from holosoma.fada.common.current_command import extract_current_command_torch
    from holosoma.fada.planner_idm.config import FADAConfig
    from holosoma.fada.planner_idm.eval import _clip_actions
    from holosoma.fada.planner_idm.trainer import _resolve_eval_config
    from holosoma.utils.sim_utils import setup_simulation_environment
    from holosoma.fada.planner_idm.trainer import set_seed
    from genedynamics.solvers.single.fada import FADASolver
    from genedynamics.learning.baselines.fada.native_diagnostics import (
        FixedTargetDynamics, SourceExecutionParity, reset_public_histories)

    payload = torch.load(args.student_checkpoint, map_location="cpu", weights_only=True)
    policy = FADASolver.from_checkpoint(payload, device=args.device)
    cfg = FADAConfig(expert_checkpoint=str(oracle_checkpoint), num_envs=args.num_envs,
                     headless=True, device=args.device, seed=args.seed, simulator_override="keep")
    experiment, _, _ = _resolve_eval_config(cfg)
    experiment = dataclasses.replace(experiment, simulator=dataclasses.replace(
        experiment.simulator, config=dataclasses.replace(experiment.simulator.config, link_mass_scale=args.target_mass_scale)))
    startup_seed = args.seed if args.startup_seed is None else args.startup_seed
    experiment = dataclasses.replace(experiment, training=dataclasses.replace(experiment.training, seed=startup_seed))
    set_seed(startup_seed)
    env, device, app = setup_simulation_environment(experiment, device=args.device)
    env.reset_all()
    fixed_dynamics = FixedTargetDynamics(env) if args.fixed_target_dynamics else None
    set_seed(args.seed)
    env.reset_all()
    parity = SourceExecutionParity(payload, device) if args.audit_source_equivalence else None
    dimensions = (args.num_envs, policy.model.history_len)
    history_obs = torch.zeros(*dimensions, policy.model.obs_dim, device=device)
    history_act = torch.zeros(*dimensions, policy.model.act_dim, device=device)
    valid = torch.zeros(*dimensions, device=device, dtype=torch.bool)
    previous_action = torch.zeros(args.num_envs, policy.model.act_dim, device=device)
    n_steps = math.ceil(args.target_seconds / (float(env.dt) * args.num_envs)) if collect else args.steps
    # Generate exogenous velocity commands independently of policy-driven resets.
    # Gait phase remains measured controller context, not a task command draw.
    command_term = env.command_manager.get_state("locomotion_command")
    schedule_generator = torch.Generator().manual_seed(args.seed + 1701)
    blocks = math.ceil(n_steps / 200)
    schedule = torch.rand(blocks, args.num_envs, 3, generator=schedule_generator)
    for axis, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
        lower, upper = command_term.command_ranges[name]
        schedule[:, :, axis] = lower + schedule[:, :, axis] * (upper - lower)
    standing = torch.rand(blocks, args.num_envs, generator=schedule_generator) < command_term.stand_prob
    schedule[standing] = 0
    schedule = schedule.repeat_interleave(200, dim=0)[:n_steps]
    schedule_file = args.output / f"velocity_command_schedule_{n_steps}_envs{args.num_envs}_seed{args.seed}.npy"
    if schedule_file.exists():
        if not np.array_equal(np.load(schedule_file), schedule.numpy()):
            raise RuntimeError("Existing matched command schedule differs")
    else:
        np.save(schedule_file, schedule.numpy())
    env.command_manager.command_cfg.locomotion_command_resampling_time = 0.0
    schedule = schedule.to(device)
    buffers = [dict(dynamics_obs=[], actions=[], current_command=[], dones=[]) for _ in range(args.num_envs)]
    dataset = args.output / f"target_{args.run_label}.h5"
    file = h5py.File(dataset, "w") if collect else None
    episodes = file.create_group("episodes") if file is not None else None
    episode_count = 0

    def finish_episode(index):
        nonlocal episode_count
        buffer = buffers[index]
        if not buffer["actions"]:
            return
        if episodes is not None:
            group = episodes.create_group(f"episode_{episode_count:06d}")
            group.attrs["collection_session"] = 0
            group.attrs["source_env"] = index
            for key, values in buffer.items():
                group.create_dataset(key, data=np.asarray(values), compression="gzip")
        episode_count += 1
        buffers[index] = {key: [] for key in buffer}

    if file is not None:
        file.attrs.update(collection_status="collecting", planned_steps=n_steps * args.num_envs,
                          planned_steps_source="aggregate target simulation action seconds",
                          collected_steps=0, has_rewards=False, public_observations_only=True,
                          student_sha256=digest(args.student_checkpoint))
    reward_sum = 0.0
    resets = 0
    failure_resets = 0
    timeout_resets = 0
    first_failure_step = torch.full((args.num_envs,), n_steps, dtype=torch.long, device=device)
    start = time.monotonic()
    try:
        for step in range(n_steps):
            env.command_manager.commands[:, :3] = schedule[step]
            observation = extract_compact_obs(env, term_scale=payload["compact_obs"]["term_scale"])
            command = extract_current_command_torch(env, profile=payload["cfg"]["command_profile"])
            history_obs = torch.roll(history_obs, -1, 1)
            history_act = torch.roll(history_act, -1, 1)
            valid = torch.roll(valid, -1, 1)
            history_obs[:, -1] = observation
            history_act[:, -1] = previous_action
            valid[:, -1] = True
            if parity is not None and step % 25 == 0:
                parity.check(env, policy, history_obs, history_act, command, valid)
            action = _clip_actions(env, policy.act(dict(history_obs=history_obs, history_act=history_act,
                                      current_command=command, history_valid_mask=valid)))
            if not bool(torch.isfinite(action).all()):
                raise RuntimeError("Nonfinite native student action")
            _, reward, done, info = env.step({"actions": action})
            # The released native environment returns reset_buf as torch.long.
            # History reset uses a boolean mask, never integer row indices.
            done = done.to(dtype=torch.bool)
            reward_sum += float(reward.sum().item())
            resets += int(done.sum().item())
            timeouts = info.get("time_outs", torch.zeros_like(done)).to(dtype=torch.bool)
            timeout_resets += int((done & timeouts).sum().item())
            failure_resets += int((done & ~timeouts).sum().item())
            first_failure_step = torch.where((done & ~timeouts) & (first_failure_step == n_steps),
                torch.full_like(first_failure_step, step + 1), first_failure_step)
            if collect:
                public = {"dynamics_obs": observation.cpu().numpy(), "actions": action.cpu().numpy(),
                          "current_command": command.cpu().numpy(), "dones": done.cpu().numpy()}
                for index in range(args.num_envs):
                    for key in buffers[index]:
                        buffers[index][key].append(public[key][index].copy())
                    if bool(done[index]):
                        finish_episode(index)
                file.attrs["collected_steps"] = (step + 1) * args.num_envs
            previous_action = action.clone()
            if bool(done.any()):
                reset_public_histories(done, history_obs, history_act, valid, previous_action)
                policy.reset()
        if file is not None:
            for index in range(args.num_envs):
                finish_episode(index)
            file.attrs["collection_status"] = "complete"
            file.attrs["collection_sessions"] = json.dumps([dict(
                session_index=0, collection_status="complete", planned_steps=n_steps * args.num_envs,
                collected_steps=n_steps * args.num_envs, planned_steps_source="aggregate target simulation action seconds")])
    finally:
        if file is not None:
            file.close()
    metrics = {"run_label": args.run_label, "student_checkpoint": str(args.student_checkpoint),
               "student_sha256": digest(args.student_checkpoint), "control_steps_per_env": n_steps,
               "num_envs": args.num_envs, "aggregate_action_seconds": n_steps * args.num_envs * float(env.dt),
               "wall_seconds": time.monotonic() - start, "episode_resets": resets,
               "failure_resets": failure_resets, "timeout_resets": timeout_resets,
               "first_failure_seconds": (first_failure_step * float(env.dt)).tolist(),
               "history_reset_mask_dtype": "torch.bool",
               "mean_reward": reward_sum / (n_steps * args.num_envs), "episodes": episode_count,
               "matched_velocity_schedule_sha256": digest(schedule_file),
               "reset_seed": args.seed, "startup_draw_seed": startup_seed,
               "physics": physics_fingerprint(env, args.engine),
               "gait_phase_context": "measured endogenous controller phase",
               "target_mass_scale_evaluator_only": args.target_mass_scale,
               "fit_information": "public scaled dynamics observations, commands, executed actions and done boundaries; no rewards",
               "qualification": "real native student closed-loop on simulator-adapted source robot; not paper reproduction"}
    if fixed_dynamics is not None:
        metrics["fixed_target_dynamics"] = fixed_dynamics.report()
    if parity is not None:
        metrics["source_execution_parity"] = parity.report()
    if collect:
        metrics.update(dataset=str(dataset), dataset_sha256=digest(dataset))
    (args.output / f"student_{args.run_label}_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


def adaptation_run(args):
    import numpy as np
    import torch
    from torch.utils.data._utils.collate import default_collate
    from holosoma.fada.common.lora_utils import load_trajectories_from_h5
    from genedynamics.learning.baselines.fada import FADAIDMAdapter, TrajectoryWindowDataset

    payload = torch.load(args.student_checkpoint, map_location="cpu", weights_only=True)
    trajectories, data_provenance = load_trajectories_from_h5(
        [args.target_dataset], obs_key="dynamics_obs", command_key="current_command",
        obs_dim=payload["obs_dim"], act_dim=payload["act_dim"], cmd_dim=payload["cmd_dim"],
        raw_obs_preprocess=None, pred_horizon_k=payload["cfg"]["pred_horizon"], require_command_key=True,
    )
    dataset = TrajectoryWindowDataset(trajectories, history_len=payload["cfg"]["history_len"],
                                      pred_horizon_k=payload["cfg"]["pred_horizon"])
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(dataset), generator=generator)
    cursor = 0

    def next_batch():
        nonlocal order, cursor
        if cursor >= len(order):
            order = torch.randperm(len(dataset), generator=generator)
            cursor = 0
        indices = order[cursor:cursor + args.batch_size].tolist()
        cursor += len(indices)
        return default_collate([dataset[index] for index in indices])

    adapter = FADAIDMAdapter(payload, device=args.device)
    losses = []
    batch_sizes = []
    started = time.monotonic()
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    for _ in range(args.adaptation_steps):
        batch = next_batch()
        batch_sizes.append(int(batch["history_obs"].shape[0]))
        losses.append(adapter.step(batch)["loss"])
        if torch.device(args.device).type == "cuda" and torch.cuda.max_memory_reserved(args.device) > args.max_gpu_mib * 1024**2:
            raise RuntimeError("Native IDM adaptation exceeded reserved CUDA memory budget")
        if time.monotonic() - started > args.max_seconds:
            raise RuntimeError("Native IDM adaptation exceeded wall-time budget")
    resume = dict(adapter=adapter.training_state_dict(), sampler_generator=generator.get_state(),
                  sampler_order=order, sampler_cursor=cursor, dataset_sha256=digest(args.target_dataset))
    resume_path = args.output / "idm_training_resume.pt"
    torch.save(resume, resume_path)
    next_record_batch = next_batch()
    expected_loss = adapter.step(next_record_batch)
    expected_weights = {name: parameter.detach().clone() for name, parameter in adapter.model.named_parameters()}
    restored = torch.load(resume_path, weights_only=True)
    resumed = FADAIDMAdapter.from_training_state(restored["adapter"], device=args.device)
    actual_loss = resumed.step(next_record_batch)
    if expected_loss != actual_loss:
        raise RuntimeError("Native target adaptation next-update loss did not resume exactly")
    for name, parameter in resumed.model.named_parameters():
        torch.testing.assert_close(parameter, expected_weights[name], rtol=0, atol=0)
    # Select the budgeted snapshot; two extra verification updates are reported,
    # and neither selects the evaluated adapted weights.
    selected = FADAIDMAdapter.from_training_state(restored["adapter"], device=args.device)
    checkpoint = args.output / "student_idm_adapted.pt"
    torch.save(selected.export_checkpoint(), checkpoint)
    report = {"target_dataset": str(args.target_dataset), "dataset_sha256": digest(args.target_dataset),
              "loaded_trajectories": len(trajectories), "valid_windows": len(dataset),
              "adaptation_steps": args.adaptation_steps, "verification_updates": 2,
              "batch_size": args.batch_size, "first_loss": losses[0], "last_loss": losses[-1],
              "actual_batch_sizes": sorted(set(batch_sizes)), "drop_last": False,
              "wall_seconds": time.monotonic() - started,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if torch.device(args.device).type == "cuda" else 0,
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(args.device) if torch.device(args.device).type == "cuda" else 0,
              "optimizer_rng_normalization_next_update_exact": True,
              "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
              "resume_checkpoint_sha256": digest(resume_path), "data_provenance": data_provenance,
              "qualification": "real target-record IDM adaptation; no paper-performance claim"}
    (args.output / "adaptation_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "data_provenance"}, indent=2), flush=True)


def dagger_resume_run(args, oracle_checkpoint):
    import torch
    from holosoma.fada.common.dataset import ReplayBuffer
    from holosoma.fada.planner_idm.config import FADAConfig
    from holosoma.fada.planner_idm.model import build_student_policy
    from holosoma.fada.planner_idm.trainer import FADATrainer
    from genedynamics.learning.baselines.fada.checkpoint import capture_rng_state, restore_rng_state

    payload = torch.load(args.student_checkpoint, map_location="cpu", weights_only=True)
    cfg = FADAConfig(**payload["cfg"])
    env, app, expert, device, experiment = build_oracle(args, oracle_checkpoint)
    definitions = dict(capacity=cfg.replay_capacity, obs_dim=payload["obs_dim"], act_dim=payload["act_dim"],
                       cmd_dim=payload["cmd_dim"], history_len=cfg.history_len, pred_horizon=cfg.pred_horizon,
                       require_future_obs_targets=True)
    offline, weak = (ReplayBuffer(**definitions, growable=True) for _ in range(2))
    online = ReplayBuffer(**definitions, eviction_policy=cfg.online_buffer_eviction_policy)
    offline.load_npz(cfg.offline_cache_path)
    weak.load_npz(cfg.offline_suboptimal_cache_path)
    # Native warmup partitions complete cached trajectories before training.
    # Reconstruct those same train subsets, rather than resuming on held-out data.
    if cfg.val_trajectory_ratio > 0:
        offline, _ = offline.split_by_trajectory(val_ratio=cfg.val_trajectory_ratio, seed=cfg.seed)
        weak, _ = weak.split_by_trajectory(val_ratio=cfg.val_trajectory_ratio, seed=cfg.seed + 1)
    online_path = args.student_checkpoint.parent / "checkpoints/online_buffer_latest.npz"
    online.load_npz(online_path)
    offline.freeze()
    weak.freeze()

    def make_trainer():
        model = build_student_policy(cfg, obs_dim=payload["obs_dim"], act_dim=payload["act_dim"],
                                     cmd_dim=payload["cmd_dim"]).to(device)
        trainer = FADATrainer(
            cfg=cfg, student_model=model, expert_policy=expert, offline_buffer=offline,
            suboptimal_offline_buffer=weak, online_buffer=online,
            optimizer=torch.optim.AdamW(model.planner_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay),
            idm_optimizer=torch.optim.AdamW(model.idm_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay),
            lr_scheduler=None, idm_lr_scheduler=None, device=device, obs_dim=payload["obs_dim"],
            act_dim=payload["act_dim"], cmd_dim=payload["cmd_dim"],
            resolved_expert_checkpoint=str(oracle_checkpoint), compact_obs_source_checkpoint=str(oracle_checkpoint),
            experiment_config_payload=payload["experiment_config"], strict_label_env=env,
        )
        trainer.load_student_checkpoint(args.student_checkpoint, load_optimizer=True)
        return trainer

    uninterrupted = make_trainer()
    rng = capture_rng_state()
    rng_file = args.output / "dagger_resume_rng.pt"
    torch.save(rng, rng_file)
    expected = uninterrupted.train_step(mode="mixed")
    expected_weights = {name: value.detach().clone() for name, value in uninterrupted.student_model.named_parameters()}
    resumed = make_trainer()
    restore_rng_state(torch.load(rng_file, weights_only=True))
    actual = resumed.train_step(mode="mixed")
    if expected != actual:
        raise RuntimeError("Real native DAgger loss/sample/metric update did not resume exactly")
    for name, value in resumed.student_model.named_parameters():
        torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)
    checkpoint = args.output / "dagger_resume_verified.pt"
    resumed.save_student_checkpoint(checkpoint, extra={"stage": "qualification_next_update"})
    report = {"student_checkpoint": str(args.student_checkpoint), "student_sha256": digest(args.student_checkpoint),
              "starting_train_step": payload["train_step_count"], "resulting_train_step": resumed.train_step_count,
              "optimal_records": len(offline), "weak_records": len(weak), "online_records": len(online),
              "planner_idm_optimizer_rng_normalization_next_update_exact": True,
              "loss": actual[0], "sample_counts": actual[1], "metrics": actual[2],
              "checkpoint_sha256": digest(checkpoint),
              "offline_replay_reconstruction": "same source trajectory split, seed and train-only subsets",
              "scope": "Exact next learner update on saved real replay; not bitwise interrupted physics collection"}
    (args.output / "dagger_resume_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "oracle", "dagger", "dagger-resume", "student", "collect", "adapt"))
    parser.add_argument("--source", type=Path, default=ROOT / "baselines/FADA")
    parser.add_argument("--runtime", type=Path, default=ROOT / ".runtime/fada_native")
    parser.add_argument("--oracle-checkpoint", type=Path, default=ROOT / ".runtime/fada_native/assets/g1/model_24999.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=("keep", "mjwarp", "isaacsim"), default="mjwarp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--oracle-updates", type=int, default=1)
    parser.add_argument("--profile", choices=("qualification", "release"), default="qualification")
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--target-dataset", type=Path)
    parser.add_argument("--target-seconds", type=float, default=120.0)
    parser.add_argument("--target-mass-scale", type=float, default=1.15)
    parser.add_argument("--run-label", default="pre_sft")
    parser.add_argument("--startup-seed", type=int)
    parser.add_argument("--fixed-target-dynamics", action="store_true")
    parser.add_argument("--audit-source-equivalence", action="store_true")
    parser.add_argument("--adaptation-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dagger-run-name", default="g1_native_qualification")
    parser.add_argument("--warmup-updates", type=int)
    parser.add_argument("--dagger-updates", type=int)
    parser.add_argument("--dagger-iterations", type=int)
    parser.add_argument("--dagger-rollout-steps", type=int)
    parser.add_argument("--offline-steps", type=int)
    parser.add_argument("--dagger-batch-size", type=int)
    parser.add_argument("--warmup-episodes", type=int)
    parser.add_argument("--warmup-horizon", type=int)
    parser.add_argument("--replay-capacity", type=int)
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--max-gpu-mib", type=int, default=12288)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    for name in ("source", "runtime", "oracle_checkpoint", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if "latex" in args.output.parts or min(args.num_envs, args.steps) <= 0 or args.oracle_updates < 0:
        parser.error("Invalid output or native execution budget")
    if args.action in {"student", "collect", "adapt", "dagger-resume"} and args.student_checkpoint is None:
        parser.error("Native student stages require --student-checkpoint")
    if args.action == "adapt" and args.target_dataset is None:
        parser.error("Native adaptation requires --target-dataset")
    if min(args.target_seconds, args.target_mass_scale, args.adaptation_steps, args.batch_size) <= 0:
        parser.error("Native collection/adaptation budgets must be positive")
    positive_budgets = (args.dagger_rollout_steps, args.offline_steps, args.dagger_batch_size,
                       args.warmup_episodes, args.warmup_horizon, args.replay_capacity,
                       args.max_seconds, args.max_gpu_mib)
    if any(value is not None and value <= 0 for value in positive_budgets):
        parser.error("Native DAgger and resource budgets must be positive")
    if any(value is not None and value < 0 for value in (args.warmup_updates, args.dagger_updates, args.dagger_iterations)):
        parser.error("Native optimizer and iteration budgets must be nonnegative")
    for label in (args.run_label, args.dagger_run_name):
        if not label or Path(label).name != label or label in {".", ".."}:
            parser.error("Native run labels must be single path components")
    if args.fixed_target_dynamics and args.num_envs != 1:
        parser.error("Fixed target qualification requires --num-envs 1")
    args.output.mkdir(parents=True, exist_ok=True)
    args.runtime.mkdir(parents=True, exist_ok=True)
    from scripts.baselines.environment import runtime_environment
    os.environ.update(runtime_environment(args.runtime))
    os.environ["WARP_CACHE_PATH"] = str(args.runtime / "warp-cache")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    os.environ.update(WANDB_MODE="disabled", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    source_setup(args.source)
    import torch
    from holosoma.fada.planner_idm.trainer import set_seed
    set_seed(args.seed)
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested native CUDA execution is unavailable")
    if args.action == "adapt":
        adaptation_run(args)
        return
    checkpoint = prepare_oracle(args)
    if args.action == "oracle":
        oracle_run(args, checkpoint)
    elif args.action == "dagger":
        dagger_run(args, checkpoint)
    elif args.action == "dagger-resume":
        dagger_resume_run(args, checkpoint)
    elif args.action in {"student", "collect"}:
        student_run(args, checkpoint, collect=args.action == "collect")


if __name__ == "__main__":
    main()
