"""Source-only FADA task diagnostics and complete-episode DAgger tools.

The released Planner-IDM architecture and two-pass action objective are kept.
All physics here is the declared nominal Panda source, never a target query.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch

from genedynamics.experiments.baselines import instantiate_task
from genedynamics.solvers.single.fada import FADASolver
from .task_training import (TaskTrajectory, TaskWindowDataset, load_source_trajectory, project_trajectory,
                            task_action_bounds, portable_history, portable_observation, bounded_task_action)


class SourceAccountingLedger:
    """Irreversible source charges independent of algorithm/environment restore."""
    def __init__(self, path):
        self.path = Path(path)
        self.values = json.loads(self.path.read_text()) if self.path.exists() else {"target_calls": 0, "roles": {}}
        self.default_role = "unclassified_source"

    def save(self):
        from genedynamics.experiments.artifacts import atomic_json
        atomic_json(self.path, self.values)

    @contextmanager
    def role(self, name):
        previous = self.default_role; self.default_role = name
        try: yield
        finally: self.default_role = previous

    @contextmanager
    def installed(self):
        from genedynamics.experiments.baselines.contact_task import CpuTaskAdapter
        original = CpuTaskAdapter._charge
        def charge(task, key, value=1):
            if task.config.robot != "panda": raise RuntimeError("source-only accounting refuses a non-Panda plant")
            original(task, key, value)
            role = getattr(task, "source_accounting_role", self.default_role)
            row = self.values["roles"].setdefault(role, {})
            row[key] = row.get(key, 0) + int(value)
            if key in {"physics_steps", "reset_worlds", "preparation_control_steps"}: self.save()
        CpuTaskAdapter._charge = charge
        try: yield self
        finally:
            CpuTaskAdapter._charge = original; self.save()


def _observed_mode(task, record):
    if not record.traces: return -1
    positions = np.asarray([row["tcp_position"] for row in record.traces])[None]
    valid = np.asarray([row["valid"] for row in record.traces])[None]
    if task.spec.alias == "T1":
        from genedynamics.learning.cross_embodiment.data import observed_route_modes
        return int(observed_route_modes(positions[:, :, None], valid[:, :, None])[0])
    from genedynamics.learning.cross_embodiment.contact_reference import classify_contact_modes
    force = np.asarray([row["normal_force"] for row in record.traces])[None]
    return int(classify_contact_modes(positions, force, valid, insertion=True)[0])


def _copy_teacher(teacher, task):
    copied = type(teacher).__new__(type(teacher))
    copied.__dict__ = {key: deepcopy(value) for key, value in teacher.__dict__.items() if key != "task"}
    copied.task = task
    return copied


def diagnose_source_checkpoint(checkpoint_path, demonstrations, *, task_name="T3", sample_limit=256):
    """Read public source demonstrations only; no reset, step or target call."""
    from scripts.baselines.qualify_tasks import PublicTeacher

    task = instantiate_task(task_name, robot="panda")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    solver = FADASolver.from_checkpoint(payload, device="cpu")
    stats = solver.backend.stats
    groups = {"linear_m_per_s": slice(0, 3), "angular_rad_per_s": slice(3, 6),
              "log_stiffness": slice(6, 12), "force_N": slice(12, 13)}
    rows = []
    all_actions = []
    for mode, path in enumerate(demonstrations):
        trajectory = project_trajectory(load_source_trajectory(path, mode=mode), task)
        data = TaskWindowDataset([trajectory], history_len=payload["cfg"]["history_len"],
                                 pred_horizon_k=payload["cfg"]["pred_horizon"])
        indices = np.unique(np.linspace(0, len(data) - 1, min(sample_limit, len(data)), dtype=int))
        predictions, inverse, futures, targets, true_futures = [], [], [], [], []
        for start in range(0, len(indices), 64):
            samples = [data[int(index)] for index in indices[start:start + 64]]
            batch = {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}
            action, future = solver.predict(batch["history_obs"], batch["history_act"], batch["current_command"],
                history_valid_mask=batch["history_valid_mask"], return_obs=True)
            history = (batch["history_obs"] - stats["obs"]["mean"]) / stats["obs"]["std"]
            history_act = (batch["history_act"] - stats["act"]["mean"]) / stats["act"]["std"]
            true_future = (batch["future_observations"] - stats["obs"]["mean"]) / stats["obs"]["std"]
            with torch.inference_mode():
                prediction = solver.model.predict_idm_actions(history, history_act, None, true_future,
                    history_valid_mask=batch["history_valid_mask"])
            predictions.append(action.numpy()); futures.append(future.numpy())
            inverse.append((prediction * stats["act"]["std"] + stats["act"]["mean"]).numpy())
            targets.append(batch["target_actions"].numpy()); true_futures.append(batch["future_observations"].numpy())
        predicted, idm, future, target, actual = map(lambda value: np.concatenate(value),
            (predictions, inverse, futures, targets, true_futures))
        with np.load(path) as source:
            public = {name[7:]: source[name] for name in source.files if name.startswith("public.")}
            teacher, phases, recomputed, sensitivity = PublicTeacher(task, mode=mode), [], [], []
            for step in range(len(trajectory.actions)):
                observation = {key: value[step:step + 1].copy() for key, value in public.items()}
                if not teacher.commands:
                    alternative = deepcopy(teacher)
                    shifted = deepcopy(observation); shifted["reference_position"][:, 0] += .002
                    original_action = deepcopy(teacher).act(observation)
                    sensitivity.append(float(np.linalg.norm(alternative.act(shifted)[:6] - original_action[:6])))
                recomputed.append(teacher.act(observation)); phases.append(int(teacher.memory.phase[0]))
        phase_values = np.asarray(phases)[indices]
        act_scale = stats["act"]["std"].numpy()
        low, high = task_action_bounds(task)
        by_phase = {}
        for phase in np.unique(phase_values):
            select = phase_values == phase
            by_phase[int(phase)] = {"windows": int(select.sum()),
                "TF0_action_normalized_mse": float(np.mean(((predicted[select, 0] - target[select, 0]) / act_scale) ** 2)),
                "TF1_IDM_action_normalized_mse": float(np.mean(((idm[select, 0] - target[select, 0]) / act_scale) ** 2))}
        rows.append({"mode": mode, "data": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "episode_controls": len(trajectory.actions), "sampled_windows": len(indices), "phase_metrics": by_phase,
            "TF0_first_action_RMSE_SI": {key: float(np.sqrt(np.mean((predicted[:, 0, selection] - target[:, 0, selection]) ** 2))) for key, selection in groups.items()},
            "TF1_first_action_RMSE_SI": {key: float(np.sqrt(np.mean((idm[:, 0, selection] - target[:, 0, selection]) ** 2))) for key, selection in groups.items()},
            "first_future_position_RMSE_m": float(np.sqrt(np.mean((future[:, 0, :3] - actual[:, 0, :3]) ** 2))),
            "last_future_position_RMSE_m": float(np.sqrt(np.mean((future[:, -1, :3] - actual[:, -1, :3]) ** 2))),
            "future_rotation_matrix_RMSE": float(np.sqrt(np.mean((future[..., 3:12] - actual[..., 3:12]) ** 2))),
            "raw_first_action_outside_public_Box_fraction": float(np.mean(np.any((predicted[:, 0] < low) | (predicted[:, 0] > high), axis=-1))),
            "unloaded_teacher_windows": int(np.sum(target[:, 0, 12] == 0)),
            "force_mode_false_positive_fraction_on_unloaded_teacher_states": float(np.mean(predicted[target[:, 0, 12] == 0, 0, 12] > .01)),
            "unloaded_teacher_predicted_force_mean_N": float(np.mean(np.maximum(predicted[target[:, 0, 12] == 0, 0, 12], 0.))),
            "teacher_action_reconstruction_max_abs_error": float(np.max(np.abs(np.asarray(recomputed) - trajectory.actions))),
            "teacher_twist_action_sensitivity_to_public_reference_dx_2mm": {"max": max(sensitivity), "mean": float(np.mean(sensitivity))}})
        all_actions.append(trajectory.actions)
    actions = np.concatenate(all_actions)
    return {"scope": "offline original35D source model; no physics reset/step or target execution", "source_calls": 0, "target_calls": 0,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "episodes": rows, "source_action_std_SI": actions.std(0).tolist(), "source_action_centered_matrix_rank": int(np.linalg.matrix_rank(actions - actions.mean(0))),
        "identifiability_limit": "Public scripted teacher uses self-issued reference, phase/integral memory and ten-step command chunks. Reference sensitivity is a conditional-input dependence check, not proof that full physical history and future cannot identify actions. Constant stiffness channels and unexcited action directions cannot establish structural inverse-dynamics identifiability from these two demonstrations.",
        "existing_online_coverage": "Previous attempt uses60primary+60shadow source controls per mode (1.2s);80%online sampling therefore overweights the initial recovery phase."}


def source_force_support(trajectories):
    """Derive requested force support from every frame of the teacher bank."""
    forces = np.concatenate([row.actions[:, 12] for row in trajectories])
    values, counts = np.unique(np.round(forces.astype(np.float64), 6), return_counts=True)
    return {"values_N": values.tolist(), "counts": counts.tolist(), "frames": len(forces),
            "scope": "all requested-force labels in complete source teacher episodes; not rate-limited issued actions"}


def bind_source_action(task, action, codec):
    value = bounded_task_action(task, action)
    kind = codec.get("kind", "continuous13D")
    if kind == "source_gap_off_deadzone":
        support = np.asarray(codec["values_N"], dtype=np.float32)
        if (support.ndim != 1 or len(support) < 2 or not np.isfinite(support).all()
                or np.any(support < 0) or np.any(support > task.config.force_limit)):
            raise ValueError("invalid source teacher requested-force support")
        positive = support[support > 0]
        threshold = float(codec["threshold_N"])
        if not np.any(support == 0) or not len(positive) or not np.isclose(threshold, float(positive.min()) / 2, rtol=1e-6, atol=1e-7):
            raise ValueError("force off threshold must be half the smallest actual positive teacher support")
        value[..., 12] = np.where(value[..., 12] < threshold, 0., value[..., 12])
    elif kind != "continuous13D":
        raise ValueError("unknown policy-owned source action decoder")
    return value


class SourceTaskPolicy:
    """Task-bound FADA proposals with the checkpoint's explicit action decoder."""
    def __init__(self, checkpoint, *, task, device="cpu"):
        self.task = task
        if checkpoint["cfg"]["obs_dim"] != 35 or task.history_len != checkpoint["cfg"]["history_len"]:
            raise ValueError("task-bound FADA requires the checkpoint35D physical view and matching public history length")
        self.decoder = deepcopy(checkpoint.get("genedynamics_task_adapter", {}).get("action_decoder", {"kind": "continuous13D"}))
        if self.decoder.get("kind") != "continuous13D" and task.spec.family != "contact":
            raise ValueError("force decoder requires the declared contact task action contract")
        declared = checkpoint.get("genedynamics_task_adapter", {}).get("task_name")
        if declared is not None and declared != task.spec.name: raise ValueError("checkpoint task action contract differs from supplied task")
        self.solver = FADASolver.from_checkpoint(checkpoint, device=device, task_action_binding=True)

    @classmethod
    def from_checkpoint(cls, checkpoint, *, task, device="cpu"):
        return cls(checkpoint, task=task, device=device)

    def act(self, public):
        return bind_source_action(self.task, self.solver.act(public), self.decoder)

    def reset(self): self.solver.reset()

    @property
    def model(self): return self.solver.model


def _json(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def save_source_episode(output, task, public_rows, actions, *, mode, role, extras=None):
    """Retain every attempted source episode, including all failures/partial rows."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    obs = np.concatenate([portable_observation(row) for row in public_rows])
    previous = np.concatenate([row["previous_action"] for row in public_rows]).astype(np.float32)
    command = np.concatenate([np.concatenate((row["goal_position"], row["goal_rotation"].reshape(1, 9),
        np.eye(2, dtype=np.float32)[mode][None]), axis=-1) for row in public_rows]).astype(np.float32)
    trajectory = TaskTrajectory(obs, np.asarray(actions, dtype=np.float32).reshape(-1, 13), command, previous, role)
    arrays = {key: getattr(trajectory, key) for key in ("obs", "actions", "current_command", "previous_issued")}
    for name in public_rows[0]: arrays["public." + name] = np.concatenate([row[name] for row in public_rows])
    if task.records[0].traces:
        for name in task.records[0].traces[0]: arrays["trace." + name] = np.asarray([row[name] for row in task.records[0].traces])
    np.savez_compressed(output / "episode.npz", **arrays)
    report = {"role": role, "requested_mode": mode, "evaluation": task.evaluate(),
              "observed_mode": _observed_mode(task, task.records[0]),
              "accounting": dict(task.lifetime_accounting), "episode_sha256": hashlib.sha256((output / "episode.npz").read_bytes()).hexdigest(),
              "extras": extras or {}}
    (output / "report.json").write_text(json.dumps(report, default=_json, indent=2) + "\n")
    (output / "episode_records.json").write_text(json.dumps([asdict(row) for row in task.records], default=_json, indent=2) + "\n")
    return trajectory, report


def load_saved_episode(path, sha256):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256: raise ValueError("source episode SHA changed")
    with np.load(path) as data:
        return TaskTrajectory(*(np.array(data[name], dtype=np.float32) for name in
            ("obs", "actions", "current_command", "previous_issued")), "saved_source_episode")


def validate_episode_split(train_refs, val_refs, train_rows, val_rows):
    """Check path, immutable raw bytes and canonical model-data identity."""
    def canonical(row):
        digest = hashlib.sha256()
        for name in ("obs", "actions", "current_command", "previous_issued"):
            value = np.ascontiguousarray(getattr(row, name), dtype=np.float32)
            digest.update(str(value.shape).encode()); digest.update(value.tobytes())
        return digest.hexdigest()
    checks = {
        "path_disjoint": not ({str(Path(row["path"]).resolve()) for row in train_refs} & {str(Path(row["path"]).resolve()) for row in val_refs}),
        "raw_SHA_disjoint": not ({row["sha256"] for row in train_refs} & {row["sha256"] for row in val_refs}),
        "canonical_model_data_disjoint": not ({canonical(row) for row in train_rows} & {canonical(row) for row in val_rows})}
    if not all(checks.values()): raise ValueError("source train/validation episodes overlap by path, SHA or canonical model data")
    return {**checks, "train_canonical_sha256": [canonical(row) for row in train_rows], "validation_canonical_sha256": [canonical(row) for row in val_rows]}


def collect_source_teacher(task_name, *, seed, mode, output, guard):
    from scripts.baselines.qualify_tasks import PublicTeacher
    task = instantiate_task(task_name, robot="panda", seed=seed, history_len=30)
    task.source_accounting_role = "teacher_collection"
    task.reset((seed,), seed=seed); task.set_modes(mode)
    teacher = PublicTeacher(task, mode=mode)
    public, actions = [task.observe()], []
    try:
        while task.state.status[0] == 0:
            guard()
            action = teacher.act(task.observe()); result = task.step(action)
            actions.append(action.copy()); public.append(result.observation)
    finally:
        row, report = save_source_episode(output, task, public, actions, mode=mode, role="source_teacher")
    return row, report


def collect_full_dagger(task_name, checkpoint, *, seed, mode, teacher_probability, output, device, guard):
    """Complete source episode, with real expert shadow branches at every K steps.

    The main behavior is the continuous13D student/teacher mixture. Both decoder
    variants later use this exact same data and training, isolating decoding.
    """
    from scripts.baselines.qualify_tasks import PublicTeacher
    from genedynamics.experiments.baselines.contact_task import CpuTaskAdapter
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    task = instantiate_task(task_name, robot="panda", seed=seed, history_len=checkpoint["cfg"]["history_len"])
    task.source_accounting_role = "DAgger_primary_collection"
    task.reset((seed,), seed=seed); task.set_modes(mode)
    teacher, policy = PublicTeacher(task, mode=mode), FADASolver.from_checkpoint(checkpoint, device=device)
    branch = CpuTaskAdapter(task.spec, mechanism_spec=task._provider.mechanism, seed=seed, history_len=task.history_len)
    branch.source_accounting_role = "expert_shadow"
    public, actions, labels, branch_rows = [task.observe()], [], [], []
    horizon = checkpoint["cfg"]["pred_horizon"]
    phases = []
    try:
        while task.state.status[0] == 0:
            guard()
            history = portable_history(task)
            branch.restore_checkpoint(task.get_checkpoint())
            expert = _copy_teacher(teacher, branch)
            before = dict(branch.lifetime_accounting)
            future, expert_actions = [], []
            try:
                for _ in range(horizon):
                    if branch.state.status[0] != 0: break
                    guard()
                    action = expert.act(branch.observe()); result = branch.step(action)
                    future.append(portable_observation(result.observation)[0]); expert_actions.append(action)
            finally:
                # Recover suffixes from the physical ledger, even if interrupted
                # after step() but before appending a training-array row.
                record = branch.records[0]
                suffix = record.traces[len(actions):]
                branch_rows.append({"source_prefix_controls": len(actions), "suffix_controls": len(suffix),
                    "complete_requested_horizon": len(suffix) == horizon,
                    "inherited_prefix_reused_not_new_execution": True, "phase": int(expert.memory.phase[0]),
                    "actions": np.asarray(record.requested_actions[len(actions):]), "traces": suffix,
                    "physical_delta": {key: branch.lifetime_accounting[key] - before[key] for key in before}})
            if len(future) == horizon:
                labels.append({"history_obs": torch.from_numpy(history["history_obs"][0]),
                    "history_act": torch.from_numpy(history["history_act"][0]), "history_valid_mask": torch.from_numpy(history["history_valid_mask"][0]),
                    "current_command": torch.from_numpy(history["current_command"][0]),
                    "future_observations": torch.as_tensor(np.asarray(future), dtype=torch.float32),
                    "target_actions": torch.as_tensor(np.asarray(expert_actions), dtype=torch.float32)})
                phases.append(int(expert.memory.phase[0]))
            for _ in range(horizon):
                if task.state.status[0] != 0: break
                guard()
                oracle = teacher.act(task.observe())
                student = bounded_task_action(task, policy.act(portable_history(task)))[0]
                action = oracle if np.random.random() < teacher_probability else student
                result = task.step(action); actions.append(action.copy()); public.append(result.observation)
    finally:
        (output / "shadow_audit.json").write_text(json.dumps({"accounting": branch.lifetime_accounting,
            "branches": branch_rows, "complete_horizon_phase_labels": phases}, default=_json) + "\n")
        row, report = save_source_episode(output / "primary", task, public, actions, mode=mode,
            role="full_episode_continuous_source_student_teacher_mixture",
            extras={"teacher_probability": teacher_probability, "shadow_accounting": dict(branch.lifetime_accounting),
                    "complete_shadow_horizons": len(labels), "phase_counts": {str(p): phases.count(p) for p in set(phases)}})
    return row, labels, report


def sample_shared_dagger(dataset, real_data, shadows, batch_size, *, offline_ratio=.2, online_real_ratio=.5):
    """Native mixed offline=.2; online budget split real/expert-shadow equally."""
    from .task_training import sample_dataset
    if not shadows or real_data is None: return sample_dataset(dataset, batch_size)
    if not 0 < offline_ratio < 1 or not 0 < online_real_ratio < 1:
        raise ValueError("offline and online real sampling ratios must be strictly between zero and one")
    offline = max(1, round(batch_size * offline_ratio))
    real_count = max(1, round((batch_size - offline) * online_real_ratio))
    shadow_count = batch_size - offline - real_count
    if shadow_count < 1: raise ValueError("batch is too small for all three configured source pools")
    batches = [sample_dataset(dataset, offline), sample_dataset(real_data, real_count)]
    ids = torch.randint(len(shadows), (shadow_count,)).tolist()
    batches.append({key: torch.stack([shadows[index][key] for index in ids]) for key in shadows[0]})
    return {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}


def evaluate_source_variant(task_name, checkpoint, *, seed, output, device, guard, ledger=None):
    """Two complete source modes plus individually charged exact CPU replay."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    codec = checkpoint["genedynamics_task_adapter"].get("action_decoder", {"kind": "continuous13D"})
    task = instantiate_task(task_name, robot="panda", seed=seed, history_len=checkpoint["cfg"]["history_len"])
    task.source_accounting_role = "evaluation"
    task.reset((seed, seed + 1), seed=seed); task.set_modes([0, 1])
    policy = SourceTaskPolicy.from_checkpoint(checkpoint, task=task, device=device)
    try:
        for _ in range(task.config.timeout_steps):
            guard(); result = task.step(policy.act(portable_history(task)))
            if result.done.all(): break
    finally:
        (output / "episode_records.json").write_text(json.dumps([asdict(row) for row in task.records], default=_json) + "\n")
        (output / "execution_accounting.json").write_text(json.dumps(task.lifetime_accounting) + "\n")
    evaluation = task.evaluate()
    replays = []
    for record in task.records:
        guard()
        if ledger is None:
            replays.append(task.replay_episode(record))
        else:
            with ledger.role("replay"): replays.append(task.replay_episode(record))
        (output / "replay_accounting.json").write_text(json.dumps([row["accounting"] for row in replays]) + "\n")
    modes = [_observed_mode(task, record) for record in task.records]
    report = {"evaluation": evaluation, "replays": replays, "action_decoder": codec, "target_calls": 0,
        "requested_modes": [0, 1], "observed_modes": modes,
        "source_evaluation_accounting": dict(task.lifetime_accounting), "source_replay_accounting": [row["accounting"] for row in replays],
        "gate_passed": evaluation["successes"] == 2 and evaluation["status_mismatches"] == 0 and modes == [0, 1]
            and all(row["matches_status"] and row["trace_exact"] for row in replays)}
    (output / "report.json").write_text(json.dumps(report, default=_json, indent=2) + "\n")
    return report
