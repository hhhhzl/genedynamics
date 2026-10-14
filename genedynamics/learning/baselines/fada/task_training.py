"""FADA Planner-IDM two-pass training on an explicit portable task interface.

This task adaptation retains the released Transformer, inverse-dynamics pass,
frozen-IDM planner action pass, oracle-shadow DAgger and target-record LoRA.
Public scripted task teachers replace the native humanoid PPO oracle. It does
not reproduce native humanoid assets, weak-policy checkpoints or paper scores.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from genedynamics.experiments.baselines.contact_task import FLAT_FIELDS
from genedynamics.solvers.single.fada import FADASolver
from genedynamics.solvers.single.fada.backends.model import PlannerIDMPolicy
from .checkpoint import capture_rng_state, restore_rng_state, to_cpu
from .data import Trajectory, TrajectoryWindowDataset
from .loss import weighted_horizon_mse

LEGACY_PORTABLE_FIELDS = tuple(key for key in FLAT_FIELDS if key not in {"joint_position", "joint_velocity"})
PORTABLE_FIELDS = ("tcp_position", "tcp_rotation", "tcp_twist", "measured_wrench",
                   "body_position", "body_twist", "controller_feedback")
PUBLIC_CHART_VERSION = "portable-physical-state35-box-v3"
LEGACY_TO_PHYSICAL_COLUMNS = tuple(range(24)) + tuple(range(75, 86))


def task_action_bounds(task):
    """Known controller Box, without inspecting hidden plant parameters."""
    if task.spec.alias == "T1":
        skill = task.runner.skill
        high = np.asarray([skill.linear_speed] * 3 + [skill.angular_speed] * 3
                          + [skill.log_stiffness_limit] * 6 + [0.], dtype=np.float32)
    else:
        high = np.asarray([.08] * 3 + [.4] * 3 + [1.] * 6 + [task.config.force_limit], dtype=np.float32)
    low = -high.copy(); low[-1] = 0.
    return low, high


def bounded_task_action(task, action):
    """Native FADA also clips deployment commands before environment stepping."""
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
    low, high = task_action_bounds(task)
    return np.clip(np.asarray(action, dtype=np.float32), low, high)


def project_trajectory(row, task):
    """Public amplitude projection of an immutable raw record, never relabeling truth.

    Old collectors logged requests before the known amplitude limiter. Those
    magnitudes have identical actuator meaning after clipping. Inverse-dynamics
    labels use that same Box. Future physical states exclude action feedback,
    self-issued references, goals and clocks to prevent an action-label shortcut.
    The raw data and its collector provenance are retained by the caller.
    """
    if row.obs.shape[-1] == 86:
        obs = row.obs[..., LEGACY_TO_PHYSICAL_COLUMNS].copy()
    elif row.obs.shape[-1] == 35:
        obs = row.obs.copy()
    else:
        raise ValueError("portable physical trajectory requires35D state or explicit86D legacy projection")
    return TaskTrajectory(obs.astype(np.float32), bounded_task_action(task, row.actions), row.current_command.astype(np.float32),
                          bounded_task_action(task, row.previous_issued), row.source)


def portable_observation(public):
    """No Panda/target joint correspondence is presumed or learned by the planner."""
    batch = np.asarray(public["tcp_position"]).shape[0]
    return np.concatenate([np.asarray(public[key]).reshape(batch, -1) for key in PORTABLE_FIELDS], axis=-1).astype(np.float32)


def portable_history(task):
    value = task.fada_observation()
    public = task.observe()
    drop = public["joint_position"].shape[-1] + public["joint_velocity"].shape[-1]
    value["history_obs"] = value["history_obs"][..., drop:][..., LEGACY_TO_PHYSICAL_COLUMNS]
    return value


@dataclass
class TaskTrajectory:
    obs: np.ndarray
    actions: np.ndarray
    current_command: np.ndarray
    previous_issued: np.ndarray
    source: str


class TaskWindowDataset(TrajectoryWindowDataset):
    """Native episode windows with the task's public issued-command history.

    Labels are policy-requested actions; history records known, rate-limited
    issued commands, before unknown actuator delay. This explicit task chart
    prevents revealing the hidden delay or crossing episode boundaries.
    """
    def __init__(self, trajectories, *, history_len=30, pred_horizon_k=6):
        self.task_trajectories = list(trajectories)
        native = [Trajectory(row.obs, row.actions, row.current_command, row.source) for row in trajectories]
        super().__init__(native, history_len=history_len, pred_horizon_k=pred_horizon_k)

    def __getitem__(self, index):
        value = super().__getitem__(index)
        episode, step = self._resolve(index)
        row = self.task_trajectories[episode]
        start = max(0, step - self.history_len + 1)
        history = np.zeros((self.history_len, row.actions.shape[-1]), dtype=np.float32)
        history[-(step - start + 1):] = row.previous_issued[start:step + 1]
        value["history_act"] = torch.from_numpy(history)
        return value


def load_source_trajectory(path, *, mode):
    import hashlib
    import json
    from pathlib import Path

    path = Path(path)
    receipt = json.loads((path.parent / "report.json").read_text())
    if not receipt["valid_source_demo"] or hashlib.sha256(path.read_bytes()).hexdigest() != receipt["episode_sha256"]:
        raise ValueError("source demonstration is not independently qualified or its SHA changed")
    if "observed_mode" in receipt and receipt["observed_mode"] != mode:
        raise ValueError("source demonstration mode differs from physical trace evidence")
    audit = path.parent / "mode-audit.json"
    if audit.exists() and json.loads(audit.read_text())["observed_mode"] != mode:
        raise ValueError("source mode audit contradicts this label")
    with np.load(path) as source:
        positions = source["public.tcp_position"][1:][None]
        valid = source["trace.valid"][None]
        task_name = receipt["contract"]["task"]["name"]
        if task_name == "obstacle_transfer":
            from genedynamics.learning.cross_embodiment.data import observed_route_modes
            observed = int(observed_route_modes(positions[:, :, None], valid[:, :, None])[0])
        else:
            from genedynamics.learning.cross_embodiment.contact_reference import classify_contact_modes
            observed = int(classify_contact_modes(positions, source["public.measured_wrench"][1:, 2][None],
                                                  valid, insertion=True)[0])
        if observed != mode:
            raise ValueError(f"source physical mode is {observed}, not requested label {mode}; unknown modes cannot be forced")
        public = {key: source["public." + key] for key in PORTABLE_FIELDS}
        # Here the sample axis is time; the portable chart is identical.
        obs = portable_observation(public)
        action = np.array(source["actions"], dtype=np.float32)
        previous = np.array(source["public.previous_action"], dtype=np.float32)
        command = np.concatenate((source["public.goal_position"], source["public.goal_rotation"].reshape(len(obs), 9),
                                  np.repeat(np.eye(2, dtype=np.float32)[mode][None], len(obs), axis=0)), axis=-1)
    return TaskTrajectory(obs, action, command, previous, "qualified_source_teacher")


def released_model_config(obs_dim, *, history_len=30, pred_horizon=6):
    return dict(obs_dim=int(obs_dim), act_dim=13, cmd_dim=14, history_len=int(history_len), pred_horizon=int(pred_horizon),
                planner_d_model=128, planner_nhead=4, planner_num_layers=3, planner_dim_feedforward=512,
                planner_dropout=.1, planner_use_learned_positional_encoding=False, planner_use_action_history=False,
                planner_predict_delta=True, idm_d_model=128, idm_nhead=4, idm_encoder_num_layers=3, idm_decoder_num_layers=2,
                idm_dim_feedforward=512, idm_dropout=.1, idm_use_learned_positional_encoding=False,
                idm_use_current_command_for_history=False)


def source_statistics(trajectories, *, task):
    """Fixed public SI chart; constant source channels never acquire huge gains.

    Only the public task goal and declared controller/scene scales enter this
    transform. No target data or hidden mechanism is used to fit statistics.
    Empirical whitening of near-constant rotation/stiffness/request channels
    would amplify small target deviations into self-exciting model inputs.
    """
    goal = trajectories[0].current_command[0, :12]
    low, high = task_action_bounds(task)
    action_mean = (low + high) / 2
    action_scale = (high - low) / 2
    action_scale[action_scale == 0] = 1.
    mean = np.zeros(35, dtype=np.float32); scale = np.ones(35, dtype=np.float32)
    position_scale = .05
    if task.spec.alias == "T1":
        position_scale = max(.05, float(np.linalg.norm(np.asarray(task.config.goal_position)
                                                      - np.asarray(task.config.start_position))))
    mean[:3], scale[:3] = goal[:3], position_scale
    mean[3:12] = goal[3:12]  # matrix entries retain unit physical scale
    scale[12:18] = high[:6]
    scale[18:21] = getattr(task.config, "force_limit", 25.)
    scale[21:24] = getattr(task.config, "orientation_wrench_limit", 4.)
    command_mean = np.r_[goal, [0., 0.]].astype(np.float32)
    command_scale = np.r_[[position_scale] * 3, np.ones(11)].astype(np.float32)
    def statistics(mean, scale, prefix):
        return {prefix + "_mean": torch.as_tensor(mean), prefix + "_std": torch.as_tensor(scale), "eps": 1e-6}
    return dict(obs_norm_stats=statistics(mean, scale, "obs"),
                action_norm_stats=statistics(action_mean, action_scale, "action"),
                command_norm_stats=statistics(command_mean, command_scale, "command"))


class TaskPlannerIDMTrainer:
    def __init__(self, config, statistics, *, device="cpu", learning_rate=3e-4, weight_decay=1e-2):
        self.config, self.statistics = deepcopy(config), to_cpu(statistics)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("FADA CUDA requested but unavailable")
        self.model = PlannerIDMPolicy(**config).to(self.device)
        self.planner_optimizer = torch.optim.AdamW(self.model.planner_parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.idm_optimizer = torch.optim.AdamW(self.model.idm_parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.steps = 0

    def normalize(self, batch):
        value = {key: tensor.to(self.device) for key, tensor in batch.items()}
        for names, stats_key, prefix in ((["history_obs", "future_observations"], "obs_norm_stats", "obs"),
                                         (["history_act", "target_actions"], "action_norm_stats", "action"),
                                         (["current_command"], "command_norm_stats", "command")):
            stats = self.statistics[stats_key]
            mean, std = stats[prefix + "_mean"].to(self.device), stats[prefix + "_std"].to(self.device)
            for name in names:
                value[name] = (value[name] - mean) / std
        return value

    def step(self, batch):
        batch = self.normalize(batch)
        self.model.train()
        args = (batch["history_obs"], batch["history_act"], batch["current_command"])
        kwargs = dict(history_valid_mask=batch["history_valid_mask"])
        # Native first pass: true future, teacher forcing=1, IDM only.
        command = self.model._resolve_idm_current_command(args[2], idm_use_current_command_for_history=False)
        predicted = self.model.predict_idm_actions(args[0], args[1], command, batch["future_observations"], **kwargs)
        idm_loss = weighted_horizon_mse(predicted, batch["target_actions"], gamma=0.)
        self.idm_optimizer.zero_grad(set_to_none=True)
        idm_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.model.idm_parameters()), 1.)
        self.idm_optimizer.step()
        # Native second pass: freeze IDM weights but differentiate through its
        # input to the planner; TF=0 and planner_obs_loss_coef=0.
        parameters = list(self.model.idm_parameters())
        flags = [value.requires_grad for value in parameters]
        for value in parameters:
            value.requires_grad_(False)
        self.planner_optimizer.zero_grad(set_to_none=True)
        try:
            predicted = self.model(*args, **kwargs, detach_planner_future_obs=False)
            planner_loss = weighted_horizon_mse(predicted, batch["target_actions"], gamma=0.)
            planner_loss.backward()
        finally:
            for parameter, flag in zip(parameters, flags):
                parameter.requires_grad_(flag)
        torch.nn.utils.clip_grad_norm_(list(self.model.planner_parameters()), 1.)
        self.planner_optimizer.step()
        self.steps += 1
        return {"idm_action_loss": float(idm_loss.detach().cpu()), "planner_via_idm_action_loss": float(planner_loss.detach().cpu())}

    def checkpoint(self):
        return dict(cfg=deepcopy(self.config), obs_dim=self.config["obs_dim"], act_dim=13, cmd_dim=14,
                    normalization={"io_enabled": True}, model_state_dict=to_cpu(self.model.state_dict()),
                    **to_cpu(self.statistics),
                    genedynamics_task_adapter={"portable_fields": PORTABLE_FIELDS, "oracle": "public scripted task teacher replaces native PPO oracle",
                                              "source_training": "released IDM TF1 then frozen-IDM planner action TF0; two AdamW optimizers",
                                              "io_normalization": "fixed public physical scales and controller Box; no target-fitted statistics",
                                              "public_chart_version": PUBLIC_CHART_VERSION,
                                              "future_state_excludes": ["actions", "references", "goals", "clock"],
                                              "native_method_qualified": False, "updates": self.steps})

    def get_training_state(self):
        return {"checkpoint": self.checkpoint(), "planner_optimizer": to_cpu(self.planner_optimizer.state_dict()),
                "idm_optimizer": to_cpu(self.idm_optimizer.state_dict()), "steps": self.steps, "rng": capture_rng_state()}

    def restore_training_state(self, state):
        if self.config != state["checkpoint"]["cfg"]:
            raise ValueError("FADA task training checkpoint architecture/chart differs")
        self.statistics = {name: to_cpu(state["checkpoint"][name]) for name in
                           ("obs_norm_stats", "action_norm_stats", "command_norm_stats")}
        self.model.load_state_dict(state["checkpoint"]["model_state_dict"], strict=True)
        self.planner_optimizer.load_state_dict(state["planner_optimizer"])
        self.idm_optimizer.load_state_dict(state["idm_optimizer"])
        self.steps = int(state["steps"])
        restore_rng_state(state["rng"])


def sample_dataset(dataset, batch_size):
    ids = torch.randint(len(dataset), (int(batch_size),)).tolist()
    return {key: torch.stack([dataset[index][key] for index in ids]) for key in dataset[ids[0]]}


def sample_pools(dataset, shadows, batch_size):
    if not shadows:
        return sample_dataset(dataset, batch_size)
    offline = max(1, int(batch_size * .2))  # native mixed offline/online ratio
    initial = sample_dataset(dataset, offline)
    indices = torch.randint(len(shadows), (batch_size - offline,)).tolist()
    return {key: torch.cat((value, torch.stack([shadows[index][key] for index in indices]))) for key, value in initial.items()}


def oracle_shadow_dagger(source_task, solver, *, mode, chunks=10, horizon=6, seed=0):
    """True expert-shadow source rollouts from each student's physical state.

    Restoring a new source-model branch does not fabricate future labels. Its
    K real MuJoCo steps are charged separately. No target labels are obtained.
    """
    from genedynamics.experiments.baselines.contact_task import CpuTaskAdapter
    from scripts.baselines.qualify_tasks import PublicTeacher

    source_task.reset((seed,), seed=seed)
    source_task.set_modes(mode)
    teacher = PublicTeacher(source_task, mode=mode, recipe="upper_clearance" if source_task.spec.alias == "T1" and mode == 0 else "default")
    solver.reset()
    from dataclasses import asdict
    labels, shadow_calls, shadow_records = [], 0, []
    random = np.random.default_rng(seed)
    for _ in range(chunks):
        history = portable_history(source_task)
        shadow = CpuTaskAdapter(source_task.spec, mechanism_spec=source_task._provider.mechanism,
                                seed=seed, history_len=source_task.history_len)
        shadow.restore_checkpoint(source_task.get_checkpoint())
        expert = deepcopy(teacher); expert.task = shadow
        future, actions = [], []
        for _ in range(horizon):
            if shadow.state.status[0] != 0:
                break
            action = expert.act(shadow.observe())
            result = shadow.step(action)
            future.append(portable_observation(result.observation)[0]); actions.append(action)
        shadow_calls += shadow.lifetime_accounting["control_steps"]
        shadow_records.append({"physical_suffix_control_steps": shadow.lifetime_accounting["control_steps"],
                               "scope": "inherited source prefix is reused data, not newly executed or a statistical repeat",
                               "episodes": [asdict(record) for record in shadow.records]})
        if len(future) == horizon:
            labels.append({"history_obs": torch.from_numpy(history["history_obs"][0]),
                           "history_act": torch.from_numpy(history["history_act"][0]),
                           "history_valid_mask": torch.from_numpy(history["history_valid_mask"][0]),
                           "current_command": torch.from_numpy(history["current_command"][0]),
                           "future_observations": torch.as_tensor(np.asarray(future)),
                           "target_actions": torch.as_tensor(np.asarray(actions))})
        for _ in range(horizon):
            if source_task.state.status[0] != 0:
                break
            oracle = teacher.act(source_task.observe())
            student = bounded_task_action(source_task, solver.act(portable_history(source_task)))[0]
            source_task.step(oracle if random.random() < .5 else student)
        if source_task.state.status[0] != 0:
            break
    return labels, {"student_source_calls": dict(source_task.lifetime_accounting), "expert_shadow_control_steps": shadow_calls,
                    "source_episode": source_task.evaluate(), "expert_mix_probability": .5,
                    "student_episode_records": [asdict(record) for record in source_task.records],
                    "expert_shadow_episode_records": shadow_records,
                    "scope": "bounded scripted-oracle task adaptation; native PPO/weak-policy source is separately qualified"}


def collect_target_records(task, solver, *, seconds=120., seed=0):
    """Unrewarded B-log self-rollouts, with failures/reset boundaries retained."""
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("positive B-log seconds are required")
    target_steps = int(np.ceil(seconds / task.config.control_dt))
    trajectories, public_rows, actions, mode, generation = [], [], [], 0, 0
    task.reset((seed,), seed=seed); task.set_modes(mode); solver.reset()
    public_rows.append(task.observe())
    for control in range(target_steps):
        # No target reward, expert command, hidden mechanism or target success
        # label is consumed for inverse-dynamics fitting or policy selection.
        action = bounded_task_action(task, solver.act(portable_history(task)))[0]
        result = task.step(action)
        actions.append(action.copy()); public_rows.append(result.observation)
        if result.done[0] or control + 1 == target_steps:
            obs = np.concatenate([portable_observation(row) for row in public_rows])
            previous = np.concatenate([row["previous_action"] for row in public_rows])
            command = np.concatenate([np.concatenate((row["goal_position"], row["goal_rotation"].reshape(1, 9),
                                                      np.eye(2, dtype=np.float32)[mode][None]), axis=-1) for row in public_rows])
            trajectories.append(TaskTrajectory(obs, np.asarray(actions), command, previous, "unrewarded_target_B_log"))
            if control + 1 < target_steps:
                generation += 1; mode = 1 - mode
                task.reset((seed + generation,), seed=seed + generation); task.set_modes(mode); solver.reset()
                public_rows, actions = [task.observe()], []
    return trajectories, {"protocol": "B-log", "reward_access": False, "target_expert_access": False,
                          "requested_seconds": float(seconds), "actual_control_seconds": task.lifetime_accounting["control_steps"] * task.config.control_dt,
                          "accounting": dict(task.lifetime_accounting), "episodes": task.evaluate(),
                          "control_policy": "frozen source Planner-IDM student; no scripted target teacher"}
