"""Explicit held-object task adaptation of H2S2R's object-trajectory mechanism.

Uses its unchanged reward definition and original PPO learner. Human video,
hand exploration initialization and FABRICS are replaced by a declared robot
state demonstration, task reset and shared skill controller; this adaptation
must never be reported as native Human2Sim2Robot reproduction.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import numpy as np
import torch

from .native_task_math import compute_reward_jit


class ObjectTrajectoryTaskEnv:
    def __init__(self, task, demonstration, *, num_envs=2, seed=0, demo_dt=None,
                 action_smoothing_weight=-.01):
        import gym

        self.task, self.num_envs, self.seed = task, int(num_envs), int(seed)
        self.reset_generation = 0
        self.demonstration = np.asarray(demonstration, dtype=np.float32)
        if (self.demonstration.ndim != 3 or self.demonstration.shape[1:] != (4, 4)
                or len(self.demonstration) < 2 or not np.isfinite(self.demonstration).all()):
            raise ValueError("demonstration must be an actual finite world object-pose trajectory [T,4,4]")
        self.demo_dt = float(task.config.control_dt if demo_dt is None else demo_dt)
        if not np.isfinite(self.demo_dt) or self.demo_dt <= 0:
            raise ValueError("demo_dt must be finite and positive")
        self.action_smoothing_weight = float(action_smoothing_weight)
        contact = task.spec.family == "contact"
        if contact:
            linear, angular, stiffness = .08, .4, 1.
        else:
            linear, angular, stiffness = (task.runner.skill.linear_speed, task.runner.skill.angular_speed,
                                          task.runner.skill.log_stiffness_limit)
        low = np.array([-linear] * 3 + [-angular] * 3 + [-stiffness] * 6 + [0.], np.float32)
        high = np.array([linear] * 3 + [angular] * 3 + [stiffness] * 6 + [task.config.force_limit if contact else 0.], np.float32)
        self.action_space = gym.spaces.Box(low, high)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(task.observation_size + 12,), dtype=np.float32)
        self.reference_index = np.zeros(self.num_envs, np.float64)
        self.previous_actions = np.zeros((self.num_envs, 13), np.float32)
        self.smallest_distance = np.full(self.num_envs, np.inf, np.float32)
        self._reset_initialized = False
        self.demonstration_sha256 = hashlib.sha256(self.demonstration.tobytes()).hexdigest()

    def get_env_info(self):
        return {"observation_space": self.observation_space, "action_space": self.action_space,
                "agents": 1, "value_size": 1}

    def _reference(self):
        return self.demonstration[np.minimum(self.reference_index.astype(int), len(self.demonstration) - 1)]

    def _observation(self, observation=None):
        reference = self._reference()
        return np.concatenate((self.task.flat_observation(observation), reference[:, :3, 3],
                               reference[:, :3, :3].reshape(self.num_envs, 9)), axis=-1).astype(np.float32)

    def reset(self):
        self.reset_generation += 1
        self.task.reset(tuple(range(self.num_envs)), seed=self.seed + self.reset_generation)
        self.reference_index[:] = 0
        self.previous_actions[:] = 0
        self.smallest_distance[:] = np.inf
        self._reset_initialized = True
        return self._observation()

    def step(self, actions, *, auto_reset=True):
        actions = np.asarray(actions, np.float32)
        center = (self.action_space.high + self.action_space.low) / 2
        half_range = (self.action_space.high - self.action_space.low) / 2
        native_actions = np.divide(actions - center, half_range,
                                   out=np.zeros_like(actions), where=half_range > 0)
        transition = self.task.step(actions)
        public = transition.observation
        reference = self._reference()
        # Known rigidly held object: pose feature uses identical origin plus
        # rotation terms. Hand distance/lift gates are fixed by this task's
        # grasp assumption, rather than inventing a dexterous hand state.
        # Native OBJECT_KEYPOINT_OFFSETS: three +axis points at 0.2 m.
        # Retain the native mean keypoint error, rather than inventing a new
        # weighted translation/rotation distance for this adaptation.
        offsets = np.eye(3, dtype=np.float32) * .2
        actual_points = np.einsum("bij,kj->bki", public["tcp_rotation"], offsets) + public["tcp_position"][:, None]
        desired_points = np.einsum("bij,kj->bki", reference[:, :3, :3], offsets) + reference[:, None, :3, 3]
        distance = np.linalg.norm(actual_points - desired_points, axis=-1).mean(axis=-1).astype(np.float32)
        self.smallest_distance = np.minimum(self.smallest_distance, distance)
        zero = torch.zeros(self.num_envs)
        one = torch.ones(self.num_envs, dtype=torch.bool)
        success = torch.from_numpy(transition.status == 1)
        reward_terms = compute_reward_jit(
            rew_buf=zero, is_object_lifted=one, is_goal_object_lifted=one,
            smallest_this_episode_indexfingertip_goal_distance=zero, indexfingertip_goal_distance=zero,
            smallest_this_episode_fingertips_object_distance=zero, fingertips_object_distance=zero,
            smallest_this_episode_object_goal_distance=torch.from_numpy(self.smallest_distance.copy()),
            object_goal_distance=torch.from_numpy(distance), is_fingertips_object_close=one,
            is_in_success_region=success, has_enough_consecutive_successes_to_end_episode=success,
            max_episode_length=self.task.config.timeout_steps,
            progress_buf=torch.from_numpy(np.asarray(self.task.state.steps).astype(np.float32)),
            raw_actions=torch.from_numpy(native_actions), prev_raw_actions=torch.from_numpy(self.previous_actions.copy()),
            object_and_goal_far_apart_need_stop_reference_motion=torch.from_numpy(distance > .2),
        )
        # Native FORCE_REFERENCE_TRAJECTORY_TRACKING reward weights.
        reward = (reward_terms["Object Tracking Reward"]
                  + self.action_smoothing_weight * reward_terms["Action Smoothing Penalty"]).numpy()
        self.previous_actions[:] = native_actions
        self.reference_index += np.where(distance > .2, 0., self.task.config.control_dt / self.demo_dt)
        done = transition.done.copy()
        info = {"task_status": transition.status.copy(), "reference_tracking_error": distance,
                "time_outs": transition.status == 5, "source_method": "human2sim2robot_adapted"}
        if auto_reset and done.any():
            self.reset_generation += 1
            self.task.reset(tuple(range(self.num_envs)), seed=self.seed + self.reset_generation, mask=done)
            self.reference_index[done] = 0
            self.previous_actions[done] = 0
            self.smallest_distance[done] = np.inf
        return self._observation(), reward.astype(np.float32), done.astype(np.uint8), info

    def set_train_info(self, frame, agent):
        self.training_frame = int(frame)

    def get_env_state(self):
        return deepcopy({"schema": "h2s2r-held-object-adaptation-v2", "task": self.task.get_checkpoint(),
                         "demonstration_sha256": self.demonstration_sha256, "demo_dt": self.demo_dt,
                         "reset_generation": self.reset_generation, "reference_index": self.reference_index,
                         "previous_actions": self.previous_actions, "smallest_distance": self.smallest_distance,
                         "training_frame": getattr(self, "training_frame", 0)})

    def set_env_state(self, payload):
        if payload is None or payload.get("schema") != "h2s2r-held-object-adaptation-v2":
            raise ValueError("complete environment state is required for continuation")
        if payload["demonstration_sha256"] != self.demonstration_sha256 or payload["demo_dt"] != self.demo_dt:
            raise ValueError("continuation demonstration or sampling clock differs")
        self.task.restore_checkpoint(payload["task"])
        self.reset_generation = int(payload["reset_generation"])
        for name in ("reference_index", "previous_actions", "smallest_distance"):
            setattr(self, name, np.asarray(payload[name]).copy())
        self.training_frame = int(payload["training_frame"])
        self._reset_initialized = True
