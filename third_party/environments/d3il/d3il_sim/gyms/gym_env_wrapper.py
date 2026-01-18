from abc import ABC, abstractmethod

import gym
from gym.utils import seeding
import numpy as np

from environments.d3il.d3il_sim.controllers.Controller import ControllerBase
from environments.d3il.d3il_sim.core import Scene


class GymEnvWrapper(gym.Env, ABC):
    """Open AI gym_envs environment for tasks using a Panda Franka robot arm with MuJoCo physics."""

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(
        self,
        scene: Scene,
        controller: ControllerBase,
        max_steps_per_episode,
        n_substeps,
        debug: bool = False,
    ):
        self.scene = scene
        self.robot = scene.robots[0]

        self.controller = controller

        self.max_steps_per_episode = max_steps_per_episode
        self.n_substeps = n_substeps
        self.env_step_counter = 0

        self.episode = 0
        self.terminated = False
        self.debug = debug

    def start(self):
        self.scene.start()
        self.controller.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, gripper_width=None, desired_vel=None, desired_acc=None):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if gripper_width is not None:
            self.robot.set_gripper_width = gripper_width

        # self.controller.set_action(action)
        # self.controller.execute_action(n_time_steps=self.n_substeps)

        self.robot.open_fingers()

        # self.robot.cartesianPosQuatTrackingController.setSetPoint(action)
        # self.robot.cartesianPosQuatTrackingController.executeControllerTimeSteps(
        #     self.robot, self.n_substeps, block=False
        # )

        # if self.env_step_counter == 0:
        #     action = self.robot.current_c_pos[:2] + action
        # else:
        #     action = self.robot.des_c_pos[:2] + action
        #
        # action[0] = np.clip(action[0], 0.3, 0.8)
        # action[1] = np.clip(action[1], -0.45, 0.45)
        # action = np.concatenate((action, [0.12, 0, 1, 0, 0]))

        self.controller.setSetPoint(action)
        self.controller.executeControllerTimeSteps(
            self.robot, self.n_substeps, block=False
        )

        observation = self.get_observation()
        reward = self.get_reward()
        done = self.is_finished()

        for i in range(self.n_substeps):
            self.scene.next_step()

        debug_info = {}
        if self.debug:
            debug_info = self.debug_msg()

        self.env_step_counter += 1
        return observation, reward, done, debug_info

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Compute an observation of your gym.
        Can be the complete robot and environment state, a camera image, etc.
        Returns:
            observation: a numpy array
        """
        raise NotImplementedError

    @abstractmethod
    def get_reward(self):
        """Calculate your Gym's reward.

        Returns:
            Scalar reward.
        """
        raise NotImplementedError

    @abstractmethod
    def _check_early_termination(self) -> bool:
        return self.terminated

    def is_finished(self):
        """Checks if the robot either exceeded the maximum number of steps or is terminated according to another task
        dependent metric.

        Returns:
            True if robot should terminate
        """
        if (
            self.terminated
            or self._check_early_termination()
            or self.env_step_counter >= self.max_steps_per_episode - 1
        ):
            return True
        return False

    def debug_msg(self) -> dict:
        """
        Use this function to return a debug message after each step.
        This object is for debugging purposes only and must not influence your agent.
        Returns:
            dict of debug messages
        """
        return {}

    @abstractmethod
    def _reset_env(self):
        raise NotImplementedError

    def reset(self):
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1
        obs = self._reset_env()

        return obs

    def robot_state(self):
        # Update Robot State
        self.robot.receiveState()

        # joint state
        joint_pos = self.robot.current_j_pos
        joint_vel = self.robot.current_j_vel

        # gripper state
        gripper_vel = self.robot.current_fing_vel
        gripper_width = [self.robot.gripper_width]

        # end effector state
        tcp_pos = self.robot.current_c_pos
        tcp_vel = self.robot.current_c_vel
        tcp_quad = self.robot.current_c_quat

        return tcp_pos

        # return np.concatenate(
        #     [
        #         joint_pos,
        #         joint_vel,
        #         gripper_vel,
        #         gripper_width,
        #         tcp_pos,
        #         tcp_vel,
        #         tcp_quad,
        #     ]
        # )
