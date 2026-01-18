import numpy as np
import copy
import time

import sys

from gym.spaces import Box

from d3il_sim.utils.sim_path import d3il_path
from d3il_sim.controllers.Controller import ControllerBase
from d3il_sim.core import Scene
from d3il_sim.core.logger import ObjectLogger, CamLogger
from d3il_sim.gyms.gym_env_wrapper import GymEnvWrapper
from d3il_sim.gyms.gym_utils.helpers import obj_distance
from d3il_sim.utils.geometric_transformation import euler2quat, quat2euler

from d3il_sim.sims.mj_beta.MjRobot import MjRobot
from d3il_sim.sims.mj_beta.MjFactory import MjFactory
from d3il_sim.sims import MjCamera

from d3il_sim.gyms.gym_controllers import GymCartesianVelController
from .objects.gate_insertion_objects import get_obj_list, init_end_eff_pos

obj_list, push_box1, push_box2, push_box3, target_box1, target_box2, target_box3, maze = get_obj_list()

class BPCageCam(MjCamera):
    """
    Cage camera. Extends the camera base class.
    """
    def __init__(self, width: int = 512, height: int = 512, *args, **kwargs):
        super().__init__(
            "bp_cam",
            width,
            height,
            init_pos=[1.05, 0, 1.2],
            init_quat=[
                0.6830127,
                0.1830127,
                0.1830127,
                0.683012,
            ],  # Looking with 30 deg to the robot
            *args,
            **kwargs,
        )


class BlockContextManager:
    def __init__(self, scene, index=0, seed=42) -> None:
        self.scene = scene

        np.random.seed(seed)

        self.box1_space = Box(
            low=np.array([0.35, -0.2, -90]), high=np.array([0.5, -0.15, 90])  # , seed=seed
        )

        self.box2_space = Box(
            low=np.array([0.55, -0.1, -90]), high=np.array([0.7, -0.05, 90])  # , seed=seed
        )

        self.box3_space = Box(
            low=np.array([0.35, 0, -90]), high=np.array([0.5, 0.05, 90])  # , seed=seed
        )

        # Reduced context space size
        self.deg_list = np.random.random_sample(60) * 90 - 45
        self.x1_list = np.random.random_sample(30) * 0.1 + 0.4
        self.x2_list = np.random.random_sample(30) * 0.1 + 0.55
        self.y_list = np.random.random_sample(60) * 0.15 - 0.15

        self.index = index

    def start(self, random=True, context=None):
        if random:
            self.context = self.sample()
        else:
            self.context = context

        self.set_context(self.context)

    def sample(self):
        box1_pos = self.box1_space.sample()
        box2_pos = self.box2_space.sample()
        box3_pos = self.box3_space.sample()

        goal_angle1 = [0, 0, box1_pos[-1] * np.pi / 180]
        quat1 = euler2quat(goal_angle1)

        goal_angle2 = [0, 0, box2_pos[-1] * np.pi / 180]
        quat2 = euler2quat(goal_angle2)

        goal_angle3 = [0, 0, box3_pos[-1] * np.pi / 180]
        quat3 = euler2quat(goal_angle3)

        return [box1_pos, quat1, box2_pos, quat2, box3_pos, quat3]

    def set_context(self, context):
        box1_pos, quat1, box2_pos, quat2, box3_pos, quat3 = context

        self.scene.set_obj_pos_and_quat(
            [box1_pos[0], box1_pos[1], 0.00],
            quat1,
            obj_name="push_box1",
        )

        self.scene.set_obj_pos_and_quat(
            [box2_pos[0], box2_pos[1], 0.00],
            quat2,
            obj_name="push_box2",
        )

        self.scene.set_obj_pos_and_quat(
            [box3_pos[0], box3_pos[1], 0.00],
            quat3,
            obj_name="push_box3",
        )

    def olb_set_context(self, index):
        goal_angle1 = [0, 0, self.deg_list[index] * np.pi / 180]
        quat1 = euler2quat(goal_angle1)

        self.scene.set_obj_pos_and_quat(
            [self.x1_list[index], self.y_list[index], 0.00],
            quat1,
            obj_name="push_box1",
        )

        goal_angle2 = [0, 0, self.deg_list[index] * np.pi / 180]
        quat2 = euler2quat(goal_angle2)

        self.scene.set_obj_pos_and_quat(
            [self.x1_list[index], self.y_list[index], 0.00],
            quat2,
            obj_name="push_box2",
        )

        goal_angle3 = [0, 0, self.deg_list[index] * np.pi / 180]
        quat3 = euler2quat(goal_angle3)

        self.scene.set_obj_pos_and_quat(
            [self.x1_list[index], self.y_list[index], 0.00],
            quat3,
            obj_name="push_box3",
        )

    def next_context(self):
        self.index = (self.index + 1) % len(self.x1_list)
        self.olb_set_context(self.index)

    def set_index(self, index):
        self.index = index


class Gate_Insertion_Env(GymEnvWrapper):
    def __init__(
        self,
        n_substeps: int = 35,
        max_steps_per_episode: int = 2e3,
        debug: bool = False,
        random_env: bool = False,
        interactive: bool = False,
        render: bool = True
    ):

        sim_factory = MjFactory()
        render_mode = Scene.RenderMode.HUMAN if render else Scene.RenderMode.BLIND
        scene = sim_factory.create_scene(
            object_list=obj_list, render=render_mode, dt=0.001
        )
        robot = MjRobot(
            scene,
            xml_path=d3il_path("./models/mj/robot/panda_rod_invisible.xml"),
        )
        controller = robot.cartesianPosQuatTrackingController
        # controller = robot.jointTrackingController
        # controller = GymCartesianVelController(robot, fixed_orientation=[0,1,0,0])

        super().__init__(
            scene=scene,
            controller=controller,
            max_steps_per_episode=max_steps_per_episode,
            n_substeps=n_substeps,
            debug=debug,
        )

        self.action_space = Box(
            low=np.array([-0.01, -0.01]), high=np.array([0.01, 0.01])
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(14, )
        )

        self.interactive = interactive

        self.random_env = random_env
        self.manager = BlockContextManager(scene, index=2)

        self.bp_cam = BPCageCam()
        self.inhand_cam = robot.inhand_cam

        self.push_box1 = push_box1
        self.push_box2 = push_box2
        self.push_box3 = push_box3

        self.target_box1 = target_box1
        self.target_box2 = target_box2
        self.target_box3 = target_box3

        self.maze_1 = maze[0]
        self.maze_2 = maze[1]
        self.maze_3 = maze[2]
        self.maze_4 = maze[3]
        self.maze_5 = maze[4]
        self.maze_6 = maze[5]
        self.maze_7 = maze[6]
        self.maze_8 = maze[7]
        self.maze_9 = maze[8]
        self.maze_10 = maze[9]
        self.maze_11 = maze[10]
        self.maze_12 = maze[11]
        self.maze_13 = maze[12]
        self.maze_14 = maze[13]
        self.maze_15 = maze[14]
        self.maze_16 = maze[15]
        self.maze_17 = maze[16]
        self.maze_18 = maze[17]
        self.maze_19 = maze[18]

        for obj in [
            self.push_box1,
            self.push_box2,
            self.push_box3,
            self.target_box1,
            self.target_box2,
            self.target_box3,
            # self.maze_1,
            # self.maze_2,
            self.maze_3,
            self.maze_4,
            self.maze_5,
            self.maze_6,
            self.maze_7,
            self.maze_8,
            self.maze_9,
            self.maze_10,
            self.maze_11,
            self.maze_12,
            self.maze_13,
            self.maze_14,
            self.maze_15,
            self.maze_16,
            self.maze_17,
            self.maze_18,
            self.maze_19
        ]:
            self.scene.add_object(obj)

        self.scene.add_object(self.bp_cam)

        self.log_dict = {
            "box-1": ObjectLogger(scene, self.push_box1),
            "box-target-1": ObjectLogger(scene, self.target_box1),
            "box-2": ObjectLogger(scene, self.push_box2),
            "box-target-2": ObjectLogger(scene, self.target_box2),
            "box-3": ObjectLogger(scene, self.push_box3),
            "box-target-3": ObjectLogger(scene, self.target_box3),
        }

        self.cam_dict = {
            "bp-cam": CamLogger(scene, self.bp_cam),
            "inhand-cam": CamLogger(scene, self.inhand_cam)
        }

        for _, v in self.log_dict.items():
            scene.add_logger(v)

        # for _, v in self.cam_dict.items():
        #     scene.add_logger(v)

        self.target_min_dist = 0.01
        self.bp_mode = None
        self.first_visit = -1

        self.modes = []
        self.mode_dict = {'rgb': 1, 'rbg': 2, 'grb': 3, 'gbr': 4, 'brg': 5, 'bgr': 6}


    def get_observation(self) -> np.ndarray:
        robot_pos = self.robot_state()[:2]

        box1_pos = self.scene.get_obj_pos(self.push_box1)[:2]  # - robot_pos
        box1_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.push_box1))[-1:])

        box2_pos = self.scene.get_obj_pos(self.push_box2)[:2]  # - robot_pos
        box2_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.push_box2))[-1:])

        box3_pos = self.scene.get_obj_pos(self.push_box3)[:2]  # - robot_pos
        box3_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.push_box3))[-1:])

        target_box1_pos = self.scene.get_obj_pos(self.target_box1)[:2]  # - robot_pos
        target_box2_pos = self.scene.get_obj_pos(self.target_box2)[:2]  # - robot_pos
        target_box3_pos = self.scene.get_obj_pos(self.target_box3)[:2]  # - robot_pos

        env_state = np.concatenate(
            [
                robot_pos,
                box1_pos,
                box1_quat,
                box2_pos,
                box2_quat,
                box3_pos,
                box3_quat,
                # target_box1_pos,
                # target_box2_pos,
                # target_box3_pos
            ]
        )

        return env_state.astype(np.float32)
        # return np.concatenate([robot_state, env_state])

    def start(self):
        self.scene.start()

        # reset view of the camera
        if self.scene.viewer is not None:
            self.scene.viewer.cam.elevation = -60
            self.scene.viewer.cam.distance = 1.8
            self.scene.viewer.cam.lookat[0] += -0.19
            self.scene.viewer.cam.lookat[2] -= 0.2

            # self.scene.viewer.cam.elevation = -60
            # self.scene.viewer.cam.distance = 1.6
            # self.scene.viewer.cam.lookat[0] += 0.05
            # self.scene.viewer.cam.lookat[2] -= 0.1

        # reset the initial state of the robot
        initial_cart_position = copy.deepcopy(init_end_eff_pos)
        # initial_cart_position[2] = 0.12
        self.robot.gotoCartPosQuatController.setDesiredPos(
            [
                initial_cart_position[0],
                initial_cart_position[1],
                initial_cart_position[2],
                0,
                1,
                0,
                0,
            ]
        )
        self.robot.gotoCartPosQuatController.initController(self.robot, 1)

        self.robot.init_qpos = self.robot.gotoCartPosQuatController.trajectory[
            -1
        ].copy()
        self.robot.init_tcp_pos = initial_cart_position
        self.robot.init_tcp_quat = [0, 1, 0, 0]

        self.robot.beam_to_joint_pos(
            self.robot.gotoCartPosQuatController.trajectory[-1]
        )
        # self.robot.gotoJointPosition(self.robot.init_qpos, duration=0.05)
        # self.robot.wait(duration=2.0)

        self.robot.gotoCartPositionAndQuat(
            desiredPos=initial_cart_position, desiredQuat=[0, 1, 0, 0], duration=0.5, log=False
        )

    def step(self, action, gripper_width=None, desired_vel=None, desired_acc=None):
        observation, reward, done, _ = super().step(action, gripper_width, desired_vel=desired_vel, desired_acc=desired_acc)
        self.success = self._check_early_termination()
        mean_distance = self.check_mean_dist()

        mode = self.check_mode()
        mode = ''.join(mode)

        one_box = 0
        two_box = 0
        three_box = 0

        if len(mode) == 1:
            one_box = 1
        elif len(mode) == 2:
            one_box = 1
            two_box = 1
        elif len(mode) == 3:
            one_box = 1
            two_box = 1
            three_box = 1

        return observation, reward, done, {'success':  self.success,
                                           'mean_distance': mean_distance,
                                           'mode': self.mode_dict[mode] if len(mode) == 3 else 0,
                                           'one_box_success': one_box,
                                           'two_box_success': two_box,
                                           'three_box_success': three_box}

    def check_mode(self):

        box1_pos = self.scene.get_obj_pos(self.push_box1)
        box2_pos = self.scene.get_obj_pos(self.push_box2)
        box3_pos = self.scene.get_obj_pos(self.push_box3)

        target_box1_pos = self.scene.get_obj_pos(self.target_box1)
        target_box2_pos = self.scene.get_obj_pos(self.target_box2)
        target_box3_pos = self.scene.get_obj_pos(self.target_box3)

        dist_box1_target1, _ = obj_distance(box1_pos, target_box1_pos)
        dist_box2_target2, _ = obj_distance(box2_pos, target_box2_pos)
        dist_box3_target3, _ = obj_distance(box3_pos, target_box3_pos)

        if dist_box1_target1 <= self.target_min_dist and 'r' not in self.modes:
            self.modes.append('r')

        if dist_box2_target2 <= self.target_min_dist and 'g' not in self.modes:
            self.modes.append('g')

        if dist_box3_target3 <= self.target_min_dist and 'b' not in self.modes:
            self.modes.append('b')

        return self.modes

    def check_mean_dist(self):
        box1_pos = self.scene.get_obj_pos(self.push_box1)
        box2_pos = self.scene.get_obj_pos(self.push_box2)
        box3_pos = self.scene.get_obj_pos(self.push_box3)

        target_box1_pos = self.scene.get_obj_pos(self.target_box1)
        target_box2_pos = self.scene.get_obj_pos(self.target_box2)
        target_box3_pos = self.scene.get_obj_pos(self.target_box3)

        dist_box1_target1, _ = obj_distance(box1_pos, target_box1_pos)
        dist_box2_target2, _ = obj_distance(box2_pos, target_box2_pos)
        dist_box3_target3, _ = obj_distance(box3_pos, target_box3_pos)

        mean_distance = (dist_box1_target1 + dist_box2_target2 + dist_box3_target3) / 3
        return mean_distance

    def get_reward(self, if_sparse=False):
        if if_sparse:
            return 0

        robot_pos = self.robot_state()[:2]

        box1_pos = self.scene.get_obj_pos(self.push_box1)
        box2_pos = self.scene.get_obj_pos(self.push_box2)
        box3_pos = self.scene.get_obj_pos(self.push_box3)

        target_box1_pos = self.scene.get_obj_pos(self.target_box1)
        target_box2_pos = self.scene.get_obj_pos(self.target_box2)
        target_box3_pos = self.scene.get_obj_pos(self.target_box3)

        dist_robot_box1, _ = obj_distance(robot_pos, box1_pos[:2])
        dist_robot_box2, _ = obj_distance(robot_pos, box2_pos[:2])
        dist_robot_box3, _ = obj_distance(robot_pos, box3_pos[:2])
        min_robot_box = min(dist_robot_box1, dist_robot_box2, dist_robot_box3)

        dist_box1_target1, _ = obj_distance(box1_pos, target_box1_pos)
        dist_box2_target2, _ = obj_distance(box2_pos, target_box2_pos)
        dist_box3_target3, _ = obj_distance(box3_pos, target_box3_pos)

        return (-1) * (min_robot_box + dist_box1_target1 + dist_box2_target2 + dist_box3_target3)

    def _check_early_termination(self) -> bool:
        # calculate the distance from end effector to object
        box1_pos = self.scene.get_obj_pos(self.push_box1)
        box2_pos = self.scene.get_obj_pos(self.push_box2)
        box3_pos = self.scene.get_obj_pos(self.push_box3)

        target_box1_pos = self.scene.get_obj_pos(self.target_box1)
        target_box2_pos = self.scene.get_obj_pos(self.target_box2)
        target_box3_pos = self.scene.get_obj_pos(self.target_box3)

        dist_box1_target1, _ = obj_distance(box1_pos, target_box1_pos)
        dist_box2_target2, _ = obj_distance(box2_pos, target_box2_pos)
        dist_box3_target3, _ = obj_distance(box3_pos, target_box3_pos)

        if dist_box1_target1 <= self.target_min_dist and dist_box2_target2 <= self.target_min_dist \
                and dist_box3_target3 <= self.target_min_dist:
            # terminate if end effector is close enough
            self.terminated = True
            return True

        return False

    def reset(self, random=True, context=None):
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1
        self.first_visit = -1

        self.modes = []

        self.bp_mode = None
        obs = self._reset_env(random=random, context=context)

        return obs

    def _reset_env(self, random=True, context=None):

        if self.interactive:
            for log_name, s in self.cam_dict.items():
                s.reset()

            for log_name, s in self.log_dict.items():
                s.reset()

        self.scene.reset()
        self.robot.beam_to_joint_pos(self.robot.init_qpos)
        self.manager.start(random=random, context=context)
        self.scene.next_step(log=False)

        observation = self.get_observation()

        return observation