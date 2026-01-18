import numpy as np
import copy
import time

import sys

from gym.spaces import Box

from environments.d3il.d3il_sim.utils.sim_path import d3il_path
from environments.d3il.d3il_sim.core import Scene
from environments.d3il.d3il_sim.core.logger import ObjectLogger, CamLogger
from environments.d3il.d3il_sim.gyms.gym_env_wrapper import GymEnvWrapper
from environments.d3il.d3il_sim.gyms.gym_utils.helpers import obj_distance
from environments.d3il.d3il_sim.utils.geometric_transformation import euler2quat, quat2euler

from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot
from environments.d3il.d3il_sim.sims.mj_beta.MjFactory import MjFactory
from environments.d3il.d3il_sim.sims import MjCamera

from .objects.pushing_objects import get_obj_list, init_end_eff_pos

obj_list, push_box1, push_box2, target_box_1, target_box_2 = get_obj_list()


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

        self.red_box_space = Box(
            low=np.array([0.4, -0.15, -90]), high=np.array([0.5, 0, 90])#, seed=seed
        )
        self.green_box_space = Box(
            low=np.array([0.55, -0.15, -90]), high=np.array([0.65, 0, 90])#, seed=seed
        )

        # self.deg_list = np.random.random_sample(60) * 180 - 90
        # self.x1_list = np.random.random_sample(30) * 0.1 + 0.4
        # self.x2_list = np.random.random_sample(30) * 0.1 + 0.55
        # self.y_list = np.random.random_sample(60) * 0.2 - 0.15

        # Reduced context space size
        self.deg_list = np.random.random_sample(60) * 90 - 45
        self.x1_list = np.random.random_sample(30) * 0.1 + 0.4
        self.x2_list = np.random.random_sample(30) * 0.1 + 0.55
        self.y_list = np.random.random_sample(60) * 0.15 - 0.15

        # 0, rr_gg
        # 1, gg_rr
        # 2, rg_gr
        # 3, gr_rg
        self.index = index

    def start(self, random=True, context=None):

        if random:
            self.context = self.sample()
        else:
            self.context = context

        self.set_context(self.context)

    def sample(self):

        red_pos = self.red_box_space.sample()
        green_pos = self.green_box_space.sample()

        goal_angle = [0, 0, red_pos[-1] * np.pi / 180]
        quat = euler2quat(goal_angle)

        goal_angle2 = [0, 0, green_pos[-1] * np.pi / 180]
        quat2 = euler2quat(goal_angle2)

        return [red_pos, quat, green_pos, quat2]

    def set_context(self, context):

        red_pos, quat, green_pos, quat2 = context

        self.scene.set_obj_pos_and_quat(
            [red_pos[0], red_pos[1], 0.00],
            quat,
            obj_name="push_box",
        )

        self.scene.set_obj_pos_and_quat(
            [green_pos[0], green_pos[1], 0.00],
            quat2,
            obj_name="push_box2",
        )
        # print(
        #     "Set Context red_pos: {}, quat_r: {}, green_pos: {}, quat_g: {}".format(
        #         red_pos, quat, green_pos, quat2
        #     )
        # )

    def random_context(self):

        red_pos = self.red_box_space.sample()
        green_pos = self.green_box_space.sample()

        goal_angle = [0, 0, red_pos[-1] * np.pi / 180]
        quat = euler2quat(goal_angle)

        self.scene.set_obj_pos_and_quat(
            [red_pos[0], red_pos[1], 0.00],
            quat,
            obj_name="push_box",
        )

        goal_angle2 = [0, 0, green_pos[-1] * np.pi / 180]
        quat2 = euler2quat(goal_angle2)
        self.scene.set_obj_pos_and_quat(
            [green_pos[0], green_pos[1], 0.00],
            quat2,
            obj_name="push_box2",
        )

        return red_pos, quat, green_pos, quat2

    def olb_set_context(self, index):
        goal_angle = [0, 0, self.deg_list[index] * np.pi / 180]
        quat = euler2quat(goal_angle)

        self.scene.set_obj_pos_and_quat(
            [self.x1_list[index], self.y_list[index], 0.00],
            quat,
            obj_name="push_box",
        )

        goal_angle2 = [0, 0, self.deg_list[len(self.x1_list) + index] * np.pi / 180]
        quat2 = euler2quat(goal_angle2)
        self.scene.set_obj_pos_and_quat(
            [self.x2_list[index], self.y_list[len(self.x1_list) + index], 0.00],
            quat2,
            obj_name="push_box2",
        )
        # print("Set Context {}".format(index))

    def next_context(self):
        self.index = (self.index + 1) % len(self.x1_list)
        self.olb_set_context(self.index)

    def set_index(self, index):
        self.index = index


class Block_Push_Env(GymEnvWrapper):
    def __init__(
        self,
        n_substeps: int = 35,
        max_steps_per_episode: int = 400,
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
        self.target_box_1 = target_box_1
        self.target_box_2 = target_box_2

        for obj in [
            self.push_box1,
            self.push_box2,
            self.target_box_1,
            self.target_box_2,
        ]:
            self.scene.add_object(obj)

        self.scene.add_object(self.bp_cam)

        self.log_dict = {
            "red-box": ObjectLogger(scene, self.push_box1),
            "red-target": ObjectLogger(scene, self.target_box_1),
            "green-box": ObjectLogger(scene, self.push_box2),
            "green-target": ObjectLogger(scene, self.target_box_2),
        }

        self.cam_dict = {
            "bp-cam": CamLogger(scene, self.bp_cam),
            "inhand-cam": CamLogger(scene, self.inhand_cam)
        }

        for _, v in self.log_dict.items():
            scene.add_logger(v)

        for _, v in self.cam_dict.items():
            scene.add_logger(v)

        self.target_min_dist = 0.05
        self.bp_mode = None
        self.first_visit = -1

    def get_observation(self) -> np.ndarray:

        robot_pos = self.robot_state()[:2]

        box_1_pos = self.scene.get_obj_pos(self.push_box1)[:2]  # - robot_pos
        box_1_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.push_box1))[-1:])

        box_2_pos = self.scene.get_obj_pos(self.push_box2)[:2]  # - robot_pos
        box_2_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.push_box2))[-1:])

        # goal_1_pos = self.scene.get_obj_pos(self.target_box_1)[:2]  # - robot_pos
        # goal_2_pos = self.scene.get_obj_pos(self.target_box_2)[:2]  # - robot_pos

        env_state = np.concatenate(
            [
                robot_pos,
                box_1_pos,
                box_1_quat,
                box_2_pos,
                box_2_quat,
                # goal_1_pos,
                # goal_2_pos,
            ]
        )

        return env_state.astype(np.float32)
        # return np.concatenate([robot_state, env_state])

    def start(self):
        self.scene.start()

        # reset view of the camera
        if self.scene.viewer is not None:
            # self.scene.viewer.cam.elevation = -55
            # self.scene.viewer.cam.distance = 1.7
            # self.scene.viewer.cam.lookat[0] += -0.1
            # self.scene.viewer.cam.lookat[2] -= 0.2

            self.scene.viewer.cam.elevation = -55
            self.scene.viewer.cam.distance = 2.0
            self.scene.viewer.cam.lookat[0] += 0
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
        mode, mean_distance = self.check_mode()
        return observation, reward, done, {'mode': mode, 'success':  self.success, 'mean_distance': mean_distance}

    def check_mode(self):
        box_1_pos = self.scene.get_obj_pos(self.push_box1)
        box_2_pos = self.scene.get_obj_pos(self.push_box2)
        goal_1_pos = self.scene.get_obj_pos(self.target_box_1)
        goal_2_pos = self.scene.get_obj_pos(self.target_box_2)

        dis_rr, _ = obj_distance(box_1_pos, goal_1_pos)
        dis_rg, _ = obj_distance(box_1_pos, goal_2_pos)
        dis_gr, _ = obj_distance(box_2_pos, goal_1_pos)
        dis_gg, _ = obj_distance(box_2_pos, goal_2_pos)
        visit = -1
        mode = -1

        if dis_rr <= self.target_min_dist and self.first_visit != 0:
            visit = 0
        elif dis_rg <= self.target_min_dist and self.first_visit != 1:
            visit = 1
        elif dis_gr <= self.target_min_dist and self.first_visit != 2:
            visit = 2
        elif dis_gg <= self.target_min_dist and self.first_visit != 3:
            visit = 3

        if self.first_visit == -1:
            self.first_visit = visit
        else:
            if self.first_visit == 0 and visit == 3:
                mode = 0  # rr -> gg
            elif self.first_visit == 3 and visit == 0:
                mode = 1  # gg -> rr
            elif self.first_visit == 1 and visit == 2:
                mode = 2  # rg -> gr
            elif self.first_visit == 2 and visit == 1:
                mode = 3  # gr -> rg

        mean_distance = 0.5 * (min(dis_rr, dis_rg) + min(dis_gr, dis_gg))

        return mode, mean_distance

    def get_reward(self, if_sparse=False):
        # return 0
        if if_sparse:
            return 0

        robot_pos = self.robot_state()[:2]

        box_1_pos = self.scene.get_obj_pos(self.push_box1)
        box_2_pos = self.scene.get_obj_pos(self.push_box2)
        goal_1_pos = self.scene.get_obj_pos(self.target_box_1)
        goal_2_pos = self.scene.get_obj_pos(self.target_box_2)

        dis_robot_box_r, _ = obj_distance(robot_pos, box_1_pos[:2])
        dis_robot_box_g, _ = obj_distance(robot_pos, box_2_pos[:2])

        dis_rr, _ = obj_distance(box_1_pos, goal_1_pos)
        dis_rg, _ = obj_distance(box_1_pos, goal_2_pos)
        dis_gr, _ = obj_distance(box_2_pos, goal_1_pos)
        dis_gg, _ = obj_distance(box_2_pos, goal_2_pos)

        # reward for pushing red box to red target area.

        # reward_red = 5 - (dis_robot_box_r + dis_rr)
        # reward_red = reward_red if reward_red > 0 else 0

        # reward_green = 100 - (dis_robot_box_g + dis_gg) * 3
        # reward_green = reward_green if reward_green > 50 else 50

        return (-1) * (dis_robot_box_r + dis_rr)

        # if dis_rr > self.target_min_dist:
        #     robot_factor = 1
        # else:
        #     robot_factor = 0
        #
        # return (-1) * (dis_robot_box_r * robot_factor + dis_rr - 2) + (-1) * (dis_robot_box_g * (1 - robot_factor) + dis_gg)

        # # reward for pushing two boxes
        # if dis_rr > self.target_min_dist:
        #
        #     self.reward_offset = dis_robot_box_g + dis_gg
        #
        #     return (-1) * (dis_robot_box_r + dis_rr)
        #
        # else:
        #     return (-1) * (dis_robot_box_g + dis_gg) + self.reward_offset

        dis_modes = np.array([dis_rr, dis_rg, dis_gr, dis_gg])

        min_ind = np.argmin(dis_modes)

        # four modes: [rr, gg], [rg, gr], [gr, rg], [gg, rr]
        min_dis = dis_modes[min_ind]
        if self.bp_mode is None and min_dis <= self.target_min_dist:
            self.bp_mode = min_ind

        if min_ind == 0 or min_ind == 3:
            return (-1) * (dis_rr + dis_gg)
        else:
            return (-1) * (dis_rg + dis_gr)

    def _check_early_termination(self) -> bool:
        # calculate the distance from end effector to object
        box_1_pos = self.scene.get_obj_pos(self.push_box1)
        box_2_pos = self.scene.get_obj_pos(self.push_box2)
        goal_1_pos = self.scene.get_obj_pos(self.target_box_1)
        goal_2_pos = self.scene.get_obj_pos(self.target_box_2)

        dis_rr, _ = obj_distance(box_1_pos, goal_1_pos)
        dis_rg, _ = obj_distance(box_1_pos, goal_2_pos)
        dis_gr, _ = obj_distance(box_2_pos, goal_1_pos)
        dis_gg, _ = obj_distance(box_2_pos, goal_2_pos)

        if (dis_rr <= self.target_min_dist and dis_gg <= self.target_min_dist) or (
            dis_rg <= self.target_min_dist and dis_gr <= self.target_min_dist
        ):
            # terminate if end effector is close enough
            self.terminated = True
            return True

        return False

    def reset(self, random=True, context=None):
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1
        self.first_visit = -1

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

        # if self.random_env:
        #     new_box1 = [self.push_box1, self.push_box1_space.sample()]
        #     new_box2 = [self.push_box2, self.push_box2_space.sample()]
        #
        #     self.scene.reset([new_box1, new_box2])
        # else:
        #     self.scene.reset()
