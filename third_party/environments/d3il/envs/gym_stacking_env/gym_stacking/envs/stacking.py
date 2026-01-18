
import copy

import cv2
import numpy as np
from gym.spaces import Box

from environments.d3il.d3il_sim.core import Scene
from environments.d3il.d3il_sim.core.logger import CamLogger, ObjectLogger
from environments.d3il.d3il_sim.gyms.gym_env_wrapper import GymEnvWrapper
from environments.d3il.d3il_sim.gyms.gym_utils.helpers import obj_distance
from environments.d3il.d3il_sim.sims import MjCamera
from environments.d3il.d3il_sim.sims.mj_beta.MjFactory import MjFactory
from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot
from environments.d3il.d3il_sim.utils.geometric_transformation import (
    euler2quat, quat2euler)
from environments.d3il.d3il_sim.utils.sim_path import d3il_path

from .objects.stacking_objects import get_obj_list, init_end_eff_pos

obj_list = get_obj_list()


class BPCageCam(MjCamera):
    """
    Cage camera. Extends the camera base class.
    """

    def __init__(self, width: int = 96, height: int = 96, *args, **kwargs):
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

        self.red_space = Box(
            low=np.array([0.35, -0.25, -90]), high=np.array([0.45, -0.15, 90])  # , seed=seed
        )

        self.green_space = Box(
            low=np.array([0.35, -0.1, -90]), high=np.array([0.45, 0, 90])  # , seed=seed
        )

        self.blue_space = Box(
            low=np.array([0.55, -0.2, -90]), high=np.array([0.6, 0, 90])  # , seed=seed
        )

        self.target_space = Box(
            low=np.array([0.4, 0.15, -90]), high=np.array([0.6, 0.25, 90])  # , seed=seed
        )

        self.index = index

    def start(self, random=True, context=None):

        if random:
            self.context = self.sample()
        else:
            self.context = context

        self.set_context(self.context)

    def sample(self):

        pos_1 = self.red_space.sample()
        angle_1 = [0, 0, pos_1[-1] * np.pi / 180]
        quat_1 = euler2quat(angle_1)

        pos_2 = self.green_space.sample()
        angle_2 = [0, 0, pos_2[-1] * np.pi / 180]
        quat_2 = euler2quat(angle_2)

        pos_3 = self.blue_space.sample()
        angle_3 = [0, 0, pos_3[-1] * np.pi / 180]
        quat_3 = euler2quat(angle_3)

        pos_4 = self.target_space.sample()
        angle_4 = [0, 0, pos_4[-1] * np.pi / 180]
        quat_4 = euler2quat(angle_4)

        return [pos_1, quat_1], [pos_2, quat_2], [pos_3, quat_3], [pos_4, quat_4]

    def set_context(self, context):

        red_pos = context[0][0]
        red_quat = context[0][1]

        green_pos = context[1][0]
        green_quat = context[1][1]

        blue_pos = context[2][0]
        blue_quat = context[2][1]

        target_pos = context[3][0]
        target_quat = context[3][1]

        self.scene.set_obj_pos_and_quat(
            [red_pos[0], red_pos[1], 0],
            red_quat,
            obj_name="red_box",
        )

        self.scene.set_obj_pos_and_quat(
            [green_pos[0], green_pos[1], 0],
            green_quat,
            obj_name="green_box",
        )

        self.scene.set_obj_pos_and_quat(
            [blue_pos[0], blue_pos[1], 0],
            blue_quat,
            obj_name="blue_box",
        )

    def set_index(self, index):
        self.index = index


class CubeStacking_Env(GymEnvWrapper):
    def __init__(
        self,
        n_substeps: int = 30,
        max_steps_per_episode: int = 50000,
        debug: bool = False,
        random_env: bool = False,
        interactive: bool = False,
        render: bool = True,
        if_vision: bool = False
    ):

        sim_factory = MjFactory()
        render_mode = Scene.RenderMode.HUMAN if render else Scene.RenderMode.BLIND
        scene = sim_factory.create_scene(
            object_list=obj_list, render=render_mode, dt=0.001
        )
        robot = MjRobot(
            scene,
            xml_path=d3il_path("./models/mj/robot/panda_invisible.xml"),
        )
        controller = robot.jointTrackingController

        super().__init__(
            scene=scene,
            controller=controller,
            max_steps_per_episode=max_steps_per_episode,
            n_substeps=n_substeps,
            debug=debug,
        )

        self.if_vision = if_vision

        self.action_space = Box(
            low=np.array([-0.01, -0.01]), high=np.array([0.01, 0.01])
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(8, )
        )

        self.interactive = interactive

        self.random_env = random_env
        self.manager = BlockContextManager(scene, index=0)

        self.bp_cam = BPCageCam()
        self.inhand_cam = robot.inhand_cam

        self.scene.add_object(self.bp_cam)

        self.red_box = obj_list[0]
        self.green_box = obj_list[1]
        self.blue_box = obj_list[2]
        self.target_box = obj_list[3]

        self.log_dict = {
            "red-box": ObjectLogger(scene, self.red_box),
            "green-box": ObjectLogger(scene, self.green_box),
            "blue-box": ObjectLogger(scene, self.blue_box),
            "target-box": ObjectLogger(scene, self.target_box),
        }

        self.cam_dict = {
            "bp-cam": CamLogger(scene, self.bp_cam),
            "inhand-cam": CamLogger(scene, self.inhand_cam)
        }

        for _, v in self.log_dict.items():
            scene.add_logger(v)

        for _, v in self.cam_dict.items():
            scene.add_logger(v)

        self.pos_min_dist = 0.06

        self.min_inds = []
        self.mode_encoding = []

    def robot_state(self):
        # Update Robot State
        self.robot.receiveState()

        # joint state
        joint_pos = self.robot.current_j_pos
        joint_vel = self.robot.current_j_vel
        gripper_width = np.array([self.robot.gripper_width])

        tcp_pos = self.robot.current_c_pos
        tcp_quad = self.robot.current_c_quat

        return np.concatenate((joint_pos, gripper_width)), joint_pos, tcp_quad
        # return np.concatenate((tcp_pos, tcp_quad, gripper_width))

    def get_observation(self) -> np.ndarray:

        j_state, robot_c_pos, robot_c_quat = self.robot_state()

        if self.if_vision:

            bp_image = self.bp_cam.get_image(depth=False)
            bp_image = cv2.cvtColor(bp_image, cv2.COLOR_RGB2BGR)

            inhand_image = self.inhand_cam.get_image(depth=False)
            inhand_image = cv2.cvtColor(inhand_image, cv2.COLOR_RGB2BGR)

            return j_state, bp_image, inhand_image

        # robot_state = self.robot_state()

        red_box_pos = self.scene.get_obj_pos(self.red_box)
        red_box_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.red_box))[-1:])
        # red_box_quat = np.concatenate((np.sin(red_box_quat), np.cos(red_box_quat)))

        green_box_pos = self.scene.get_obj_pos(self.green_box)
        green_box_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.green_box))[-1:])
        # green_box_quat = np.concatenate((np.sin(green_box_quat), np.cos(green_box_quat)))

        blue_box_pos = self.scene.get_obj_pos(self.blue_box)
        blue_box_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.blue_box))[-1:])
        # blue_box_quat = np.concatenate((np.sin(blue_box_quat), np.cos(blue_box_quat)))

        target_pos = self.scene.get_obj_pos(self.target_box) #- robot_c_pos
        target_quat = self.scene.get_obj_quat(self.target_box)

        env_state = np.concatenate(
            [
                # robot_state[:-1],
                # quat2euler(robot_c_quat),
                # joint_state[:-1],
                # gripper_width,
                # robot_c_pos,
                # robot_c_quat,
                red_box_pos,
                red_box_quat,
                green_box_pos,
                green_box_quat,
                blue_box_pos,
                blue_box_quat,
                # target_pos,
            ]
        )

        return env_state.astype(np.float32)

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

        j_pos = action[:7]
        # j_vel = action[7:14]
        gripper_width = action[-1]

        if gripper_width > 0.075:

            self.robot.open_fingers()

            # if self.gripper_flag == 0:
            #     print(0)
            #     self.robot.open_fingers()
            #     self.gripper_flag = 1
        else:
            self.robot.close_fingers(duration=0.0)
            # if self.gripper_flag == 1:
            #
            #     print(1)
            #     self.robot.close_fingers(duration=0.5)
            #     print(self.robot.set_gripper_width)
            #
            #     self.gripper_flag = 0

        # self.robot.set_gripper_width = gripper_width

        # c_pos, c_quat = self.robot.getForwardKinematics(action)
        # c_action = np.concatenate((c_pos, c_quat))

        # c_pos = action[:3]
        # c_quat = euler2quat(action[3:6])
        # c_action = np.concatenate((c_pos, c_quat))

        self.controller.setSetPoint(action[:-1])#, desired_vel=desired_vel, desired_acc=desired_acc)
        # self.controller.setSetPoint(action)#, desired_vel=j_vel, desired_acc=desired_acc)
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

        self.success = self._check_early_termination()
        mode_encoding, mean_distance = self.check_mode()

        mode = ''
        mode = mode.join(mode_encoding)

        return observation, reward, done, {'mode': mode,
                                           'success':  self.success,
                                           'success_1': len(mode) > 0,
                                           'success_2': len(mode) > 1,
                                           'mean_distance': mean_distance}

    def check_mode(self):

        modes = ['r', 'g', 'b']

        red_pos = self.scene.get_obj_pos(self.red_box)[:2]
        green_pos = self.scene.get_obj_pos(self.green_box)[:2]
        blue_pos = self.scene.get_obj_pos(self.blue_box)[:2]

        target_pos = self.scene.get_obj_pos(self.target_box)[:2]

        box_pos = np.vstack((red_pos, green_pos, blue_pos))

        dists = np.linalg.norm(box_pos - np.reshape(target_pos, (1, -1)), axis=-1)
        mean_dists = np.mean(dists)

        dists[self.min_inds] = 100000

        min_ind = np.argmin(dists)

        if dists[min_ind] <= self.pos_min_dist:

            self.mode_encoding.append(modes[min_ind])
            self.min_inds.append(min_ind)

        return self.mode_encoding, mean_dists

    def get_reward(self, if_sparse=False):

        return 0

    def _check_early_termination(self) -> bool:

        red_pos = self.scene.get_obj_pos(self.red_box)
        green_pos = self.scene.get_obj_pos(self.green_box)
        blue_pos = self.scene.get_obj_pos(self.blue_box)

        diff_z = min([np.linalg.norm(red_pos[-1]-green_pos[-1]),
                      np.linalg.norm(red_pos[-1] - blue_pos[-1]),
                      np.linalg.norm(green_pos[-1] - blue_pos[-1])])

        target_pos = self.scene.get_obj_pos(self.target_box)[:2]

        dis_rt, _ = obj_distance(red_pos[:2], target_pos)
        dis_gt, _ = obj_distance(green_pos[:2], target_pos)
        dis_bt, _ = obj_distance(blue_pos[:2], target_pos)

        if (dis_rt <= self.pos_min_dist) and (dis_gt <= self.pos_min_dist) \
                and (dis_bt <= self.pos_min_dist) and (diff_z > 0.03):
            # terminate if end effector is close enough
            self.terminated = True
            return True

        return False

    def reset(self, random=True, context=None):
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1

        self.min_inds = []
        self.mode_encoding = []

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

        self.robot.open_fingers()

        self.manager.start(random=random, context=context)
        self.scene.next_step(log=False)

        observation = self.get_observation()

        return observation
