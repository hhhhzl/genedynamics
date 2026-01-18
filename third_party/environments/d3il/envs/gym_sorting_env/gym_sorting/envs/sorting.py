import random

import cv2
import numpy as np
import copy

from gym.spaces import Box

from environments.d3il.d3il_sim.utils.sim_path import d3il_path
from environments.d3il.d3il_sim.core import Scene
from environments.d3il.d3il_sim.core.logger import ObjectLogger, CamLogger
from environments.d3il.d3il_sim.gyms.gym_env_wrapper import GymEnvWrapper
from environments.d3il.d3il_sim.utils.geometric_transformation import euler2quat, quat2euler

from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot
from environments.d3il.d3il_sim.sims.mj_beta.MjFactory import MjFactory
from environments.d3il.d3il_sim.sims import MjCamera

from .objects.sorting_objects import get_obj_list, init_end_eff_pos

red_boxes, blue_boxes, target_list = get_obj_list()


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
    def __init__(self, scene, index=0, num_boxes=2, seed=42) -> None:
        self.scene = scene

        np.random.seed(seed)

        self.box1_space = Box(
            low=np.array([0.4, -0.15, -90]), high=np.array([0.5, -0.1, 90])  # , seed=seed
        )

        self.box2_space = Box(
            low=np.array([0.4, -0.05, -90]), high=np.array([0.5, 0, 90])  # , seed=seed
        )

        self.box3_space = Box(
            low=np.array([0.4, 0.05, -90]), high=np.array([0.5, 0.1, 90])  # , seed=seed
        )

        self.box4_space = Box(
            low=np.array([0.55, -0.15, -90]), high=np.array([0.65, -0.1, 90])  # , seed=seed
        )

        self.box5_space = Box(
            low=np.array([0.55, -0.05, -90]), high=np.array([0.65, 0, 90])  # , seed=seed
        )

        self.box6_space = Box(
            low=np.array([0.55, 0.05, -90]), high=np.array([0.65, 0.1, 90])  # , seed=seed
        )

        self.index = index
        self.num_boxes = num_boxes

    def start(self, random=True, context=None):

        if random:
            self.context = self.sample()
        else:
            self.context = context

        self.set_context(self.context)

    def sample(self):

        pos_1 = self.box1_space.sample()
        angle_1 = [0, 0, pos_1[-1] * np.pi / 180]
        quat_1 = euler2quat(angle_1)

        pos_2 = self.box2_space.sample()
        angle_2 = [0, 0, pos_2[-1] * np.pi / 180]
        quat_2 = euler2quat(angle_2)

        pos_3 = self.box3_space.sample()
        angle_3 = [0, 0, pos_3[-1] * np.pi / 180]
        quat_3 = euler2quat(angle_3)

        pos_4 = self.box4_space.sample()
        angle_4 = [0, 0, pos_4[-1] * np.pi / 180]
        quat_4 = euler2quat(angle_4)

        pos_5 = self.box5_space.sample()
        angle_5 = [0, 0, pos_5[-1] * np.pi / 180]
        quat_5 = euler2quat(angle_5)

        pos_6 = self.box6_space.sample()
        angle_6 = [0, 0, pos_6[-1] * np.pi / 180]
        quat_6 = euler2quat(angle_6)

        contexts = [[pos_1, quat_1], [pos_2, quat_2], [pos_3, quat_3],
                    [pos_4, quat_4], [pos_5, quat_5], [pos_6, quat_6]]

        random.shuffle(contexts)

        return contexts

    def set_context(self, context):

        if self.num_boxes == 2:
            box_range = [0, 1]

            for i, num in enumerate(box_range[:1]):
                pos, quat = context[num]

                self.scene.set_obj_pos_and_quat(
                    [pos[0], pos[1], 0.05],
                    quat,
                    obj_name="red_" + str(i + 1),
                )

            for i, num in enumerate(box_range[1:]):
                pos, quat = context[num]

                self.scene.set_obj_pos_and_quat(
                    [pos[0], pos[1], 0.05],
                    quat,
                    obj_name="blue_" + str(i + 1),
                )
        #######################################################
        elif self.num_boxes == 4:
            box_range = [0, 1, 2, 3]

            for i, num in enumerate(box_range[:2]):
                pos, quat = context[num]

                self.scene.set_obj_pos_and_quat(
                    [pos[0], pos[1], 0.05],
                    quat,
                    obj_name="red_" + str(i + 1),
                )

            for i, num in enumerate(box_range[2:]):
                pos, quat = context[num]

                self.scene.set_obj_pos_and_quat(
                    [pos[0], pos[1], 0.05],
                    quat,
                    obj_name="blue_" + str(i + 1),
                )
        ########################################################
        elif self.num_boxes == 6:
            box_range = [0, 1, 2, 3, 4, 5]

            for i, num in enumerate(box_range[:3]):
                pos, quat = context[num]

                self.scene.set_obj_pos_and_quat(
                    [pos[0], pos[1], 0.05],
                    quat,
                    obj_name="red_" + str(i + 1),
                )

            for i, num in enumerate(box_range[3:]):
                pos, quat = context[num]

                self.scene.set_obj_pos_and_quat(
                    [pos[0], pos[1], 0.05],
                    quat,
                    obj_name="blue_" + str(i + 1),
                )
        #######################################################
        else:
            assert False

    def set_index(self, index):
        self.index = index


class Sorting_Env(GymEnvWrapper):
    def __init__(
        self,
        n_substeps: int = 35,
        max_steps_per_episode: int = 2e3,
        debug: bool = False,
        random_env: bool = False,
        interactive: bool = False,
        render: bool = True,
        num_boxes: int = 2,
        if_vision: bool = False
    ):

        sim_factory = MjFactory()
        render_mode = Scene.RenderMode.HUMAN if render else Scene.RenderMode.BLIND

        if num_boxes == 2:
            obj_list = red_boxes[:1] + blue_boxes[:1] + target_list
        elif num_boxes == 4:
            obj_list = red_boxes[:2] + blue_boxes[:2] + target_list
        elif num_boxes == 6:
            obj_list = red_boxes + blue_boxes + target_list
        else:
            assert False, "no such num_boxes"

        scene = sim_factory.create_scene(
            object_list=obj_list, render=render_mode, dt=0.001
        )
        robot = MjRobot(
            scene,
            xml_path=d3il_path("./models/mj/robot/panda_rod_invisible.xml"),
        )
        controller = robot.cartesianPosQuatTrackingController

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
        self.manager = BlockContextManager(scene, index=0, num_boxes=num_boxes)

        self.bp_cam = BPCageCam()
        self.inhand_cam = robot.inhand_cam

        self.scene.add_object(self.bp_cam)

        self.red_box_1 = red_boxes[0]
        self.red_box_2 = red_boxes[1]
        self.red_box_3 = red_boxes[2]

        self.blue_box_1 = blue_boxes[0]
        self.blue_box_2 = blue_boxes[1]
        self.blue_box_3 = blue_boxes[2]

        if num_boxes == 2:
            self.log_dict = {
                "red-box1": ObjectLogger(scene, self.red_box_1),
                "blue-box1": ObjectLogger(scene, self.blue_box_1),
            }
        elif num_boxes == 4:
            self.log_dict = {
                "red-box1": ObjectLogger(scene, self.red_box_1),
                "red-box2": ObjectLogger(scene, self.red_box_2),
                "blue-box1": ObjectLogger(scene, self.blue_box_1),
                "blue-box2": ObjectLogger(scene, self.blue_box_2),
            }
        elif num_boxes == 6:
            self.log_dict = {
                "red-box1": ObjectLogger(scene, self.red_box_1),
                "red-box2": ObjectLogger(scene, self.red_box_2),
                "red-box3": ObjectLogger(scene, self.red_box_3),
                "blue-box1": ObjectLogger(scene, self.blue_box_1),
                "blue-box2": ObjectLogger(scene, self.blue_box_2),
                "blue-box3": ObjectLogger(scene, self.blue_box_3),
            }
        else:
            assert False, "no such num boxes"

        self.num_boxes = num_boxes

        self.cam_dict = {
            "bp-cam": CamLogger(scene, self.bp_cam),
            "inhand-cam": CamLogger(scene, self.inhand_cam)
        }

        for _, v in self.log_dict.items():
            scene.add_logger(v)

        for _, v in self.cam_dict.items():
            scene.add_logger(v)

        self.target_min_dist = 0.05

        self.red_target_pos = np.array([0.4, 0.32])
        self.blue_target_pos = np.array([0.625, 0.32])

        self.mode = np.array([-1, -1, -1, -1, -1, -1])
        self.mode_step = 0
        self.min_inds = []

    def get_observation(self) -> np.ndarray:

        robot_pos = self.robot_state()[:2]

        if self.if_vision:

            bp_image = self.bp_cam.get_image(depth=False)
            bp_image = cv2.cvtColor(bp_image, cv2.COLOR_RGB2BGR)

            inhand_image = self.inhand_cam.get_image(depth=False)
            inhand_image = cv2.cvtColor(inhand_image, cv2.COLOR_RGB2BGR)

            return robot_pos, bp_image, inhand_image

        red_box_1_pos = self.scene.get_obj_pos(self.red_box_1)[:2]
        red_box_1_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.red_box_1))[-1:])

        red_box_2_pos = self.scene.get_obj_pos(self.red_box_2)[:2]
        red_box_2_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.red_box_2))[-1:])

        red_box_3_pos = self.scene.get_obj_pos(self.red_box_3)[:2]
        red_box_3_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.red_box_3))[-1:])

        blue_box_1_pos = self.scene.get_obj_pos(self.blue_box_1)[:2]
        blue_box_1_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.blue_box_1))[-1:])

        blue_box_2_pos = self.scene.get_obj_pos(self.blue_box_2)[:2]
        blue_box_2_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.blue_box_2))[-1:])

        blue_box_3_pos = self.scene.get_obj_pos(self.blue_box_3)[:2]
        blue_box_3_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.blue_box_3))[-1:])

        if self.num_boxes == 2:

            env_state = np.concatenate(
                [
                    robot_pos,
                    red_box_1_pos,
                    red_box_1_quat,
                    blue_box_1_pos,
                    blue_box_1_quat,
                ]
            )
        elif self.num_boxes == 4:

            env_state = np.concatenate(
                [
                    robot_pos,
                    red_box_1_pos,
                    red_box_1_quat,
                    red_box_2_pos,
                    red_box_2_quat,
                    blue_box_1_pos,
                    blue_box_1_quat,
                    blue_box_2_pos,
                    blue_box_2_quat,
                ]
            )

        elif self.num_boxes == 6:

            env_state = np.concatenate(
                [
                    robot_pos,
                    red_box_1_pos,
                    red_box_1_quat,
                    red_box_2_pos,
                    red_box_2_quat,
                    red_box_3_pos,
                    red_box_3_quat,
                    blue_box_1_pos,
                    blue_box_1_quat,
                    blue_box_2_pos,
                    blue_box_2_quat,
                    blue_box_3_pos,
                    blue_box_3_quat
                ]
            )

        else:
            assert False, "no such num boxes"

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
            self.scene.viewer.cam.lookat[0] += -0.01
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
        mode, min_inds = self.check_mode()

        if self.num_boxes == 2:
            mode = mode[:2]
        elif self.num_boxes == 4:
            mode = mode[:4]
        elif self.num_boxes == 6:
            mode = mode[:6]

        mode = self.decode_mode(mode)

        return observation, reward, done, {'mode': mode, 'success':  self.success, 'min_inds': min_inds}

    def decode_mode(self, mode):

       return int(np.packbits(mode)[0])

    def check_mode(self):

        if self.mode_step > 5:
            return self.mode, self.min_inds

        red_box_1_pos = self.scene.get_obj_pos(self.red_box_1)[:2]
        red_box_2_pos = self.scene.get_obj_pos(self.red_box_2)[:2]
        red_box_3_pos = self.scene.get_obj_pos(self.red_box_3)[:2]

        blue_box_1_pos = self.scene.get_obj_pos(self.blue_box_1)[:2]
        blue_box_2_pos = self.scene.get_obj_pos(self.blue_box_2)[:2]
        blue_box_3_pos = self.scene.get_obj_pos(self.blue_box_3)[:2]

        box_pos = np.vstack((red_box_1_pos, red_box_2_pos, red_box_3_pos, blue_box_1_pos, blue_box_2_pos, blue_box_3_pos))

        red_dist = np.linalg.norm(box_pos[:3] - np.reshape(self.red_target_pos, (1, -1)), axis=-1)
        blue_dist = np.linalg.norm(box_pos[3:] - np.reshape(self.blue_target_pos, (1, -1)), axis=-1)

        dists = np.concatenate((red_dist, blue_dist))
        dists[self.min_inds] = 100000

        min_ind = np.argmin(dists)
        min_box_pos = box_pos[min_ind]

        if min_ind < 3:
            # manipulate red box, 0
            if_finish = min_box_pos[0] > 0.3 and min_box_pos[0] < 0.5 and min_box_pos[1] > 0.22 and min_box_pos[1] < 0.41

            if if_finish:
                self.mode[self.mode_step] = 0
                self.mode_step += 1

                self.min_inds.append(min_ind)
        else:
            # manipulate blue box
            if_finish = min_box_pos[0] > 0.525 and min_box_pos[0] < 0.725 and min_box_pos[1] > 0.22 and min_box_pos[1] < 0.41

            if if_finish:
                self.mode[self.mode_step] = 1
                self.mode_step += 1

                self.min_inds.append(min_ind)

        return self.mode, self.min_inds

    def get_reward(self, if_sparse=False):

        return 0

    def _check_early_termination(self) -> bool:

        red_box_1_pos = self.scene.get_obj_pos(self.red_box_1)[:2]
        red_box_2_pos = self.scene.get_obj_pos(self.red_box_2)[:2]
        red_box_3_pos = self.scene.get_obj_pos(self.red_box_3)[:2]

        blue_box_1_pos = self.scene.get_obj_pos(self.blue_box_1)[:2]
        blue_box_2_pos = self.scene.get_obj_pos(self.blue_box_2)[:2]
        blue_box_3_pos = self.scene.get_obj_pos(self.blue_box_3)[:2]

        if self.num_boxes == 2:
            red = np.expand_dims(red_box_1_pos, axis=0)
            blue = np.expand_dims(blue_box_1_pos, axis=0)
        elif self.num_boxes == 4:
            red = np.vstack((red_box_1_pos, red_box_2_pos))
            blue = np.vstack((blue_box_1_pos, blue_box_2_pos))
        elif self.num_boxes == 6:
            red = np.vstack((red_box_1_pos, red_box_2_pos, red_box_3_pos))
            blue = np.vstack((blue_box_1_pos, blue_box_2_pos, blue_box_3_pos))
        else:
            assert False, "no such num boxes"

        red_finished = (red[:, 0] > 0.3).all() and (red[:, 0] < 0.5).all() and (red[:, 1] > 0.22).all() and (red[:, 1] < 0.41).all()
        blue_finished = (blue[:, 0] > 0.525).all() and (blue[:, 0] < 0.725).all() and (blue[:, 1] > 0.22).all() and (blue[:, 1] < 0.41).all()

        if red_finished and blue_finished:
            # terminate if end effector is close enough
            self.terminated = True
            return True

        return False

    def reset(self, random=True, context=None, if_vision=False):
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1

        self.mode = np.array([-1, -1, -1, -1, -1, -1])
        self.mode_step = 0
        self.min_inds = []

        self.bp_mode = None
        obs = self._reset_env(random=random, context=context, if_vision=if_vision)

        return obs

    def _reset_env(self, random=True, context=None, if_vision=False):

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