import cv2
import numpy as np
import copy

from gym.spaces import Box

from environments.d3il.d3il_sim.utils.sim_path import d3il_path
from environments.d3il.d3il_sim.core import Scene
from environments.d3il.d3il_sim.core.logger import ObjectLogger, CamLogger
from environments.d3il.d3il_sim.gyms.gym_env_wrapper import GymEnvWrapper
from environments.d3il.d3il_sim.utils.geometric_transformation import euler2quat

from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot
from environments.d3il.d3il_sim.sims.mj_beta.MjFactory import MjFactory
from environments.d3il.d3il_sim.sims import MjCamera

from .objects.aligning_objects import get_obj_list, init_end_eff_pos

obj_list = get_obj_list()


def rotation_distance(p: np.array, q: np.array):
    """
    Calculates the rotation angular between two quaternions
    param p: quaternion
    param q: quaternion
    theta: rotation angle between p and q (rad)
    """
    assert p.shape == q.shape, "p and q should be quaternion"
    theta = 2 * np.arccos(abs(p @ q))
    return theta


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

        self.box_space = Box(
            low=np.array([0.4, -0.25, -90]), high=np.array([0.6, -0.1, 90])#, seed=seed
        )

        self.target_space = Box(
            low=np.array([0.4, 0.2, -90]), high=np.array([0.6, 0.35, 90])#, seed=seed
        )

        # index = 0, push from inside
        # index = 1, push from outside
        self.index = index

    def start(self, random=True, context=None):

        if random:
            self.context = self.sample()
        else:
            self.context = context

        self.set_context(self.context)

        pos, quat, target_pos, target_quat = self.context

        return target_pos, target_quat

    def sample(self):

        pos = self.box_space.sample()

        goal_angle = [0, 0, pos[-1] * np.pi / 180]
        quat = euler2quat(goal_angle)

        target_pos = self.target_space.sample()
        target_angle = [0, 0, target_pos[-1] * np.pi / 180]
        target_quat = euler2quat(target_angle)

        return [pos, quat, target_pos, target_quat]

    def sample_target_pos(self):

        pos = self.target_space.sample()

        goal_angle = [0, 0, pos[-1] * np.pi / 180]
        quat = euler2quat(goal_angle)

        return [pos, quat]

    def set_context(self, context):

        pos, quat, target_pos, target_quat = context

        self.scene.set_obj_pos_and_quat(
            [pos[0], pos[1], 0.00],
            quat,
            obj_name="aligning_box",
        )

        self.scene.set_obj_pos_and_quat(
            [target_pos[0], target_pos[1], 0.00],
            target_quat,
            obj_name="target_box",
        )

    def set_index(self, index):
        self.index = index


class Robot_Push_Env(GymEnvWrapper):
    def __init__(
        self,
        n_substeps: int = 35,
        max_steps_per_episode: int = 400,
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
        self.manager = BlockContextManager(scene, index=1)

        self.bp_cam = BPCageCam()
        self.inhand_cam = robot.inhand_cam

        self.push_box = obj_list[0]
        self.target_box = obj_list[1]

        self.scene.add_object(self.bp_cam)

        self.log_dict = {
            "push-box": ObjectLogger(scene, self.push_box),
            "target-box": ObjectLogger(scene, self.target_box)
        }

        self.cam_dict = {
            "bp-cam": CamLogger(scene, self.bp_cam),
            "inhand-cam": CamLogger(scene, self.inhand_cam)
        }

        for _, v in self.log_dict.items():
            scene.add_logger(v)

        for _, v in self.cam_dict.items():
            scene.add_logger(v)

        self.pos_min_dist = 0.018
        self.rot_min_dist = 0.048

        self.robot_box_dist = 0.051

        self.first_visit = -1

    def get_observation(self) -> np.ndarray:

        robot_pos = self.robot_state()

        if self.if_vision:

            bp_image = self.bp_cam.get_image(depth=False)
            bp_image = cv2.cvtColor(bp_image, cv2.COLOR_RGB2BGR)

            inhand_image = self.inhand_cam.get_image(depth=False)
            inhand_image = cv2.cvtColor(inhand_image, cv2.COLOR_RGB2BGR)

            return robot_pos, bp_image, inhand_image

        box_pos = self.scene.get_obj_pos(self.push_box)  # - robot_pos
        box_quat = self.scene.get_obj_quat(self.push_box)

        target_pos = self.scene.get_obj_pos(self.target_box)
        target_quat = self.scene.get_obj_quat(self.target_box)

        env_state = np.concatenate(
            [
                robot_pos,
                box_pos,
                box_quat,
                target_pos,
                target_quat
            ]
        )

        return env_state.astype(np.float32)#, bp_image, inhand_image

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

        mode = -1

        box_pos = self.scene.get_obj_pos(self.push_box)
        box_quat = self.scene.get_obj_quat(self.push_box)

        target_pos = self.scene.get_obj_pos(self.target_box)
        target_quat = self.scene.get_obj_quat(self.target_box)

        box_goal_pos_dist = np.linalg.norm(box_pos - target_pos)
        box_goal_rot_dist = rotation_distance(box_quat, target_quat) / np.pi

        robot_pos = self.robot_state()

        robot_box_dist = np.linalg.norm(box_pos[:2] - robot_pos[:2])

        if robot_box_dist < self.robot_box_dist:
            mode = 0
        else:
            mode = 1

        mean_distance = 0.5 * (box_goal_pos_dist + box_goal_rot_dist)

        return mode, mean_distance

    def get_reward(self, if_sparse=False):

        box_pos = self.scene.get_obj_pos(self.push_box)
        box_quat = self.scene.get_obj_quat(self.push_box)

        target_pos = self.scene.get_obj_pos(self.target_box)
        target_quat = self.scene.get_obj_quat(self.target_box)

        box_goal_pos_dist_reward = -3.5 * np.linalg.norm(box_pos - target_pos)
        box_goal_rot_dist_reward = -rotation_distance(box_quat, target_quat) / np.pi

        return box_goal_rot_dist_reward + box_goal_pos_dist_reward

    def _check_early_termination(self) -> bool:

        box_pos = self.scene.get_obj_pos(self.push_box)
        box_quat = self.scene.get_obj_quat(self.push_box)

        target_pos = self.scene.get_obj_pos(self.target_box)
        target_quat = self.scene.get_obj_quat(self.target_box)

        box_goal_pos_dist = np.linalg.norm(box_pos - target_pos)
        box_goal_rot_dist = rotation_distance(box_quat, target_quat) / np.pi

        if (box_goal_pos_dist <= self.pos_min_dist) and (
            box_goal_rot_dist <= self.rot_min_dist
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
