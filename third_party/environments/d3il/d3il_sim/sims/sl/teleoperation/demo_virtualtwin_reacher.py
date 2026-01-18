import copy
import sys
from datetime import datetime
from functools import partial
from os import rmdir
from tokenize import Number

import numpy as np

from environments.d3il.d3il_sim.sims.mujoco.mj_interactive.ia_robots.mj_push_robot import MjPushRobot
from environments.d3il.d3il_sim.sims.SimFactory import SimRepository
from environments.d3il.d3il_sim.sims.sl.teleoperation.demo_teacher import connect_sl_robot
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.controller import teleop_controller as tele_ctrl
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.controller.virtualtwin_controller import (
    VirtualTwinController,
)
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.schedulers.virtualtwin_scheduler import (
    VirtualTwinMetaScheduler,
)
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.ui.cli_prompts import HumanTeacherReacherCliMap
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.util.reacher_log import ReacherMetaLogger
from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import Sphere
from environments.d3il.d3il_sim.utils.geometric_transformation import euler2quat

goal_positions = [[0.5, 0.2, 0.6], [0.3, -0.1, 0.3]]

original_goal_positions = [[0.5, 0.2, 0.6], [0.3, -0.1, 0.3]]


def virtual_twin(render=True, goal_min_dist=0.05):
    # np.random.seed(seed)

    goal_one_next = Sphere(
        "goal_one_next",
        goal_positions[0],
        [0, 0, 0, 1],
        size=[0.05],
        rgba=[0, 1, 0, 0.3],
        static=True,
        visual_only=True,
    )

    goal_one_shadow = Sphere(
        "goal_one_shadow",
        [-100, 0.0, 0.0],
        [0, 0, 0, 1],
        size=[0.05],
        rgba=[0, 0, 1, 0.3],
        static=True,
        visual_only=True,
    )

    goal_two = Sphere(
        "goal_two",
        goal_positions[1],
        [0, 0, 0, 1],
        size=[0.05],
        rgba=[1, 0, 0, 0.3],
        static=True,
        visual_only=True,
    )

    goal_two_shadow = Sphere(
        "goal_two_shadow",
        [-100, 0.0, 0.0],
        [0, 0, 0, 1],
        size=[0.05],
        rgba=[0, 0, 1, 0.3],
        static=True,
        visual_only=True,
    )

    goal_two_next = Sphere(
        "goal_two_next",
        [-100, 0.0, 0.0],
        [0, 0, 0, 1],
        size=[0.05],
        rgba=[0, 1, 0, 0.3],
        static=True,
        visual_only=True,
    )

    factory = SimRepository.get_factory("mujoco")

    rm = factory.RenderMode.BLIND
    if render:
        rm = factory.RenderMode.HUMAN
    object_list = [
        goal_one_next,
        goal_one_shadow,
        goal_two,
        goal_two_shadow,
        goal_two_next,
    ]
    sphere_list = copy.deepcopy(object_list)
    s = factory.create_scene(object_list=copy.deepcopy(object_list), render=rm)
    virtual_bot = factory.create_robot(scene=s)

    def callback(goal_min_dist: Number):

        try:
            pos_ee = virtual_bot.current_c_pos

            body_id_one_next = s.sim.model.body_name2id(goal_one_next.name)
            body_id_one_shadow = s.sim.model.body_name2id(goal_one_shadow.name)
            pos_one_next = s.get_obj_pos(obj_name=goal_one_next.name)
            pos_one_shadow = s.get_obj_pos(obj_name=goal_one_shadow.name)

            body_id_two = s.sim.model.body_name2id(goal_two.name)
            body_id_two_next = s.sim.model.body_name2id(goal_two_next.name)
            body_id_two_shadow = s.sim.model.body_name2id(goal_two_shadow.name)
            pos_two = s.get_obj_pos(obj_name=goal_two.name)
            pos_two_next = s.get_obj_pos(obj_name=goal_two_next.name)
            pos_two_shadow = s.get_obj_pos(obj_name=goal_two_shadow.name)

            diff_ee_one = np.linalg.norm(pos_ee - pos_one_next)
            diff_ee_two = np.linalg.norm(pos_ee - pos_two_next)

            if diff_ee_one < goal_min_dist:
                s.model.body_pos[body_id_one_next] = pos_one_shadow
                s.model.body_pos[body_id_one_shadow] = pos_one_next

                s.model.body_pos[body_id_two] = pos_two_next
                s.model.body_pos[body_id_two_next] = pos_two

            if diff_ee_two < goal_min_dist:
                s.model.body_pos[body_id_two_shadow] = pos_two_next
                s.model.body_pos[body_id_two_next] = pos_two_shadow

        except Exception:
            pass

    def reset_callback():
        for sphere in sphere_list:
            obj_id = s.sim.model.body_name2id(sphere.name)
            s.model.body_pos[obj_id] = sphere.init_pos

    s.register_callback(callback, goal_min_dist=goal_min_dist)

    return s, virtual_bot, reset_callback


if __name__ == "__main__":
    contexts = np.load("contexts.npy")
    print(contexts.shape)

    context_idx = 9

    goal_positions = contexts[context_idx]

    REAL_ROBOT = True

    if REAL_ROBOT:
        primary_robot = connect_sl_robot(
            robot_name="panda2",
            backend_addr="tcp://141.3.53.152:51468",
            local_addr="141.3.53.206",
            gripper_actuation=True,
        )
    else:
        primary_robot = connect_sl_robot(robot_name="panda2")

    goal_min_dist = 0.05

    # Save Scene, so that Pybullet does not get Garbage Collected!
    replica_scene, replica_robot, reset_callback = virtual_twin(goal_min_dist)

    date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%p")
    # Create TeleopLog Directly without Prompt. Might cause issues with Mujoco somehow??
    teleop_log = ReacherMetaLogger(
        root_dir="./",
        log_name=f"reacher_{date}_c{context_idx}",
        additional_info={
            "goal_positions": goal_positions,
            "goal_min_dist": goal_min_dist,
        },  # TODO ACTUAL SEED
    )

    primary_ctrl = tele_ctrl.TeleopPrimaryController(primary_robot)
    replica_ctrl = VirtualTwinController(digital_scene=replica_scene)

    meta_thread = VirtualTwinMetaScheduler(
        primary_robot, replica_robot, primary_ctrl, replica_ctrl, teleop_log
    )

    # allow manuel control without a controller
    primary_robot.activeController = None
    replica_robot.activeController = None

    cli_map = HumanTeacherReacherCliMap(meta_thread, reset_callback)

    meta_thread.start_control()
    while cli_map.get_input():
        continue

    meta_thread.stop_control()
