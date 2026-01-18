import sys

import numpy as np

from environments.d3il.d3il_sim.sims.mujoco.mj_interactive.ia_objects.pushing_object import (
    VirtualPushObject,
)
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
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.ui.cli_prompts import HumanTeacherCliMap
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.util.teaching_log import TeleopMetaLogger
from environments.d3il.d3il_sim.utils.geometric_transformation import euler2quat, quat2euler


def virtual_twin(seed=0, render=True, proc_id="", hint_alpha=0.3):
    np.random.seed(seed)
    degrees = np.random.random_sample()

    goal_angle = [0, 0, degrees * np.pi / 180]
    quat = euler2quat(goal_angle)
    x_pos = np.random.random_sample() * 0.2 + 0.2
    y_pos = np.random.random_sample() * 0.8 - 0.4

    goal_vis = VirtualPushObject(
        [x_pos, y_pos, 0.0],
        quat,
        scale=1.5,
        rgba=[0, 1, 0, 0.3],
        visual_only=True,
    )
    push_box = VirtualPushObject(
        [0.33, 0.0, 0.016], [1, 0, 0, 0], rgba=[1, 0, 0, 1], scale=1.3, name="push_box"
    )

    factory = SimRepository.get_factory("mujoco")

    rm = factory.RenderMode.BLIND
    if render:
        rm = factory.RenderMode.HUMAN

    s = factory.create_scene(
        object_list=[goal_vis, push_box], render=rm, proc_id=proc_id
    )
    virtual_bot = MjPushRobot(scene=s)

    s.get_obj_pos

    return s, virtual_bot


def save_context(s):
    xy = s.get_obj_pos(obj_name="push_box")[:2]
    z = quat2euler(s.get_obj_quat(obj_name="push_box"))[-1]
    with open("context.txt", "a") as f:
        f.write(str(xy[0]) + " " + str(xy[1]) + " " + str(z) + "\n")


if __name__ == "__main__":
    # DEGREES = int(sys.argv[1])
    DEGREES = 5
    REAL_ROBOT = True

    if REAL_ROBOT:
        primary_robot = connect_sl_robot(
            robot_name="panda2",
            backend_addr="tcp://141.3.53.152:51468",
            local_addr="141.3.53.158",
            gripper_actuation=True,
        )
    else:
        primary_robot = connect_sl_robot(robot_name="panda2")

    # Save Scene, so that Pybullet does not get Garbage Collected!
    replica_scene, replica_robot = virtual_twin(DEGREES)

    # Create TeleopLog Directly without Prompt. Might cause issues with Mujoco somehow??
    teleop_log = TeleopMetaLogger(
        root_dir="./box_log", log_name="twin_seed_{:03d}".format(DEGREES)
    )

    primary_ctrl = tele_ctrl.TeleopPrimaryController(primary_robot)
    replica_ctrl = VirtualTwinController(digital_scene=replica_scene)

    meta_thread = VirtualTwinMetaScheduler(
        primary_robot,
        replica_robot,
        primary_ctrl,
        replica_ctrl,
        teleop_log,
        virtual_callback=save_context,
    )

    # allow manuel control without a controller
    primary_robot.activeController = None
    replica_robot.activeController = None

    cli_map = HumanTeacherCliMap(meta_thread)

    meta_thread.start_control()
    while cli_map.get_input():
        continue

    meta_thread.stop_control()
