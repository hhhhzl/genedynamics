import os

import pandas as pd

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
from environments.d3il.d3il_sim.sims.sl.teleoperation.virt_obst_sim import remix_goal, virtual_obst_setup

DIRNAME = os.path.dirname(__file__)
DATA_DIR = os.path.join(DIRNAME, "..", "data")
CONTEXT_DF_FILE = os.path.join(DATA_DIR, "context_df.pkl")

if __name__ == "__main__":
    REAL_ROBOT = True

    ### CONNECT REAL ROBOT
    if REAL_ROBOT:
        primary_robot = connect_sl_robot(
            robot_name="panda2",
            backend_addr="tcp://141.3.53.152:51468",
            local_addr="141.3.53.158",
            gripper_actuation=True,
        )
    else:
        primary_robot = connect_sl_robot(robot_name="panda2")

    ### CONNECT VIRTUAL TWIN
    replica_scene, replica_robot = virtual_obst_setup()

    ### CREATE LOGGER
    teleop_log = TeleopMetaLogger(
        root_dir=DATA_DIR,
        log_name="virt_obst_log",
    )

    ### CREATE CONTROL THREADS
    primary_ctrl = tele_ctrl.TeleopPrimaryController(primary_robot)
    replica_ctrl = VirtualTwinController(digital_scene=replica_scene)
    meta_thread = VirtualTwinMetaScheduler(
        primary_robot, replica_robot, primary_ctrl, replica_ctrl, teleop_log
    )

    primary_robot.activeController = None
    replica_robot.activeController = None

    ### CREATE CLI
    cli_map = HumanTeacherCliMap(meta_thread)

    df = pd.DataFrame()
    if os.path.exists("virt_obst_contexts.pkl"):
        df = pd.read_pickle("virt_obst_contexts.pkl")

    def reset_and_mix():
        global df
        meta_thread.stop_logging()
        c = remix_goal(replica_scene)
        df = df.append(c, ignore_index=True)
        df.to_pickle("virt_obst_contexts.pkl")

    def print_j_pos():
        print(primary_robot.current_j_pos)

    def return_home():
        primary_robot.gotoJointController.trajectory = None
        primary_robot.gotoJointPosition(
            [-0.2028, 0.6555, -0.7554, -1.7188, 0.5067, 2.1350, 1.0885], 6
        )
        primary_ctrl.reset()
        # allow manuel control without a controller.
        primary_robot.activeController = None

    cli_map._map["P"] = ("Stop log and Remix", reset_and_mix)
    cli_map._map["S"] = ("Current Pos", print_j_pos)
    cli_map._map["R"] = ("Reset Pos", return_home)
    # cli_map._map["M"] = ("Remix", reset_and_mix)

    ### START
    meta_thread.start_control()

    # replica_robot.set_gripper_width = primary_robot.gripper_width
    # replica_robot.beam_to_joint_pos(primary_robot.current_j_pos)

    ### MAIN LOOP
    while cli_map.get_input():
        continue

    ### STOP
    meta_thread.stop_control()
