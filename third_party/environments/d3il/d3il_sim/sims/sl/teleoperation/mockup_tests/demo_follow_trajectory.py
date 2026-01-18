import os
import time

from environments.d3il.d3il_sim.sims.sl.teleoperation.demo_teacher import connect_sl_robot
from environments.d3il.d3il_sim.sims.sl.teleoperation.mockup_tests.follow_trajectory.trajectory_controller import (
    TrajectoryController,
)
from environments.d3il.d3il_sim.sims.sl.teleoperation.mockup_tests.follow_trajectory.trajectory_robot import (
    TrajectoryMockupRobot,
)
from environments.d3il.d3il_sim.sims.sl.teleoperation.mockup_tests.follow_trajectory.trajectory_scheduler import (
    MetaTrajectoryScheduler,
)
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.controller import teleop_controller as tele_ctrl
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.ui.cli_prompts import TrajectoryCliMap
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.util.teaching_log import TeleopMetaLogger

"""
Demo script for the Follow trajectory test: The primary robot follows a predefined trajectory; the replica robot follows
it.
"""


if __name__ == "__main__":
    replica_robot = connect_sl_robot(
        robot_name="panda2",
        backend_addr="tcp://141.3.53.152:51468",
        local_addr="141.3.53.26",
        gripper_actuation=True,
    )

    val_name = "val_003"
    log_file = os.path.join(
        "/home/philipp/phd_projects/teleop_data/validation_trajectories/data/",
        val_name + ".pkl",
    )

    primary_robot = TrajectoryMockupRobot(log_file)

    # Mirror Robot Positions
    replica_robot.gotoJointPosition(primary_robot.current_j_pos)

    teleop_log = TeleopMetaLogger(
        root_dir=os.path.join(
            "/home/philipp/phd_projects/teleop_data/follow_trajectory/", val_name
        ),
        log_name="exec_7",
    )

    primary_ctrl = TrajectoryController(primary_robot)
    replica_ctrl = tele_ctrl.TeleopReplicaController(replica_robot)

    meta_thread = MetaTrajectoryScheduler(
        log_file, primary_robot, replica_robot, primary_ctrl, replica_ctrl, teleop_log
    )

    cli_map = TrajectoryCliMap(meta_thread)
    time.sleep(1)
    meta_thread.start_control()
    while cli_map.get_input():
        continue

    meta_thread.stop_control()
