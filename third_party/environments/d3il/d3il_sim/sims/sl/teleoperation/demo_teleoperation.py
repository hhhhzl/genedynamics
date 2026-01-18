from environments.d3il.d3il_sim.sims.sl.teleoperation.demo_teacher import connect_sl_robot
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.controller import teleop_controller as tele_ctrl
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.schedulers.teleop_scheduler import (
    TeleopMetaScheduler,
)
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.ui.cli_prompts import TeleopCliMap
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.util.teaching_log import TeleopMetaLogger

if __name__ == "__main__":
    primary_robot = connect_sl_robot(
        robot_name="panda2",
        backend_addr="tcp://141.3.53.152:51468",
        local_addr="141.3.53.158",
        gripper_actuation=True,
    )
    replica_robot = connect_sl_robot(
        robot_name="panda1",
        backend_addr="tcp://141.3.53.151:51468",
        local_addr="141.3.53.158",
        gripper_actuation=True,
    )

    # Mirror Robot Positions
    replica_robot.set_gripper_width = primary_robot.gripper_width
    replica_robot.gotoJointPosition(primary_robot.current_j_pos)

    # allow manuel control without a controller
    primary_robot.activeController = None
    replica_robot.activeController = None

    # change the path to your current system
    teleop_log = TeleopMetaLogger(
        root_dir="/home/philipp/phd_projects/teleop_data/raw_loads", log_name="exec_10"
    )

    primary_ctrl = tele_ctrl.TeleopPrimaryController(primary_robot)
    replica_ctrl = tele_ctrl.TeleopReplicaController(replica_robot)
    meta_thread = TeleopMetaScheduler(
        primary_robot, replica_robot, primary_ctrl, replica_ctrl, teleop_log
    )

    cli_map = TeleopCliMap(meta_thread)

    meta_thread.start_control()
    while cli_map.get_input():
        continue

    meta_thread.stop_control()
