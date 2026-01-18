import environments.d3il.d3il_sim.sims.SimFactory as sims
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.controller.human_controller import (
    HumanController,
)
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.schedulers.human_scheduler import (
    HumanControlScheduler,
)
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.ui.cli_prompts import HumanTeacherCliMap
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.util.teaching_log import TeachingLog


def connect_sl_robot(*args, **kwargs):
    # Start SL Scene and Robot
    sl = sims.SimRepository.get_factory("sl")
    sl_scene = sl.create_scene(skip_home=True)
    sl_robot = sl.create_robot(sl_scene, *args, **kwargs)
    sl_scene.start()

    print("*" * 20)
    print("CONNECTED TO {}".format(sl_robot.robot_name))
    print("*" * 20)

    return sl_robot


if __name__ == "__main__":
    robot = connect_sl_robot(
        robot_name="panda2",
        backend_addr="tcp://141.3.53.152:51468",
        local_addr="141.3.53.158",
        gripper_actuation=True,
    )
    teaching_log = TeachingLog(
        root_dir="/home/philipp/phd_projects/teleop_data/gravity",
        log_name="val_no_kalman",
    )
    # allow manuel control without a controller
    robot.activeController = None
    human_controller = HumanController(robot)
    control_thread = HumanControlScheduler(robot, human_controller, teaching_log)

    cli_map = HumanTeacherCliMap(control_thread)

    control_thread.start_control()

    while cli_map.get_input():
        continue

    control_thread.stop_control()
