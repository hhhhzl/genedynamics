import numpy as np

from environments.d3il.d3il_sim.core.logger import RobotPlotFlags
from environments.d3il.d3il_sim.sims.SimFactory import SimRepository

if __name__ == "__main__":
    sim_factory = SimRepository.get_factory("sl")
    scene = sim_factory.create_scene()
    robot = sim_factory.create_robot(
        scene=scene,
        robot_name="panda2",
        backend_addr="tcp://141.3.53.152:51468",
        local_addr="141.3.53.158",
    )
    scene.start()
    robot.use_inv_dyn = False

    # gotoController = SLRobot.JointTrajectoryTracker()

    # Set desired joint position and go there. The controller is using a bspline to plan the trajectory from the current
    # state to the desired state. Duration of the movement is default 4 secs (can be changed by robot.gotoJointPosController
    desired_joint_pos1 = np.array([-0.2, 0, 0, -1.562, 0, 1.914, -0.2])
    desired_joint_pos2 = np.array([0.4, 0.2, 0.2, -1.362, 0.2, 1.7, 0.2])

    robot.start_logging()
    robot.gotoJointPosition(desired_joint_pos1, duration=4)
    robot.gotoJointPosition(desired_joint_pos2, duration=4)
    robot.gotoJointPosition(desired_joint_pos1, duration=4)
    robot.stop_logging()

    # robot.logger.plot(RobotPlotFlags.JOINTS)
    import matplotlib.pyplot as plt

    robot.robot_logger.plot(
        plotSelection=RobotPlotFlags.JOINT_POS
        | RobotPlotFlags.JOINT_VEL
        | RobotPlotFlags.JOINT_ACC
        | RobotPlotFlags.TORQUES
        | RobotPlotFlags.COMMAND
        | RobotPlotFlags.TIME_STAMPS
    )

    plt.show()
    print("hello")
