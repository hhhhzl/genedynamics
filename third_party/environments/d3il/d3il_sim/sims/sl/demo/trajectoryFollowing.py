import numpy as np

from environments.d3il.d3il_sim.controllers import JointTrajectoryTracker
from environments.d3il.d3il_sim.sims.SimFactory import SimRepository

if __name__ == "__main__":
    sim_factory = SimRepository.get_factory("sl")

    scene = sim_factory.create_scene()
    robot = sim_factory.create_robot(scene, robot_name="franka_m")
    scene.start()

    trajectoryFollowing = JointTrajectoryTracker(dt=0.001)

    desired_joint_pos = np.array([0, 0, 0, -1.562, 0, 1.914, 0])
    robot.gotoJointPosition(desired_joint_pos)

    init_joint_state = robot.current_j_pos

    trajectory = np.zeros((5000, 7))

    for i in range(5000):
        trajectory[i, :] = init_joint_state
        trajectory[i, 3] = trajectory[i, 3] + np.sin(i * 0.001 / 5 * 2 * np.pi)
    trajectoryFollowing.setTrajectory(trajectory)

    scene.start_logging()
    trajectoryFollowing.executeController(robot)
    scene.stop_logging()

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(robot.robot_logger.joint_pos)

    plt.figure()
    plt.plot(robot.robot_logger.time_stamp, robot.robot_logger.joint_pos)

    plt.figure()
    plt.plot(robot.robot_logger.time_stamp, robot.robot_logger.uff)

    plt.figure()
    plt.plot(robot.robot_logger.time_stamp, robot.robot_logger.last_cmd, "--")

    plt.show()
    print("hello")
