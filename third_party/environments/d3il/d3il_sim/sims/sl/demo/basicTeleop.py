import time

import numpy as np

from environments.d3il.d3il_sim.sims.sl.SlRobot import SlRobot

if __name__ == "__main__":
    """
    A script demonstrating simple teleoperation. Legacy demo from the SimulationFramework_SL
    """

    # Control gains

    pgain = 0.1 * np.array(
        [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0], dtype=np.float64
    )
    # dgain = 0.2 * np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0], dtype=np.float64)
    dgain = 0.1 * np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0], dtype=np.float64)

    robotSlave = SlRobot(
        backend_addr="tcp://192.168.3.4:51468",
        local_addr="192.168.3.3",
        gripper_actuation=True,
    )
    robotMaster = SlRobot(robot_name="franka5")

    robotSlave.receiveState()
    robotMaster.receiveState()

    robotSlave.use_inv_dyn = False
    robotMaster.use_inv_dyn = False

    desired_joint_pos1 = np.array([0, 0, 0, -1.562, 0, 1.914, 0])

    desired_joint_pos2 = np.array([0, 0, 0, -1.562, 0, 1.914, 0])
    desired_joint_pos2[0] = desired_joint_pos2[0] - 0.5

    initState = robotMaster.current_j_pos
    robotSlave.set_gripper_width = robotMaster.gripper_width
    robotSlave.gotoJointPosition(initState)

    robotMaster.robot_logger.max_time_steps = 10000
    robotSlave.robot_logger.max_time_steps = 10000

    robotMaster.start_logging()
    robotSlave.start_logging()
    while robotSlave.step_count < 10000:

        recv_start = time.clock_gettime(time.CLOCK_MONOTONIC)

        robotMaster.command = np.zeros((7,))

        j_posdes_slave = robotMaster.current_j_pos
        j_veldes_slave = robotMaster.current_j_vel

        target_j_acc = (
            pgain * (j_posdes_slave - robotSlave.current_j_pos)
            - dgain * robotSlave.current_j_vel
        )
        robotSlave.command = target_j_acc

        if robotSlave.step_count % 30 == 0:
            robotSlave.set_gripper_width = robotMaster.gripper_width
            robotSlave.gripper_send_acutation = True

        robotMaster.nextStep()
        robotSlave.nextStep()

    robotMaster.stop_logging()
    robotSlave.stop_logging()

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(robotMaster.robot_logger.joint_pos)
    plt.plot(robotSlave.robot_logger.joint_pos, "--")

    plt.figure()
    plt.plot(robotMaster.robot_logger.time_stamp, robotMaster.robot_logger.joint_pos)

    plt.figure()
    plt.plot(robotMaster.robot_logger.time_stamp, robotMaster.robot_logger.uff)

    plt.figure()
    plt.plot(
        robotMaster.robot_logger.time_stamp, robotMaster.robot_logger.last_cmd, "--"
    )

    plt.show()
