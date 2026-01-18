import time

import numpy as np

from environments.d3il.d3il_sim.sims.SimFactory import SimRepository

if __name__ == "__main__":

    # Control gains

    pgain = 0.1 * np.array(
        [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0], dtype=np.float64
    )
    # dgain = 0.2 * np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0], dtype=np.float64)
    dgain = 0.1 * np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0], dtype=np.float64)

    sim_factory = SimRepository.get_factory("sl")
    robot = sim_factory.create_robot(robot_name="panda2")
    scene = sim_factory.create_scene(robot)
    scene.start()

    robot.use_inv_dyn = False

    robot.robot_logger.max_time_steps = 10000

    scene.start_logging()

    while robot.step_count < 1000000:
        recv_start = time.clock_gettime(time.CLOCK_MONOTONIC)

        A = np.random.normal(0, 1, size=(20, 20))
        temp = np.linalg.inv(A)

        robot.command = np.zeros((7,))

        robot.nextStep()

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
