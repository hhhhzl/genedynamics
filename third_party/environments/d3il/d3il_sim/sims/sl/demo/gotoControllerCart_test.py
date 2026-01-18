import matplotlib.pyplot as plt
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

    home_joint_pos = np.array(
        [
            3.46311499e-05,
            9.08202901e-02,
            -7.06741586e-04,
            -1.58921993e00,
            -1.13537759e-02,
            1.90483785e00,
            7.85356522e-01,
        ]
    )

    robot.gotoJointPosition(home_joint_pos, duration=4)
    print("reached home position")

    target = np.hstack((np.array([0.4, 0.0, 0.2]), np.array([0, 1, 0, 0])))
    robot.robot_logger.max_time_steps = 100000000
    scene.start_logging()
    robot.gotoCartPositionAndQuat(target[:3], target[3:], duration=6)

    scene.stop_logging()
    robot.robot_logger.plot(RobotPlotFlags.END_EFFECTOR | RobotPlotFlags.JOINTS)

    plt.show()
