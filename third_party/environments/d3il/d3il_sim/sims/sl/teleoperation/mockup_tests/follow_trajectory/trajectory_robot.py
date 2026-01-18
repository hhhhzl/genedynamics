import pickle

import numpy as np


class TrajectoryMockupRobot:
    """
    Mockup of a robot which acts as the primary. It gets its values from a log file and saves them here for the
    replica robot to PD control on these values.
    """

    def __init__(self, trajectory_log):
        logs = pickle.load(open(trajectory_log, "rb"))
        data = logs["data"]
        # inital position
        self.current_j_pos = data["j_pos"][0]
        # no initial velocity
        self.current_j_vel = np.zeros(7)

        # dummy values
        self.current_load = np.zeros(7)
        self.tau_ext_hat_filtered = np.zeros(7)
        self.current_c_pos = None
        self.current_c_quat = None
        self.gripper_width = None

    def set_state(self, current_j_pos, current_j_vel):
        self.current_j_pos = current_j_pos
        self.current_j_vel = current_j_vel
