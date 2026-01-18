import threading

import numpy as np

from ..util.kalman_filter import KalmanFilter


class HumanController:
    def __init__(self, robot):
        self.zero_tau = np.zeros((7,))
        self.time_delay = 0

        self.load_filter = None
        self.paramsLock = threading.Lock()
        self.robot = robot

        # see self.get_signal() for use
        self.use_tau_ext_hat_filtered = True
        # the tau_ext_hat_filtered has a small gravity bias: the robot gets "sleepy", we compensate this with this gain
        self.anti_gravity_gain = 0.0 * np.array([1.0, 1.2, 1.0, 0.5, 0.5, 0.5, 0.5])

    def get_signal(self):
        """
        The tau_ext_hat_filtered sensor reading is preferred over the other choice: It measures wrong external forces.
        """
        if self.use_tau_ext_hat_filtered:
            return (
                self.robot.tau_ext_hat_filtered
                - self.anti_gravity_gain * self.robot.gravity
            )
        else:
            return self.robot.current_load - self.robot.gravity - self.robot.coriolis

    def initialize(
        self,
    ):
        # initialise Kalman filter for both robots to get a filtered load value.
        self.load_filter = KalmanFilter(self.get_signal())

    def get_tau(self, *args):
        return self.zero_tau

    def get_load(self, receive_state=True):
        self.paramsLock.acquire()
        robot = self.robot
        if receive_state:
            robot.receiveState()
        curr_load = self.load_filter.get_filtered(self.get_signal())
        self.paramsLock.release()
        return curr_load

    def reset(self):
        pass
