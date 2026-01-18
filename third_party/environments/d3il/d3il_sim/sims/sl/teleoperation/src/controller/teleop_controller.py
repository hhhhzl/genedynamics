import abc
from collections import deque

import numpy as np

from ..util.force_feedback import ForceFeedback, NoForceFeedback
from .human_controller import HumanController


class TeleopController(HumanController, abc.ABC):
    @abc.abstractmethod
    def get_tau(self, other_teleop_ctrl):
        raise NotImplementedError


class TeleopPrimaryController(TeleopController):
    def __init__(self, robot):
        super(TeleopPrimaryController, self).__init__(robot)

        self.force = NoForceFeedback()

        self.history_length = 150

        self.history = None

    def initialize(self):
        super(TeleopPrimaryController, self).initialize()
        self.history = deque([self.robot.current_j_pos], maxlen=self.history_length)

    def get_tau(self, other_teleop_ctrl):
        self.paramsLock.acquire()
        primary = self.robot
        self.history.append(primary.current_j_pos)

        tau = self.force.feedback(self, other_teleop_ctrl)

        self.paramsLock.release()
        return tau

    def set_feedback_mode(self, force_fb: ForceFeedback):
        self.paramsLock.acquire()
        self.force = force_fb
        self.paramsLock.release()

    def reset(self):
        self.history.clear()


class TeleopReplicaController(TeleopController):
    def __init__(self, robot):
        super(TeleopReplicaController, self).__init__(robot)
        # PD gains
        self.pgain = 0.45 * np.array(
            [150.0, 150.0, 150.0, 150.0, 100.0, 75.0, 50.0], dtype=np.float64
        )
        self.dgain = 0.1 * np.array(
            [50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0], dtype=np.float64
        )
        self.virtual_obstacles = []

    def get_tau(self, other_teleop_ctrl):
        primary = other_teleop_ctrl.robot
        replica = self.robot
        self.paramsLock.acquire()
        tau = self._pd_control(primary, replica)
        self.paramsLock.release()
        return tau

    def _pd_control(self, primary, replica):
        return (
            self.pgain * (primary.current_j_pos - replica.current_j_pos)
            - self.dgain * replica.current_j_vel
        )

    """
    Functions for Feedback. Deactivated for now since the virtual feedback is not tested yet with the current Force
    Feedback.
    def set_virtual_obstacles(self, v_obst):
        self.paramsLock.acquire()
        self.virtual_obstacles = v_obst
        self.paramsLock.release()

    def virtual_repel_force(self, k=2.5, repel_dist=0.055):
        robot = self.robot
        endeffector_pos = robot.current_c_pos

        force_vect = np.zeros((6,))
        if np.array(self.virtual_obstacles).size == 0:
            return force_vect

        diffs = np.array(endeffector_pos - self.virtual_obstacles)
        diffs = diffs[np.linalg.norm(diffs, axis=1) < repel_dist]

        if not diffs.size == 0:
            net_force = np.sum(diffs, axis=0)
            force_vect[:3] = k * net_force / np.linalg.norm(net_force)
        return force_vect
    """
