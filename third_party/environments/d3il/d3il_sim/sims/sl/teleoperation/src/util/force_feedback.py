import abc

import numpy as np


class ForceFeedback(abc.ABC):
    def __init__(self, dof=7):
        self.zero_tau = np.zeros((dof,))
        self.fb_gain = -1.5 * np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64
        )

    @abc.abstractmethod
    def feedback(self, primary_ctrl, replica_ctrl):
        raise NotImplementedError

    def total_force_clip(self, replica_load, max_force=8, a=6, b=4):
        """
        computes l1 norm of the replica load (maybe with incorporated d-gain control of the primary velocity).
        If it is smaller than b: return zero force
        If it is between a and b: return linear interpolation
        If it is bigger than a: return replica_load
        if it is bigger than max_force: scale it down such that total force == max_force
        """
        total_force = np.sum(np.abs(replica_load))
        if total_force < b:
            return np.zeros(7)
        elif total_force < a:
            return (1 / (a - b) * total_force - b / (a - b)) * replica_load
        elif total_force < max_force:
            return replica_load
        else:
            return replica_load * (max_force / total_force)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        return "Force Feedback"


class NoForceFeedback(ForceFeedback):
    def feedback(self, primary_ctrl, replica_ctrl):
        return self.zero_tau

    @property
    def name(self) -> str:
        return "No Force Feedback"


class TorqueForceFeedback(ForceFeedback):
    def __init__(self):
        super().__init__()
        self.torque_dgain = 0.5 * np.array(
            [50.0, 50.0, 50.0, 50.0, 15.0, 8.0, 8.0], dtype=np.float64
        )
        self.tau_moving_average = np.zeros(7)
        self.alpha = 0.9

    def feedback(self, primary_ctrl, replica_ctrl):
        """
        General idea of this torque feedback:
        Have a D controller on the primary velocity which activates only when external forces are present. This feature
        combines the safety of the primary robot with its useabilty: if the
        D controller were always active the human would not be able to move the primary robot easily. However, without
        a D controller the primary robot (which is next to a human) can move very fast if the external forces are high.
        This needs to be prevented. This is the algorithm:
        1. Get the filtered external forces from the replica robot
        2. Use total_force_clip() to filter out small forces, this is crucial to filter out the inherent sensor noise.
           While we loose the feedback of small forces, this improves the general control of the primary robot.
           While this also clips the forces with 'max_force', the current D controller on the primary velocity handles
           too high forces acting on the primary robot as well by regularizing the primary velocity.
        3. Compute the D control feedback 'd_feedback' based on the primary joint velocities
        4. clip 'd_feedback' with the absolute value of the current external forces. However, for a safer
           environment, also include the latest external forces with a moving average.
        returns the force feedback - 'd_feedback'
        """
        # check if initialization takes longer --> first iteration robot may be NONE, we catch that here
        if replica_ctrl.robot is None or primary_ctrl.robot is None:
            return self.zero_tau
        # 1. get force feedback from replica
        replica_load = replica_ctrl.get_load(receive_state=False)
        plain_feedback = self.total_force_clip(
            self.fb_gain * replica_load, a=7, b=5, max_force=60
        )
        # 3. get d control feedback from primary
        primary_j_vel = primary_ctrl.robot.current_j_vel
        d_feedback = self.torque_dgain * primary_j_vel
        # update moving average of the latest external forces
        self.tau_moving_average = self.alpha * self.tau_moving_average + (
            1 - self.alpha
        ) * np.abs(plain_feedback)
        # 4. compute clipping boundaries for d_feedback
        m = np.maximum(np.abs(plain_feedback), self.tau_moving_average)
        # clip and return
        d_feedback = np.clip(d_feedback, -m, m)
        return plain_feedback - d_feedback

    @property
    def name(self) -> str:
        return "Torque Force Feedback"


# Buggy - Do not Use
"""
class PositionForceFeedback(ForceFeedback):
    def __init__(self):
        super().__init__()
        # PD gains
        self.pgain = 0.1 * np.array(
            [400.0, 400.0, 400.0, 400.0, 100.0, 100.0, 20.0], dtype=np.float64
        )
        self.dgain = 0.3 * np.array(
            [50.0, 50.0, 50.0, 50.0, 15.0, 15.0, 5.0], dtype=np.float64
        )
    def feedback(self, primary_ctrl, replica_ctrl):
        # hacky trick since initalization takes longer --> first iteration robot may be NONE, we catch that here
        if replica_ctrl.robot is None:
            print("Robot is NONE")
            return self.zero_tau
        primary_bot = primary_ctrl.robot
        replica_bot = replica_ctrl.robot

        # calculate external endeffector force from virtual fixtures
        # r_J = primary_bot.getJacobian()
        # force_vect = replica_ctrl.virtual_repel_force(replica_bot)

        td = 80
        if td > len(primary_ctrl.history):
            td = len(primary_ctrl.history)
        tau = self.pgain * (
                replica_bot.current_j_pos - primary_ctrl.history[-td]) - self.dgain * primary_bot.current_j_vel

        tau = self._gainsquish(tau)
        # return tau + r_J.T.dot(force_vect)
        return tau

    @property
    def name(self) -> str:
        return "Position Force Feedback"
"""
