import numpy as np

from environments.d3il.d3il_sim.sims.sl.multibot_teleop.src.human_controller import HumanController
from environments.d3il.d3il_sim.sims.sl.multibot_teleop.src.kalman_filter import KalmanFilter


def total_force_clip(replica_load, max_force=8.0, a=6.0, b=4.0):
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


class ForceFeedbackController(HumanController):
    def __init__(self, primary_robot, replica_robot):
        super().__init__(primary_robot)
        # Force feedback parameters
        self.fb_gain = -1.0 * np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64
        )

        self.torque_dgain = 0.5 * np.array(
            [50.0, 50.0, 50.0, 50.0, 15.0, 8.0, 8.0], dtype=np.float64
        )
        self.tau_moving_average = np.zeros(7)
        self.alpha = 0.9

        self._replica_load_filter = KalmanFilter(self.get_raw_load(replica_robot))
        self.replica_robot = replica_robot

        self._active_force_feedback = False

    def get_force_feedback(self):
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
        # 1. get force feedback from replica
        replica_load = self._replica_load_filter.get_filtered(
            self.get_raw_load(self.replica_robot)
        )
        plain_feedback = total_force_clip(
            self.fb_gain * replica_load, a=4.66, b=4.0, max_force=60.0
        )

        # 3. get d control feedback from primary
        primary_j_vel = self.primary_robot.current_j_vel
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

    def enable_force_feedback(self):
        print("Enabled Force Feedback!")
        self._active_force_feedback = True

    def disable_force_feedback(self):
        print("Disabled Force Feedback!")
        self._active_force_feedback = False

    def getControl(self, robot):
        human_control = super().getControl(robot)
        force_feedback = self.get_force_feedback()
        if self._active_force_feedback:
            return human_control + force_feedback
        else:
            return human_control
