import pickle
import time

import numpy as np

from environments.d3il.d3il_sim.sims.sl.teleoperation.mockup_tests.follow_trajectory.trajectory_controller import (
    TrajectoryController,
)
from environments.d3il.d3il_sim.sims.sl.teleoperation.mockup_tests.follow_trajectory.trajectory_robot import (
    TrajectoryMockupRobot,
)
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.controller.teleop_controller import (
    TeleopReplicaController,
)
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.schedulers.human_scheduler import (
    HumanControlScheduler,
    SchedulerInterface,
)
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.schedulers.teleop_scheduler import (
    TeleopReplicaScheduler,
)
from environments.d3il.d3il_sim.sims.sl.teleoperation.src.util.teaching_log import (
    TeachingLog,
    TeleopMetaLogger,
)


class TrajectoryScheduler(HumanControlScheduler):
    """
    This is a research and testing class for teleoperation. It implements a replica Scheduler, which follows a
    fixed logged trajectory instead of a human input on the primary robot. This deterministic behavior can be used
    to study the behavior of different gains on the same environment.
    """

    def __init__(
        self,
        trajectory_log,
        primary_robot: TrajectoryMockupRobot,
        primary_ctrl: TrajectoryController,
        replica_ctrl: TeleopReplicaController,
        teleop_log: TeachingLog,
        use_inv_dyn: bool = False,
    ):
        super().__init__(
            primary_robot,
            primary_ctrl,
            teleop_log,
            use_inv_dyn,
            name="Primary",
        )
        self.replica_ctrl = replica_ctrl
        logs = pickle.load(open(trajectory_log, "rb"))
        data = logs["data"]
        self.pos_data = data["j_pos"]
        self.vel_data = data["j_vel"]
        # time period between an update of pos and vel of the robot
        self.sleep_length = np.mean(np.diff(data["timestamp"]))
        self.index = 0
        self.follow_trajectory = False

    def _ctrl_step(self):
        """
        Main method of the trajectory scheduler: Loads a new postion for the robot from the logs
        """
        if self.follow_trajectory:
            self.robot.set_state(self.pos_data[self.index], self.vel_data[self.index])
            self.index += 1
            if self.index >= len(self.pos_data):
                # stop
                self.index -= 1
                self.follow_trajectory = False
                print("Reached end of trajectory!")
        else:
            self.robot.set_state(self.pos_data[self.index], np.zeros(7))
        # waits to match the original pace
        time.sleep(self.sleep_length)
        # return tau, control action, curr_load as dummy values
        return 0.0, 0.0, 0.0

    def set_follow_trajectory(self, follow_trajectory):
        self.follow_trajectory = follow_trajectory


class MetaTrajectoryScheduler(SchedulerInterface):
    """
    To obtain the same structure as the real teleoperation setup, a Meta scheduler is implemented here. The primary
    robot is just a dummy implementation.
    """

    def __init__(
        self,
        trajectory_log,
        primary_robot,
        replica_robot,
        primary_ctrl,
        replica_ctrl,
        teleop_log: TeleopMetaLogger,
        use_inv_dyn: bool = False,
    ):
        self.primary_thread = TrajectoryScheduler(
            trajectory_log,
            primary_robot,
            primary_ctrl,
            replica_ctrl,
            teleop_log.primary_log,
            use_inv_dyn,
        )
        self.replica_thread = TeleopReplicaScheduler(
            replica_robot,
            replica_ctrl,
            primary_ctrl,
            teleop_log.replica_log,
            use_inv_dyn,
        )

    def start_control(self):
        self.primary_thread.start_control()
        self.replica_thread.start_control()

    def stop_control(self):
        self.primary_thread.stop_control()
        self.replica_thread.stop_control()

    def reset_position(self):
        self.primary_thread.reset_position()
        self.replica_thread.reset_position()

    def start_logging(self, start_time=None):
        if start_time is None:
            start_time = time.time()
        self.primary_thread.start_logging(start_time=start_time)
        self.replica_thread.start_logging(start_time=start_time)

    def stop_logging(self):
        self.primary_thread.stop_logging()
        self.replica_thread.stop_logging()

    def snapshot(self):
        self.primary_thread.snapshot()
        self.replica_thread.snapshot()
