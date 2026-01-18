import time

from ..controller.teleop_controller import (
    TeleopPrimaryController,
    TeleopReplicaController,
)
from ..util import force_feedback
from ..util.teaching_log import TeachingLog, TeleopMetaLogger
from .human_scheduler import HumanControlScheduler, SchedulerInterface


class TeleopPrimaryScheduler(HumanControlScheduler):
    def __init__(
        self,
        primary_robot,
        primary_ctrl: TeleopPrimaryController,
        replica_ctrl: TeleopReplicaController,
        teleop_log: TeachingLog,
        use_inv_dyn: bool = False,
    ):
        super(TeleopPrimaryScheduler, self).__init__(
            primary_robot, primary_ctrl, teleop_log, use_inv_dyn, name="Primary"
        )
        self.replica_ctrl = replica_ctrl

    def _ctrl_step(self):
        tau = self.controller.get_tau(self.replica_ctrl)
        curr_load = self.controller.get_load()
        ctrl_action = self.get_human_control_action(tau, curr_load)

        self.robot.command = ctrl_action
        self.robot.nextStep()

        if self.robot.gripper_width <= 0.003:
            self.replica_ctrl.robot.set_gripper_cmd_type = 1  # Grasp
        else:
            self.replica_ctrl.robot.set_gripper_cmd_type = 2  # Move
        self.replica_ctrl.robot.set_gripper_width = self.robot.gripper_width

        return tau, ctrl_action, curr_load

    def set_feedback_mode(self, fb: force_feedback.ForceFeedback):
        self.controller.set_feedback_mode(fb)


class TeleopReplicaScheduler(HumanControlScheduler):
    def __init__(
        self,
        replica_robot,
        replica_ctrl: TeleopReplicaController,
        primary_ctrl: TeleopPrimaryController,
        teleop_log: TeachingLog,
        use_inv_dyn: bool = False,
    ):
        super(TeleopReplicaScheduler, self).__init__(
            replica_robot, replica_ctrl, teleop_log, use_inv_dyn, name="Replica"
        )
        self.primary_ctrl = primary_ctrl

    def _ctrl_step(self):
        tau = self.controller.get_tau(self.primary_ctrl)
        self.robot.command = tau
        self.robot.nextStep()
        return tau, tau, self.controller.get_load()

    def reset_position(self):
        self.controller.reset()


class TeleopMetaScheduler(SchedulerInterface):
    def __init__(
        self,
        primary_robot,
        replica_robot,
        primary_ctrl,
        replica_ctrl,
        teleop_log: TeleopMetaLogger,
        use_inv_dyn: bool = False,
    ):
        self.primary_thread = TeleopPrimaryScheduler(
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

        self.fb_modes = [
            force_feedback.NoForceFeedback(),
            force_feedback.TorqueForceFeedback(),
        ]

    def set_fb_mode(self, i: int):
        try:
            if not isinstance(i, int):
                i = int(i)
            self.primary_thread.set_feedback_mode(self.fb_modes[i])
            print("Activating {}".format(self.fb_modes[i].name))
        except IndexError as e:
            "Could not set Feedback Mode {}".format(i)

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
