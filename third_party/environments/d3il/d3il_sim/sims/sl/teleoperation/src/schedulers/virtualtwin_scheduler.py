from .teleop_scheduler import TeleopMetaScheduler, TeleopReplicaScheduler


class VirtualTwinReplicaScheduler(TeleopReplicaScheduler):
    def __init__(
        self,
        replica_robot,
        replica_ctrl,
        primary_ctrl,
        teleop_log,
        use_inv_dyn: bool = False,
        virtual_callback=None,
    ):
        super().__init__(
            replica_robot, replica_ctrl, primary_ctrl, teleop_log, use_inv_dyn
        )
        self._virt_callback = virtual_callback

    def _ctrl_step(self):
        prim_j = self.primary_ctrl.robot.current_j_pos
        prim_j_vel = self.primary_ctrl.robot.current_j_vel
        # Skip ReplicaController and do JPos Control directly instead
        tau = -1
        self.robot.jointTrackingController.setSetPoint(prim_j, prim_j_vel)
        self.robot.jointTrackingController.executeControllerTimeSteps(
            self.robot, timeSteps=1
        )
        return tau, tau, tau

    def reset_position(self):
        with self.controllerLock:
            if self._virt_callback is not None:
                self._virt_callback(self.robot.scene)

            self.robot.scene.reset()
            self.robot.beam_to_joint_pos(self.primary_ctrl.robot.current_j_pos)


class VirtualTwinMetaScheduler(TeleopMetaScheduler):
    def __init__(
        self,
        primary_robot,
        replica_robot,
        primary_ctrl,
        replica_ctrl,
        teleop_log,
        use_inv_dyn: bool = False,
        virtual_callback=None,
    ):
        super(VirtualTwinMetaScheduler, self).__init__(
            primary_robot,
            replica_robot,
            primary_ctrl,
            replica_ctrl,
            teleop_log,
            use_inv_dyn,
        )
        self.replica_thread = VirtualTwinReplicaScheduler(
            replica_robot,
            replica_ctrl,
            primary_ctrl,
            teleop_log.replica_log,
            use_inv_dyn,
            virtual_callback,
        )
