import abc
import os
import pickle
import threading
import time
import warnings

import numpy as np

from ..controller.human_controller import HumanController
from ..util.teaching_log import TeachingLog


class SchedulerInterface(abc.ABC):
    @abc.abstractmethod
    def start_control(self):
        raise NotImplementedError

    @abc.abstractmethod
    def stop_control(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reset_position(self):
        raise NotImplementedError

    @abc.abstractmethod
    def start_logging(self, start_time=None):
        raise NotImplementedError

    @abc.abstractmethod
    def stop_logging(self):
        raise NotImplementedError

    @abc.abstractmethod
    def snapshot(self):
        raise NotImplementedError


class HumanControlScheduler(SchedulerInterface, threading.Thread):
    def __init__(
        self,
        robot,
        controller: HumanController,
        teaching_log: TeachingLog,
        use_inv_dyn: bool = False,
        name="teleoperation",
    ):

        threading.Thread.__init__(self, name=name)
        self.controllerLock = threading.Lock()

        self.controller = controller
        self.robot = robot
        print("robot initialized")
        self.robot.use_inv_dyn = use_inv_dyn

        self.default_position = np.asarray(
            [0.0, 0.2758, 0.0, -2.0917, 0.0, 2.3341, 0.7854]
        )

        self.teaching_log = teaching_log
        self.doControl = False

    def start_control(self):
        self.doControl = True
        self.start()

    def run(self):
        if self.controller is None:
            warnings.warn("HumanScheduler: Controller is None")
            return

        with self.controllerLock:
            self.controller.initialize()
        # short period of waiting to make sure that the other controller is initialized in a teleop setup
        time.sleep(1)

        while self.doControl:
            with self.controllerLock:
                tau, ctrl_action, curr_load = self._ctrl_step()
                self.teaching_log.log_entry(
                    self.robot, tau, ctrl_action, curr_load, time.time()
                )

    def get_human_control_action(self, tau, curr_load):
        ext_tau_fb_gain = 0.3 * np.array([1.4, 1.4, 1.4, 1.4, 1.2, 1.7, 2.0])
        ctrl_action = tau - ext_tau_fb_gain * curr_load
        return ctrl_action

    def _ctrl_step(self):
        tau = self.controller.get_tau()
        curr_load = self.controller.get_load()

        ctrl_action = self.get_human_control_action(tau, curr_load)

        self.robot.command = ctrl_action
        self.robot.nextStep()
        return tau, ctrl_action, curr_load

    def stop_control(self):
        self.doControl = False
        self.join()
        self.stop_logging()

    def reset_position(self):
        with self.controllerLock:
            self.robot.gotoJointController.trajectory = None
            # self.robot.gotoJointPosition(self.default_position)
            self.controller.reset()
            # allow manuel control without a controller.
            self.robot.activeController = None

    def start_logging(self, start_time=None):
        with self.controllerLock:
            self.teaching_log.start_logging(start_time=start_time)

    def stop_logging(self):
        with self.controllerLock:
            self.teaching_log.stop_logging()

    def snapshot(self):
        """
        method to capture the current position and the current external forces and log it. It was used to gain data
        for a neural network correction of the load.
        """
        root_dir = self.teaching_log.root_dir
        file_name = os.path.join(root_dir, f"snapshot_{self.name}.dat")
        try:
            f = open(file_name, "rb")
        except IOError:
            # create file
            data = {"pos": [], "load": [], "tau_ext_hat_filtered": []}
            pickle.dump(data, open(file_name, "wb"))
        with open(os.path.join(root_dir, f"snapshot_{self.name}.dat"), "rb") as file:
            current_data = pickle.load(file)
        current_data["pos"].append(self.robot.current_j_pos)
        current_data["load"].append(self.controller.get_load(receive_state=False))
        current_data["tau_ext_hat_filtered"].append(self.robot.tau_ext_hat_filtered)
        pickle.dump(current_data, open(file_name, "wb"))
