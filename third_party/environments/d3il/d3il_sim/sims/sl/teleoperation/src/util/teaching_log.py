import os
import pickle
import time
from collections import defaultdict

import numpy as np


class TeachingLog:
    def __init__(self, root_dir: str, log_name: str, real_robot=True):
        self.root_dir = root_dir
        self.log_name = log_name
        self.logging = False
        self.real_robot = real_robot

        # dictionary to store key-value pairs of variable names and lists
        self._log_data = defaultdict(list)
        self.start_time = 0
        self.log_counter = 0

    def start_logging(self, start_time=None):
        if self.logging:
            return

        print("START LOG {} {:03d}".format(self.log_name, self.log_counter))
        if start_time is None:
            self.start_time = time.time()
        self.logging = True
        self._log_data.clear()

    def stop_logging(self):
        if not self.logging:
            return

        print("STOP LOG {} {:03d}".format(self.log_name, self.log_counter))

        self.logging = False
        self.save_log()
        self.log_counter += 1

    def log_entry(self, robot, tau_raw, tau_cmd, curr_load, ts):
        self._log_data["tau"].append(tau_cmd)
        self._log_data["tau_raw"].append(tau_raw)
        self._log_data["curr_load"].append(curr_load)
        # for extensive logging, uncomment these:
        # self._log_data["raw_load"].append(robot.current_load)
        # self._log_data["raw_gravity"].append(robot.gravity)
        # self._log_data["raw_coriolis"].append(robot.coriolis)
        # self._log_data["tau_ext_hat_filtered"].append(robot.tau_ext_hat_filtered)

        self._log_data["c_pos"].append(robot.current_c_pos)
        self._log_data["c_vel"].append(robot.current_c_vel)
        self._log_data["c_quat"].append(robot.current_c_quat)
        self._log_data["gripper"].append(robot.gripper_width)

        self._log_data["j_pos"].append(robot.current_j_pos)
        self._log_data["j_vel"].append(robot.current_j_vel)

        self._log_data["power"].append(np.dot(curr_load, robot.current_j_vel))

        self._log_data["timestamp"].append(ts)
        self._log_data["rel_timestamp"].append(ts - self.start_time)

    def save_log(self):
        log_dir = os.path.join(self.root_dir, "data")
        os.makedirs(log_dir, exist_ok=True)

        log_obj = {
            "name": self.log_name,
            "trial_number": self.log_counter,
            "data": self._numpy_log(),
        }

        f_name = "{}_{:03d}.pkl".format(self.log_name, self.log_counter)
        while os.path.exists(os.path.join(log_dir, f_name)):
            self.log_counter += 1
            f_name = "{}_{:03d}.pkl".format(self.log_name, self.log_counter)

        with open(os.path.join(log_dir, f_name), "wb") as f:
            pickle.dump(log_obj, f)

    def _numpy_log(self) -> np.ndarray:
        list_data = self._log_data.copy()
        np_data = {key: np.array(log) for (key, log) in list_data.items()}
        return np_data

    @staticmethod
    def _get_cli_input():
        print()
        root_dir = str(input("Enter LogDir Path: [./] ") or "./")
        log_name = str(
            input("Enter LogFile Name: [trajectory] ") or "trajectory"
        ).casefold()
        return root_dir, log_name

    @classmethod
    def cli_prompt(cls):
        root_dir, log_name = cls._get_cli_input()
        teaching_log = cls(root_dir, log_name)
        return teaching_log


class TeleopMetaLogger:
    def __init__(self, root_dir: str, log_name: str):
        """
        Container for the two loggers, which both get distributed to the two schedulers.
        """
        self.primary_log = TeachingLog(root_dir, log_name + "_primary")
        self.replica_log = TeachingLog(root_dir, log_name + "_replica")
