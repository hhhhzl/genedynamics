import sys
import threading

import numpy as np

from environments.d3il.d3il_sim.sims.sl.multibot_teleop.src.force_feedback_controller import (
    ForceFeedbackController,
)


class GeneralCLI(threading.Thread):
    def __init__(self):
        super().__init__(name="Teleoperation CLI")
        self._func_map = {"Q": lambda: False}
        self._instruction_map = {"Q": "Exit"}

    def register_function(self, key: str, label: str, fn: callable):
        _k = key.upper()
        self._instruction_map[_k] = label
        self._func_map[_k] = fn

    def remove_function(self, key: str):
        _k = key.upper()
        if _k in self._func_map:
            del self._instruction_map[_k]
            del self._func_map[_k]

    def print_instructions(self):
        print()
        for key, text in self._instruction_map.items():
            print("({}) {}".format(key, text))

    def exec_cmd(self, cmd: str):
        if cmd not in self._func_map:
            print("Wrong Command!")
            return True

        func = self._func_map[cmd]
        cli_continue = func()

        if cli_continue is None:
            cli_continue = True
        return cli_continue

    def get_input(self) -> bool:
        self.print_instructions()
        cmd = str(input("Enter Command... ")).upper()
        return self.exec_cmd(cmd)

    def run(self):
        while self.get_input():
            continue


class ForceFeedbackCliMap(GeneralCLI):
    def __init__(self, force_feedback_controller: ForceFeedbackController):
        super().__init__()
        self.register_function(
            "1",
            "Enable Force Feedback",
            force_feedback_controller.enable_force_feedback,
        )
        self.register_function(
            "0",
            "Disable Force Feedback",
            force_feedback_controller.disable_force_feedback,
        )
        # self.register_function("R", "Reset Robot", scheduler.reset_position)


class VTwinCLI(GeneralCLI):
    def __init__(self, primary_robot, primary_ctrl, replica_robot, replica_ctrl):
        super().__init__()
        self.register_function("Q", "close", self.cmd_close)
        self.register_function("L", "log", self.cmd_start_log)
        self.register_function("P", "pause", self.cmd_stop_log)
        self.register_function("R", "reset", self.cmd_reset)

        self.prim_robot = primary_robot
        self.prim_ctrl = primary_ctrl
        self.repl_robot = replica_robot
        self.repl_ctrl = replica_ctrl

    def cmd_close(self):
        # Hack to shutdown the controllers and "break" Run Loop
        self.repl_ctrl._max_duration = 0
        self.prim_ctrl._max_duration = 0
        return False

    def cmd_start_log(self):
        print("Start Logging")
        self.prim_robot.scene.start_logging()
        self.repl_robot.scene.start_logging()

    def cmd_stop_log(self):
        print("Stop Logging")
        self.prim_robot.scene.stop_logging()
        self.repl_robot.scene.stop_logging()

    def cmd_reset(self):
        print("Reset All")
        self.prim_ctrl._max_duration = 0
        self.prim_robot.go_home()

        self.repl_robot.scene.reset()
        self.repl_ctrl.up()
        self.repl_robot.beam_to_joint_pos(self.prim_robot.current_j_pos)

        self.prim_ctrl.executeController(self.prim_robot, maxDuration=1000, block=False)
        self.repl_ctrl.executeController(self.repl_robot, maxDuration=1000, block=False)
