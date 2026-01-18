import threading

import numpy as np


class GainAdjuster(threading.Thread):
    def __init__(self, controller_gain):
        super().__init__(name="gain_adjuster")

        self.controller_gain = controller_gain
        self.length = len(self.controller_gain)
        self.pos = 0
        self.increment = 0.1
        self._float_precision = 2

    def print_gain(self):
        space_per_number = self._float_precision + 3
        print("*" * (space_per_number * self.length + 4))
        gain_str = "* "
        for g in self.controller_gain:
            gain_str += f"{g:.{self._float_precision}f} "
        gain_str += " *"
        print(gain_str)
        arrow_str = "*"
        arrow_str += " " * (space_per_number // 2)
        arrow_str += " " * (self.pos * space_per_number)
        arrow_str += "^"
        arrow_str += " " * (space_per_number // 2 + space_per_number % 2 + 1)
        arrow_str += " " * ((self.length - self.pos - 1) * space_per_number)
        arrow_str += "*"
        print(arrow_str)
        print("*" * (space_per_number * self.length + 4))

    def convert_to_numpy(self, gain_str):
        gain = np.zeros(self.length)
        gain_strs = gain_str.split(" ")
        assert len(gain_strs) == self.length
        for idx, g_str in enumerate(gain_strs):
            gain[idx] = g_str
        return gain

    def run(self):
        # load manual gains if necessary
        # self.controller.gain = self.convert_to_numpy("0.96 0.14 0.70 0.91 0.70 1.90 1.05")
        while True:
            print("Adjust with WASD")
            self.print_gain()
            cmd = str(input("")).upper()
            if cmd == "A":
                self.pos = (self.pos - 1) % self.length
            elif cmd == "D":
                self.pos = (self.pos + 1) % self.length
            elif cmd == "W":
                self.controller_gain[self.pos] += self.increment
            elif cmd == "S":
                self.controller_gain[self.pos] -= self.increment
