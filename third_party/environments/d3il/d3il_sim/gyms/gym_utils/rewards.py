import numpy as np


class Distance:
    @staticmethod
    def linear(start, goal):
        return np.linalg.norm(start - goal, axis=-1)

    @staticmethod
    def quadratic(start, goal):
        ...

    @staticmethod
    def tanh(start, goal):
        ...
