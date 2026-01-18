import logging
import os
import pprint
import time

import matplotlib.pyplot as plt
import pandas as pd

from environments.d3il.d3il_sim.gym.gym_controllers import CartPosQuatCartesianRobotController
from environments.d3il.d3il_sim.gym.gym_utils.multiprocessing.parallel_sampling import (
    GenericPolicy,
    ParallelSampler,
)
from envs.reach_env.reach import ReachEnv

"""
Small test policy sampling randomly from a MocapController
"""


class RandomPolicy(GenericPolicy):
    def __init__(self):
        self.action_space = CartPosQuatCartesianRobotController().action_space

    def predict(self, obs):
        return self.action_space.sample()


DEFAULT_LOG = os.path.join(os.path.dirname(__file__), "timelog.csv")


def get_log_file(path: str = DEFAULT_LOG):
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    df = pd.DataFrame()
    return df


def save_log_file(df, path: str = DEFAULT_LOG):
    df.to_csv(path)


def main_time_sampling():
    N_CORES = 4
    SAMPLES = 50000

    df = get_log_file()

    start_t = time.perf_counter()
    p = ParallelSampler(ReachEnv, N_CORES, render=False)
    samples = p.sample(RandomPolicy(), SAMPLES)
    end_t = time.perf_counter()
    df = df.append(
        pd.Series({"cores": N_CORES, "samples": SAMPLES, "seconds": end_t - start_t}),
        ignore_index=True,
    )

    pprint.pprint(df)
    save_log_file(df)


def main_boxplot():
    df = get_log_file()
    df.boxplot("seconds", "cores", figsize=(8, 5))
    # df.groupby('cores')['seconds'].mean().plot.bar()
    plt.show()


if __name__ == "__main__":
    main_boxplot()
