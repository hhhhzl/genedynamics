import copy
import logging
import multiprocessing
from multiprocessing import Pipe, Process

import numpy as np

from environments.d3il.d3il_sim import gym


def _subprocess_sampling(pipe, policy, env: gym.Env, n_steps: int):
    """
    private function for multiprocessing sampling
    gathers n_steps trajectories in the environment according to the policy
    Args:
        pipe: a Pipe() connection tuple for communicating with the parent process
        policy: object which predicts an action for each env state. See GenericPolicy
        env: a gym.Env instance
        n_steps: number of steps to be collected in this sub_process

    Returns:
        a list of all observations, actions, rewards
    """
    # State, action, next_state, reward
    # --> numpy Array mit initialisierter groesse

    # Use multiprocessing.get_logger() for printing
    mpl = multiprocessing.get_logger()
    mpl.info("Starting...")

    # Save initial state and initialize lists
    samples = []
    start_obs = env.reset()

    for _ in range(n_steps):
        # Use Policy to predict an action
        action = policy.predict(start_obs)

        # Observe the next state
        next_obs, reward, done, _ = env.step(action)
        samples.append(np.array([start_obs, action, reward, next_obs], dtype=object))

        # If an episode is finished, reset the environment
        if done:
            start_obs = env.reset()
        else:
            start_obs = next_obs

    trajectories = np.array(samples)
    mpl.info("Closing down...")
    pipe[1].send(trajectories)
    pipe[1].close()


class ParallelSampler:
    """
    Class to wrap all the multiprocessing sampling functionality
    """

    def __init__(
        self, env_cls: gym.Env.__class__, num_processes: int = 2, *args, **kwargs
    ):
        """
        Args:
            env_cls: Class (!) to a gym.Env
            num_processes: number of parallel processes
            *args: args for Env.__init__()
            **kwargs: kwargs for Env.__init__()
        """
        self.env_cls = env_cls
        self.n = num_processes
        self.args = args
        self.kwargs = kwargs

        self.pipes = []
        self.ps = []

        # Initialize the logger for multiprocessing
        mpl = multiprocessing.log_to_stderr()
        mpl.setLevel(logging.INFO)
        pass

    def sample(self, policy, num_samples):
        """
        Collect n samples divided across all n processes
        Args:
            policy: a policy object predicting an action for each env state
            num_samples: the number of samples to be collected

        Returns:
            a list of observation action reward lists
        """
        # Equally divide work.
        sub_samples = int(num_samples / self.n)
        overhang = num_samples % self.n
        step_packages = [sub_samples for _ in range(self.n)]
        for i in range(overhang):
            step_packages[i] += 1

        self.pipes = [Pipe() for _ in range(self.n)]
        self.ps = [
            Process(
                target=_subprocess_sampling,
                args=(
                    pipe,
                    copy.deepcopy(policy),
                    self.env_cls(*self.args, **self.kwargs),
                    steps,
                ),
            )
            for (pipe, steps) in zip(self.pipes, step_packages)
        ]

        for p in self.ps:
            p.start()

        tmp = []
        for pipe in self.pipes:
            tmp.append(pipe[0].recv())
            pipe[0].close()
        return np.vstack(tmp)


"""
Function interface for a policy.
Stable Baselines already supports this signature
"""


class GenericPolicy:
    def predict(self, observations):
        return [None]
