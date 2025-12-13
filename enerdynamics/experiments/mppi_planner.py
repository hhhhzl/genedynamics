from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from enerdynamics.solvers.mppi import MPPIArgs, run_mppi


# Legacy implementation kept for reference
# New code should use run_mppi from enerdynamics.solvers.mppi


def main():
    import argparse

    parser = argparse.ArgumentParser("MPPI planner for enerdynamics environments.")
    parser.add_argument("--seed", type=int, default=MPPIArgs.seed)
    parser.add_argument("--env_name", type=str, default=MPPIArgs.env_name)
    parser.add_argument("--horizon", type=int, default=MPPIArgs.horizon)
    parser.add_argument("--dt", type=float, default=MPPIArgs.dt)
    parser.add_argument("--num_samples", type=int, default=MPPIArgs.num_samples)
    parser.add_argument("--num_iterations", type=int, default=MPPIArgs.num_iterations)
    parser.add_argument("--noise_sigma", type=float, default=MPPIArgs.noise_sigma)
    parser.add_argument("--lambda_", type=float, default=MPPIArgs.lambda_)
    parser.add_argument("--action_limit", type=float, default=MPPIArgs.action_limit)
    parser.add_argument("--verbose", action="store_true", default=False)

    cli_args = parser.parse_args()
    args = MPPIArgs(
        seed=cli_args.seed,
        env_name=cli_args.env_name,
        horizon=cli_args.horizon,
        dt=cli_args.dt,
        num_samples=cli_args.num_samples,
        num_iterations=cli_args.num_iterations,
        noise_sigma=cli_args.noise_sigma,
        lambda_=cli_args.lambda_,
        action_limit=cli_args.action_limit,
        verbose=cli_args.verbose,
    )
    run_mppi(args)


if __name__ == "__main__":
    main()

