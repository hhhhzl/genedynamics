from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from enerdynamics.solvers.cem import CEMArgs, run_cem


# Legacy implementation kept for reference
# New code should use run_cem from enerdynamics.solvers.cem


def main():
    import argparse

    parser = argparse.ArgumentParser("CEM planner for enerdynamics environments.")
    parser.add_argument("--seed", type=int, default=CEMArgs.seed)
    parser.add_argument("--env_name", type=str, default=CEMArgs.env_name)
    parser.add_argument("--horizon", type=int, default=CEMArgs.horizon)
    parser.add_argument("--dt", type=float, default=CEMArgs.dt)
    parser.add_argument("--num_samples", type=int, default=CEMArgs.num_samples)
    parser.add_argument("--num_iterations", type=int, default=CEMArgs.num_iterations)
    parser.add_argument("--elite_frac", type=float, default=CEMArgs.elite_frac)
    parser.add_argument("--init_std", type=float, default=CEMArgs.init_std)
    parser.add_argument("--min_std", type=float, default=CEMArgs.min_std)
    parser.add_argument("--action_limit", type=float, default=CEMArgs.action_limit)
    parser.add_argument("--verbose", action="store_true", default=False)

    cli_args = parser.parse_args()
    args = CEMArgs(
        seed=cli_args.seed,
        env_name=cli_args.env_name,
        horizon=cli_args.horizon,
        dt=cli_args.dt,
        num_samples=cli_args.num_samples,
        num_iterations=cli_args.num_iterations,
        elite_frac=cli_args.elite_frac,
        init_std=cli_args.init_std,
        min_std=cli_args.min_std,
        action_limit=cli_args.action_limit,
        verbose=cli_args.verbose,
    )
    run_cem(args)


if __name__ == "__main__":
    main()
