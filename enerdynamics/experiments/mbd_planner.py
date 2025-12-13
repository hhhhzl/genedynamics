from __future__ import annotations

import argparse

from enerdynamics.solvers.mbd import DiffusionArgs, run_diffusion

# Legacy implementation kept for reference
# New code should use run_diffusion from enerdynamics.solvers.mbd


def main():
    parser = argparse.ArgumentParser("Diffusion planner for the double-integrator box environment.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_name", type=str, default="double_integrator_box")
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--Nsample", type=int, default=2048)
    parser.add_argument("--Ndiffuse", type=int, default=100)
    parser.add_argument("--temp_sample", type=float, default=0.1)
    parser.add_argument("--beta0", type=float, default=1e-4)
    parser.add_argument("--betaT", type=float, default=1e-2)
    parser.add_argument("--action_limit", type=float, default=1.0)
    parser.add_argument("--no_verbose", action="store_true")

    cli_args = parser.parse_args()
    diff_args = DiffusionArgs(
        seed=cli_args.seed,
        horizon=cli_args.horizon,
        dt=cli_args.dt,
        Nsample=cli_args.Nsample,
        Ndiffuse=cli_args.Ndiffuse,
        temp_sample=cli_args.temp_sample,
        beta0=cli_args.beta0,
        betaT=cli_args.betaT,
        action_limit=cli_args.action_limit,
        verbose=not cli_args.no_verbose,
    )
    diff_args.env_name = cli_args.env_name
    run_diffusion(diff_args)


if __name__ == "__main__":
    main()

