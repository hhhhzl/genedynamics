from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import time
from enerdynamics.control.edoc import EDOCArgs, run_edoc
from enerdynamics.experiments.mbd_planner import DiffusionArgs, run_diffusion


def compare(seed: int, horizon: int, dt: float, noise_std: float,
            Nsample: int, Ndiffuse: int, temp_sample: float, action_limit: float,
            beta0: float, betaT: float, save_path: str | None):
    edoc_args = EDOCArgs(
        seed=seed,
        horizon=horizon,
        dt=dt,
        noise_std=noise_std,
        verbose=False,
        action_nsample=Nsample
    )
    start = time.time()
    edoc_out = run_edoc(edoc_args)
    print("Running for edoc:", round(time.time()-start, 2))

    diff_args = DiffusionArgs(
        seed=seed,
        horizon=horizon,
        dt=dt,
        Nsample=Nsample,
        Ndiffuse=Ndiffuse,
        temp_sample=temp_sample,
        action_limit=action_limit,
        beta0=beta0,
        betaT=betaT,
        verbose=False,
    )
    start = time.time()
    diff_out = run_diffusion(diff_args)
    print("Running for mbd:", round(time.time() - start, 2))

    edoc_states = np.asarray(edoc_out["states"])
    diff_states = np.asarray(diff_out["states"])
    edoc_energies = np.asarray(edoc_out.get("energies", []), dtype=np.float32)
    diff_energies = np.asarray(diff_out.get("energies", []), dtype=np.float32)

    T_edoc = edoc_states.shape[0] - 1
    T_diff = diff_states.shape[0] - 1
    time_edoc = np.arange(T_edoc + 1) * dt
    time_diff = np.arange(T_diff + 1) * dt

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)

    axs[0].plot(time_edoc, edoc_states[:, 0], label="EDOC position")
    axs[0].plot(time_diff, diff_states[:, 0], label="MBD position", linestyle="--")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("position")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(time_edoc, edoc_states[:, 1], label="EDOC velocity")
    axs[1].plot(time_diff, diff_states[:, 1], label="MBD velocity", linestyle="--")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("velocity")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    fig.suptitle("Double Integrator Trajectories")
    fig.tight_layout()

    fig_energy, ax_energy = plt.subplots(1, 1, figsize=(6, 4))
    if edoc_energies.size > 0:
        time_energy_edoc = np.arange(edoc_energies.shape[0]) * dt
        ax_energy.plot(time_energy_edoc, edoc_energies, label="EDOC energy")
    if diff_energies.size > 0:
        time_energy_diff = np.arange(diff_energies.shape[0]) * dt
        ax_energy.plot(time_energy_diff, diff_energies, label="MBD energy", linestyle="--")
    ax_energy.set_xlabel("time")
    ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Energy along rollout")
    ax_energy.legend()
    ax_energy.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        energy_path = save_path.replace(".png", "_energy.png")
        fig_energy.savefig(energy_path, dpi=150)
    else:
        plt.show()

    print(f"EDOC total reward: {float(np.sum(np.asarray(edoc_out['rewards']))):.3f}")
    print(f"MBD total reward: {float(diff_out['total_reward']):.3f}")
    if edoc_energies.size > 0:
        print(f"EDOC total energy: {float(np.sum(edoc_energies)):.3f}")
        print(f"EDOC final energy: {float(edoc_energies[-1]):.3f}")
    if diff_energies.size > 0:
        print(f"MBD total energy: {float(np.sum(diff_energies)):.3f}")
        print(f"MBD final energy: {float(diff_energies[-1]):.3f}")
    print(f"Initial state (MBD): {edoc_out['initial_state']}")
    print(f"Initial state (Diffusion): {np.asarray(diff_out['initial_state'])}")
    print(f"Final state (EDOC): {edoc_states[-1]}")
    print(f"Final state (MBD): {diff_states[-1]}")


def main():
    parser = argparse.ArgumentParser("Compare EDOC and diffusion planners on Double Integrator.")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--Nsample", type=int, default=128)
    parser.add_argument("--Ndiffuse", type=int, default=100)
    parser.add_argument("--temp_sample", type=float, default=0.1)
    parser.add_argument("--action_limit", type=float, default=1.0)
    parser.add_argument("--beta0", type=float, default=1e-4)
    parser.add_argument("--betaT", type=float, default=1e-2)
    parser.add_argument("--save_path", type=str, default='trajectories.png')

    args = parser.parse_args()
    compare(
        seed=args.seed,
        horizon=args.horizon,
        dt=args.dt,
        noise_std=args.noise_std,
        Nsample=args.Nsample,
        Ndiffuse=args.Ndiffuse,
        temp_sample=args.temp_sample,
        action_limit=args.action_limit,
        beta0=args.beta0,
        betaT=args.betaT,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()

