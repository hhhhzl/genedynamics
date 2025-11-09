from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
from enerdynamics.control.edoc import EDOCArgs, run_edoc, make_energy, make_env
from enerdynamics.experiments.mbd_planner import DiffusionArgs, run_diffusion
from enerdynamics.experiments.mppi_planner import MPPIArgs, run_mppi
from enerdynamics.experiments.cem_planner import CEMArgs, run_cem

EDOC_COLOR = "#1f77b4"
MBD_COLOR = "#35B779"
MPPI_COLOR = "#ff7f0e"
CEM_COLOR = "#d95f02"
DIFFUSION_FRACTIONS = (0.1, 0.5, 0.9)
MAX_SAMPLE_TRAJ_PLOT = 80


def compare(seed: int, horizon: int, dt: float, noise_std: float,
            Nsample: int, Ndiffuse: int, temp_sample: float, action_limit: float,
            beta0: float, betaT: float, env_name: str, save_path: str | None):
    energy_fn = make_energy(env_name)
    env_template = make_env(env_name)
    if hasattr(env_template, "dt"):
        env_template.dt = dt
    if hasattr(env_template, "horizon"):
        env_template.horizon = horizon
    if hasattr(env_template, "control_limit"):
        env_template.control_limit = action_limit

    edoc_args = EDOCArgs(
        seed=seed,
        env_name=env_name,
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
        env_name=env_name,
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
    edoc_rewards = np.asarray(edoc_out.get("rewards", []), dtype=np.float32)
    diff_rewards = np.asarray(diff_out.get("rewards", []), dtype=np.float32)
    edoc_diffusion_actions = np.asarray(edoc_out.get("diffusion_actions_traj", []), dtype=np.float32)
    edoc_diffusion_samples = np.asarray(edoc_out.get("diffusion_sampled_actions", []), dtype=np.float32)
    mbd_diffusion_actions = np.asarray(diff_out.get("diffusion_actions_traj", []), dtype=np.float32)
    mbd_diffusion_samples = np.asarray(diff_out.get("diffusion_sampled_actions", []), dtype=np.float32)

    T_edoc = edoc_states.shape[0] - 1
    T_diff = diff_states.shape[0] - 1
    time_edoc = np.arange(T_edoc + 1) * dt
    time_diff = np.arange(T_diff + 1) * dt
    reward_time = np.arange(edoc_rewards.shape[0]) * dt if edoc_rewards.size else None

    state_dim = edoc_states.shape[1]
    pos_dim = state_dim // 2

    fig_state_timeseries = None
    fig_energy_field = None

    if pos_dim <= 1:
        fig_traj, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)
        axs[0].plot(time_edoc, edoc_states[:, 0], label="EDOC position")
        axs[0].plot(time_diff, diff_states[:, 0], label="MBD position", linestyle="--")
        axs[0].set_xlabel("time")
        axs[0].set_ylabel("position")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(time_edoc, edoc_states[:, pos_dim], label="EDOC velocity")
        axs[1].plot(time_diff, diff_states[:, pos_dim], label="MBD velocity", linestyle="--")
        axs[1].set_xlabel("time")
        axs[1].set_ylabel("velocity")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        fig_traj.suptitle("Double Integrator Trajectories")
        fig_traj.tight_layout()
    else:
        fig_traj, ax_traj = plt.subplots(1, 1, figsize=(6, 6))
        ax_traj.plot(edoc_states[:, 0], edoc_states[:, 1], label="EDOC trajectory")
        ax_traj.plot(diff_states[:, 0], diff_states[:, 1], label="MBD trajectory", linestyle="--")
        ax_traj.scatter(edoc_states[0, 0], edoc_states[0, 1], c="green", marker="o", label="start")
        ax_traj.scatter(edoc_states[-1, 0], edoc_states[-1, 1], c="blue", marker="x", label="EDOC final")
        ax_traj.scatter(diff_states[-1, 0], diff_states[-1, 1], c="red", marker="^", label="MBD final")
        ax_traj.set_xlabel("x position")
        ax_traj.set_ylabel("y position")
        ax_traj.set_title("Planar Trajectories")
        ax_traj.legend()
        ax_traj.grid(True, alpha=0.3)

        fig_state_timeseries, axs_state = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
        axs_state = axs_state.ravel()
        labels = [
            ("x position", 0),
            ("y position", 1),
            ("x velocity", pos_dim),
            ("y velocity", pos_dim + 1),
        ]
        series = [
            (time_edoc, edoc_states[:, labels[i][1]], "EDOC", "-", axs_state[i])
            for i in range(4)
        ]
        series += [
            (time_diff, diff_states[:, labels[i][1]], "MBD", "--", axs_state[i])
            for i in range(4)
        ]
        for t_vals, values, lbl, style, ax in series:
            ax.plot(t_vals, values, label=lbl, linestyle=style)
            ax.set_xlabel("time")
        for ax, (title, _) in zip(axs_state, labels):
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig_state_timeseries.tight_layout()

        # Energy field visualization with diffusion snapshots
        grid_points = 101
        x_vals = np.linspace(-env_template.p_max, env_template.p_max, grid_points)
        y_vals = np.linspace(-env_template.p_max, env_template.p_max, grid_points)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        energy_grid = np.zeros_like(X_grid, dtype=np.float32)
        zero_action = np.zeros(env_template.act_dim, dtype=np.float32)
        for j in range(y_vals.size):
            for i in range(x_vals.size):
                state = np.array([X_grid[j, i], Y_grid[j, i], 0.0, 0.0], dtype=np.float32)
                energy_grid[j, i] = float(energy_fn.compute(state, zero_action, {"t": 0}))

        stages = [
            ("Diffusion 10%", 0.1),
            ("Diffusion 50%", 0.5),
            ("Diffusion 90%", 0.9),
            ("Final Plan", None),
        ]
        fig_energy_field, axs_field = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
        if not isinstance(axs_field, np.ndarray):
            axs_field = np.array([axs_field])

        for ax, (stage_label, frac) in zip(axs_field, stages):
            cs = ax.contourf(X_grid, Y_grid, energy_grid, levels=40, cmap="viridis")

            def plot_trajs(out, color_base, diffusion_actions, diffusion_samples):
                initial_state = np.asarray(out["initial_state"], dtype=np.float32)
                if frac is None:
                    action_seq = np.asarray(out["actions"], dtype=np.float32)
                    title = "Final"
                else:
                    if diffusion_actions.size == 0 or diffusion_actions.ndim != 3:
                        return
                    total_steps_local = diffusion_actions.shape[0]
                    step = int((total_steps_local + 1) * frac)
                    if step > total_steps_local:
                        return
                    idx_local = total_steps_local - step
                    title = f"{int(frac * 100)}%"
                    action_seq = diffusion_actions[idx_local]

                    if diffusion_samples.size > 0 and diffusion_samples.ndim == 4:
                        samples_actions = diffusion_samples[idx_local]
                        samples_actions = np.asarray(samples_actions, dtype=np.float32)
                        if samples_actions.shape[0] > 0:
                            if samples_actions.shape[0] > 80:
                                sample_idx = np.linspace(
                                    0, samples_actions.shape[0] - 1, 80, dtype=int
                                )
                                samples_actions = samples_actions[sample_idx]
                            light_rgba = mcolors.to_rgba(color_base, alpha=0.3)
                            for acts in samples_actions:
                                sample_states = env_template.rollout_actions(initial_state, acts)
                                ax.plot(
                                    sample_states[:, 0],
                                    sample_states[:, 1],
                                    color=light_rgba,
                                    linewidth=1.0,
                                )

                traj_states = env_template.rollout_actions(initial_state, action_seq)
                ax.plot(traj_states[:, 0], traj_states[:, 1], color=color_base, linewidth=2.5)
                ax.scatter(
                    traj_states[0, 0],
                    traj_states[0, 1],
                    marker="o",
                    s=30,
                    facecolors="white",
                    edgecolors=color_base,
                    linewidths=1.0,
                )
                ax.scatter(
                    traj_states[-1, 0],
                    traj_states[-1, 1],
                    marker="*",
                    s=55,
                    facecolors=color_base,
                    edgecolors="black",
                    linewidths=0.5,
                )

            plot_trajs(edoc_out, "#1f77b4", edoc_diffusion_actions, edoc_diffusion_samples)
            plot_trajs(diff_out, "#ff7f0e", mbd_diffusion_actions, mbd_diffusion_samples)

            ax.set_title(stage_label)
            ax.set_xlabel("x position")
            ax.set_ylabel("y position")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)

        fig_energy_field.colorbar(cs, ax=axs_field.tolist(), shrink=0.9, label="Energy")

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

    fig_reward, ax_reward = plt.subplots(1, 1, figsize=(6, 4))
    if edoc_rewards.size > 0 and reward_time is not None:
        ax_reward.plot(reward_time, edoc_rewards, label="EDOC reward")
    if diff_rewards.size > 0:
        diff_reward_time = np.arange(diff_rewards.shape[0]) * dt
        ax_reward.plot(diff_reward_time, diff_rewards, label="MBD reward", linestyle="--")
    ax_reward.set_xlabel("time")
    ax_reward.set_ylabel("Reward")
    ax_reward.set_title("Instantaneous reward")
    ax_reward.legend()
    ax_reward.grid(True, alpha=0.3)

    if save_path is not None:
        base = save_path.rsplit(".", 1)[0]
        traj_path = save_path if save_path.endswith(".png") else f"{base}_traj.png"
        energy_path = f"{base}_energy.png"
        reward_path = f"{base}_reward.png"
        state_path = f"{base}_states.png"
        energy_field_path = f"{base}_energy_field.png"
        fig_traj.savefig(traj_path, dpi=150)
        fig_energy.savefig(energy_path, dpi=150)
        fig_reward.savefig(reward_path, dpi=150)
        if fig_state_timeseries is not None:
            fig_state_timeseries.savefig(state_path, dpi=150)
        if fig_energy_field is not None:
            fig_energy_field.savefig(energy_field_path, dpi=150)
    else:
        plt.show()

    print(f"EDOC total reward: {float(np.sum(edoc_rewards)):.3f}")
    print(f"MBD total reward: {float(np.sum(diff_rewards)):.3f}")
    if edoc_energies.size > 0:
        print(f"EDOC total energy: {float(np.sum(edoc_energies)):.3f}")
        print(f"EDOC final energy: {float(edoc_energies[-1]):.3f}")
    if diff_energies.size > 0:
        print(f"MBD total energy: {float(np.sum(diff_energies)):.3f}")
        print(f"MBD final energy: {float(diff_energies[-1]):.3f}")
    print(f"Initial state (EDOC): {np.asarray(edoc_out['initial_state'])}")
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
    parser.add_argument("--temp_sample", type=float, default=0.5)
    parser.add_argument("--action_limit", type=float, default=1.0)
    parser.add_argument("--beta0", type=float, default=1e-4)
    parser.add_argument("--betaT", type=float, default=1e-2)
    parser.add_argument("--env_name", type=str, default="double_integrator_box")
    parser.add_argument("--save_path", type=str, default="compare_di.png")

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
        env_name=args.env_name,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()

