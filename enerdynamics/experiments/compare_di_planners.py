from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
from enerdynamics.solvers.single.edoc import run_edoc
from enerdynamics.solvers.single.mbd import run_mbd
from enerdynamics.solvers.single.mppi import run_mppi
from enerdynamics.solvers.single.cem import run_cem
from enerdynamics.envs.factories import make_energy, make_env
from configs.double_integrator_box.edoc import EDOCArgs
from configs.double_integrator_box.mbd import DiffusionArgs
from configs.double_integrator_box.mppi import MPPIArgs
from configs.double_integrator_box.cem import CEMArgs

EDOC_COLOR = "#1f77b4"
MBD_COLOR = "#ff7f0e"
MPPI_COLOR = "#35B779"
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
    diff_out = run_mbd(diff_args)
    print("Running for mbd:", round(time.time() - start, 2))

    shared_initial_state = np.asarray(edoc_out["initial_state"], dtype=np.float32)

    mppi_args = MPPIArgs(
        seed=seed,
        env_name=env_name,
        horizon=horizon,
        dt=dt,
        num_samples=Nsample,
        num_iterations=max(3, min(8, Ndiffuse // 10)) if Ndiffuse > 0 else 5,
        noise_sigma=noise_std if noise_std > 0 else 0.3,
        lambda_=1.0,
        action_limit=action_limit,
        verbose=False,
    )
    start = time.time()
    mppi_out = run_mppi(mppi_args, initial_state=shared_initial_state)
    print("Running for mppi:", round(time.time() - start, 2))

    cem_args = CEMArgs(
        seed=seed,
        env_name=env_name,
        horizon=horizon,
        dt=dt,
        num_samples=Nsample,
        num_iterations=max(3, min(8, Ndiffuse // 10)) if Ndiffuse > 0 else 5,
        elite_frac=0.1,
        init_std=max(0.1, noise_std),
        min_std=0.05,
        action_limit=action_limit,
        verbose=False,
    )
    start = time.time()
    cem_out = run_cem(cem_args, initial_state=shared_initial_state)
    print("Running for cem:", round(time.time() - start, 2))

    edoc_states = np.asarray(edoc_out["states"])
    diff_states = np.asarray(diff_out["states"])
    mppi_states = np.asarray(mppi_out["states"])
    cem_states = np.asarray(cem_out["states"])

    edoc_energies = np.asarray(edoc_out.get("energies", []), dtype=np.float32)
    diff_energies = np.asarray(diff_out.get("energies", []), dtype=np.float32)
    mppi_energies = np.asarray(mppi_out.get("energies", []), dtype=np.float32)
    cem_energies = np.asarray(cem_out.get("energies", []), dtype=np.float32)

    edoc_rewards = np.asarray(edoc_out.get("rewards", []), dtype=np.float32)
    diff_rewards = np.asarray(diff_out.get("rewards", []), dtype=np.float32)
    mppi_rewards = np.asarray(mppi_out.get("rewards", []), dtype=np.float32)
    cem_rewards = np.asarray(cem_out.get("rewards", []), dtype=np.float32)
    edoc_diffusion_actions = np.asarray(edoc_out.get("diffusion_actions_traj", []), dtype=np.float32)
    edoc_diffusion_samples = np.asarray(edoc_out.get("diffusion_sampled_actions", []), dtype=np.float32)
    mbd_diffusion_actions = np.asarray(diff_out.get("diffusion_actions_traj", []), dtype=np.float32)
    mbd_diffusion_samples = np.asarray(diff_out.get("diffusion_sampled_actions", []), dtype=np.float32)

    T_edoc = edoc_states.shape[0] - 1
    T_diff = diff_states.shape[0] - 1
    T_mppi = mppi_states.shape[0] - 1
    T_cem = cem_states.shape[0] - 1
    time_edoc = np.arange(T_edoc + 1) * dt
    time_diff = np.arange(T_diff + 1) * dt
    time_mppi = np.arange(T_mppi + 1) * dt
    time_cem = np.arange(T_cem + 1) * dt
    reward_times = {
        "EDOC": np.arange(edoc_rewards.shape[0]) * dt if edoc_rewards.size else None,
        "MBD": np.arange(diff_rewards.shape[0]) * dt if diff_rewards.size else None,
        "MPPI": np.arange(mppi_rewards.shape[0]) * dt if mppi_rewards.size else None,
        "CEM": np.arange(cem_rewards.shape[0]) * dt if cem_rewards.size else None,
    }

    state_dim = edoc_states.shape[1]
    pos_dim = state_dim // 2

    fig_state_timeseries = None
    fig_energy_field = None

    planner_series = [
        ("EDOC", time_edoc, edoc_states, EDOC_COLOR, "-"),
        ("MBD", time_diff, diff_states, MBD_COLOR, "--"),
        ("MPPI", time_mppi, mppi_states, MPPI_COLOR, ":"),
        ("CEM", time_cem, cem_states, CEM_COLOR, "-."),
    ]

    if pos_dim <= 1:
        fig_traj, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)
        for label, t_axis, states_arr, color, style in planner_series:
            axs[0].plot(
                t_axis,
                states_arr[:, 0],
                label=f"{label} position",
                linestyle=style,
                color=color,
            )
            axs[1].plot(
                t_axis,
                states_arr[:, pos_dim],
                label=f"{label} velocity",
                linestyle=style,
                color=color,
            )
        for ax, y_label in zip(axs, ["position", "velocity"]):
            ax.set_xlabel("time")
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig_traj.suptitle("Double Integrator Trajectories")
        fig_traj.tight_layout()
    else:
        fig_traj, ax_traj = plt.subplots(1, 1, figsize=(6, 6))
        for idx, (label, _, states_arr, color, style) in enumerate(planner_series):
            ax_traj.scatter(
                states_arr[0, 0],
                states_arr[0, 1],
                marker="o",
                s=35,
                facecolors="white",
                edgecolors=color,
                linewidths=1.0,
                label="Start" if idx == 0 else "_nolegend_",
            )
            ax_traj.plot(
                states_arr[:, 0],
                states_arr[:, 1],
                linestyle=style,
                color=color,
                linewidth=2.0,
                label=f"{label} trajectory",
            )
            ax_traj.scatter(
                states_arr[-1, 0],
                states_arr[-1, 1],
                marker="o",
                s=60,
                facecolors=color,
                edgecolors=color,
                linewidths=0.5,
                # label=f"{label} final",
            )
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
        for ax, (title, idx_state) in zip(axs_state, labels):
            for label, t_axis, states_arr, color, style in planner_series:
                ax.plot(
                    t_axis,
                    states_arr[:, idx_state],
                    label=label,
                    linestyle=style,
                    color=color,
                )
            ax.set_xlabel("time")
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

        stages = [(f"Diffusion {int(frac * 100)}%", frac) for frac in DIFFUSION_FRACTIONS]
        stages.append(("Final Plan", None))
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
                            if samples_actions.shape[0] > MAX_SAMPLE_TRAJ_PLOT:
                                sample_idx = np.linspace(
                                    0, samples_actions.shape[0] - 1, MAX_SAMPLE_TRAJ_PLOT, dtype=int
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

            plot_trajs(edoc_out, EDOC_COLOR, edoc_diffusion_actions, edoc_diffusion_samples)
            plot_trajs(diff_out, MBD_COLOR, mbd_diffusion_actions, mbd_diffusion_samples)

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
    if mppi_energies.size > 0:
        time_energy_mppi = np.arange(mppi_energies.shape[0]) * dt
        ax_energy.plot(time_energy_mppi, mppi_energies, label="MPPI energy", linestyle=":")
    if cem_energies.size > 0:
        time_energy_cem = np.arange(cem_energies.shape[0]) * dt
        ax_energy.plot(time_energy_cem, cem_energies, label="CEM energy", linestyle="-.")
    ax_energy.set_xlabel("time")
    ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Energy along rollout")
    ax_energy.legend()
    ax_energy.grid(True, alpha=0.3)

    fig_reward, ax_reward = plt.subplots(1, 1, figsize=(6, 4))
    for label, rewards_arr, color, style in [
        ("EDOC", edoc_rewards, EDOC_COLOR, "-"),
        ("MBD", diff_rewards, MBD_COLOR, "--"),
        ("MPPI", mppi_rewards, MPPI_COLOR, ":"),
        ("CEM", cem_rewards, CEM_COLOR, "-."),
    ]:
        t_axis = reward_times[label]
        if rewards_arr.size > 0 and t_axis is not None:
            ax_reward.plot(t_axis, rewards_arr, label=f"{label} reward", linestyle=style, color=color)
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
    print(f"MPPI total reward: {float(np.sum(mppi_rewards)):.3f}")
    print(f"CEM total reward: {float(np.sum(cem_rewards)):.3f}")
    if edoc_energies.size > 0:
        print(f"EDOC total energy: {float(np.sum(edoc_energies)):.3f}")
        print(f"EDOC final energy: {float(edoc_energies[-1]):.3f}")
    if diff_energies.size > 0:
        print(f"MBD total energy: {float(np.sum(diff_energies)):.3f}")
        print(f"MBD final energy: {float(diff_energies[-1]):.3f}")
    if mppi_energies.size > 0:
        print(f"MPPI total energy: {float(np.sum(mppi_energies)):.3f}")
        print(f"MPPI final energy: {float(mppi_energies[-1]):.3f}")
    if cem_energies.size > 0:
        print(f"CEM total energy: {float(np.sum(cem_energies)):.3f}")
        print(f"CEM final energy: {float(cem_energies[-1]):.3f}")
    print(f"Initial state (EDOC): {np.asarray(edoc_out['initial_state'])}")
    print(f"Initial state (Diffusion): {np.asarray(diff_out['initial_state'])}")
    print(f"Final state (EDOC): {edoc_states[-1]}")
    print(f"Final state (MBD): {diff_states[-1]}")
    print(f"Final state (MPPI): {mppi_states[-1]}")
    print(f"Final state (CEM): {cem_states[-1]}")


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
    parser.add_argument("--env_name", type=str, default="double_integrator_box_2d")
    parser.add_argument("--save_path", type=str, default="compare_di_2d.png")

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

