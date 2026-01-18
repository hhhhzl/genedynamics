import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

"""
Some basic plotting methods to access the logged data from the teachinglog. Adapt the file paths to your system.
"""


path = "/home/philipp/phd_projects/teleop_data/raw_loads/data"
file_name = "exec_10"
file_number = "000"
plotting_keys = ["curr_load", "j_vel", "tau_raw", "curr_load"]
# plotting_keys = ["j_pos", "j_vel"]


def _sync_logs(primary_data, replica_data, key):
    """
    returns: primary_data, synced replica_data of the specified key
    """
    # make sure that replica time stamps are in order
    assert np.all(np.diff(replica_data["timestamp"]) > 0)
    synced_to_primary_values = np.zeros_like(primary_data[key])
    for j in range(synced_to_primary_values.shape[1]):
        synced_to_primary_values[:, j] = np.interp(
            primary_data["timestamp"],
            replica_data["timestamp"],
            replica_data[key][:, j],
        )
    return primary_data[key], synced_to_primary_values


def compare_validation_trajectories(path, replica_data_files, key):
    rows = 7
    fig, axs = plt.subplots(rows, 1)
    for file in replica_data_files:
        # data_primary = pickle.load(open(os.path.join(path, file[0]), "rb"))["data"]
        data_replica = pickle.load(open(os.path.join(path, file[1]), "rb"))["data"]
        # _, synced_data = _sync_logs(data_primary, data_replica, key)
        np_data = data_replica[key]
        for j in range(rows):
            axs[j].plot(np_data[:, j], label=file)
        axs[0].legend()
    plt.show()


def load_analysis():
    file = file_name + "_replica_" + file_number + ".pkl"
    logs = pickle.load(open(os.path.join(path, file), "rb"))
    data = logs["data"]
    file = file_name + "_primary_" + file_number + ".pkl"
    logs = pickle.load(open(os.path.join(path, file), "rb"))
    data_primary = logs["data"]
    keys = ["raw_load", "raw_gravity", "raw_coriolis"]
    # plot_joint_values(data["raw_load"], "raw_load")
    #
    # plot_joint_values(data["raw_gravity"], "gravity")
    # fig, axs = plot_joint_values(data["raw_load"] - data["raw_gravity"], "raw_load - gravity", set_y_lim=False)
    # plot_joint_values(data["raw_coriolis"], "neg_raw_coriolis", set_y_lim=False, fig=fig, axs=axs)
    # plot_joint_values(data["raw_gravity"], "gravity", set_y_lim=False, fig=fig, axs=axs)
    # plot_joint_values(data["raw_load"] - data["raw_coriolis"], "raw_load - coriolis")

    # plot_joint_values(data["raw_load"] - data["raw_coriolis"] - data["raw_gravity"], "raw_load - gravity - coriolis")
    plot_joint_values(data_primary["tau"], "tau")
    plot_joint_values(data_primary["tau_raw"], "tau_raw")
    total_tau = np.sqrt(np.sum(data_primary["tau_raw"] ** 2, axis=1))
    plot_joint_values(data["curr_load"], "tau_ext_hat_filtered_filtered")
    _, curr_load = _sync_logs(data_primary, data, "curr_load")
    total_force = np.sqrt(np.sum(curr_load * curr_load, axis=1))
    fig, axs = plt.subplots()
    axs.plot(total_force, label="total force")
    axs.plot(total_tau, label="total tau")
    axs.legend()
    # plot_joint_values(data["tau_ext_hat_filtered"], "tau external")
    plt.show()


def plot_joint_values(np_data, title, set_y_lim=True, fig=None, axs=None):
    min_elt = np.min(np_data)
    max_elt = np.max(np_data)
    rows = np_data.shape[1]
    if fig is None or axs is None:
        fig, axs = plt.subplots(rows, 1)
    for j in range(rows):
        axs[j].plot(np_data[:, j])
        axs[j].set_ylim(min_elt, max_elt)
    fig.suptitle(title)
    return fig, axs


def standard_plot(robots=("primary", "replica")):
    for robot in robots:
        file = file_name + "_" + robot + "_" + file_number + ".pkl"
        logs = pickle.load(open(os.path.join(path, file), "rb"))
        data = logs["data"]
        for key in plotting_keys:
            np_data = data[key]
            plot_joint_values(np_data, robot + "_" + key)
    plt.show()


def synced_plot(keys):
    file_primary = file_name + "_primary_" + file_number + ".pkl"
    file_replica = file_name + "_replica_" + file_number + ".pkl"
    data_primary = pickle.load(open(os.path.join(path, file_primary), "rb"))["data"]
    data_replica = pickle.load(open(os.path.join(path, file_replica), "rb"))["data"]
    for key in keys:
        synced_primary, synced_replica = _sync_logs(data_primary, data_replica, key)
        rows = synced_primary.shape[1]
        fig, axs = plt.subplots(rows, 1)
        for j in range(rows):
            axs[j].plot(synced_primary[:, j], label="primary")
            axs[j].plot(synced_replica[:, j], label="replica")
        fig.suptitle(key)
        axs[0].legend()
    plt.show()


def teacher_plot():
    file = file_name + "_" + file_number + ".pkl"
    logs = pickle.load(open(os.path.join(path, file), "rb"))
    data = logs["data"]
    for key in plotting_keys:
        np_data = data[key]
        plot_joint_values(np_data, key)
    plt.show()


if __name__ == "__main__":
    standard_plot()
