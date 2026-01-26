import numpy as np
import einops
from scipy.spatial.transform import Rotation as R
import pdb

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#


def compose(*fns):
    def _fn(x):
        for fn in fns:
            x = fn(x)
        return x

    return _fn


def get_preprocess_fn(fn_names, env):
    fns = [eval(name)(env) for name in fn_names]
    return compose(*fns)


def get_policy_preprocess_fn(fn_names):
    fns = [eval(name) for name in fn_names]
    return compose(*fns)


#-----------------------------------------------------------------------------#
#-------------------------- preprocessing functions --------------------------#
#-----------------------------------------------------------------------------#

#------------------------ @TODO: remove some of these ------------------------#


def arctanh_actions(*args, **kwargs):
    epsilon = 1e-4

    def _fn(dataset):
        actions = dataset["actions"]
        assert (
            actions.min() >= -1 and actions.max() <= 1
        ), f"applying arctanh to actions in range [{actions.min()}, {actions.max()}]"
        actions = np.clip(actions, -1 + epsilon, 1 - epsilon)
        dataset["actions"] = np.arctanh(actions)
        return dataset

    return _fn


def add_deltas(env):
    def _fn(dataset):
        deltas = dataset["next_observations"] - dataset["observations"]
        dataset["deltas"] = deltas
        return dataset

    return _fn


def maze2d_set_terminals(env):
    goal = np.array(env._target)
    threshold = 0.5

    def _fn(dataset):
        xy = dataset["observations"][:, :2]
        distances = np.linalg.norm(xy - goal, axis=-1)
        at_goal = distances < threshold
        timeouts = np.zeros_like(dataset["timeouts"])

        # timeout at time t iff
        #      at goal at time t and
        #      not at goal at time t + 1
        timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

        timeout_steps = np.where(timeouts)[0]
        path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        print(
            f"[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | "
            f"min length: {path_lengths.min()} | max length: {path_lengths.max()}"
        )

        dataset["timeouts"] = timeouts
        return dataset

    return _fn


#-------------------------- block-stacking --------------------------#


def blocks_quat_to_euler(observations):
    """
    input : [ N x robot_dim + n_blocks * 8 ] = [ N x 39 ]
        xyz: 3
        quat: 4
        contact: 1

    returns : [ N x robot_dim + n_blocks * 10] = [ N x 47 ]
        xyz: 3
        sin: 3
        cos: 3
        contact: 1
    """
    robot_dim = 7
    block_dim = 8
    n_blocks = 4
    assert observations.shape[-1] == robot_dim + n_blocks * block_dim

    X = observations[:, :robot_dim]

    for i in range(n_blocks):
        start = robot_dim + i * block_dim
        end = start + block_dim

        block_info = observations[:, start:end]

        xpos = block_info[:, :3]
        quat = block_info[:, 3:-1]
        contact = block_info[:, -1:]

        euler = R.from_quat(quat).as_euler("xyz")
        sin = np.sin(euler)
        cos = np.cos(euler)

        block_info = np.concatenate([xpos, sin, cos, contact], axis=-1)
        start = robot_dim + i * block_info.shape[-1]
        end = start + block_info.shape[-1]
        X = np.concatenate([X[:, :start], block_info, X[:, start:]], axis=-1)

    return X


#-------------------------- antmaze preprocessing ---------------------------#


def antmaze_linear_transform_actions(env):
    lb, ub = env.action_space.low, env.action_space.high

    def _fn(dataset):
        dataset["actions"] = (dataset["actions"] - lb) / (ub - lb)
        return dataset

    return _fn


def antmaze_standardize_obs(dataset):
    dataset["observations"] = np.clip(dataset["observations"], -10, 10)
    dataset["next_observations"] = np.clip(dataset["next_observations"], -10, 10)
    return dataset


def antmaze_make_hard(dataset):
    dataset["rewards"] = (dataset["rewards"] - 0.5) * 2
    dataset["terminals"] = dataset["terminals"] * 0
    dataset["timeouts"] = dataset["timeouts"] * 0
    return dataset


def antmaze_keep_maze_obs(dataset):
    dataset["observations"] = dataset["observations"][:, 8:]
    dataset["next_observations"] = dataset["next_observations"][:, 8:]
    return dataset


#-----------------------------------------------------------------------------#
#-------------------------- sequence transformations -------------------------#
#-----------------------------------------------------------------------------#


def get_trajectories(dataset):
    dones_float = np.zeros_like(dataset["rewards"])
    dones_float[-1] = 1
    traj = []
    data_ = (dataset["observations"], dataset["actions"], dataset["rewards"], dataset["masks"], dones_float)
    traj.append(data_)
    dataset["observations"], dataset["actions"], dataset["rewards"], dataset["masks"], dones_float = utils.multi_concat(traj)
    traj_dict = dataset
    return traj_dict


#-----------------------------------------------------------------------------#
#------------------------------------ misc -----------------------------------#
#-----------------------------------------------------------------------------#


def normalise_double_pendulum_vel(dataset):
    dataset["observations"][:, -2:] = dataset["observations"][:, -2:] / 10
    dataset["actions"] = dataset["actions"] / 3.0
    return dataset


def bias_double_pendulum(dataset):
    n = dataset["observations"].shape[0]
    angles = dataset["observations"][:, :2]
    vel = dataset["observations"][:, 2:]
    vel_mean = vel.mean(axis=0)
    vel_std = vel.std(axis=0) + 0.01
    vel_norm = (vel - vel_mean[None, :]) / vel_std[None, :]
    angles = angles % np.pi
    vel = vel * 0 + vel_norm
    dataset["observations"] = np.concatenate([angles, vel], axis=-1)
    return dataset


def soft_mocap_normalize(dataset):
    # Normalize mocap observations. Akin to the normalization done to actions & the like.
    # Note: Should normally come after soft_mocap_to_displacement().
    keys = ["observations", "next_observations", "actions", "deltas"]
    # Compute variance as mean squared expectation of samples:
    for key in keys:
        data = dataset[key]
        norm = np.sqrt(np.mean(np.square(data), axis=0))  # double check with mean squared expectation of the dims.
        data /= (norm + 1e-5)
        dataset[key] = data
    dataset["observations"] = np.concatenate([dataset["observations"][:, :5], dataset["observations"][:, -9:]], axis=1)
    dataset["next_observations"] = np.concatenate(
        [dataset["next_observations"][:, :5], dataset["next_observations"][:, -9:]], axis=1
    )
    dataset["actions"] = np.concatenate([dataset["actions"][:, :5], dataset["actions"][:, -9:]], axis=1)
    dataset["deltas"] = np.concatenate([dataset["deltas"][:, :5], dataset["deltas"][:, -9:]], axis=1)
    return dataset


def mul5(x):
    return x * 5


def add(a):
    def _fn(x):
        return x + a

    return _fn


#-----------------------------------------------------------------------------#
#--------------------------------- trajectories ------------------------------#
#-----------------------------------------------------------------------------#

def action_qpos_to_setpoints(dataset):
    # convert actions and qpos to residual actions
    actions = dataset["actions"]
    qpos = dataset["observations"][..., :7]

    new_actions = actions - qpos
    dataset["actions"] = new_actions
    dataset["observations"] = np.concatenate([dataset["observations"], qpos], axis=-1)
    return dataset


def pre_crop(dataset):
    L = 120
    dataset["observations"] = dataset["observations"][:, :L]
    dataset["next_observations"] = dataset["next_observations"][:, :L]
    return dataset


def crop_observations(dataset, num_keep_obs=53):
    dataset["observations"] = dataset["observations"][:, :num_keep_obs]
    dataset["next_observations"] = dataset["next_observations"][:, :num_keep_obs]
    return dataset

