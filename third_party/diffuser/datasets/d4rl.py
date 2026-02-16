import os
import numpy as np
import pickle
from pathlib import Path


def _resolve_avoiding_data_dir() -> str:
    """
    Resolve the local directory that stores avoiding trajectory state files.

    Priority:
    1) Explicit env overrides:
       - DPCC_AVOIDING_DATA_DIR
       - D3IL_AVOIDING_DATA_DIR
    2) Common repository-relative locations.
    """
    env_override = os.environ.get("DPCC_AVOIDING_DATA_DIR") or os.environ.get("D3IL_AVOIDING_DATA_DIR")
    if env_override:
        if os.path.isdir(env_override):
            return env_override
        raise FileNotFoundError(
            f"Avoiding data directory from env var not found: {env_override}"
        )

    project_root = Path(__file__).resolve().parents[3]
    candidates = [
        project_root / "environments" / "dataset" / "data" / "avoiding" / "data",
        project_root / "third_party" / "environments" / "dataset" / "data" / "avoiding" / "data",
    ]

    for path in candidates:
        if path.is_dir():
            return str(path)

    raise FileNotFoundError(
        "Could not locate avoiding dataset directory. Set DPCC_AVOIDING_DATA_DIR "
        "or D3IL_AVOIDING_DATA_DIR to the folder containing trajectory state files."
    )

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """

    if env in ('avoiding-d3il', 'd3il-avoiding', 'avoiding-d3il-9d', 'd3il-avoiding-9d'):
        data_dir = _resolve_avoiding_data_dir()
        state_files = os.listdir(data_dir)

        for file in state_files:
            with open(os.path.join(data_dir, file), 'rb') as f:
                env_state = pickle.load(f)
                robot_state = env_state['robot']

                if env in ('avoiding-d3il-9d', 'd3il-avoiding-9d'):
                    # 9D with target: obs = [x_des, y_des, x, y, q1..q7] (11D).
                    tcp_xy = robot_state['c_pos'][:, :2]
                    q = robot_state['j_pos'][:, :7]
                    qdot = robot_state['j_vel'][:, :7]
                    try:
                        des_xy = robot_state['des_c_pos'][:, :2]
                        input_state = np.concatenate((des_xy, tcp_xy, q), axis=-1)  # 11D
                    except (KeyError, TypeError):
                        import warnings
                        warnings.warn(
                            "avoiding-d3il-9d: 'des_c_pos' not in robot_state, using 9D obs without target. "
                            "Add target to data or use data that includes des_c_pos for goal-conditioned 9D."
                        )
                        input_state = np.concatenate((tcp_xy, q), axis=-1)  # 9D fallback
                    action = qdot[:-1]
                else:
                    # Legacy DPCC avoiding setup.
                    robot_des_pos = robot_state['des_c_pos'][:, :2]
                    robot_c_pos = robot_state['c_pos'][:, :2]
                    input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)
                    action = robot_des_pos[1:] - robot_des_pos[:-1]

                valid_len = len(action)

            # Rewards: both 4D and 9D use zeros here. Target-reaching signal may be added
            # elsewhere (e.g. preprocessing) for 4D; 9D has no target in obs so no alignment.
            episode_data = {
                'observations': input_state[:-1],
                'actions': action,
                'rewards': np.zeros(valid_len),
                'terminals': np.concatenate((np.zeros(valid_len-1), np.array([1])))
            }

            yield episode_data
    else:
        raise NotImplementedError(f'Unsupported dataset without Minari: {env}')
