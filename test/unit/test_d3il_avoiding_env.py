import numpy as np
import pytest


@pytest.mark.unit
@pytest.mark.requires_mujoco
def test_d3il_avoiding_env_start_and_reset():
    """Ensure the vendored D3IL avoiding env can start and reset."""
    try:
        import pybullet  
    except ImportError:
        pytest.skip("pybullet not available")
    try:
        import pinocchio  
    except ImportError:
        pytest.skip("pinocchio not available")

    from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import (  # noqa: E501
        ObstacleAvoidanceEnv,
    )

    env = ObstacleAvoidanceEnv()
    env.start()
    obs = env.reset()

    obs_arr = np.asarray(obs, dtype=np.float32)
    assert obs_arr.shape == (2,)
    assert np.isfinite(obs_arr).all()
