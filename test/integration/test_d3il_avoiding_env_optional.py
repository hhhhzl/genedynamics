import pytest


def test_d3il_avoiding_import_and_reset_step_optional():
    """
    Optional smoke test for D3IL avoiding.

    This test is skipped unless the user has installed:
    - gym (classic)
    - mujoco
    and has D3IL vendored/submodule available under third_party/.
    """
    try:
        import gym  # noqa: F401
    except Exception:
        pytest.skip("classic gym not installed")

    try:
        import mujoco  # noqa: F401
    except Exception:
        pytest.skip("mujoco not installed")

    from enerdynamics.envs.external.d3il import D3ILAvoidingEnv, D3ILAvoidingConfig

    env = D3ILAvoidingEnv(D3ILAvoidingConfig(render=False))
    s0, info0 = env.reset()
    assert len(s0) == 4

    s1, cost, done, info = env.step(s0, [0.0, 0.0])
    assert len(s1) == 4
    assert isinstance(cost, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    env.close()


