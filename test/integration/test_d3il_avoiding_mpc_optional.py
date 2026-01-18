import pytest


def test_d3il_avoiding_edoc_mpc_optional():
    """
    Optional smoke test: create d3il_avoiding env + run a couple MPC steps with EDOC.

    Skips unless classic gym + mujoco are installed and D3IL is available.
    """
    try:
        import gym  # noqa: F401
    except Exception:
        pytest.skip("classic gym not installed")

    try:
        import mujoco  # noqa: F401
    except Exception:
        pytest.skip("mujoco not installed")

    import jax

    from enerdynamics.envs.external.d3il import D3ILAvoidingEnv, D3ILAvoidingConfig
    from enerdynamics.envs.external.d3il.avoiding_plan_env import AvoidingPlanEnv, AvoidingPlanSpec
    from enerdynamics.solvers.single.edoc import EDOCPlanner
    from enerdynamics.core.energy import LegacyEnergyFunctional, EnergyTerm
    import jax.numpy as jnp

    exec_env = D3ILAvoidingEnv(D3ILAvoidingConfig(render=False))
    plan_env = AvoidingPlanEnv(AvoidingPlanSpec(horizon=8))

    goal_xy = jnp.array([0.4, 0.35], dtype=jnp.float32)

    def task_energy(x, u, ctx):
        _ = (u, ctx)
        return jnp.sum((x[2:4] - goal_xy) ** 2)

    energy = LegacyEnergyFunctional({"task": EnergyTerm(task_energy, 1.0)})
    planner = EDOCPlanner(env=plan_env, energy=energy, horizon=8, dt=plan_env.dt, action_nsample=16, action_diffuse_steps=10)

    x0, _ = exec_env.reset()
    plan_env.set_initial_state(x0)
    out = planner.plan(jax.random.PRNGKey(0))
    assert "actions" in out
    assert len(out["actions"]) > 0
    exec_env.close()


