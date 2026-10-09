"""Use 2GO and a planar environment to reach the origin on CPU.

Run after installing the package: python examples/plan_to_goal.py
"""

import jax
import numpy as np

from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.envs import make_env, make_energy
from genedynamics.solvers import TwoGOSolver


def main() -> None:
    RuntimeBackendManager.set_backend("jax", device="cpu")
    env_name = "single_integrator_box_2d"
    env = make_env(env_name)
    planner = TwoGOSolver(
        dynamics=env,
        energy=make_energy(env_name),
        backend=RuntimeBackendManager.get_backend(),
        dt=env.dt,
        horizon=20,
        Nsample=128,
        Ndiffuse=16,
    )

    state = np.array([0.8, 0.8], dtype=np.float32)
    for step in range(60):
        key = jax.random.fold_in(jax.random.PRNGKey(0), step)
        plan = planner.solve(state, horizon=20, rng_key=key)
        assert np.isfinite(np.asarray(plan.states)).all(), "Non-finite plan states"
        assert np.isfinite(np.asarray(plan.actions)).all(), "Non-finite plan actions"
        state = env.transition(state, plan.actions[0])
        assert np.isfinite(state).all(), "Non-finite executed state"
        if np.linalg.norm(state) < 0.1:
            break

    error = float(np.linalg.norm(state))
    assert error < 0.1, f"Goal not reached after {step + 1} steps: error={error:.3f}"
    print(f"Reached goal in {step + 1} steps (error={error:.3f}).")


if __name__ == "__main__":
    main()
