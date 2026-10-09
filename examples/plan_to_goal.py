"""Use 2GO and a planar environment to reach the origin on CPU or GPU.

Run after installing the package: python examples/plan_to_goal.py
"""

import argparse

import jax
import numpy as np

from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.envs import make_env, make_energy
from genedynamics.solvers import TwoGOSolver


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def run(device: jax.Device, samples: int) -> None:
    RuntimeBackendManager.set_backend("jax", device=device.platform)
    backend = RuntimeBackendManager.get_backend()
    probe = backend.tensor([0.0])
    assert probe.devices() == {device}, "Backend array is on the wrong device"
    print(f"Planning on {device} with {samples} candidates per refinement step.")

    env_name = "single_integrator_box_2d"
    env = make_env(env_name)
    planner = TwoGOSolver(
        dynamics=env,
        energy=make_energy(env_name),
        backend=backend,
        dt=env.dt,
        horizon=20,
        Nsample=samples,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--samples", type=positive_int, default=128)
    args = parser.parse_args()
    try:
        device = jax.devices(args.device)[0]
    except (RuntimeError, IndexError) as exc:
        raise SystemExit(
            f"Requested {args.device} is unavailable. For NVIDIA GPU setup, see "
            "docs/recipes/gpu-planning.md. This example does not fall back to CPU."
        ) from exc
    with jax.default_device(device):
        run(device, args.samples)


if __name__ == "__main__":
    main()
