from dataclasses import dataclass

import numpy as np
import torch
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.constraints.action_filters import (
    TorchTrajectoryFeasibilityFilter,
)
from genedynamics.core.energy import EnergyTerm, LegacyEnergyFunctional
from genedynamics.solvers.single.mdoc import MDOCSolver


@dataclass(frozen=True)
class Evaluation:
    costs: torch.Tensor
    physical_states: torch.Tensor
    model_calls: int
    model_call_unit: str = "scalar_dynamics_transition"
    maximum_violation: torch.Tensor | None = None
    predicted_feasible: torch.Tensor | None = None


class QuadraticTrajectoryBinding:
    action_dim = 2
    device = "cpu"

    def __init__(self):
        self.model_call_count = 0

    def evaluate_action_sequences(self, candidates):
        batch, horizon, _ = candidates.shape
        self.model_call_count += batch * horizon
        positions = torch.cumsum(candidates, dim=1)
        initial = torch.zeros(batch, 1, 2)
        states = torch.cat((initial, positions), dim=1)
        costs = positions[:, -1].square().sum(dim=-1) + 0.01 * candidates.square().sum(
            dim=(1, 2)
        )
        violation = candidates.abs().amax(dim=(1, 2)) - 1.0
        return Evaluation(
            costs=costs,
            physical_states=states,
            model_calls=batch * horizon,
            maximum_violation=violation,
            predicted_feasible=violation <= 0,
        )

    def step(self, state, action):
        return state + action


def _solver(seed=7, constraint_filter=None):
    RuntimeBackendManager.set_backend("torch", device="cpu")
    binding = QuadraticTrajectoryBinding()
    energy = LegacyEnergyFunctional(
        {"unused": EnergyTerm(lambda state, action, info: 0.0)}
    )
    return MDOCSolver(
        dynamics=binding,
        energy=energy,
        backend=RuntimeBackendManager.get_backend(),
        horizon=4,
        Nsample=8,
        Ndiffuse=4,
        temp_sample=0.3,
        action_limit=1.0,
        seed=seed,
        constraint_filter=constraint_filter,
    )


def test_torch_mdoc_uses_batched_task_evaluator_and_exact_call_count():
    result = _solver().solve(
        np.zeros(2, dtype=np.float32),
        horizon=4,
        rng_key=7,
    )

    assert np.asarray(result.actions).shape == (4, 2)
    assert np.asarray(result.states).shape == (5, 2)
    assert np.isfinite(result.info["candidate_costs"]).all()
    assert result.info["model_calls"] == (3 * 8 + 1) * 4
    assert result.info["model_call_unit"] == "scalar_dynamics_transition"
    assert result.info["termination_reason"] == "torch_mdoc_plan_returned"


def test_torch_mdoc_is_seed_deterministic():
    first = _solver(11).solve(np.zeros(2), horizon=4, rng_key=11)
    second = _solver(11).solve(np.zeros(2), horizon=4, rng_key=11)

    np.testing.assert_array_equal(first.info["actions"], second.info["actions"])
    np.testing.assert_array_equal(
        first.info["diffusion_sampled_actions"],
        second.info["diffusion_sampled_actions"],
    )


def test_torch_mdoc_counts_filter_rollouts_in_model_budget():
    solver = _solver(
        constraint_filter=TorchTrajectoryFeasibilityFilter((0.5,))
    )

    result = solver.solve(np.zeros(2), horizon=4, rng_key=7)

    # Each of three diffusion steps evaluates the 8 sampled plans, the filtered
    # plans and the filtered mean; the terminal mean is then evaluated once.
    assert result.info["model_calls"] == 3 * (8 * 4 + 8 * 4 + 4) + 4
    assert (
        solver._env_adapter.dynamics.model_call_count
        == result.info["model_calls"]
    )
