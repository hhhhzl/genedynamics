"""Source-style AdaptSim train/evaluate/update loop with explicit resource counts.

The original AdaptSim agent repeatedly trains a task policy under a sampled
simulator-parameter distribution, observes the target rollout, chooses a
branching-Q update to that distribution, and repeats. The task adapter owns
all simulator interaction and returns charged counts. This loop never reads
hidden target parameters or substitutes synthetic rewards for target rollout.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..adaptsim import AdaptSimSolver, ParameterDistribution


@dataclass(frozen=True)
class AdaptationTraining:
    policy: Any
    simulation_steps: int

    def __post_init__(self) -> None:
        if self.simulation_steps < 0:
            raise ValueError("simulation_steps must be nonnegative")


@dataclass(frozen=True)
class AdaptationEvaluation:
    reward: float
    features: np.ndarray
    target_control_steps: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.reward) or self.target_control_steps < 0:
            raise ValueError("evaluation reward and target control steps must be valid")


@dataclass(frozen=True)
class ParticleSelection:
    """Charged, same-reset target comparison of trained simulator particles."""

    selected_index: int
    distribution: ParameterDistribution
    policy: Any
    simulation_steps: int
    target_control_steps: int
    records: tuple[dict[str, Any], ...]


class AdaptSimAdaptation:
    """Run one full target-adaptation trajectory under task-provided callbacks.

    `train_policy` receives the current simulation parameter distribution,
    optional warm-start policy, and deterministic seed. `evaluate_target`
    receives that trained policy and a deterministic seed; it must execute an
    actual target rollout and return its reward, public trajectory features,
    and number of target control steps. The callbacks do not receive target
    mechanism parameters from this class.
    """

    def __init__(
        self,
        solver: AdaptSimSolver,
        train_policy: Callable[[ParameterDistribution, Any | None, int], AdaptationTraining],
        evaluate_target: Callable[[Any, int], AdaptationEvaluation],
        *,
        seed: int,
    ) -> None:
        self.solver = solver
        self.train_policy = train_policy
        self.evaluate_target = evaluate_target
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)

    def select_particle(
        self, particles: Sequence[ParameterDistribution], *, seed: int | None = None,
    ) -> ParticleSelection:
        """Train every particle and choose by reward on one shared target reset."""
        if not particles:
            raise ValueError("at least one initial particle required")
        shared_seed = self.seed if seed is None else int(seed)
        best_index = -1
        best_reward = -np.inf
        best_policy: Any = None
        records: list[dict[str, Any]] = []
        simulation_steps = 0
        target_steps = 0
        for index, distribution in enumerate(particles):
            self.solver._check_distribution(distribution)
            training = self.train_policy(distribution, None, shared_seed + 2 * index)
            if not isinstance(training, AdaptationTraining):
                raise TypeError("train_policy must return AdaptationTraining")
            evaluation = self.evaluate_target(training.policy, shared_seed + 1000000)
            if not isinstance(evaluation, AdaptationEvaluation):
                raise TypeError("evaluate_target must return AdaptationEvaluation")
            simulation_steps += training.simulation_steps
            target_steps += evaluation.target_control_steps
            records.append({
                "particle_index": index,
                "distribution": {
                    "kind": distribution.kind,
                    "left": distribution.left.tolist(),
                    "right": distribution.right.tolist(),
                },
                "target_reward": float(evaluation.reward),
                "simulation_steps": int(training.simulation_steps),
                "target_control_steps": int(evaluation.target_control_steps),
                "target_reset_seed": shared_seed + 1000000,
            })
            if evaluation.reward > best_reward:
                best_index = index
                best_reward = evaluation.reward
                best_policy = training.policy
        return ParticleSelection(
            selected_index=best_index,
            distribution=particles[best_index],
            policy=best_policy,
            simulation_steps=simulation_steps,
            target_control_steps=target_steps,
            records=tuple(records),
        )

    def run(
        self,
        initial: ParameterDistribution,
        *,
        updates: int,
        exploration_probability: float = 0.0,
        train_inference: bool = False,
        simulated_meta_target: bool = False,
        reuse_policy: bool = True,
        initial_policy: Any | None = None,
    ) -> dict[str, Any]:
        if updates < 0 or not 0 <= exploration_probability <= 1:
            raise ValueError("invalid update count or exploration probability")
        if train_inference and not simulated_meta_target:
            raise ValueError("inference policy may train only on simulated meta-targets, not deployment target rollouts")
        distribution = initial
        warm_start: Any | None = initial_policy
        records: list[dict[str, Any]] = []
        previous: tuple[np.ndarray, np.ndarray] | None = None
        simulated_steps = 0
        target_steps = 0
        policy: Any | None = None
        for index in range(updates + 1):
            training = self.train_policy(distribution, warm_start, self.seed + 2 * index)
            if not isinstance(training, AdaptationTraining):
                raise TypeError("train_policy must return AdaptationTraining")
            simulated_steps += training.simulation_steps
            policy = training.policy
            evaluation = self.evaluate_target(policy, self.seed + 2 * index + 1)
            if not isinstance(evaluation, AdaptationEvaluation):
                raise TypeError("evaluate_target must return AdaptationEvaluation")
            target_steps += evaluation.target_control_steps
            features = np.asarray(evaluation.features, dtype=np.float32)
            state = self.solver.state(distribution, features)
            if train_inference and previous is not None:
                previous_state, previous_action = previous
                self.solver.train_batch(
                    torch.from_numpy(previous_state[None]),
                    torch.from_numpy(previous_action[None]),
                    torch.tensor([evaluation.reward], dtype=torch.float32),
                    torch.from_numpy(state[None]),
                    torch.tensor([float(index == updates)], dtype=torch.float32),
                )
            record = {
                "index": index,
                "distribution": {
                    "kind": distribution.kind,
                    "left": distribution.left.tolist(),
                    "right": distribution.right.tolist(),
                },
                "target_reward": float(evaluation.reward),
                "simulation_steps": int(training.simulation_steps),
                "target_control_steps": int(evaluation.target_control_steps),
            }
            if index < updates:
                if self.rng.random() < exploration_probability:
                    maximum = 5 if self.solver.kind == "uniform" else 3
                    action = self.rng.integers(maximum, size=len(self.solver.specs))
                    next_distribution = self.solver.apply_action(distribution, action)
                else:
                    action, next_distribution = self.solver.propose_update(distribution, features)
                record["action"] = np.asarray(action, dtype=int).tolist()
                previous = (state, np.asarray(action, dtype=np.int64))
                distribution = next_distribution
                warm_start = policy if reuse_policy else None
            records.append(record)
        return {
            "domain": "simulated_meta_target" if simulated_meta_target else "deployment_target",
            "records": records,
            "simulation_steps": simulated_steps,
            "target_control_steps": target_steps,
            "target_rollouts": updates + 1,
            "final_policy": policy,
            "final_distribution": distribution,
        }

