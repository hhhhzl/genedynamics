"""AdaptSim inference training on simulated target tasks.

The source implementation samples a simulator-parameter particle, trains a
task policy in that particle, evaluates it on a separately sampled simulated
target, updates the particle distribution, and trains a branching-Q inference
network from those transitions.  This method-local loop preserves that
separation; the task owns simulation and policy optimization.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..adaptsim import AdaptSimSolver, ParameterDistribution
from .adaptation import AdaptationEvaluation, AdaptationTraining


@dataclass(frozen=True)
class MetaTrainingResult:
    episodes: int
    transitions: int
    inference_updates: int
    simulation_steps: int
    simulated_target_rollouts: int
    simulated_target_control_steps: int
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PopulationMetaTrainingResult:
    iterations: int
    inner_episodes: int
    inference_updates: int
    simulation_steps: int
    simulated_target_rollouts: int
    simulated_target_control_steps: int
    final_particles: tuple[ParameterDistribution, ...]
    records: tuple[dict[str, Any], ...]


class AdaptSimMetaTraining:
    """Train the parameter-update Q network without deployment target data.

    `sample_simulated_target` returns an opaque task token.  The solver never
    reads its physical parameters; only `evaluate_simulated_target` receives
    it.  `train_policy` must train a task policy from the proposed simulator
    distribution and report all environment steps it consumed.
    """

    def __init__(
        self,
        solver: AdaptSimSolver,
        train_policy: Callable[[ParameterDistribution, Any | None, int], AdaptationTraining],
        sample_simulated_target: Callable[[int], Any],
        evaluate_simulated_target: Callable[[Any, Any, int], AdaptationEvaluation],
        *,
        seed: int,
    ) -> None:
        self.solver = solver
        self.train_policy = train_policy
        self.sample_simulated_target = sample_simulated_target
        self.evaluate_simulated_target = evaluate_simulated_target
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        initial_particles: Sequence[ParameterDistribution],
        *,
        episodes: int,
        adaptation_steps: int,
        replay_capacity: int = 10000,
        batch_size: int = 64,
        update_every: int = 10,
        gradient_steps: int = 1,
        exploration_probability: float = 0.1,
        reuse_policy: bool = True,
        initial_policies: Sequence[Any | None] | None = None,
        inner_spawn_perturbation: float = 0.0,
        replay_buffer: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] | None = None,
    ) -> MetaTrainingResult:
        if not initial_particles or episodes <= 0 or adaptation_steps <= 0:
            raise ValueError("positive episodes/steps and at least one particle required")
        if min(replay_capacity, batch_size, update_every, gradient_steps) <= 0:
            raise ValueError("replay and update sizes must be positive")
        if not 0 <= exploration_probability <= 1:
            raise ValueError("exploration probability outside [0, 1]")
        for particle in initial_particles:
            self.solver._check_distribution(particle)
        if initial_policies is not None and len(initial_policies) != len(initial_particles):
            raise ValueError("initial policy count must match particle count")
        if not np.isfinite(inner_spawn_perturbation) or inner_spawn_perturbation < 0:
            raise ValueError("inner particle perturbation must be finite and nonnegative")

        replay: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = (
            deque(maxlen=replay_capacity) if replay_buffer is None else replay_buffer
        )
        if replay.maxlen is None or replay.maxlen < batch_size:
            raise ValueError("replay buffer must have bounded capacity at least batch_size")
        records: list[dict[str, Any]] = []
        simulation_steps = 0
        target_steps = 0
        target_rollouts = 0
        inference_updates = 0
        transitions = 0
        for episode in range(episodes):
            particle_index = int(self.rng.integers(len(initial_particles)))
            distribution = initial_particles[particle_index]
            if inner_spawn_perturbation:
                distribution = self.solver.clone_particle(
                    distribution, self.rng, perturbation=inner_spawn_perturbation
                )
            target = self.sample_simulated_target(self.seed + episode * 100000)
            warm_start = None if initial_policies is None else initial_policies[particle_index]
            training = self.train_policy(distribution, warm_start, self.seed + episode * 100000 + 1)
            if not isinstance(training, AdaptationTraining):
                raise TypeError("train_policy must return AdaptationTraining")
            simulation_steps += training.simulation_steps
            policy = training.policy
            evaluation = self.evaluate_simulated_target(
                policy, target, self.seed + episode * 100000 + 2
            )
            if not isinstance(evaluation, AdaptationEvaluation):
                raise TypeError("evaluate_simulated_target must return AdaptationEvaluation")
            target_steps += evaluation.target_control_steps
            target_rollouts += 1
            row: dict[str, Any] = {
                "episode": episode,
                "particle_index": particle_index,
                "simulated_target_seed": self.seed + episode * 100000,
                "initial_distribution": {
                    "kind": distribution.kind,
                    "left": distribution.left.tolist(),
                    "right": distribution.right.tolist(),
                },
                "initial_reward": float(evaluation.reward),
                "initial_simulation_steps": int(training.simulation_steps),
                "initial_simulated_target_control_steps": int(evaluation.target_control_steps),
                "steps": [],
            }
            for step in range(adaptation_steps):
                state = self.solver.state(distribution, evaluation.features)
                if self.rng.random() < exploration_probability:
                    bins = 5 if self.solver.kind == "uniform" else 3
                    action = self.rng.integers(bins, size=len(self.solver.specs))
                    next_distribution = self.solver.apply_action(distribution, action)
                else:
                    action, next_distribution = self.solver.propose_update(
                        distribution, evaluation.features
                    )
                training = self.train_policy(
                    next_distribution, policy if reuse_policy else None,
                    self.seed + episode * 100000 + 3 + 2 * step,
                )
                if not isinstance(training, AdaptationTraining):
                    raise TypeError("train_policy must return AdaptationTraining")
                simulation_steps += training.simulation_steps
                policy = training.policy
                evaluation = self.evaluate_simulated_target(
                    policy, target, self.seed + episode * 100000 + 4 + 2 * step
                )
                if not isinstance(evaluation, AdaptationEvaluation):
                    raise TypeError("evaluate_simulated_target must return AdaptationEvaluation")
                target_steps += evaluation.target_control_steps
                target_rollouts += 1
                next_state = self.solver.state(next_distribution, evaluation.features)
                done = float(step == adaptation_steps - 1)
                replay.append((state, np.asarray(action, dtype=np.int64),
                               float(evaluation.reward), next_state, done))
                transitions += 1
                row["steps"].append({
                    "action": np.asarray(action, dtype=int).tolist(),
                    "reward": float(evaluation.reward),
                    "distribution": {
                        "kind": next_distribution.kind,
                        "left": next_distribution.left.tolist(),
                        "right": next_distribution.right.tolist(),
                    },
                    "simulation_steps": int(training.simulation_steps),
                    "simulated_target_control_steps": int(evaluation.target_control_steps),
                })
                distribution = next_distribution
                if transitions % update_every == 0 and len(replay) >= batch_size:
                    for _ in range(gradient_steps):
                        indices = self.rng.integers(len(replay), size=batch_size)
                        batch = [replay[int(index)] for index in indices]
                        self.solver.train_batch(
                            torch.from_numpy(np.stack([item[0] for item in batch])),
                            torch.from_numpy(np.stack([item[1] for item in batch])),
                            torch.tensor([item[2] for item in batch], dtype=torch.float32),
                            torch.from_numpy(np.stack([item[3] for item in batch])),
                            torch.tensor([item[4] for item in batch], dtype=torch.float32),
                        )
                        inference_updates += 1
            records.append(row)
        return MetaTrainingResult(
            episodes=episodes,
            transitions=transitions,
            inference_updates=inference_updates,
            simulation_steps=simulation_steps,
            simulated_target_rollouts=target_rollouts,
            simulated_target_control_steps=target_steps,
            records=tuple(records),
        )

    def run_population(
        self,
        initial_particles: Sequence[ParameterDistribution],
        *,
        iterations: int,
        inner_episodes: int,
        adaptation_steps: int,
        kill_frequency: int,
        spawn_reward_threshold: float,
        meta_spawn_perturbation: float,
        inner_spawn_perturbation: float = 0.0,
        meta_policy_reuse: bool = True,
        batch_size: int = 64,
        update_every: int = 10,
        gradient_steps: int = 1,
        exploration_probability: float = 0.1,
    ) -> PopulationMetaTrainingResult:
        """Run source-style outer particle evaluation, replacement, and inner Q learning.

        One simulated meta-target is held fixed across outer iterations. Every
        particle is trained and evaluated on it before a possible best-to-worst
        replacement. Inner adaptation samples separate simulated targets; no
        deployment target feedback updates the inference network.
        """
        if not initial_particles or min(iterations, inner_episodes, adaptation_steps) <= 0:
            raise ValueError("positive iterations/episodes/steps and particles required")
        if kill_frequency < 0 or not np.isfinite(spawn_reward_threshold):
            raise ValueError("invalid particle replacement schedule")
        for particle in initial_particles:
            self.solver._check_distribution(particle)
        particles = list(initial_particles)
        policies: list[Any | None] = [None] * len(particles)
        meta_target = self.sample_simulated_target(self.seed + 90000000)
        records: list[dict[str, Any]] = []
        total_simulation = 0
        total_target_steps = 0
        total_target_rollouts = 0
        total_updates = 0
        replay: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=10000)
        for iteration in range(iterations):
            meta_rows: list[dict[str, Any]] = []
            rewards: list[float] = []
            shared_reset = self.seed + 60000000 + iteration * 100000
            for index, distribution in enumerate(particles):
                training = self.train_policy(
                    distribution, policies[index] if meta_policy_reuse else None,
                    self.seed + 50000000 + iteration * 100000 + 2 * index,
                )
                if not isinstance(training, AdaptationTraining):
                    raise TypeError("train_policy must return AdaptationTraining")
                evaluation = self.evaluate_simulated_target(
                    training.policy, meta_target, shared_reset
                )
                if not isinstance(evaluation, AdaptationEvaluation):
                    raise TypeError("evaluate_simulated_target must return AdaptationEvaluation")
                policies[index] = training.policy if meta_policy_reuse else None
                rewards.append(float(evaluation.reward))
                total_simulation += training.simulation_steps
                total_target_steps += evaluation.target_control_steps
                total_target_rollouts += 1
                meta_rows.append({
                    "particle_index": index,
                    "distribution": {
                        "kind": distribution.kind,
                        "left": distribution.left.tolist(),
                        "right": distribution.right.tolist(),
                    },
                    "reward": float(evaluation.reward),
                    "simulation_steps": int(training.simulation_steps),
                    "simulated_target_control_steps": int(evaluation.target_control_steps),
                    "simulated_target_reset_seed": shared_reset,
                })
            replacement: dict[str, Any] | None = None
            if (iteration > 0 and kill_frequency and iteration % kill_frequency == 0
                    and len(particles) > 1 and max(rewards) > spawn_reward_threshold):
                best = int(np.argmax(rewards))
                worst = int(np.argmin(rewards))
                if best != worst:
                    new_distribution = self.solver.clone_particle(
                        particles[best], self.rng, perturbation=meta_spawn_perturbation
                    )
                    particles[worst] = new_distribution
                    policies[worst] = policies[best]
                    replacement = {
                        "replaced_index": worst,
                        "parent_index": best,
                        "distribution": {
                            "kind": new_distribution.kind,
                            "left": new_distribution.left.tolist(),
                            "right": new_distribution.right.tolist(),
                        },
                    }
            inner = AdaptSimMetaTraining(
                self.solver, self.train_policy, self.sample_simulated_target,
                self.evaluate_simulated_target,
                seed=self.seed + iteration * 1000000,
            ).run(
                particles, episodes=inner_episodes, adaptation_steps=adaptation_steps,
                batch_size=batch_size, update_every=update_every,
                gradient_steps=gradient_steps,
                exploration_probability=exploration_probability,
                initial_policies=policies,
                inner_spawn_perturbation=inner_spawn_perturbation,
                replay_buffer=replay,
            )
            total_simulation += inner.simulation_steps
            total_target_steps += inner.simulated_target_control_steps
            total_target_rollouts += inner.simulated_target_rollouts
            total_updates += inner.inference_updates
            records.append({
                "iteration": iteration,
                "meta_evaluations": meta_rows,
                "replacement": replacement,
                "inner_records": list(inner.records),
                "inner_inference_updates": inner.inference_updates,
            })
        return PopulationMetaTrainingResult(
            iterations=iterations,
            inner_episodes=iterations * inner_episodes,
            inference_updates=total_updates,
            simulation_steps=total_simulation,
            simulated_target_rollouts=total_target_rollouts,
            simulated_target_control_steps=total_target_steps,
            final_particles=tuple(particles),
            records=tuple(records),
        )


__all__ = ["AdaptSimMetaTraining", "MetaTrainingResult", "PopulationMetaTrainingResult"]
