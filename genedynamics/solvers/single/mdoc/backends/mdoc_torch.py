"""Torch MDOC backend for task-owned batched trajectory evaluators.

The reverse diffusion update and constraint-filter placement match the existing
JAX/NumPy MDOC backends.  Task semantics stay outside GD: the wrapped dynamics
must expose ``evaluate_action_sequences([batch, horizon, action])`` and return
cost, state, feasibility and exact model-call evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from genedynamics.core.constraints.action_filters import (
    ConstraintFilter,
    NoOpConstraintFilter,
)
from genedynamics.core.types import State, Trajectory


def _host(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class MDOCBackendTorch:
    """Weighted-mean MDOC diffusion over an exact Torch trajectory binding."""

    def __init__(
        self,
        *,
        env_adapter,
        legacy_energy,
        horizon: int,
        dt: float,
        Nsample: int,
        Ndiffuse: int,
        temp_sample: float,
        beta0: float,
        betaT: float,
        action_limit: float,
        seed: int = 0,
        scheduler: Any = None,
        show_tqdm: bool = False,
        constraint_filter: ConstraintFilter | None = None,
        obstacles: Any = None,
        **kwargs: Any,
    ):
        del legacy_energy, show_tqdm, kwargs
        if not hasattr(env_adapter, "evaluate_action_sequences"):
            raise TypeError(
                "Torch MDOC requires dynamics.evaluate_action_sequences"
            )
        if min(horizon, Nsample, Ndiffuse) <= 0 or temp_sample <= 0:
            raise ValueError("Torch MDOC dimensions and temperature must be positive")
        if action_limit <= 0 or not 0 <= beta0 <= betaT < 1:
            raise ValueError("Torch MDOC diffusion parameters are invalid")
        self.env = env_adapter
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.Nsample = int(Nsample)
        self.Ndiffuse = int(Ndiffuse)
        self.temp_sample = float(temp_sample)
        self.beta0 = float(beta0)
        self.betaT = float(betaT)
        self.action_limit = float(action_limit)
        self.seed = int(seed)
        self.act_dim = int(env_adapter.act_dim)
        self.scheduler = scheduler
        self.obstacles = obstacles
        self.constraint_filter = constraint_filter or NoOpConstraintFilter()
        self.num_modes = 1

    def _device(self):
        device = getattr(self.env, "device", None)
        if device is None:
            dynamics = getattr(self.env, "dynamics", None)
            device = getattr(dynamics, "device", None)
        return torch.device(device or "cpu")

    def _generator(self, rng_key, device):
        if isinstance(rng_key, torch.Generator):
            if torch.device(rng_key.device) != device:
                raise ValueError("Torch MDOC generator and task binding devices differ")
            return rng_key
        seed = self.seed if not isinstance(rng_key, (int, np.integer)) else int(rng_key)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def _schedule(self, step, total):
        margin, rho, qp_gate, qp_prob, top_k = 0.0, 1.0, True, 1.0, -1
        if self.scheduler is not None:
            constraint_schedulers = getattr(
                self.scheduler, "constraint_schedulers", ()
            )
            if constraint_schedulers:
                from genedynamics.core.constraints.core.types import ScheduleState

                values = constraint_schedulers[0].constraint_params(
                    ScheduleState(k=step, K=total)
                ) or {}
                margin = float(values.get("margin", margin))
                rho = float(values.get("rho", rho))
                qp_gate = bool(values.get("qp_gate", qp_gate))
                qp_prob = float(values.get("qp_prob", qp_prob))
                top_k = int(values.get("topK", top_k))
        return (
            {"k": step, "K": total},
            {
                "margin": margin,
                "rho": rho,
                "qp_gate": qp_gate,
                "qp_prob": qp_prob,
                "topK": top_k,
            },
        )

    def _evaluate(self, candidates):
        result = self.env.evaluate_action_sequences(candidates)
        required = (
            "costs",
            "physical_states",
            "model_calls",
            "model_call_unit",
        )
        if any(not hasattr(result, name) for name in required):
            raise TypeError("Torch MDOC task evaluation lacks required evidence")
        costs = result.costs
        if (
            not isinstance(costs, torch.Tensor)
            or costs.shape != (candidates.shape[0],)
            or not torch.isfinite(costs).all()
        ):
            raise ValueError("Torch MDOC task costs are invalid")
        expected_calls = int(candidates.shape[0] * candidates.shape[1])
        if (
            result.model_calls != expected_calls
            or result.model_call_unit != "scalar_dynamics_transition"
        ):
            raise ValueError("Torch MDOC task model-call evidence is not exact")
        return result

    def plan(self, x0: State, rng_key: Any | None = None) -> dict[str, Any]:
        device = self._device()
        generator = self._generator(rng_key, device)
        dtype = torch.float32
        betas = torch.linspace(
            self.beta0, self.betaT, self.Ndiffuse, dtype=dtype, device=device
        )
        sigmas = torch.sqrt(1.0 - torch.cumprod(1.0 - betas, dim=0))
        mean = torch.zeros(
            self.horizon, self.act_dim, dtype=dtype, device=device
        )
        reward_history, mean_history, sample_history = [], [], []
        model_calls = 0
        total_steps = max(0, self.Ndiffuse - 1)

        for index in range(self.Ndiffuse - 1, 0, -1):
            noise = torch.randn(
                self.Nsample,
                self.horizon,
                self.act_dim,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            candidates = (noise * sigmas[index] + mean).clamp(
                -self.action_limit, self.action_limit
            )
            step = self.Ndiffuse - 1 - index
            schedule_state, schedule_params = self._schedule(step, total_steps)
            candidates = self.constraint_filter.apply_actions_batch(
                x0,
                candidates,
                env=self.env,
                obstacles=self.obstacles,
                schedule_state=schedule_state,
                schedule_params=schedule_params,
            )
            if not isinstance(candidates, torch.Tensor):
                candidates = torch.as_tensor(
                    candidates, dtype=dtype, device=device
                )
            evaluation = self._evaluate(candidates)
            model_calls += int(evaluation.model_calls)
            rewards = -evaluation.costs
            deviation = rewards.std(unbiased=False)
            if float(deviation) < 1.0e-4:
                deviation = torch.ones((), dtype=dtype, device=device)
            temperature = self.temp_sample
            if self.scheduler is not None:
                diffusion_schedulers = getattr(
                    self.scheduler, "diffusion_schedulers", ()
                )
                if diffusion_schedulers:
                    from genedynamics.core.constraints.core.types import ScheduleState

                    values = diffusion_schedulers[0].diffusion_params(
                        ScheduleState(k=step, K=total_steps)
                    ) or {}
                    temperature = float(values.get("T_k", temperature))
            weights = torch.softmax(
                (rewards - rewards.mean()) / (deviation * temperature), dim=0
            )
            mean = torch.einsum("n,nha->ha", weights, candidates)
            mean = self.constraint_filter.apply_actions(
                x0,
                mean,
                env=self.env,
                obstacles=self.obstacles,
                schedule_state=schedule_state,
                schedule_params=schedule_params,
            )
            if not isinstance(mean, torch.Tensor):
                mean = torch.as_tensor(mean, dtype=dtype, device=device)
            reward_history.append(rewards.mean().detach().cpu())
            mean_history.append(mean.detach().cpu())
            sample_history.append(candidates.detach().cpu())

        final_actions = mean.clamp(-self.action_limit, self.action_limit)
        final = self._evaluate(final_actions.unsqueeze(0))
        model_calls += int(final.model_calls)
        states = final.physical_states[0]
        cost = float(final.costs[0].detach().cpu())
        maximum_violation = getattr(final, "maximum_violation", None)
        predicted_feasible = getattr(final, "predicted_feasible", None)
        return {
            "actions": _host(final_actions).astype(np.float32, copy=False),
            "states": _host(states).astype(np.float32, copy=False),
            "rewards": np.asarray([-cost], dtype=np.float32),
            "initial_state": _host(states[0]).astype(np.float32, copy=False),
            "candidate_states": [
                _host(states).astype(np.float32, copy=False)
            ],
            "candidate_actions": [
                _host(final_actions).astype(np.float32, copy=False)
            ],
            "candidate_costs": np.asarray([cost], dtype=np.float32),
            "best_idx": 0,
            "diffusion_rewards": np.asarray(reward_history, dtype=np.float32),
            "diffusion_actions_traj": np.asarray(mean_history, dtype=np.float32),
            "diffusion_sampled_actions": np.asarray(
                sample_history, dtype=np.float32
            ),
            "model_calls": model_calls,
            "model_call_unit": "scalar_dynamics_transition",
            "maximum_violation": (
                None
                if maximum_violation is None
                else float(maximum_violation[0].detach().cpu())
            ),
            "predicted_feasible": (
                None
                if predicted_feasible is None
                else bool(predicted_feasible[0].detach().cpu())
            ),
            "termination_reason": "torch_mdoc_plan_returned",
        }

    def plan_batch(self, x0: State, keys):
        return [self.plan(x0, key) for key in keys]

    def sample_trajectories(
        self,
        x0: State,
        n_samples: int,
        rng_key: Any | None = None,
    ) -> list[Trajectory]:
        return [
            Trajectory(
                states=list(result["states"]),
                actions=list(result["actions"]),
                info=result,
            )
            for result in (
                self.plan(x0, self.seed + index)
                for index in range(max(1, int(n_samples)))
            )
        ]


__all__ = ["MDOCBackendTorch"]
