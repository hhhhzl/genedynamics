"""Torch 2GO backend for task-owned differentiable trajectory evaluators.

This is the eager Torch counterpart of the JAX scan core.  GD still owns the
2GO update order: sample/evaluate, budgeted probe retraction, constraint
geometry, CVaR + route gating, manifold-projected score/noise, transport-family
update, and conditional mean retraction.  A task binding supplies only rollout
costs, feasibility, and differentiable constraint geometry.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from genedynamics.core.constraints.action_filters import (
    ConstraintFilter,
    NoOpConstraintFilter,
)
from genedynamics.core.types import State, Trajectory
from genedynamics.genemetry import (
    CfsRetraction,
    MultimodalGate,
    ProbeModulator,
    ProbePipeline,
    ScheduleOverlay,
    SdfManifold,
    SlidingWindow,
    resolve_overlay_config,
)


def _host(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class TwoGOBackendTorch:
    """Geometry-aware gated diffusion over an exact Torch task binding."""

    def __init__(
        self,
        *,
        solver,
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
        action_extra_sigma: float = 0.0,
        **kwargs: Any,
    ):
        del legacy_energy, show_tqdm, kwargs
        if not hasattr(env_adapter, "evaluate_action_sequences"):
            raise TypeError("Torch 2GO requires dynamics.evaluate_action_sequences")
        if not hasattr(env_adapter, "constraint_geometry"):
            raise TypeError("Torch 2GO requires dynamics.constraint_geometry")
        if min(horizon, Nsample, Ndiffuse) <= 0 or Nsample < 2 or temp_sample <= 0:
            raise ValueError("Torch 2GO dimensions and temperature are invalid")
        if action_limit <= 0 or not 0 <= beta0 <= betaT < 1:
            raise ValueError("Torch 2GO diffusion parameters are invalid")

        self.env = env_adapter
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.Nsample = int(Nsample)
        self.Ndiffuse = int(Ndiffuse)
        self.temp_sample = float(temp_sample)
        self.beta0 = float(beta0)
        self.betaT = float(betaT)
        self.action_limit = float(action_limit)
        self.action_extra_sigma = float(action_extra_sigma)
        self.seed = int(seed)
        self.act_dim = int(env_adapter.act_dim)
        self.scheduler = scheduler
        self.obstacles = obstacles
        self.constraint_filter = constraint_filter or NoOpConstraintFilter()
        self.num_modes = 1

        cfg = solver.config
        self.cvar_alpha = float(cfg["twogo_cvar_alpha"])
        if not 0.0 <= self.cvar_alpha < 1.0:
            raise ValueError("Torch 2GO CVaR alpha must be in [0, 1)")
        self.window = SlidingWindow(
            int(cfg["twogo_window_size"]), int(cfg["twogo_window_stride"])
        )
        self.active_topk = int(cfg["twogo_active_topk"])
        self.agp_eta = float(cfg["twogo_agp_eta"])
        self.multi_scale = float(cfg["twogo_multi_scale"])
        self.enable_local_gating = bool(cfg["twogo_enable_local_gating"])
        self.retract_qp_boost = bool(cfg["twogo_retract_qp_boost"])
        self.retract_qp_every = max(1, int(cfg["twogo_retract_qp_every"]))
        self.stability_eps = max(1.0e-8, float(cfg["twogo_stability_eps"]))
        self.gamma_init = float(np.clip(cfg["twogo_gamma_init"], 0.0, 1.0))
        self.probe_enable = bool(cfg["twogo_probe_enable"])
        self.probe_geom_alpha = float(cfg["twogo_probe_geom_alpha"])
        configured_probes = cfg.get("twogo_probe_b")
        self.probe_count = (
            int(configured_probes)
            if configured_probes is not None
            else round(float(cfg["twogo_probe_frac"]) * self.Nsample)
        )
        cap = cfg.get("twogo_probe_m_cap")
        if cap is not None:
            self.probe_count = min(self.probe_count, int(cap))
        self.probe_count = max(0, min(self.Nsample, self.probe_count))
        self.gate_ema_beta = float(np.clip(cfg["twogo_gate_ema_beta"], 0.0, 0.99))
        self.gate_risk_theta = float(cfg["twogo_gate_risk_theta"])
        self.task_dir_alpha = float(cfg["twogo_task_dir_alpha"])
        self.enable_geometry = bool(cfg["twogo_enable_geometry"])
        schedule = cfg.get("twogo_transport_schedule")
        self.transport_schedule = None if schedule is None else str(schedule).lower()
        if self.transport_schedule not in {
            None,
            "ddpm",
            "ddim",
            "fm",
            "progress",
            "adaptive",
        }:
            raise ValueError("Torch 2GO transport schedule is unsupported")
        self.transport_ddim_frac = float(cfg["twogo_transport_ddim_frac"])
        self.transport_fm_frac = float(cfg["twogo_transport_fm_frac"])

        overlay_config = resolve_overlay_config(
            cfg.get("twogo_overlay"), rho_ref_default=500.0
        )
        self.overlay = ScheduleOverlay(config=overlay_config, backend="torch")
        self.manifold = SdfManifold(backend="torch")
        self.gate = MultimodalGate(
            backend="torch",
            multi_scale=self.multi_scale,
            enable_local_gating=self.enable_local_gating,
        )
        self.modulator = ProbeModulator()
        self.retraction = CfsRetraction(backend="torch", filter_fn=self._filter_single)
        self.probes = ProbePipeline(
            backend="torch",
            tail_mix=float(cfg["twogo_probe_tail_mix"]),
            pool_ratio=float(cfg["twogo_probe_tail_pool_ratio"]),
            max_probes=max(1, self.probe_count),
        )

    def _device(self):
        device = getattr(self.env, "device", None)
        if device is None:
            device = getattr(getattr(self.env, "dynamics", None), "device", None)
        return torch.device(device or "cpu")

    def _generator(self, rng_key, device):
        if isinstance(rng_key, torch.Generator):
            if torch.device(rng_key.device) != device:
                raise ValueError("Torch 2GO generator and binding devices differ")
            return rng_key
        seed = self.seed if not isinstance(rng_key, (int, np.integer)) else int(rng_key)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def _binding_model_call_count(self):
        dynamics = getattr(self.env, "dynamics", None)
        count = getattr(dynamics, "model_call_count", None)
        if count is None:
            count = getattr(self.env, "model_call_count", None)
        return None if count is None else int(count)

    def _schedule(self, step, total):
        values = {
            "margin": 0.0,
            "rho": 1.0,
            "qp_gate": True,
            "qp_prob": 1.0,
            "I_QP": 1,
            "eps": 1.0e-4,
            "topK": -1,
        }
        if self.scheduler is not None:
            schedulers = getattr(self.scheduler, "constraint_schedulers", ())
            if schedulers:
                from genedynamics.core.constraints.core.types import ScheduleState

                resolved = (
                    schedulers[0].constraint_params(ScheduleState(k=step, K=total))
                    or {}
                )
                values.update({key: resolved[key] for key in values if key in resolved})
        state = {"k": int(step), "K": int(total)}
        values["margin"] = float(values["margin"])
        values["rho"] = float(values["rho"])
        values["qp_gate"] = bool(values["qp_gate"])
        values["qp_prob"] = float(values["qp_prob"])
        values["I_QP"] = int(values["I_QP"])
        values["eps"] = float(values["eps"])
        values["topK"] = int(values["topK"])
        return state, values

    def _temperature(self, step, total):
        if self.scheduler is None:
            return self.temp_sample
        schedulers = getattr(self.scheduler, "diffusion_schedulers", ())
        if not schedulers:
            return self.temp_sample
        from genedynamics.core.constraints.core.types import ScheduleState

        values = schedulers[0].diffusion_params(ScheduleState(k=step, K=total)) or {}
        return float(values.get("T_k", self.temp_sample))

    def _evaluate(self, candidates):
        before = self._binding_model_call_count()
        if before is None:
            raise TypeError("Torch 2GO requires exact binding model_call_count")
        result = self.env.evaluate_action_sequences(candidates)
        after = self._binding_model_call_count()
        expected = int(candidates.shape[0] * candidates.shape[1])
        if after - before != expected:
            raise RuntimeError("Torch 2GO evaluator model-call count is not exact")
        costs = getattr(result, "costs", None)
        violations = getattr(result, "maximum_violation", None)
        if (
            not isinstance(costs, torch.Tensor)
            or costs.shape != candidates.shape[:1]
            or not torch.isfinite(costs).all()
            or not isinstance(violations, torch.Tensor)
            or violations.shape != candidates.shape[:1]
            or not torch.isfinite(violations).all()
        ):
            raise ValueError("Torch 2GO task evaluation is invalid")
        if (
            getattr(result, "model_calls", None) != expected
            or getattr(result, "model_call_unit", None) != "scalar_dynamics_transition"
        ):
            raise ValueError("Torch 2GO task evaluation lacks exact call evidence")
        return result

    def _constraint_geometry(self, actions):
        before = self._binding_model_call_count()
        if before is None:
            raise TypeError("Torch 2GO geometry requires exact model-call accounting")
        result = self.env.constraint_geometry(actions)
        after = self._binding_model_call_count()
        vectors = getattr(result, "vectors", result)
        reported = getattr(result, "model_calls", after - before)
        if (
            not isinstance(vectors, torch.Tensor)
            or vectors.shape != actions.shape
            or not torch.isfinite(vectors).all()
            or reported != after - before
            or getattr(result, "model_call_unit", "scalar_dynamics_transition")
            != "scalar_dynamics_transition"
        ):
            raise ValueError("Torch 2GO task geometry is invalid or unaccounted")
        return vectors

    def _filter_single(self, state, trajectory, schedule_state, schedule_params):
        return self.constraint_filter.apply_actions(
            state,
            trajectory,
            env=self.env,
            obstacles=self.obstacles,
            schedule_state=schedule_state,
            schedule_params=schedule_params,
        )

    def _retract(self, state, trajectory, schedule_state, schedule_params):
        before = self._binding_model_call_count()
        result = self.retraction.retract(
            state,
            trajectory,
            {"sched_state": schedule_state, "sched_params": schedule_params},
        ).trajectory
        after = self._binding_model_call_count()
        if before is None or after is None or after < before:
            raise RuntimeError("Torch 2GO retraction call accounting changed")
        return result

    @staticmethod
    def _cvar(values, count):
        return torch.topk(values, count, largest=True, sorted=False).values.mean()

    def _transport_family(self, step, cvar, pi_multi, delta, theta):
        mode = self.transport_schedule
        if mode == "adaptive":
            if float(cvar) > float(delta):
                return 0
            return 1 if float(pi_multi) > float(theta) else 2
        if mode in {"ddpm", "ddim", "fm"}:
            return {"ddpm": 0, "ddim": 1, "fm": 2}[mode]
        if mode == "progress":
            fraction = step / max(self.Ndiffuse - 1, 1)
            if fraction >= self.transport_fm_frac:
                return 2
            if fraction >= self.transport_ddim_frac:
                return 1
        return 0

    def plan(self, x0: State, rng_key: Any | None = None) -> dict[str, Any]:
        device = self._device()
        generator = self._generator(rng_key, device)
        dtype = torch.float32
        betas = torch.linspace(
            self.beta0, self.betaT, self.Ndiffuse, device=device, dtype=dtype
        )
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
        diffusion_sigmas = torch.sqrt(1.0 - alpha_bars)
        mean = torch.zeros(self.horizon, self.act_dim, device=device, dtype=dtype)
        initial_calls = self._binding_model_call_count()
        if initial_calls is None:
            raise TypeError("Torch 2GO requires exact model-call accounting")

        histories = {
            name: []
            for name in (
                "reward",
                "actions",
                "samples",
                "margin",
                "r",
                "v_rate",
                "v_mean",
                "gamma",
                "sigma",
                "delta",
                "theta",
                "eta",
                "cvar",
                "activeK",
                "rho",
                "pi_multi",
                "rho_family",
                "probe_b",
                "rollout_eval_calls",
            )
        }
        gamma_previous = mean.new_tensor(self.gamma_init)
        route_ema = mean.new_tensor(1.0)
        tail_count = max(1, math.ceil((1.0 - self.cvar_alpha) * self.Nsample))
        total = max(1, self.Ndiffuse - 1)

        for step, index in enumerate(range(self.Ndiffuse - 1, -1, -1)):
            schedule_state, schedule_params = self._schedule(step, total)
            margin = mean.new_tensor(schedule_params["margin"])
            rho = mean.new_tensor(schedule_params["rho"])
            hardness = self.overlay.hardness(rho).to(device=device)
            kappa, delta = self.overlay.constraint_overlay(margin, rho)
            sigma, theta, eta = self.overlay.diffusion_overlay(hardness, self.agp_eta)
            kappa, delta = kappa.to(device=device), delta.to(device=device)
            sigma, theta, eta = (
                sigma.to(device=device),
                theta.to(device=device),
                eta.to(device=device),
            )
            window = self.window.mask_torch(step, self.horizon, device=device)
            noise = torch.randn(
                self.Nsample,
                self.horizon,
                self.act_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            candidates = (mean + diffusion_sigmas[index] * noise).clamp(
                -self.action_limit, self.action_limit
            )
            evaluation = self._evaluate(candidates)
            rewards = -evaluation.costs
            violations = evaluation.maximum_violation.clamp_min(0)
            cvar = self._cvar(violations, tail_count)
            risk = self._cvar(evaluation.maximum_violation, tail_count)
            v_rate = (evaluation.maximum_violation > 0).to(dtype).mean()
            v_mean = evaluation.maximum_violation.mean()

            do_probe = (
                self.probe_enable
                and schedule_params["qp_gate"]
                and self.probe_count > 0
            )
            if do_probe:
                before_probe = self._binding_model_call_count()
                probe_result = self.probes.sample_and_retract(
                    candidates,
                    evaluation.maximum_violation,
                    self.retraction,
                    x0,
                    {
                        "M_eff": self.probe_count,
                        "sched_state": schedule_state,
                        "sched_params": dict(
                            schedule_params, qp_gate=True, qp_prob=1.0, I_QP=1
                        ),
                    },
                    {"generator": generator},
                )
                after_probe = self._binding_model_call_count()
                if (
                    before_probe is None
                    or after_probe is None
                    or after_probe < before_probe
                ):
                    raise RuntimeError("Torch 2GO probe accounting changed")
                effective_candidates = probe_result.fixed_trajectories
                probe_geometry = probe_result.residual_geometry
                probed = int(probe_result.meta["n_probed"])
            else:
                effective_candidates = candidates
                probe_geometry = torch.zeros_like(mean)
                probed = 0

            base_geometry = self._constraint_geometry(mean) * window[:, None]
            geometry = self.modulator.modulate(
                base_geometry,
                probe_geometry=probe_geometry,
                window_mask=window,
                alpha=self.probe_geom_alpha,
            )
            active = min(max(1, self.active_topk), self.horizon)
            bundle = self.manifold.geometry(geometry, active, self.stability_eps)

            deviation = rewards.std(unbiased=False)
            if float(deviation) < 1.0e-4:
                deviation = mean.new_tensor(1.0)
            weights = torch.softmax(
                (rewards - rewards.mean())
                / (deviation * self._temperature(step, total)),
                dim=0,
            )
            weighted = torch.einsum("n,nha->ha", weights, effective_candidates)
            gate = self.gate.evaluate(
                candidates,
                mean,
                window,
                diffusion_sigmas[index],
                theta,
            )
            route_ema = (
                self.gate_ema_beta * route_ema
                + (1.0 - self.gate_ema_beta) * gate.meta["pi_multi"]
            )
            risk_threshold = self.gate_risk_theta * (1.0 + 4.0 * hardness)
            gamma = ((route_ema > theta) | (cvar > risk_threshold)).to(dtype)
            if not self.enable_local_gating:
                gamma = mean.new_tensor(1.0)
            gamma_previous = gamma

            task_delta = -mean
            task_direction = torch.zeros_like(mean)
            norm = torch.linalg.vector_norm(task_delta)
            if self.task_dir_alpha != 0.0 and float(norm) > 1.0e-6:
                task_direction = (
                    self.task_dir_alpha * (1.0 - hardness) * task_delta / norm
                )
            score = kappa * (weighted - mean) + task_direction
            extra_noise = torch.randn(
                mean.shape, device=device, dtype=dtype, generator=generator
            )
            if self.enable_geometry and bool(bundle.is_valid):
                score = self.manifold.project(score, bundle, mode="metric")
                extra_noise = self.manifold.project(extra_noise, bundle, mode="tangent")

            sigma_effective = gamma_previous * (
                sigma + self.action_extra_sigma * (1.0 - step / total)
            )
            family = self._transport_family(step, cvar, route_ema, delta, theta)
            if family == 0:
                transported = weighted
            else:
                previous_index = max(index - 1, 0)
                numerator = alpha_bars[index] * (1.0 - alpha_bars[previous_index])
                denominator = (
                    alpha_bars[previous_index] * (1.0 - alpha_bars[index]) + 1.0e-12
                )
                retain = torch.sqrt((numerator / denominator).clamp(0.0, 1.0))
                transported = (1.0 - retain) * weighted + retain * mean
            proposed = transported + eta * score + sigma_effective * extra_noise

            high_risk = bool(cvar > delta)
            retract = (
                self.retract_qp_boost
                and schedule_params["qp_gate"]
                and (step % self.retract_qp_every == 0 or high_risk)
            )
            if retract:
                proposed = self._retract(
                    x0,
                    proposed,
                    schedule_state,
                    dict(schedule_params, qp_gate=True, qp_prob=1.0),
                )
            mean = proposed.clamp(-self.action_limit, self.action_limit)

            histories["reward"].append(rewards.mean().detach().cpu())
            histories["actions"].append(mean.detach().cpu())
            histories["samples"].append(effective_candidates.detach().cpu())
            for name, value in (
                ("margin", margin),
                ("r", risk),
                ("v_rate", v_rate),
                ("v_mean", v_mean),
                ("gamma", gamma),
                ("sigma", sigma_effective),
                ("delta", delta),
                ("theta", theta),
                ("eta", eta),
                ("cvar", cvar),
                ("activeK", bundle.active_count),
                ("rho", rho),
                ("pi_multi", route_ema),
            ):
                histories[name].append(value.detach().cpu())
            histories["rho_family"].append(torch.tensor(float(family)))
            histories["probe_b"].append(torch.tensor(float(probed)))
            histories["rollout_eval_calls"].append(
                torch.tensor(float(self.Nsample * self.horizon))
            )

        final = self._evaluate(mean.unsqueeze(0))
        total_calls = self._binding_model_call_count() - initial_calls
        states = final.physical_states[0]
        cost = float(final.costs[0].detach().cpu())

        def history(name):
            return np.asarray(torch.stack(histories[name]), dtype=np.float32)

        return {
            "actions": _host(mean).astype(np.float32, copy=False),
            "states": _host(states).astype(np.float32, copy=False),
            "rewards": np.asarray([-cost], dtype=np.float32),
            "initial_state": _host(states[0]).astype(np.float32, copy=False),
            "reward_history": history("reward"),
            "diffusion_rewards": history("reward"),
            "diffusion_actions_traj": history("actions"),
            "diffusion_sampled_actions": history("samples"),
            "margin_hist": history("margin"),
            "r_hist": history("r"),
            "v_rate_hist": history("v_rate"),
            "v_mean_hist": history("v_mean"),
            "gamma_hist": history("gamma"),
            "sigma_hist": history("sigma"),
            "delta_hist": history("delta"),
            "theta_hist": history("theta"),
            "eta_hist": history("eta"),
            "cvar_hist": history("cvar"),
            "activeK_hist": history("activeK"),
            "rho_hist": history("rho"),
            "pi_multi_hist": history("pi_multi"),
            "rho_family_hist": history("rho_family"),
            "probe_b_hist": history("probe_b"),
            "rollout_eval_calls_hist": history("rollout_eval_calls"),
            "candidate_states": [_host(states).astype(np.float32, copy=False)],
            "candidate_actions": [_host(mean).astype(np.float32, copy=False)],
            "candidate_costs": np.asarray([cost], dtype=np.float32),
            "best_idx": 0,
            "model_calls": int(total_calls),
            "model_call_unit": "scalar_dynamics_transition",
            "maximum_violation": float(final.maximum_violation[0].detach().cpu()),
            "predicted_feasible": bool(final.predicted_feasible[0].detach().cpu()),
            "termination_reason": "torch_twogo_plan_returned",
        }

    def plan_batch(self, x0, rng_keys):
        return [self.plan(x0, key) for key in rng_keys]

    def sample_trajectories(self, x0, n_samples, rng_key=None):
        seed = self.seed if not isinstance(rng_key, (int, np.integer)) else int(rng_key)
        return [
            Trajectory(
                states=list(result["states"]),
                actions=list(result["actions"]),
                info=result,
            )
            for result in (
                self.plan(x0, seed + index) for index in range(max(1, int(n_samples)))
            )
        ]


__all__ = ["TwoGOBackendTorch"]
