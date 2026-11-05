"""Public-history prediction and admission for physics-substep contact peaks.

The policy clock is 20 ms while the T2 contact contract is evaluated at 1 ms.
This module keeps those semantics explicit: inputs are public observations and
requested command chunks, while labels are maxima of recorded physics-substep
force peaks over the command horizon.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import nn

from genedynamics.learning.checkpoints.torch import to_cpu
from .response import PublicResponseSpec, features, public_frame
from .task_transport import (
    T2_TASK_OWNED_COORDINATES,
    T2_TRANSPORT_COORDINATES,
)


class RiskSetEmptyError(RuntimeError):
    """No declared T2 candidate satisfies the calibrated public risk bound."""

    def __init__(self):
        super().__init__("risk_set_empty")


@dataclass(frozen=True)
class PublicSubstepRiskCalibration:
    version: str = "public-T2-substep-risk-calibration-v1"
    miscoverage: float = 0.05
    additive_log_correction: tuple[float, float] = (0.0, 0.0)
    soft_force_limits_N: tuple[float, float] = (20.0, 8.0)
    interpolation_fractions: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0)

    def __post_init__(self):
        if self.version != "public-T2-substep-risk-calibration-v1":
            raise ValueError("unknown substep-risk calibration version")
        if not 0.0 < self.miscoverage < 1.0:
            raise ValueError("miscoverage must be strictly between zero and one")
        if (len(self.additive_log_correction) != 2
                or not np.isfinite(self.additive_log_correction).all()
                or min(self.additive_log_correction) < 0.0):
            raise ValueError("two finite nonnegative log-risk corrections required")
        if (len(self.soft_force_limits_N) != 2
                or not np.isfinite(self.soft_force_limits_N).all()
                or min(self.soft_force_limits_N) <= 0.0):
            raise ValueError("two finite positive soft force limits required")
        fractions = np.asarray(self.interpolation_fractions, dtype=float)
        if (fractions.ndim != 1 or len(fractions) < 2
                or not np.isfinite(fractions).all()
                or fractions[0] != 1.0 or fractions[-1] != 0.0
                or np.any(np.diff(fractions) >= 0.0)):
            raise ValueError("interpolation fractions must descend from one to zero")


@dataclass(frozen=True)
class PublicJointAdmissionCalibration:
    """One-sided calibration for force and guide-contact viability risks."""

    version: str = "public-T2-joint-admission-calibration-v2"
    miscoverage: float = 0.05
    additive_transformed_correction: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    risk_limits: tuple[float, ...] = (20.0, 8.0, 0.15, 0.5)
    interpolation_fractions: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0)

    def __post_init__(self):
        if self.version != "public-T2-joint-admission-calibration-v2":
            raise ValueError("unknown joint-admission calibration version")
        if not 0.0 < self.miscoverage < 1.0:
            raise ValueError("miscoverage must be strictly between zero and one")
        correction = np.asarray(self.additive_transformed_correction, dtype=float)
        limits = np.asarray(self.risk_limits, dtype=float)
        if (correction.shape != (4,) or not np.isfinite(correction).all()
                or np.any(correction < 0.0)):
            raise ValueError("four finite nonnegative transformed corrections required")
        if (limits.shape != (4,) or not np.isfinite(limits).all()
                or np.any(limits <= 0.0)):
            raise ValueError("four finite positive joint-risk limits required")
        fractions = np.asarray(self.interpolation_fractions, dtype=float)
        if (fractions.ndim != 1 or len(fractions) < 2
                or not np.isfinite(fractions).all()
                or fractions[0] != 1.0 or fractions[-1] != 0.0
                or np.any(np.diff(fractions) >= 0.0)):
            raise ValueError("interpolation fractions must descend from one to zero")


class PublicSubstepRiskEnsemble(nn.Module):
    """Predict log1p(total/lateral) substep peaks from public information."""

    VERSION = "public-T2-substep-risk-ensemble-v1"

    def __init__(self, spec: PublicResponseSpec | None = None):
        super().__init__()
        self.spec = spec or PublicResponseSpec()
        self.members = nn.ModuleList()
        for _ in range(self.spec.members):
            sizes = (
                self.spec.feature_dim + 13 * self.spec.horizon,
                *self.spec.hidden, 2,
            )
            layers = []
            for index, (source, target) in enumerate(zip(sizes[:-1], sizes[1:])):
                layers.append(nn.Linear(source, target))
                if index < len(sizes) - 2:
                    layers.append(nn.SiLU())
            self.members.append(nn.Sequential(*layers))

    def forward(self, public_features, commands):
        if (public_features.shape[-1] != self.spec.feature_dim
                or commands.shape[-2:] != (self.spec.horizon, 13)):
            raise ValueError("substep-risk input schema mismatch")
        value = torch.cat((public_features, commands.flatten(start_dim=-2)), dim=-1)
        return torch.stack([member(value) for member in self.members])

    def checkpoint(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "spec": asdict(self.spec),
            "state_dict": to_cpu(self.state_dict()),
            "private_state": False,
            "task": "T2",
            "physics_dt_seconds": 0.001,
            "output_semantics": (
                "log1p exact total/lateral force maxima across all 1 ms physics "
                "substeps in the public command horizon"
            ),
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any], device: str = "cpu"):
        if (payload.get("version") != cls.VERSION
                or payload.get("private_state") is not False
                or payload.get("task") != "T2"
                or payload.get("physics_dt_seconds") != 0.001):
            raise ValueError("explicit public T2 substep-risk checkpoint required")
        values = dict(payload["spec"])
        values["hidden"] = tuple(values["hidden"])
        model = cls(PublicResponseSpec(**values)).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


class PublicJointAdmissionEnsemble(nn.Module):
    """Predict four transformed T2 risks from public history and one candidate."""

    VERSION = "public-T2-joint-force-contact-admission-ensemble-v2"
    OUTPUTS = (
        "log1p_total_force_N", "log1p_lateral_force_N",
        "log1p_contact_loss_ms", "negative_log_contact_coverage",
    )

    def __init__(self, spec: PublicResponseSpec | None = None):
        super().__init__()
        self.spec = spec or PublicResponseSpec()
        self.members = nn.ModuleList()
        for _ in range(self.spec.members):
            sizes = (
                self.spec.feature_dim + 13 * self.spec.horizon,
                *self.spec.hidden, 4,
            )
            layers = []
            for index, (source, target) in enumerate(zip(sizes[:-1], sizes[1:])):
                layers.append(nn.Linear(source, target))
                if index < len(sizes) - 2:
                    layers.append(nn.SiLU())
            self.members.append(nn.Sequential(*layers))

    def forward(self, public_features, commands):
        if (public_features.shape[-1] != self.spec.feature_dim
                or commands.shape[-2:] != (self.spec.horizon, 13)):
            raise ValueError("joint-admission input schema mismatch")
        value = torch.cat((public_features, commands.flatten(start_dim=-2)), dim=-1)
        return torch.stack([member(value) for member in self.members])

    def checkpoint(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "spec": asdict(self.spec),
            "state_dict": to_cpu(self.state_dict()),
            "private_state": False,
            "task": "T2",
            "physics_dt_seconds": 0.001,
            "outputs": self.OUTPUTS,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any], device: str = "cpu"):
        if (payload.get("version") != cls.VERSION
                or payload.get("private_state") is not False
                or payload.get("task") != "T2"
                or payload.get("physics_dt_seconds") != 0.001
                or tuple(payload.get("outputs", ())) != cls.OUTPUTS):
            raise ValueError("explicit public T2 joint-admission checkpoint required")
        values = dict(payload["spec"])
        values["hidden"] = tuple(values["hidden"])
        model = cls(PublicResponseSpec(**values)).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


class PublicSubstepRiskTrainer:
    """Tail-aware asymmetric trainer for an explicit substep-risk ensemble."""

    def __init__(self, model: PublicSubstepRiskEnsemble, *,
                 underprediction_weight: float = 4.0):
        if not np.isfinite(underprediction_weight) or underprediction_weight < 1.0:
            raise ValueError("finite underprediction weight >= 1 required")
        self.model = model
        self.underprediction_weight = float(underprediction_weight)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=model.spec.learning_rate,
        )
        self.steps = 0

    def step(self, public_features: torch.Tensor, commands: torch.Tensor,
             targets: torch.Tensor) -> dict[str, float]:
        self.model.train()
        predicted = self.model(public_features, commands)
        if targets.shape != predicted.shape[1:]:
            raise ValueError("one two-channel target per risk example required")
        error = predicted - targets.unsqueeze(0)
        weight = torch.where(
            error < 0.0,
            error.new_tensor(self.underprediction_weight),
            error.new_tensor(1.0),
        )
        loss = (weight * torch.nn.functional.smooth_l1_loss(
            predicted, targets.unsqueeze(0).expand_as(predicted), reduction="none",
        )).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.steps += 1
        return {
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(torch.as_tensor(gradient).detach().cpu()),
            "mean_prediction": float(predicted.detach().mean().cpu()),
            "mean_target": float(targets.detach().mean().cpu()),
        }


class PublicJointAdmissionTrainer:
    """Tail-aware trainer for four joint force/contact transformed risks."""

    def __init__(self, model: PublicJointAdmissionEnsemble, *, underprediction_weight: float = 4.0):
        if not np.isfinite(underprediction_weight) or underprediction_weight < 1.0:
            raise ValueError("finite underprediction weight >= 1 required")
        self.model = model
        self.underprediction_weight = float(underprediction_weight)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=model.spec.learning_rate)
        self.steps = 0

    def step(self, public_features: torch.Tensor, commands: torch.Tensor,
             targets: torch.Tensor) -> dict[str, float]:
        self.model.train()
        predicted = self.model(public_features, commands)
        if targets.shape != predicted.shape[1:]:
            raise ValueError("one four-channel target per joint-admission example required")
        error = predicted - targets.unsqueeze(0)
        weight = torch.where(
            error < 0.0, error.new_tensor(self.underprediction_weight), error.new_tensor(1.0),
        )
        loss = (weight * torch.nn.functional.smooth_l1_loss(
            predicted, targets.unsqueeze(0).expand_as(predicted), reduction="none",
        )).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.steps += 1
        return {
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(torch.as_tensor(gradient).detach().cpu()),
            "mean_prediction": float(predicted.detach().mean().cpu()),
            "mean_target": float(targets.detach().mean().cpu()),
        }


def substep_risk_windows(
    public: dict[str, np.ndarray], requests: np.ndarray, *,
    total_force_peaks_N: np.ndarray, lateral_force_peaks_N: np.ndarray,
    body: str, calibration_context: list[float] | tuple[float, float, float],
    spec: PublicResponseSpec,
) -> dict[str, np.ndarray]:
    """Create exact multi-rate risk labels from one recorded T2 episode."""
    requests = np.asarray(requests, dtype=np.float32)
    total = np.asarray(total_force_peaks_N, dtype=np.float32).reshape(-1)
    lateral = np.asarray(lateral_force_peaks_N, dtype=np.float32).reshape(-1)
    count = len(requests)
    if (requests.shape != (count, 13) or len(total) != count
            or len(lateral) != count
            or any(len(value) != count + 1 for value in public.values())):
        raise ValueError("one public transition and two substep peaks per request required")
    if (not np.isfinite(total).all() or not np.isfinite(lateral).all()
            or np.any(total < 0.0) or np.any(lateral < 0.0)):
        raise ValueError("substep peaks must be finite and nonnegative")
    starts = np.arange(max(0, count - spec.horizon + 1))
    frames = public_frame(public, body, spec)
    horizon = starts[:, None] + np.arange(spec.horizon)[None, :]
    bounds = np.asarray([.08] * 3 + [.4] * 3 + [1.] * 6 + [25.], np.float32)
    peaks = np.stack((total[horizon].max(axis=1), lateral[horizon].max(axis=1)), axis=1)
    return {
        "features": features(
            frames, starts, body, "T2", calibration_context, spec,
        ),
        "commands": (requests[horizon] / bounds).astype(np.float32),
        "targets": np.log1p(peaks).astype(np.float32),
        "ticks": starts,
    }


def fit_split_conformal_calibration(
    predicted_log_peaks: np.ndarray, true_log_peaks: np.ndarray, *,
    miscoverage: float = 0.05, soft_force_limits_N: tuple[float, float] = (20.0, 8.0),
    interpolation_fractions: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0),
) -> PublicSubstepRiskCalibration:
    """Fit per-channel one-sided split-conformal corrections.

    ``predicted_log_peaks`` has shape [members, examples, 2]. The ensemble
    maximum is the uncalibrated upper predictor. Corrections use the finite
    sample ``ceil((n+1)(1-alpha))/n`` order statistic with ``higher`` rounding.
    """
    predicted = np.asarray(predicted_log_peaks, dtype=float)
    truth = np.asarray(true_log_peaks, dtype=float)
    if (predicted.ndim != 3 or predicted.shape[2] != 2
            or truth.shape != predicted.shape[1:]
            or len(truth) == 0 or not np.isfinite(predicted).all()
            or not np.isfinite(truth).all()):
        raise ValueError("finite [members,N,2] predictions and [N,2] truth required")
    scores = np.maximum(truth - predicted.max(axis=0), 0.0)
    level = min(1.0, math.ceil((len(truth) + 1) * (1.0 - miscoverage)) / len(truth))
    correction = np.quantile(scores, level, axis=0, method="higher")
    return PublicSubstepRiskCalibration(
        miscoverage=float(miscoverage),
        additive_log_correction=tuple(float(value) for value in correction),
        soft_force_limits_N=tuple(float(value) for value in soft_force_limits_N),
        interpolation_fractions=tuple(float(value) for value in interpolation_fractions),
    )


def fit_block_conformal_calibration(
    predicted_log_peaks: np.ndarray, true_log_peaks: np.ndarray,
    episode_ids: np.ndarray, *, miscoverage: float = 0.05,
    soft_force_limits_N: tuple[float, float] = (20.0, 8.0),
    interpolation_fractions: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0),
) -> PublicSubstepRiskCalibration:
    """Fit one-sided corrections using each episode as one exchangeable block."""
    predicted = np.asarray(predicted_log_peaks, dtype=float)
    truth = np.asarray(true_log_peaks, dtype=float)
    groups = np.asarray(episode_ids).reshape(-1)
    if (predicted.ndim != 3 or predicted.shape[2] != 2
            or truth.shape != predicted.shape[1:]
            or groups.shape != (len(truth),) or len(truth) == 0
            or not np.isfinite(predicted).all() or not np.isfinite(truth).all()):
        raise ValueError("finite predictions, truth and one episode id per window required")
    point_scores = np.maximum(truth - predicted.max(axis=0), 0.0)
    unique = np.unique(groups)
    block_scores = np.stack([
        point_scores[groups == group].max(axis=0) for group in unique
    ])
    zeros = np.zeros((1, len(block_scores), 2), dtype=float)
    return fit_split_conformal_calibration(
        zeros, block_scores, miscoverage=miscoverage,
        soft_force_limits_N=soft_force_limits_N,
        interpolation_fractions=interpolation_fractions,
    )


def joint_admission_targets(
    force_peaks_N: np.ndarray, contact_loss_seconds: np.ndarray,
    contact_coverage: np.ndarray, outlet_without_contact: np.ndarray,
) -> np.ndarray:
    """Transform exact physical labels into four nonnegative upper risks."""
    force = np.asarray(force_peaks_N, dtype=np.float32)
    loss = np.asarray(contact_loss_seconds, dtype=np.float32)
    coverage = np.asarray(contact_coverage, dtype=np.float32)
    missing = np.asarray(outlet_without_contact, dtype=bool)
    if (force.ndim != 2 or force.shape[1] != 2 or loss.shape != (len(force),)
            or coverage.shape != (len(force),) or missing.shape != (len(force),)
            or not np.isfinite(force).all() or not np.isfinite(loss).all()
            or not np.isfinite(coverage).all() or np.any(force < 0.0)
            or np.any(loss < 0.0) or np.any((coverage < 0.0) | (coverage > 1.0))):
        raise ValueError("finite physical force/contact labels required")
    # Crossing the outlet without prior support is a hard contact-contract
    # failure. Encode it beyond both frozen contact limits so a regressor cannot
    # interpret the absence of a contact segment as zero risk.
    loss = np.where(missing, np.maximum(loss, np.float32(0.151)), loss)
    coverage = np.where(missing, 0.0, coverage)
    return np.stack((
        np.log1p(force[:, 0]),
        np.log1p(force[:, 1]),
        np.log1p(loss * np.float32(1000.0)),
        -np.log(np.clip(coverage, np.float32(1e-6), np.float32(1.0))),
    ), axis=1).astype(np.float32)


def decode_joint_admission_risks(transformed: np.ndarray | torch.Tensor):
    """Decode transformed outputs to N, N, seconds, and coverage deficit."""
    if isinstance(transformed, torch.Tensor):
        if transformed.shape[-1] != 4:
            raise ValueError("four transformed joint risks required")
        return torch.stack((
            torch.expm1(transformed[..., 0]).clamp(min=0.0),
            torch.expm1(transformed[..., 1]).clamp(min=0.0),
            torch.expm1(transformed[..., 2]).clamp(min=0.0) / 1000.0,
            (1.0 - torch.exp(-transformed[..., 3])).clamp(min=0.0, max=1.0),
        ), dim=-1)
    value = np.asarray(transformed, dtype=float)
    if value.shape[-1] != 4:
        raise ValueError("four transformed joint risks required")
    return np.stack((
        np.maximum(np.expm1(value[..., 0]), 0.0),
        np.maximum(np.expm1(value[..., 1]), 0.0),
        np.maximum(np.expm1(value[..., 2]), 0.0) / 1000.0,
        np.clip(1.0 - np.exp(-value[..., 3]), 0.0, 1.0),
    ), axis=-1)


def fit_joint_block_conformal_calibration(
    predicted_transformed: np.ndarray, true_transformed: np.ndarray,
    episode_ids: np.ndarray, *, miscoverage: float = 0.05,
    risk_limits: tuple[float, float, float, float] = (20.0, 8.0, 0.15, 0.5),
    interpolation_fractions: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0),
) -> PublicJointAdmissionCalibration:
    predicted = np.asarray(predicted_transformed, dtype=float)
    truth = np.asarray(true_transformed, dtype=float)
    groups = np.asarray(episode_ids).reshape(-1)
    if (predicted.ndim != 3 or predicted.shape[2] != 4
            or truth.shape != predicted.shape[1:] or groups.shape != (len(truth),)
            or len(truth) == 0 or not np.isfinite(predicted).all()
            or not np.isfinite(truth).all()):
        raise ValueError("finite joint predictions, truth and episode ids required")
    scores = np.maximum(truth - predicted.max(axis=0), 0.0)
    unique = np.unique(groups)
    block_scores = np.stack([scores[groups == group].max(axis=0) for group in unique])
    level = min(
        1.0,
        math.ceil((len(block_scores) + 1) * (1.0 - miscoverage)) / len(block_scores),
    )
    correction = np.quantile(block_scores, level, axis=0, method="higher")
    return PublicJointAdmissionCalibration(
        miscoverage=float(miscoverage),
        additive_transformed_correction=tuple(float(value) for value in correction),
        risk_limits=tuple(float(value) for value in risk_limits),
        interpolation_fractions=tuple(float(value) for value in interpolation_fractions),
    )


def calibrated_joint_upper_bound(
    predicted_transformed: torch.Tensor,
    calibration: PublicJointAdmissionCalibration,
) -> torch.Tensor:
    if predicted_transformed.ndim != 3 or predicted_transformed.shape[-1] != 4:
        raise ValueError("expected [members,batch,4] transformed predictions")
    correction = predicted_transformed.new_tensor(
        calibration.additive_transformed_correction,
    )
    transformed = (predicted_transformed.amax(dim=0) + correction).clamp(min=0.0)
    return decode_joint_admission_risks(transformed)


def select_joint_admissible_candidate(
    upper_risks: torch.Tensor, calibration: PublicJointAdmissionCalibration,
) -> tuple[torch.Tensor, torch.Tensor]:
    choices = len(calibration.interpolation_fractions) + 1
    if upper_risks.shape[-2:] != (choices, 4):
        raise ValueError("upper risks must contain every candidate and four channels")
    limits = upper_risks.new_tensor(calibration.risk_limits)
    safe = torch.all(upper_risks <= limits, dim=-1)
    admitted = torch.any(safe, dim=-1)
    first = safe.to(torch.int64).argmax(dim=-1)
    return torch.where(admitted, first, first.new_full(first.shape, -1)), admitted


def calibrated_force_upper_bound(
    predicted_log_peaks: torch.Tensor, calibration: PublicSubstepRiskCalibration,
) -> torch.Tensor:
    if predicted_log_peaks.ndim != 3 or predicted_log_peaks.shape[-1] != 2:
        raise ValueError("expected [members,batch,2] log-peak predictions")
    correction = predicted_log_peaks.new_tensor(calibration.additive_log_correction)
    return torch.expm1(predicted_log_peaks.amax(dim=0) + correction).clamp(min=0.0)


def interpolation_candidates(
    candidate: torch.Tensor, base: torch.Tensor,
    calibration: PublicSubstepRiskCalibration,
) -> torch.Tensor:
    """Build declared candidate-to-base chunks with an exact zero fallback."""
    if candidate.shape != base.shape or candidate.ndim != 3 or candidate.shape[-1] != 13:
        raise ValueError("matching [batch,horizon,13] candidate and base required")
    fractions = candidate.new_tensor(calibration.interpolation_fractions)
    values = base[:, None] + fractions[None, :, None, None] * (
        candidate[:, None] - base[:, None]
    )
    values[:, 0] = candidate
    values[:, -1] = base
    return values


def emergency_reference_brake_horizon(base: torch.Tensor) -> torch.Tensor:
    """Stop every T2 transport coordinate and preserve task-owned commands.

    The input and output are normalized command horizons.  The task continues
    to own normal motion, impedance, and force, so the emergency candidate may
    not alter those fields.  It is a candidate for the same learned risk model,
    not an automatically safe fallback.
    """
    if base.ndim != 3 or base.shape[-1] != 13:
        raise ValueError("expected a normalized [batch,horizon,13] T2 base")
    emergency = base.clone()
    emergency[..., list(T2_TRANSPORT_COORDINATES)] = 0.0
    if not torch.equal(
        emergency[..., list(T2_TASK_OWNED_COORDINATES)],
        base[..., list(T2_TASK_OWNED_COORDINATES)],
    ):
        raise AssertionError("emergency brake changed task-owned coordinates")
    return emergency


def risk_admission_candidates(
    candidate: torch.Tensor, base: torch.Tensor,
    calibration: PublicSubstepRiskCalibration,
) -> torch.Tensor:
    """Return ordered transport fractions, public base, and emergency brake."""
    interpolated = interpolation_candidates(candidate, base, calibration)
    emergency = emergency_reference_brake_horizon(base)
    return torch.cat((interpolated, emergency[:, None]), dim=1)


def select_admissible_candidate(
    upper_bounds_N: torch.Tensor, calibration: PublicSubstepRiskCalibration,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the first safe declared candidate and expose an empty safe set."""
    choices = len(calibration.interpolation_fractions) + 1
    if upper_bounds_N.shape[-2:] != (choices, 2):
        raise ValueError(
            "upper bounds must contain every interpolation and emergency candidate"
        )
    limits = upper_bounds_N.new_tensor(calibration.soft_force_limits_N)
    safe = torch.all(upper_bounds_N <= limits, dim=-1)
    admitted = torch.any(safe, dim=-1)
    first = safe.to(torch.int64).argmax(dim=-1)
    index = torch.where(admitted, first, first.new_full(first.shape, -1))
    return index, admitted


def select_admissible_fraction(
    upper_bounds_N: torch.Tensor, calibration: PublicSubstepRiskCalibration,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the largest safe interpolation fraction; expose an empty set."""
    fractions = upper_bounds_N.new_tensor(calibration.interpolation_fractions)
    if upper_bounds_N.shape[-2:] != (len(fractions), 2):
        raise ValueError("upper bounds must contain every declared interpolation fraction")
    limits = upper_bounds_N.new_tensor(calibration.soft_force_limits_N)
    safe = torch.all(upper_bounds_N <= limits, dim=-1)
    admitted = torch.any(safe, dim=-1)
    first = safe.to(torch.int64).argmax(dim=-1)
    index = torch.where(admitted, first, first.new_full(first.shape, -1))
    selected = fractions[first]
    selected = torch.where(
        admitted, selected, selected.new_full(selected.shape, float("nan")),
    )
    return selected, index
