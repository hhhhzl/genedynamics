"""Calibrated, observable reliability model for the MDAC acceptance gate.

The model deliberately stays small and CPU-friendly: standardized multi-output
ridge regression followed by split-conformal one-sided residual quantiles.  It
predicts the same task-owned risk vector used by the analytical acceptance gate
from signals that are measurable before executing a candidate's first action.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import jax.numpy as jnp
import numpy as np


FEATURE_NAMES = (
    "path_error",
    "normal_offset",
    "force_error",
    "deformation",
    "contact_loss",
    "phase",
    "action_along",
    "action_cross",
    "action_yaw",
    "action_stiffness_rms",
    "command_force_error",
    "action_delta_rms",
)
RISK_NAMES = (
    "force_violation",
    "contact_loss",
    "deformation",
    "force_mae",
)


def _finite_rows(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(y), axis=1)
    return x[keep], y[keep]


@dataclass(frozen=True)
class LinearReliabilityModel:
    """Standardized ridge predictor with calibrated per-risk upper bounds."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    upper_residual: np.ndarray
    support_radius: np.ndarray
    calibration_quantile: float = 0.95

    @classmethod
    def fit(
        cls,
        train_x: np.ndarray,
        train_y: np.ndarray,
        calibration_x: np.ndarray,
        calibration_y: np.ndarray,
        *,
        ridge: float = 1.0e-3,
        quantile: float = 0.95,
    ) -> "LinearReliabilityModel":
        train_x, train_y = _finite_rows(
            np.asarray(train_x, np.float64), np.asarray(train_y, np.float64)
        )
        calibration_x, calibration_y = _finite_rows(
            np.asarray(calibration_x, np.float64),
            np.asarray(calibration_y, np.float64),
        )
        if train_x.ndim != 2 or train_x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"train_x must have shape (N, {len(FEATURE_NAMES)})"
            )
        if train_y.ndim != 2 or train_y.shape[1] != len(RISK_NAMES):
            raise ValueError(f"train_y must have shape (N, {len(RISK_NAMES)})")
        if len(train_x) < len(FEATURE_NAMES) + 1:
            raise ValueError("not enough finite training rows for reliability fit")
        if len(calibration_x) < 2:
            raise ValueError("at least two finite calibration rows are required")
        if not 0.5 <= quantile < 1.0:
            raise ValueError("quantile must lie in [0.5, 1.0)")

        mean = np.mean(train_x, axis=0)
        scale = np.std(train_x, axis=0)
        scale = np.where(scale > 1.0e-6, scale, 1.0)
        z = (train_x - mean) / scale
        design = np.concatenate([z, np.ones((len(z), 1))], axis=1)
        penalty = np.eye(design.shape[1]) * float(ridge)
        penalty[-1, -1] = 0.0
        coeff = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
        weights, bias = coeff[:-1], coeff[-1]

        cal_pred = (calibration_x - mean) / scale @ weights + bias
        residual = calibration_y - cal_pred
        # Finite-sample split-conformal order statistic for a one-sided bound.
        n = len(residual)
        level = min(1.0, np.ceil((n + 1) * quantile) / n)
        upper = np.quantile(residual, level, axis=0, method="higher")
        calibration_z = np.abs((calibration_x - mean) / scale)
        state_scores = np.max(calibration_z[:, :6], axis=1)
        action_scores = np.max(calibration_z[:, 6:], axis=1)
        # Bonferroni split keeps the joint state-and-action support coverage at
        # the requested level without letting either group mask the other.
        support_quantile = 1.0 - (1.0 - quantile) / 2.0
        support_level = min(
            1.0, np.ceil((n + 1) * support_quantile) / n
        )
        support_radius = np.asarray([
            np.quantile(state_scores, support_level, method="higher"),
            np.quantile(action_scores, support_level, method="higher"),
        ])
        return cls(
            feature_mean=mean.astype(np.float32),
            feature_scale=scale.astype(np.float32),
            weights=weights.astype(np.float32),
            bias=bias.astype(np.float32),
            upper_residual=np.maximum(upper, 0.0).astype(np.float32),
            support_radius=np.maximum(support_radius, 1.0e-6).astype(np.float32),
            calibration_quantile=float(quantile),
        )

    def predict(self, features: Any):
        x = jnp.asarray(features, jnp.float32)
        z = (x - jnp.asarray(self.feature_mean)) / jnp.asarray(self.feature_scale)
        return z @ jnp.asarray(self.weights) + jnp.asarray(self.bias)

    def predict_upper(self, features: Any):
        raw = self.predict(features) + jnp.asarray(self.upper_residual)
        # All four risks are nonnegative; the first two are probabilities.
        bounded = jnp.maximum(raw, 0.0)
        return bounded.at[..., :2].set(jnp.clip(bounded[..., :2], 0.0, 1.0))

    def support_score(self, features: Any):
        """Maximum standardized feature excursion relative to seen support."""
        x = jnp.asarray(features, jnp.float32)
        z = jnp.abs(
            (x - jnp.asarray(self.feature_mean))
            / jnp.asarray(self.feature_scale)
        )
        radius = jnp.asarray(self.support_radius)
        state_score = jnp.max(z[..., :6], axis=-1) / radius[0]
        action_score = jnp.max(z[..., 6:], axis=-1) / radius[1]
        return jnp.maximum(state_score, action_score)

    def in_support(self, features: Any):
        """Whether confidence is calibrated for these observable features."""
        return self.support_score(features) <= 1.0

    def save(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> None:
        payload = {
            "schema_version": 1,
            "feature_names": list(FEATURE_NAMES),
            "risk_names": list(RISK_NAMES),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "upper_residual": self.upper_residual.tolist(),
            "support_radius": self.support_radius.tolist(),
            "calibration_quantile": self.calibration_quantile,
            "metadata": dict(metadata or {}),
        }
        with Path(path).open("w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LinearReliabilityModel":
        with Path(path).open() as f:
            payload = json.load(f)
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("reliability checkpoint feature contract mismatch")
        if tuple(payload.get("risk_names", ())) != RISK_NAMES:
            raise ValueError("reliability checkpoint risk contract mismatch")
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], np.float32),
            feature_scale=np.asarray(payload["feature_scale"], np.float32),
            weights=np.asarray(payload["weights"], np.float32),
            bias=np.asarray(payload["bias"], np.float32),
            upper_residual=np.asarray(payload["upper_residual"], np.float32),
            support_radius=np.asarray(payload["support_radius"], np.float32),
            calibration_quantile=float(payload["calibration_quantile"]),
        )


def samples_from_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build transition samples from existing MDAC metric records.

    ``series[t]`` is the post-action state for ``actions[t]``.  Therefore row
    ``t`` uses post-state ``t-1`` and action ``t`` to predict post-state ``t``;
    the reset transition is omitted because old result files do not store its
    equivalent signal vector.
    """
    xs, ys, provenance = [], [], []
    for record in records:
        series = record.get("series") or {}
        actions = np.asarray(series.get("actions", ()), np.float64)
        if actions.ndim != 2 or len(actions) < 2:
            continue
        force = np.asarray(series["force"], np.float64)
        force_cmd = np.asarray(series["force_cmd"], np.float64)
        contact = np.asarray(series["in_contact"], np.float64)
        deformation = np.asarray(series["deformation"], np.float64)
        path = np.asarray(series["gate_path_error"], np.float64)
        normal = np.asarray(series["normal_offset"], np.float64)
        n = min(map(len, (actions, force, force_cmd, contact, deformation, path, normal)))
        if n < 2:
            continue
        f_target = float(series["f_target"])
        f_min, f_max = float(series["f_min"]), float(series["f_max"])
        force_scale = max(f_max - f_min, 1.0e-6)
        def_scale = max(float(series["deformation_scale"]), 1.0e-6)
        path_scale = max(float(series.get("path_scale", 0.02)), 1.0e-6)
        normal_scale = max(float(series.get("normal_scale", def_scale)), 1.0e-6)
        scan_rate = float(series.get("scan_rate", 0.0))
        scan_span = max(float(series.get("scan_span", 1.0)), 1.0e-6)
        for t in range(1, n):
            prev, action = actions[t - 1], actions[t]
            x = np.asarray([
                path[t - 1] / path_scale,
                normal[t - 1] / normal_scale,
                (force[t - 1] - f_target) / force_scale,
                deformation[t - 1] / def_scale,
                float(contact[t - 1] <= 0.5),
                min(t * scan_rate / scan_span, 1.0),
                action[0], action[1], action[2],
                np.sqrt(np.mean(action[3:9] ** 2)),
                (force_cmd[t] - f_target) / force_scale,
                np.sqrt(np.mean((action - prev) ** 2)),
            ], np.float64)
            y = np.asarray([
                float(force[t] < f_min or force[t] > f_max),
                float(contact[t] <= 0.5),
                deformation[t] / def_scale,
                abs(force[t] - f_target) / force_scale,
            ], np.float64)
            xs.append(x)
            ys.append(y)
            provenance.append({
                "suite": record.get("suite", record.get("level")),
                "seed": record.get("seed"),
                "policy_training_seed": record.get("policy_training_seed"),
                "step": t,
            })
    return (
        np.asarray(xs, np.float32).reshape(-1, len(FEATURE_NAMES)),
        np.asarray(ys, np.float32).reshape(-1, len(RISK_NAMES)),
        provenance,
    )
