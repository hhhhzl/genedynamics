"""Calibrated, observable reliability model for the MGA acceptance gate.

The model deliberately stays small and CPU-friendly.  Surface scanning retains
its standardized ridge + split-conformal contract exactly.  PegInsert uses
ridge-logistic probabilities for binary force/torque/jam events and a one-sided
conformal bound for continuous force error.  Its observable features summarize
the complete committed horizon plus controller lag, so training and deployment
share the same latency-aware semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import jax
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

PEG_INSERT_FEATURE_NAMES = (
    "peg_x", "peg_y", "insertion_depth",
    "angle_x", "angle_y", "angle_z",
    "measured_lateral_force", "measured_axial_force",
    "measured_bending_torque", "measured_wrench_delta",
    "contact_count", "stall_fraction",
    "tracking_x", "tracking_y", "tracking_z",
    "tracking_angle_x", "tracking_angle_y", "tracking_angle_z",
    "motion_x", "motion_y", "motion_z",
    "motion_angle_x", "motion_angle_y", "motion_angle_z",
    "queued_action_x", "queued_action_y", "queued_action_z",
    "queued_rotation_x", "queued_rotation_y", "queued_rotation_z",
    "action_delay", "sensor_delay",
    "first_action_x", "first_action_y", "first_action_z",
    "first_rotation_norm", "first_action_delta_rms",
    "translation_rms", "rotation_rms", "stiffness_rms",
    "action_jerk_rms", "cumulative_lateral_action",
    "cumulative_axial_action", "force_action_max",
)
PEG_INSERT_RISK_NAMES = (
    "force_violation", "torque_violation", "jam", "force_mae",
)

_LINEAR_BASIS = "linear"
_CONTACT_PHASE_BASIS = "contact_phase"
_BASIS_MODES = {_LINEAR_BASIS, _CONTACT_PHASE_BASIS}


def _numpy_basis(raw: np.ndarray, z: np.ndarray, mode: str) -> np.ndarray:
    if mode == _LINEAR_BASIS:
        return z
    if mode != _CONTACT_PHASE_BASIS:
        raise ValueError(f"unsupported reliability basis_mode: {mode!r}")
    if raw.shape[-1] <= 11:
        raise ValueError("contact_phase basis requires PegInsert state features")
    stall = (raw[:, 11] > 0.0).astype(np.float64)[:, None]
    contact = (
        ((raw[:, 10] > 0.0) | (raw[:, 2] > 0.0))
        & (raw[:, 11] <= 0.0)
    ).astype(np.float64)[:, None]
    # Shared global coefficients retain statistical strength in sparse free
    # space.  Contact/stall blocks learn only phase residuals.
    return np.concatenate(
        [z, contact * z, stall * z, contact, stall], axis=1
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
    feature_names: tuple[str, ...] = FEATURE_NAMES
    risk_names: tuple[str, ...] = RISK_NAMES
    state_feature_count: int = 6
    support_state_feature_count: int = 6
    probability_risk_count: int = 2
    classification_probabilities: bool = False
    basis_mode: str = _LINEAR_BASIS

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
        feature_names: tuple[str, ...] = FEATURE_NAMES,
        risk_names: tuple[str, ...] = RISK_NAMES,
        state_feature_count: int = 6,
        support_state_feature_count: int | None = None,
        probability_risk_count: int = 2,
        classification_probabilities: bool = False,
        basis_mode: str = _LINEAR_BASIS,
    ) -> "LinearReliabilityModel":
        train_x, train_y = _finite_rows(
            np.asarray(train_x, np.float64), np.asarray(train_y, np.float64)
        )
        calibration_x, calibration_y = _finite_rows(
            np.asarray(calibration_x, np.float64),
            np.asarray(calibration_y, np.float64),
        )
        feature_names = tuple(feature_names)
        risk_names = tuple(risk_names)
        if train_x.ndim != 2 or train_x.shape[1] != len(feature_names):
            raise ValueError(
                f"train_x must have shape (N, {len(feature_names)})"
            )
        if train_y.ndim != 2 or train_y.shape[1] != len(risk_names):
            raise ValueError(f"train_y must have shape (N, {len(risk_names)})")
        if len(train_x) < len(feature_names) + 1:
            raise ValueError("not enough finite training rows for reliability fit")
        if len(calibration_x) < 2:
            raise ValueError("at least two finite calibration rows are required")
        if not 0.5 <= quantile < 1.0:
            raise ValueError("quantile must lie in [0.5, 1.0)")
        if basis_mode not in _BASIS_MODES:
            raise ValueError(f"unsupported reliability basis_mode: {basis_mode!r}")

        mean = np.mean(train_x, axis=0)
        scale = np.std(train_x, axis=0)
        scale = np.where(scale > 1.0e-6, scale, 1.0)
        z = (train_x - mean) / scale
        basis = _numpy_basis(train_x, z, basis_mode)
        design = np.concatenate([basis, np.ones((len(z), 1))], axis=1)
        penalty = np.eye(design.shape[1]) * float(ridge)
        penalty[-1, -1] = 0.0
        coeff = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
        if classification_probabilities and probability_risk_count:
            # Binary safety events require calibrated conditional
            # probabilities.  A conformal upper interval for each individual
            # Bernoulli outcome degenerates to one over broad regions and
            # cannot support a useful selective gate.  Fit ridge-logistic
            # channels instead; continuous risks retain the one-sided
            # conformal regression bound below.
            for risk_index in range(int(probability_risk_count)):
                target = train_y[:, risk_index]
                logistic_coeff = np.zeros(design.shape[1], np.float64)
                def objective(beta):
                    logits = design @ beta
                    return (
                        np.sum(np.logaddexp(0.0, logits) - target * logits)
                        + 0.5 * beta @ penalty @ beta
                    )

                for _ in range(100):
                    logit = np.clip(design @ logistic_coeff, -30.0, 30.0)
                    probability = 1.0 / (1.0 + np.exp(-logit))
                    variance = np.maximum(
                        probability * (1.0 - probability), 1.0e-5
                    )
                    hessian = (
                        design.T @ (variance[:, None] * design) + penalty
                    )
                    gradient = (
                        design.T @ (probability - target)
                        + penalty @ logistic_coeff
                    )
                    update = np.linalg.solve(hessian, gradient)
                    # Full Newton steps can diverge on nearly separable
                    # contact data. Backtracking keeps the penalized logistic
                    # objective decreasing instead of saturating all risks.
                    step = 1.0
                    current_loss = objective(logistic_coeff)
                    descent = gradient @ update
                    while step > 1.0e-8:
                        candidate = logistic_coeff - step * update
                        if objective(candidate) <= current_loss - 1.0e-4 * step * descent:
                            break
                        step *= 0.5
                    logistic_coeff -= step * update
                    if np.max(np.abs(step * update)) < 1.0e-7:
                        break
                coeff[:, risk_index] = logistic_coeff
        weights, bias = coeff[:-1], coeff[-1]

        calibration_z = (calibration_x - mean) / scale
        cal_raw = _numpy_basis(
            calibration_x, calibration_z, basis_mode
        ) @ weights + bias
        cal_pred = cal_raw.copy()
        if classification_probabilities and probability_risk_count:
            cal_pred[:, :probability_risk_count] = 1.0 / (
                1.0 + np.exp(-np.clip(
                    cal_raw[:, :probability_risk_count], -30.0, 30.0
                ))
            )
        residual = calibration_y - cal_pred
        # Finite-sample split-conformal order statistic for a one-sided bound.
        n = len(residual)
        level = min(1.0, np.ceil((n + 1) * quantile) / n)
        upper = np.quantile(residual, level, axis=0, method="higher")
        if classification_probabilities and probability_risk_count:
            upper[:probability_risk_count] = 0.0
        calibration_z = np.abs(calibration_z)
        split = int(state_feature_count)
        if split < 1 or split >= len(feature_names):
            raise ValueError("state_feature_count must split state and action features")
        if probability_risk_count < 0 or probability_risk_count > len(risk_names):
            raise ValueError("invalid probability_risk_count")
        support_split = int(
            split if support_state_feature_count is None
            else support_state_feature_count
        )
        if support_split < 1 or support_split > split:
            raise ValueError(
                "support_state_feature_count must lie within state features"
            )
        state_scores = np.max(calibration_z[:, :support_split], axis=1)
        action_scores = np.max(calibration_z[:, split:], axis=1)
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
            feature_names=feature_names,
            risk_names=risk_names,
            state_feature_count=split,
            support_state_feature_count=support_split,
            probability_risk_count=int(probability_risk_count),
            classification_probabilities=bool(classification_probabilities),
            basis_mode=basis_mode,
        )

    def predict(self, features: Any):
        x = jnp.asarray(features, jnp.float32)
        z = (x - jnp.asarray(self.feature_mean)) / jnp.asarray(self.feature_scale)
        if self.basis_mode == _LINEAR_BASIS:
            basis = z
        elif self.basis_mode == _CONTACT_PHASE_BASIS:
            stall = (x[..., 11] > 0.0).astype(x.dtype)[..., None]
            contact = (
                ((x[..., 10] > 0.0) | (x[..., 2] > 0.0))
                & (x[..., 11] <= 0.0)
            ).astype(x.dtype)[..., None]
            basis = jnp.concatenate(
                [z, contact * z, stall * z, contact, stall], axis=-1
            )
        else:
            raise ValueError(
                f"unsupported reliability basis_mode: {self.basis_mode!r}"
            )
        raw = basis @ jnp.asarray(self.weights) + jnp.asarray(self.bias)
        if not self.classification_probabilities:
            return raw
        count = int(self.probability_risk_count)
        return raw.at[..., :count].set(jax.nn.sigmoid(raw[..., :count]))

    def predict_upper(self, features: Any):
        raw = self.predict(features) + jnp.asarray(self.upper_residual)
        # All four risks are nonnegative; the first two are probabilities.
        bounded = jnp.maximum(raw, 0.0)
        n_probability = int(self.probability_risk_count)
        return bounded.at[..., :n_probability].set(
            jnp.clip(bounded[..., :n_probability], 0.0, 1.0)
        )

    def support_score(self, features: Any):
        """Maximum standardized feature excursion relative to seen support."""
        return jnp.maximum(
            self.support_score_state(features),
            self.support_score_action(features),
        )

    def support_score_state(self, features: Any):
        """Standardized excursion of observable state features only."""
        x = jnp.asarray(features, jnp.float32)
        z = jnp.abs(
            (x - jnp.asarray(self.feature_mean))
            / jnp.asarray(self.feature_scale)
        )
        radius = jnp.asarray(self.support_radius)
        split = int(self.support_state_feature_count)
        return jnp.max(z[..., :split], axis=-1) / radius[0]

    def support_score_action(self, features: Any):
        """Standardized excursion of candidate-action features only."""
        x = jnp.asarray(features, jnp.float32)
        z = jnp.abs(
            (x - jnp.asarray(self.feature_mean))
            / jnp.asarray(self.feature_scale)
        )
        split = int(self.state_feature_count)
        return jnp.max(z[..., split:], axis=-1) / jnp.asarray(
            self.support_radius
        )[1]

    def in_support(self, features: Any):
        """Whether confidence is calibrated for these observable features."""
        return self.support_score(features) <= 1.0

    def save(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> None:
        payload = {
            "schema_version": 4 if self.basis_mode != _LINEAR_BASIS else 3,
            "feature_names": list(self.feature_names),
            "risk_names": list(self.risk_names),
            "state_feature_count": int(self.state_feature_count),
            "support_state_feature_count": int(
                self.support_state_feature_count
            ),
            "probability_risk_count": int(self.probability_risk_count),
            "classification_probabilities": bool(
                self.classification_probabilities
            ),
            "basis_mode": self.basis_mode,
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
        feature_names = tuple(payload.get("feature_names", ()))
        risk_names = tuple(payload.get("risk_names", ()))
        if not feature_names or not risk_names:
            raise ValueError("reliability checkpoint feature/risk contract missing")
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], np.float32),
            feature_scale=np.asarray(payload["feature_scale"], np.float32),
            weights=np.asarray(payload["weights"], np.float32),
            bias=np.asarray(payload["bias"], np.float32),
            upper_residual=np.asarray(payload["upper_residual"], np.float32),
            support_radius=np.asarray(payload["support_radius"], np.float32),
            calibration_quantile=float(payload["calibration_quantile"]),
            feature_names=feature_names,
            risk_names=risk_names,
            state_feature_count=int(payload.get("state_feature_count", 6)),
            support_state_feature_count=int(payload.get(
                "support_state_feature_count",
                payload.get("state_feature_count", 6),
            )),
            probability_risk_count=int(payload.get("probability_risk_count", 2)),
            classification_probabilities=bool(
                payload.get("classification_probabilities", False)
            ),
            basis_mode=str(payload.get("basis_mode", _LINEAR_BASIS)),
        )


def _peg_insert_samples_from_records(records):
    xs, ys, provenance = [], [], []
    for record in records:
        series = record.get("series") or {}
        actions = np.asarray(series.get("actions", ()), np.float64)
        pose = np.asarray(series.get("pose", ()), np.float64)
        angle = np.asarray(series.get("angle_vec", ()), np.float64)
        if actions.ndim != 2 or pose.ndim != 2 or angle.ndim != 2:
            continue
        names = (
            "measured_lateral_force", "measured_axial_force",
            "measured_bending_torque", "measured_wrench_delta",
            "contact_count", "stall_steps", "force_violation",
            "torque_violation", "jammed", "axial_force",
        )
        values = {
            name: np.asarray(series.get(name, ()), np.float64)
            for name in names
        }
        n = min([len(actions), len(pose), len(angle), *map(len, values.values())])
        if n < 2:
            continue
        cfg = {
            "socket_depth": max(float(series["socket_depth"]), 1.0e-6),
            "lateral_force_limit": max(float(series["lateral_force_limit"]), 1.0e-6),
            "f_max": max(float(series["f_max"]), 1.0e-6),
            "bending_torque_limit": max(float(series["bending_torque_limit"]), 1.0e-6),
            "jam_dwell_steps": max(float(series["jam_dwell_steps"]), 1.0),
            "f_target": float(series["f_target"]),
            "f_min": float(series.get("f_min", 0.0)),
            "f_cmd_pad": float(series.get("f_cmd_pad", 10.0)),
            # New records persist the execution queue explicitly.  The suite
            # fallback keeps the established canonical OOD records auditable;
            # those records predate the metadata field but their manifest
            # fixes ood_compound to a one-step action delay.
            "action_delay_steps": int(series.get(
                "action_delay_steps",
                1 if record.get("suite") == "ood_compound" else 0,
            )),
            "sensor_delay_steps": int(series.get(
                "sensor_delay_steps",
                1 if record.get("suite") == "ood_compound" else 0,
            )),
            "approach_gap": float(series.get("approach_gap", 0.008)),
        }
        # Historical PegInsert records used a jam dwell that kept counting
        # while the delayed controller was already executing zero-force or
        # retract commands.  Re-label those records with the current task
        # contract without rerunning physics: the old positive stall counter
        # identifies a contact/load/no-progress sample, while submitted actions
        # plus the recorded delay identify whether the *applied* command still
        # had insertion intent.  New records are idempotent under this pass.
        relabeled_stall = np.zeros(n, np.float64)
        relabeled_jam = np.zeros(n, np.float64)
        dwell = int(max(round(cfg["jam_dwell_steps"]), 1))
        lo = cfg["f_min"] - cfg["f_cmd_pad"]
        hi = cfg["f_max"] + cfg["f_cmd_pad"]
        count = 0
        for step in range(n):
            applied_index = step - cfg["action_delay_steps"]
            insertion_intent = False
            if applied_index >= 0:
                applied = actions[applied_index]
                raw_force = np.clip(applied[12], -1.0, 1.0)
                force_cmd = np.clip(
                    lo + 0.5 * (raw_force + 1.0) * (hi - lo),
                    cfg["f_min"],
                    cfg["f_max"],
                )
                insertion_intent = bool(
                    applied[2] > 1.0e-8
                    or force_cmd > cfg["f_min"] + 1.0e-6
                )
            old_stalled = values["stall_steps"][step] > 0.0
            count = count + 1 if old_stalled and insertion_intent else 0
            relabeled_stall[step] = count
            relabeled_jam[step] = float(count >= dwell)
        values["stall_steps"] = relabeled_stall
        values["jammed"] = relabeled_jam
        pose_scale = np.asarray([0.01, 0.01, cfg["socket_depth"]], np.float64)
        angle_scale = np.full(3, 0.1, np.float64)
        translation_step = np.asarray(
            series.get("translation_step", (0.001, 0.001, 0.0015)),
            np.float64,
        )
        rotation_step = float(series.get("rotation_step", 0.015))
        command = np.asarray(
            [0.0, 0.0, -cfg["approach_gap"], 0.0, 0.0, 0.0],
            np.float64,
        )
        command_history = []
        command_step = np.concatenate([
            translation_step, np.full(3, rotation_step, np.float64)
        ])
        command_lo = np.asarray(
            [-0.012, -0.012, -cfg["approach_gap"], -0.18, -0.18, -0.18]
        )
        command_hi = np.asarray(
            [0.012, 0.012, cfg["socket_depth"] + 0.002, 0.18, 0.18, 0.18]
        )
        for step in range(n):
            applied_index = step - cfg["action_delay_steps"]
            if applied_index >= 0:
                command = np.clip(
                    command + actions[applied_index, :6] * command_step,
                    command_lo,
                    command_hi,
                )
            command_history.append(command.copy())
        command_history = np.asarray(command_history)
        for t in range(1, n):
            previous_action = actions[t - 1]
            # Match PegInsert.risk_commit_horizon: one submitted action in
            # free space, then the realization-probe horizon near contact.  A
            # delayed execution queue must additionally cover the full jam
            # dwell window, which is four steps in the canonical task.
            near_contact = (
                pose[t - 1, 2] >= -0.0045
                or values["contact_count"][t - 1] > 0.0
            )
            contact_horizon = max(
                3,
                int(cfg["action_delay_steps"] + cfg["jam_dwell_steps"] - 1)
                if cfg["action_delay_steps"] > 0 else 3,
            )
            free_horizon = max(cfg["action_delay_steps"] + 1, 1)
            horizon = min(
                contact_horizon if near_contact else free_horizon,
                n - t,
            )
            candidate = actions[t:t + horizon]
            if not len(candidate):
                continue
            first = candidate[0]
            previous = np.concatenate([previous_action[None], candidate[:-1]], axis=0)
            delta = candidate - previous
            cumulative_translation = np.cumsum(candidate[:, :3], axis=0)
            current_pose = np.concatenate([pose[t - 1], angle[t - 1]])
            previous_pose = (
                np.concatenate([pose[t - 2], angle[t - 2]])
                if t > 1 else current_pose
            )
            tracking = (current_pose - command_history[t - 1]) / np.concatenate([
                pose_scale, angle_scale
            ])
            motion = (current_pose - previous_pose) / command_step
            queued = previous_action[:6]
            x = np.asarray([
                pose[t - 1, 0] / 0.01,
                pose[t - 1, 1] / 0.01,
                pose[t - 1, 2] / cfg["socket_depth"],
                angle[t - 1, 0] / 0.1,
                angle[t - 1, 1] / 0.1,
                angle[t - 1, 2] / 0.1,
                values["measured_lateral_force"][t - 1] / cfg["lateral_force_limit"],
                values["measured_axial_force"][t - 1] / cfg["f_max"],
                values["measured_bending_torque"][t - 1] / cfg["bending_torque_limit"],
                values["measured_wrench_delta"][t - 1] / 30.0,
                values["contact_count"][t - 1] / 4.0,
                values["stall_steps"][t - 1] / cfg["jam_dwell_steps"],
                *tracking,
                *motion,
                *queued,
                cfg["action_delay_steps"] / 2.0,
                cfg["sensor_delay_steps"] / 2.0,
                first[0], first[1], first[2],
                np.linalg.norm(first[3:6]),
                np.sqrt(np.mean((first - previous_action) ** 2)),
                np.sqrt(np.mean(candidate[:, :3] ** 2)),
                np.sqrt(np.mean(candidate[:, 3:6] ** 2)),
                np.sqrt(np.mean(candidate[:, 6:12] ** 2)),
                np.sqrt(np.mean(delta ** 2)),
                np.max(np.linalg.norm(cumulative_translation[:, :2], axis=-1)),
                np.sum(candidate[:, 2]),
                np.max(candidate[:, 12]),
            ], np.float64)
            end = t + horizon
            y = np.asarray([
                np.max(values["force_violation"][t:end]),
                np.max(values["torque_violation"][t:end]),
                np.max(values["jammed"][t:end]),
                np.mean(np.abs(values["axial_force"][t:end] - cfg["f_target"]))
                / cfg["f_max"],
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
        np.asarray(xs, np.float32).reshape(-1, len(PEG_INSERT_FEATURE_NAMES)),
        np.asarray(ys, np.float32).reshape(-1, len(PEG_INSERT_RISK_NAMES)),
        provenance,
    )


def samples_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    task: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build transition samples from existing MGA metric records.

    ``series[t]`` is the post-action state for ``actions[t]``.  Therefore row
    ``t`` uses post-state ``t-1`` and action ``t`` to predict post-state ``t``;
    the reset transition is omitted because old result files do not store its
    equivalent signal vector.
    """
    records = list(records)
    inferred_task = task or next(
        (str(r.get("task")) for r in records if r.get("task")), ""
    )
    if inferred_task == "manipulator_peg_insert" or any(
        "pose" in (r.get("series") or {}) for r in records
    ):
        return _peg_insert_samples_from_records(records)

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
