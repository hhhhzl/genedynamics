"""Learned public-response transport for semantic task phases.

V1 multiplied a planner by a selected constant and fitted a response model only
after selection. V2 fits source and target response models first, scores every
bounded residual with an explicit task-band/response objective, and stores the
terms needed to recompute selection.
"""
from __future__ import annotations
from copy import deepcopy
import math
from typing import Any, Mapping
import numpy as np
from .ablation import treatment, treatment_metadata
from .residual_transport import BOUNDS

VERSION = "semantic-response-transport-v2"
RESPONSE_KIND = "public-linear-step-response-v2"
SELECTION_KIND = "learned-response-objective-v1"


def _phase_from_actor(actor: Any, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if id(actor) in seen:
        raise ValueError("cyclic semantic actor graph")
    seen.add(id(actor))
    if hasattr(actor, "phase"):
        value = np.asarray(actor.phase).reshape(-1)
        if value.size == 1:
            return int(value[0])
    if hasattr(actor, "stage") and np.asarray(actor.stage).size == 1:
        return int(np.asarray(actor.stage).item())
    observer = getattr(actor, "observer", None)
    memory = getattr(observer, "memory", None)
    if memory is not None and hasattr(memory, "phase"):
        value = np.asarray(memory.phase).reshape(-1)
        if value.size == 1:
            return int(value[0])
    for name in ("policy", "actor", "base"):
        child = getattr(actor, name, None)
        if child is not None and child is not actor:
            try:
                return _phase_from_actor(child, seen)
            except ValueError:
                pass
    raise ValueError("semantic actor exposes no public phase")


def _validate_response_fit(value: Mapping[str, Any]) -> None:
    required = {"weights", "train_samples", "validation_samples", "validation_mse"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("semantic response fit schema differs")
    weights = np.asarray(value["weights"], dtype=float)
    if weights.shape != (4, 3) or not np.isfinite(weights).all():
        raise ValueError("response weights must map xyz command plus bias to xyz velocity")
    if (any(int(value[name]) < 0 for name in ("train_samples", "validation_samples"))
            or not math.isfinite(float(value["validation_mse"]))
            or float(value["validation_mse"]) < 0):
        raise ValueError("semantic response diagnostic is invalid")


def _objective_row(raw, treatment_name, tolerance):
    value = treatment(treatment_name)
    task_loss = float(raw["task_loss"])
    calibrated = float(raw["calibrated_response_loss"])
    uncalibrated = float(raw["uncalibrated_response_loss"])
    response_loss = calibrated if value.calibration == "target_calibrated" else uncalibrated
    gate = (True if value.anchoring_gate == "unconditional"
            else task_loss <= float(raw["best_task_loss"]) + tolerance + 1e-12)
    objective = task_loss + value.response_weight * (response_loss if gate else 0.)
    return {
        "scale": float(raw["scale"]), "task_loss": task_loss,
        "calibrated_response_loss": calibrated,
        "uncalibrated_response_loss": uncalibrated,
        "anchoring_gate_active": bool(gate),
        "response_loss_used": response_loss, "objective": objective,
    }


def select_learned_scale(candidates, *, treatment_name, task_band_tolerance=.08):
    """Select a residual using the declared learned task/response objective."""
    treatment(treatment_name)
    if not math.isfinite(task_band_tolerance) or task_band_tolerance < 0:
        raise ValueError("finite nonnegative task-band tolerance required")
    if not candidates:
        candidates = [{"scale": 1., "task_loss": 2.,
                       "calibrated_response_loss": 0.,
                       "uncalibrated_response_loss": 0.}]
    scales = [float(row["scale"]) for row in candidates]
    if len(set(scales)) != len(scales) or any(
            not math.isfinite(scale) or not .5 <= scale <= 1.5 for scale in scales):
        raise ValueError("unique bounded candidate scales required")
    best = min(float(row["task_loss"]) for row in candidates)
    rows = [_objective_row({**dict(row), "best_task_loss": best},
                           treatment_name, task_band_tolerance)
            for row in candidates]
    selected = min(rows, key=lambda row: (row["objective"], row["scale"]))
    return {
        "kind": SELECTION_KIND, "selected_scale": selected["scale"],
        "task_band_tolerance": float(task_band_tolerance),
        "candidate_rows": rows,
        "tie_break": "lowest scale after minimum objective",
    }


def validate_checkpoint(checkpoint, *, task=None):
    expected = {"schema_version", "task", "treatment", "interaction_budget_seconds",
                "phase_scales", "active_phases", "translation_coordinates",
                "selection", "response_model"}
    if not isinstance(checkpoint, dict) or set(checkpoint) != expected:
        raise ValueError("semantic response checkpoint schema differs")
    if checkpoint["schema_version"] != VERSION:
        raise ValueError("semantic response checkpoint version differs")
    if task is not None and checkpoint["task"] != task:
        raise ValueError("semantic response checkpoint task differs")
    name = checkpoint["treatment"].get("treatment")
    if checkpoint["treatment"] != treatment_metadata(name):
        raise ValueError("semantic response treatment metadata differs")
    if checkpoint["interaction_budget_seconds"] not in (0, 30, 60, 120):
        raise ValueError("semantic response budget must use the frozen grid")
    scales = np.asarray(checkpoint["phase_scales"], dtype=float)
    phases = tuple(int(value) for value in checkpoint["active_phases"])
    coordinates = tuple(int(value) for value in checkpoint["translation_coordinates"])
    if (scales.ndim != 1 or len(scales) == 0 or not np.isfinite(scales).all()
            or np.any(scales < .5) or np.any(scales > 1.5)
            or len(set(phases)) != len(phases)
            or any(value < 0 or value >= len(scales) for value in phases)
            or not coordinates or len(set(coordinates)) != len(coordinates)
            or any(value not in (0, 1, 2) for value in coordinates)):
        raise ValueError("semantic response scales, phases, or coordinates differ")
    response = checkpoint["response_model"]
    if not isinstance(response, Mapping) or set(response) != {
            "kind", "source", "target", "input_semantics", "output_semantics"}:
        raise ValueError("semantic response model schema differs")
    if response["kind"] != RESPONSE_KIND:
        raise ValueError("semantic response model kind differs")
    _validate_response_fit(response["source"])
    _validate_response_fit(response["target"])
    selection = checkpoint["selection"]
    if not isinstance(selection, Mapping) or selection.get("kind") != SELECTION_KIND:
        raise ValueError("learned semantic selection required")
    required_row = {"scale", "task_loss", "calibrated_response_loss",
                    "uncalibrated_response_loss", "anchoring_gate_active",
                    "response_loss_used", "objective"}
    raw = []
    for row in selection.get("candidate_rows", ()):
        if set(row) != required_row:
            raise ValueError("learned semantic candidate row differs")
        raw.append({key: row[key] for key in (
            "scale", "task_loss", "calibrated_response_loss",
            "uncalibrated_response_loss")})
    recomputed = select_learned_scale(
        raw, treatment_name=name,
        task_band_tolerance=float(selection.get("task_band_tolerance", -1.)))
    if selection != recomputed:
        raise ValueError("learned semantic selection is not reproducible")
    selected = float(selection["selected_scale"])
    if any(not math.isclose(scales[phase], selected) for phase in phases):
        raise ValueError("selected learned residual is not bound to active phases")


class PublicSemanticPhaseTransport:
    """Execute one response-learned residual over a public task actor."""
    VERSION = VERSION

    def __init__(self, actor: Any, checkpoint, *, task):
        validate_checkpoint(checkpoint, task=task)
        self.actor = actor
        self.checkpoint = deepcopy(checkpoint)
        self.scales = np.asarray(checkpoint["phase_scales"], dtype=np.float32)
        self.active = frozenset(int(value) for value in checkpoint["active_phases"])
        self.coordinates = tuple(int(value) for value in checkpoint["translation_coordinates"])
        self.last_phase = None
        self.last_scale = 1.0

    def act(self, public):
        action = np.asarray(self.actor.act(public), dtype=np.float32).copy()
        phase = _phase_from_actor(self.actor)
        if phase < 0 or phase >= len(self.scales):
            raise ValueError("public phase outside semantic checkpoint")
        scale = float(self.scales[phase]) if phase in self.active else 1.0
        action[:, self.coordinates] *= scale
        bounds = np.asarray(BOUNDS, dtype=np.float32)[list(self.coordinates)]
        action[:, self.coordinates] = np.clip(action[:, self.coordinates], -bounds, bounds)
        self.last_phase, self.last_scale = phase, scale
        return action

    def reset(self, **kwargs):
        self.last_phase, self.last_scale = None, 1.0
        return self.actor.reset(**kwargs)

    def get_state(self):
        return {"version": self.VERSION, "checkpoint": deepcopy(self.checkpoint),
                "last_phase": self.last_phase, "last_scale": self.last_scale,
                "actor": self.actor.get_state() if hasattr(self.actor, "get_state") else None}


__all__ = ["PublicSemanticPhaseTransport", "RESPONSE_KIND", "SELECTION_KIND",
           "VERSION", "select_learned_scale", "validate_checkpoint"]
