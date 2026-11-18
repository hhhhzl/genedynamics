"""Versioned public task and physical action contracts (SI, world axes).

These development contracts describe the existing CPU scenes. They do not
claim that the draft's proposed disturbance ranges or native baseline bodies
have been reproduced. Hidden MuJoCo data is stored separately from this view.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

CONTRACT_VERSION = "cross-embodiment-public-cpu-v2"
EVIDENCE_VERSION = "transfer-episode-evidence-v1"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    alias: str
    config: Any
    family: str
    body_scope: str
    physics: str = "MuJoCo CPU reference"
    public_frame: str = "world; wrench at TCP; linear xyz then angular xyz"
    skill_units: tuple = ("m/s",) * 3 + ("rad/s",) * 3 + ("log ratio",) * 6 + ("N",)
    controller: str = "shared skill impedance/hybrid force controller"
    mode_semantics: tuple = ("source route 0", "source route 1")
    qualification: str = "development interface; independent scene/body qualification required"

    def describe(self):
        value = asdict(self)
        value["contract_version"] = CONTRACT_VERSION
        value["observations"] = {
            "joint_position": "encoder q, rad (mobile base translation m if present)",
            "joint_velocity": "encoder qdot, rad/s (mobile base m/s if present)",
            "tcp_position": "world xyz, m",
            "tcp_rotation": "world-from-TCP 3x3 proper rotation",
            "tcp_twist": "world spatial velocity, m/s then rad/s",
            "measured_wrench": "on tool at TCP, world N then Nm",
            "goal_position": "known task goal, world m",
            "goal_rotation": "known task goal, world-from-TCP",
            "previous_action": "previous issued shared skill after public rate limit, BEFORE hidden actuator delay",
            "previous_request": "previous policy-issued shared skill",
            "reference_position": "public controller-issued position reference, m",
            "reference_rotation": "public controller-issued rotation reference",
            "elapsed_seconds": "time since task reset; preparation charged separately",
            "body_position": "measured world xyz, zero for a fixed base",
            "body_twist": "measured world linear/angular body velocity",
            "controller_feedback": "WBC residual and base tilt; zero for fixed base",
        }
        if self.name in {"aperture_brake", "sequential_aperture_reorientation"}:
            value["observations"].update({
                "carried_object_position": "known rigid payload-envelope center, world m",
                "carried_object_rotation": "world-from-payload-envelope frame",
                "carried_object_twist": "payload-envelope center spatial velocity",
                "aperture_geometry": "known plane, opening center/half-size, and exit interval",
            })
            if self.name == "sequential_aperture_reorientation":
                value["observations"]["aperture_geometry"] = (
                    "two known orthogonal planes, opening centers/half-sizes, "
                    "reorientation chamber, and final exit interval"
                )
        if self.name == "reposition_then_dock":
            value["observations"].update({
                "body_planar_pose": "measured world x/y/yaw of the base",
                "body_target": "public task-directed world x/y/yaw target",
                "body_target_bounds": "public fixed-foot target-request bounds",
            })
        value["excluded_policy_fields"] = [
            "physics", "mechanism", "gain", "delay_steps", "payload_mass", "friction",
            "command_buffer", "plant_applied_command", "plant_delayed_reference", "qacc_warmstart", "internal_contact_whitelist", "axial_support_truth",
        ]
        value["reward_scope"] = "development shaping; not a frozen paper benchmark cost"
        return value


@dataclass
class EpisodeRecord:
    sample_id: int
    seed: int
    generation: int
    initial: str
    requested_actions: list = field(default_factory=list)
    traces: list = field(default_factory=list)
    state: str = "active"
    rejection_reason: str | None = None
    controller: str = "shared_skill"
    preparation_control_steps: int = 0
    preparation_traces: list = field(default_factory=list)
    requested_mode: int = 0
    control_diagnostics: list = field(default_factory=list)
    reset_evidence: dict = field(default_factory=dict)
    incomplete_transitions: list = field(default_factory=list)


@dataclass(frozen=True)
class TaskStep:
    observation: dict
    reward: np.ndarray
    done: np.ndarray
    status: np.ndarray
    executed_action: np.ndarray
    trace: dict


def numpy_action(value):
    """Make the host/device boundary explicit without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def controller_diagnostics(actor, *, _depth=0):
    """Copy explicit public controller memory, never learned weights or plant state.

    Read only owned attributes: wrappers with __getattr__ must not duplicate
    candidate counters as common-controller measurements. Values describe the
    command just computed, before the following physical transition.
    """
    if actor is None or _depth > 4:
        return {}
    state = vars(actor)
    result = {"implementation": f"{type(actor).__module__}.{type(actor).__qualname__}"}
    for key in (
        "stage", "tick", "last_tick", "retreating", "clear_count", "guard_tick",
        "compensation", "compensation_tick", "safe_count", "release_count",
        "reference", "reference_tick", "lateral_complete", "contact_seen",
        "maximum_precontact_z", "enabled", "common_controls", "learned_controls",
        "fallback_controls", "last_query_distance", "irreversible_network_calls",
        "irreversible_model_calls", "irreversible_candidate_rollouts",
        "irreversible_action_updates", "hold_count", "contact_count", "contact_ticks",
        "first_contact_tick", "first_outlet_tick", "axis_endpoint_reached",
        "projection_active_ticks", "projection_infeasible_ticks",
        "stage_entry_tick",
        "filtered_normal_force",
        "risk_fallback_controls", "irreversible_risk_model_calls",
        "risk_emergency_controls", "risk_empty_controls",
        "last_risk_fraction", "last_risk_index", "last_risk_upper_bounds_N",
        "last_risk_upper_bounds",
        "risk_fraction_counts",
        "contact_guard_active", "contact_guard_recovery_ticks",
        "contact_guard_controls", "contact_guard_activations",
        "contact_establishment_applicable", "contact_establishment_admitted",
        "contact_establishment_projected_controls",
        "contact_establishment_transport_controls",
        "contact_establishment_tick",
        "achieved_speed_mps", "required_speed_mps", "lead_limit_m",
        "last_selected_speed_mps", "last_force_utilization",
        "admission_controls", "progress_empty_controls",
        "authority", "tracking_lag_m", "force_utilization",
        "authority_controls", "progress_deficit_controls",
        "terminal_pose_active", "terminal_distance_m",
        "terminal_orientation_error_rad",
        "terminal_pose_hold_controls", "terminal_capture_controls",
        "terminal_reference_overshoot_m",
        "chunk_waypoint", "chunk_control_step", "execution_horizon",
        "action_down_sample_steps", "execution_control_steps",
        "clearance_reference_settled", "clearance_settling_controls",
        "clearance_lift_controls",
        "clearance_capture_latched", "clearance_capture_band_m",
        "terminal_planar_distance_m", "terminal_planar_speed_cap_active",
        "pre_descent_settling_active", "pre_descent_settling_complete",
        "pre_descent_settling_controls", "pre_descent_settled_controls",
        "pre_descent_reference_lead_norm_m",
        "pre_descent_planar_speed_mps",
        "pre_descent_reference_unloading_active",
        "pre_descent_reference_unloading_speed_mps",
        "public_reference_delay_estimate_controls",
        "public_reference_delay_identification_samples",
        "public_reference_delay_score_gap_mps",
        "pre_descent_causal_unloading_limit_mps",
        "normal_force_channel_controls",
        "last_common_action", "last_candidate_action", "last_selected_action",
        "last_selector_code",
        "last_completion_blend", "last_completion_code", "last_terminal_near",
        "task_dominance_blend_counts", "transport_admission_controls",
        "task_only_fallback_controls", "last_task_dominance_blend",
        "last_transport_task_proxy", "last_task_only_task_proxy",
        "last_linear_progress_gain", "last_angular_progress_gain",
        "last_normalized_progress_gain",
        "last_worst_member_task_proxy_gain",
        "horizon_commitment_remaining", "horizon_commitment_controls",
        "horizon_commitment_aborts",
        "completion_admission_controls", "completion_common_fallback_controls",
        "completion_terminal_controls",
        "completion_blend_counts",
        "terminal_ownership_active", "terminal_ownership_controls",
        "terminal_ownership_activations",
        "candidate_authority_fraction",
        "terminal_admission_controls", "terminal_common_fallback_controls",
        "terminal_set_controls", "terminal_blend_counts",
        "terminal_transport_controls", "terminal_force_fallback_controls",
        "handoff_authority_cap", "handoff_deadline_scale",
        "handoff_reference_lag_scale", "handoff_progress_scale",
        "handoff_clearance_debt_scale", "handoff_clearance_lift_controls",
        "handoff_completion_budget_scale", "handoff_force_speed_scale",
        "handoff_completion_budget_controls", "handoff_planar_remaining_m",
        "handoff_progress_speed_mps",
        "handoff_remaining_controls", "handoff_reference_lag_m",
        "handoff_viability_controls", "handoff_exact_common_controls",
    ):
        value = state.get(key)
        if value is not None and isinstance(value, (bool, int, float, list, tuple, np.ndarray, np.generic)):
            array = np.asarray(value)
            if array.size <= 128 and array.dtype.kind in "biuf":
                result[key] = array.copy()
    memory = state.get("memory")
    if memory is not None and hasattr(memory, "_asdict"):
        result["memory"] = {key: np.asarray(value).copy() for key, value in memory._asdict().items()}
    for key in (
        "base", "candidate", "route", "completion", "common", "gate",
        "observer",
    ):
        child = state.get(key)
        if child is not None and hasattr(child, "__dict__"):
            result[key] = controller_diagnostics(child, _depth=_depth + 1)
    return result


def _evidence_encode(value):
    """Portable JSON including nonfinite failure evidence, without JSON NaN."""
    if isinstance(value, np.ndarray):
        return _evidence_encode(value.tolist())
    if isinstance(value, np.generic):
        return _evidence_encode(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"__nonfinite_float__": str(value)}
    if isinstance(value, dict):
        return {key: _evidence_encode(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_evidence_encode(child) for child in value]
    return value


def episode_evidence(records, spec, evaluation):
    """Task-owned durable evidence, suitable for a generic compressed artifact."""
    import hashlib
    from pathlib import Path

    package = Path(__file__).resolve().parent
    return _evidence_encode({
        "schema_version": EVIDENCE_VERSION,
        "contract_version": spec.describe()["contract_version"],
        "task_spec": asdict(spec),
        "evaluation": evaluation,
        "evaluator_sources": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in _evaluator_source_names(spec)
        },
        "records": [asdict(record) for record in records],
        "scope": "physical and controller evidence for CPU reevaluation; not policy observations",
    })


def _evaluator_source_names(spec):
    names = ("episode.py", "evaluation.py", "guide_evaluation.py",
             "routes.py", "contact_reference.py")
    if spec.name == "compliant_keyed_insertion_recovery":
        names += ("compliant_socket.py",)
    if spec.name in {"aperture_brake", "sequential_aperture_reorientation"}:
        names += ("aperture.py",)
    if spec.name == "reposition_then_dock":
        names += ("reposition_dock.py",)
    return names


def load_episode_evidence(path):
    """Read recorded episodes and their exact task config without a simulator."""
    import gzip
    import json
    from pathlib import Path
    from .contact import ContactConfig
    from .runtime import ObstacleTransferConfig

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"__nonfinite_float__"}:
                return float(value["__nonfinite_float__"])
            return {key: decode(child) for key, child in value.items()}
        if isinstance(value, list):
            return [decode(child) for child in value]
        return value

    path = Path(path)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    payload = decode(json.loads(raw))
    if payload.get("schema_version") != EVIDENCE_VERSION:
        raise ValueError("unsupported episode evidence schema")
    spec = dict(payload["task_spec"])
    spec_type = TaskSpec
    expected_contract = CONTRACT_VERSION
    config_type = ObstacleTransferConfig if spec["family"] == "obstacle" else ContactConfig
    if spec["name"] == "compliant_keyed_insertion_recovery":
        from .compliant_socket import CompliantSocketConfig, CompliantSocketTaskSpec
        from .compliant_socket import CONTRACT_VERSION as expected_contract
        config_type, spec_type = CompliantSocketConfig, CompliantSocketTaskSpec
    if payload.get("contract_version") != expected_contract:
        raise ValueError("episode evidence contract version differs")
    spec["config"] = config_type(**spec["config"])
    if any("requested_mode" not in row for row in payload["records"]):
        raise ValueError("record has no requested mode evidence")
    records = [EpisodeRecord(**row) for row in payload["records"]]
    for record in records:
        if record.requested_mode not in (0, 1):
            raise ValueError("record has no valid requested mode")
        if len(record.requested_actions) != len(record.traces):
            raise ValueError("recorded action/physical trace count differs")
        if len(record.control_diagnostics) not in (0, len(record.traces)):
            raise ValueError("recorded controller/physical trace count differs")
    return records, spec_type(**spec), payload["evaluation"]


def audit_episode_evidence(path):
    """Reevaluate durable evidence on CPU, without resetting a physical scene."""
    import gzip
    import hashlib
    import json
    from pathlib import Path
    from .evaluation import evaluate_records

    path = Path(path)
    records, spec, recorded = load_episode_evidence(path)
    recomputed = evaluate_records(records, spec)
    content = path.read_bytes()
    payload = json.loads(gzip.decompress(content) if path.suffix == ".gz" else content)
    package = Path(__file__).resolve().parent
    expected_sources = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in _evaluator_source_names(spec)
    }
    recorded_scores = {key: value for key, value in recorded.items()
                       if key not in {"accounting", "lifetime_accounting"}}
    matching = json.dumps(_evidence_encode(recorded_scores), sort_keys=True, allow_nan=False) == json.dumps(
        _evidence_encode(recomputed), sort_keys=True, allow_nan=False,
    )
    sources_match = payload.get("evaluator_sources") == expected_sources
    accounting = recorded.get("accounting")
    physical_controls = sum(
        sum(bool(row["valid"]) for row in record.traces)
        + sum(bool(row["physical_trace"]["valid"]) for row in record.incomplete_transitions)
        for record in records
    )
    accounting_matches = accounting is None or (
        accounting.get("control_steps") == physical_controls
        and accounting.get("preparation_control_steps") == sum(record.preparation_control_steps for record in records)
    )
    return {
        "path": str(path), "sha256": hashlib.sha256(content).hexdigest(),
        "passed": bool(matching and sources_match and accounting_matches),
        "recorded_evaluation_matches": matching,
        "evaluator_sources_match": sources_match,
        "recorded_accounting_matches": accounting_matches,
        "physical_calls": 0, "gpu_calls": 0,
        "evaluation": _evidence_encode(recomputed),
        "reset_physical_hashes": [record.reset_evidence.get("physical_state_sha256") for record in records],
    }
