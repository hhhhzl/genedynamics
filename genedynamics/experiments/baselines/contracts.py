"""Versioned public task and physical action contracts (SI, world axes).

These development contracts describe the existing CPU scenes. They do not
claim that the draft's proposed disturbance ranges or native baseline bodies
have been reproduced. Hidden MuJoCo data is stored separately from this view.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

CONTRACT_VERSION = "cross-embodiment-public-cpu-v1"


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
