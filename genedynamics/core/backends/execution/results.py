"""Keep pre-reset transition data separate from the next collector carry."""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ResetResult:
    observation: Any
    carry: Any
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionStep:
    transition_observation: Any
    transition_state: Any
    next_observation: Any
    next_carry: Any
    reward: Any
    terminated: Any
    truncated: Any
    info: Mapping[str, Any] = field(default_factory=dict)
