"""Canonical semantic identities for transfer tasks.

``T1``--``T4`` are accepted only at artifact/config reader boundaries.  New
runtime dispatch and newly written metadata use the semantic names below.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar


CANONICAL_TASKS = (
    "obstacle_transfer",
    "guided_contact_alignment",
    "keyed_insertion_recovery",
    "mobile_keyed_insertion_recovery",
)

LEGACY_TO_CANONICAL = {
    "T1": "obstacle_transfer",
    "T2": "guided_contact_alignment",
    "T3": "keyed_insertion_recovery",
    "T4": "mobile_keyed_insertion_recovery",
}
CANONICAL_TO_LEGACY = {value: key for key, value in LEGACY_TO_CANONICAL.items()}
LEGACY_TASKS = tuple(LEGACY_TO_CANONICAL)

_Value = TypeVar("_Value")


def canonical_task_id(value: str) -> str:
    """Return the runtime task identity, accepting a legacy artifact alias."""
    if value in CANONICAL_TASKS:
        return value
    try:
        return LEGACY_TO_CANONICAL[value]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown transfer task identity: {value!r}") from error


def legacy_task_id(value: str) -> str:
    """Return the historical alias needed to read frozen T1--T4 artifacts."""
    canonical = canonical_task_id(value)
    return CANONICAL_TO_LEGACY[canonical]


def canonicalize_task_mapping(values: Mapping[str, _Value]) -> dict[str, _Value]:
    """Normalize task-keyed reader input and reject alias/name collisions."""
    normalized: dict[str, _Value] = {}
    for key, value in values.items():
        canonical = canonical_task_id(key)
        if canonical in normalized:
            raise ValueError(
                f"task mapping declares both a semantic name and legacy alias for {canonical}"
            )
        normalized[canonical] = value
    return normalized

