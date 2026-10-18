"""Shared public observation chart for transfer-task learning methods."""

import numpy as np


PORTABLE_FIELDS = (
    "tcp_position",
    "tcp_rotation",
    "tcp_twist",
    "measured_wrench",
    "body_position",
    "body_twist",
    "controller_feedback",
)
PUBLIC_CHART_VERSION = "portable-physical-state35-box-v3"
LEGACY_TO_PHYSICAL_COLUMNS = tuple(range(24)) + tuple(range(75, 86))


def portable_observation(public):
    """Flatten the public 35D physical chart without method-owned state."""
    batch = np.asarray(public["tcp_position"]).shape[0]
    return np.concatenate(
        [np.asarray(public[key]).reshape(batch, -1) for key in PORTABLE_FIELDS],
        axis=-1,
    ).astype(np.float32)


__all__ = [
    "LEGACY_TO_PHYSICAL_COLUMNS",
    "PORTABLE_FIELDS",
    "PUBLIC_CHART_VERSION",
    "portable_observation",
]

