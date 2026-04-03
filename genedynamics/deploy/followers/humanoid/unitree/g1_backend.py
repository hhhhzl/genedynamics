"""
Unitree G1 backend scaffold for mixed loco + upper-body control.
"""

from __future__ import annotations

from typing import Dict

UNITREE_G1_SDK_AVAILABLE = False
try:
    import unitree_sdk2py  # type: ignore  # noqa: F401

    UNITREE_G1_SDK_AVAILABLE = True
except ImportError:
    pass

from genedynamics.deploy.followers.common.controller_output import LocoCommand


class UnitreeG1Backend:
    """
    Thin facade for the future G1 real backend.

    We keep the methods narrow so the higher-level adapter can stabilize before
    we bind to concrete SDK topics or ROS bridges.
    """

    def __init__(self, network_interface: str = "") -> None:
        self.network_interface = network_interface

    def send_loco_command(self, command: LocoCommand) -> None:
        _ = command
        raise NotImplementedError(
            "UnitreeG1Backend.send_loco_command is not wired yet. "
            "Bind this facade to the chosen G1 SDK/ROS control path."
        )

    def send_upper_body_joint_positions(self, joint_targets: Dict[str, float]) -> None:
        _ = joint_targets
        raise NotImplementedError(
            "UnitreeG1Backend.send_upper_body_joint_positions is not wired yet."
        )
