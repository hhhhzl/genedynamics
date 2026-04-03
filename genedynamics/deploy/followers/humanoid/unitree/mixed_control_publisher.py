"""
Publisher for humanoid mixed commands on the real G1 path.
"""

from __future__ import annotations

from typing import Optional

from genedynamics.deploy.followers.common.controller_output import MixedHumanoidCommand
from genedynamics.deploy.followers.humanoid.unitree.g1_backend import UnitreeG1Backend


class MixedHumanoidControlPublisher:
    def __init__(self, backend: Optional[UnitreeG1Backend] = None) -> None:
        self.backend = backend
        self._last_command: Optional[MixedHumanoidCommand] = None

    def publish(self, command: MixedHumanoidCommand) -> None:
        self._last_command = command
        if self.backend is None:
            return
        if command.lower_body is not None:
            self.backend.send_loco_command(command.lower_body)
        if command.upper_body_joint_targets:
            self.backend.send_upper_body_joint_positions(command.upper_body_joint_targets)

    def get_last_command(self) -> Optional[MixedHumanoidCommand]:
        return self._last_command
