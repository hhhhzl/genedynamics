"""Unified-runner environment plugin for the humanoid box-push task."""

from genedynamics.solvers.single.mdac.experiment import HUMANOID_TASK

from ._contact_task import ContactTaskEnvironmentPlugin


class HumanoidBoxPushPlugin(ContactTaskEnvironmentPlugin):
    def __init__(self) -> None:
        super().__init__(HUMANOID_TASK)


__all__ = ["HumanoidBoxPushPlugin"]
