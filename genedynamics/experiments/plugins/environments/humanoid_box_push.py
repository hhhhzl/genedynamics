"""Unified-runner environment plugin for the humanoid box-push task."""

from ._contact_task import HUMANOID_TASK, ContactTaskEnvironmentPlugin


class HumanoidBoxPushPlugin(ContactTaskEnvironmentPlugin):
    def __init__(self) -> None:
        super().__init__(HUMANOID_TASK)


__all__ = ["HumanoidBoxPushPlugin"]
