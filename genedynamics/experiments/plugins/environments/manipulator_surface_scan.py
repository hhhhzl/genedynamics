"""Unified-runner environment plugin for the arm surface-scan task."""

from ._contact_task import ARM_TASK, ContactTaskEnvironmentPlugin


class ManipulatorSurfaceScanPlugin(ContactTaskEnvironmentPlugin):
    def __init__(self) -> None:
        super().__init__(ARM_TASK)


__all__ = ["ManipulatorSurfaceScanPlugin"]
