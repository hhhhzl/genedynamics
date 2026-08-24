"""Unified-runner environment plugin for the arm surface-scan task."""

from genedynamics.solvers.single.mdac.experiment import ARM_TASK

from ._contact_task import ContactTaskEnvironmentPlugin


class ManipulatorSurfaceScanPlugin(ContactTaskEnvironmentPlugin):
    def __init__(self) -> None:
        super().__init__(ARM_TASK)


__all__ = ["ManipulatorSurfaceScanPlugin"]
