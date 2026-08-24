"""Unified-runner environment plugin for the arm peg-insert task."""

from genedynamics.solvers.single.mdac.experiment import INSERT_TASK

from ._contact_task import ContactTaskEnvironmentPlugin


class ManipulatorPegInsertPlugin(ContactTaskEnvironmentPlugin):
    def __init__(self) -> None:
        super().__init__(INSERT_TASK)


__all__ = ["ManipulatorPegInsertPlugin"]
