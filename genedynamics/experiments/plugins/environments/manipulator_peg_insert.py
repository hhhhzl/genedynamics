"""Unified-runner environment plugin for the arm peg-insert task."""

from ._contact_task import INSERT_TASK, ContactTaskEnvironmentPlugin


class ManipulatorPegInsertPlugin(ContactTaskEnvironmentPlugin):
    def __init__(self) -> None:
        super().__init__(INSERT_TASK)


__all__ = ["ManipulatorPegInsertPlugin"]
