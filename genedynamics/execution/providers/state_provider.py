"""Base state provider."""

from abc import ABC, abstractmethod
from typing import Optional

from genedynamics.execution.core.contracts import HealthStatus, RobotState


class StateProviderBase(ABC):
    """Abstract base for state providers."""

    @abstractmethod
    def get_state(self) -> Optional[RobotState]:
        """Get current state. Returns None if unavailable."""
        pass

    @abstractmethod
    def get_timestamp(self) -> float:
        """Get last update timestamp."""
        pass

    @abstractmethod
    def health(self) -> HealthStatus:
        """Health status."""
        pass
