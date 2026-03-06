"""Base control publisher."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class ControlPublisherBase(ABC):
    """Abstract base for control publishers."""

    @abstractmethod
    def publish(self, action: np.ndarray, mode: str, ttl_ms: float = 0.0) -> None:
        """Publish control action."""
        pass

    @abstractmethod
    def set_safe_posture(self, posture: np.ndarray) -> None:
        """Set safe posture (hold, stand)."""
        pass

    @abstractmethod
    def emergency_stop(self) -> None:
        """Emergency stop."""
        pass

    def get_last_action(self) -> Optional[np.ndarray]:
        """Return last published action (for logging)."""
        return None
