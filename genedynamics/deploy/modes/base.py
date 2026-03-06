"""
Execution mode protocol.

Modes build state provider, control publisher, executor and run episodes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from genedynamics.deploy.config import DeployConfig
from genedynamics.deploy.profiles.base import RobotProfile


class ExecutionModeProtocol(Protocol):
    """
    Protocol for execution modes.

    Each mode (sim, real, shadow, replay) implements run() to execute
    the deploy pipeline for its context.
    """

    def run(
        self,
        config: DeployConfig,
        profile: RobotProfile,
        env: Any,
        planner: Any,
    ) -> Dict[str, Any]:
        """
        Run deploy for this mode.

        Args:
            config: Deploy configuration
            profile: Robot profile (for spec, etc.)
            env: Environment instance
            planner: Planner instance

        Returns:
            Dict with keys: results, steps, episode_paths, etc.
        """
        ...
