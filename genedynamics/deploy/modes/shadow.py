"""
Shadow mode: sim with full telemetry for sim2real preparation.
"""

from __future__ import annotations

from genedynamics.deploy.modes.sim import SimMode


class ShadowMode(SimMode):
    """
    Shadow mode: same as sim but with execution_mode=SHADOW and tags.
    Used for sim2real pipeline preparation.
    """

    def run(self, config, profile, env, planner):
        """Run shadow episodes (sim with shadow tags)."""
        config = _shadow_config(config)
        result = super().run(config, profile, env, planner)
        result["mode"] = "shadow"
        return result


def _shadow_config(config):
    """Create config copy with shadow settings."""
    from genedynamics.deploy.config import DeployConfig
    data = config.to_dict()
    data["tags"] = dict(config.tags)
    data["tags"]["mode"] = "shadow"
    data["record"] = True
    return DeployConfig.from_dict(data)
