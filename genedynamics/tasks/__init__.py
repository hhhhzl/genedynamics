"""Task specifications with lazy numerical dependencies."""

__all__ = ["UAV3DTaskSpec", "SteppingStonesScene"]


def __getattr__(name):
    if name == "UAV3DTaskSpec":
        from genedynamics.tasks.uav3d.spec import UAV3DTaskSpec

        return UAV3DTaskSpec
    if name == "SteppingStonesScene":
        from genedynamics.tasks.stepping_stones import SteppingStonesScene

        return SteppingStonesScene
    raise AttributeError(name)
