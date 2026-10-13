"""Cross-embodiment frontend. Importing this package loads no tensor runtime."""

from .contracts import ENVIRONMENT_NAME, ObstacleTransferConfig, ExecutionSpec, MechanismSpec, RandomRequest, SkillSpec
from .cross_embodiment import CrossEmbodiment
from .source_query import SourceQueryAdapter, SourceQuerySpec, SourceQueryResult, SupportReason, TaskSemantics

__all__ = ["CrossEmbodiment", "ENVIRONMENT_NAME", "ObstacleTransferConfig", "ExecutionSpec", "MechanismSpec", "RandomRequest", "SkillSpec",
           "SourceQueryAdapter", "SourceQuerySpec", "SourceQueryResult", "SupportReason", "TaskSemantics"]
