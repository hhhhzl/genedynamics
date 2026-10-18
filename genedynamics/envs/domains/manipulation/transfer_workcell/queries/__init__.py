"""Source-state query adapters used by Cross-EBDM inference."""

from .source import (
    SourceQueryAdapter,
    SourceQueryResult,
    SourceQuerySpec,
    SupportReason,
    TaskSemantics,
)

__all__ = [
    "SourceQueryAdapter",
    "SourceQueryResult",
    "SourceQuerySpec",
    "SupportReason",
    "TaskSemantics",
]
