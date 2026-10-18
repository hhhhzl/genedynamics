"""Task-owned demonstrators used to create transfer training data."""

from .acp import NativeACPTeacher, RateLimitedACPTeacher

__all__ = ["NativeACPTeacher", "RateLimitedACPTeacher"]
