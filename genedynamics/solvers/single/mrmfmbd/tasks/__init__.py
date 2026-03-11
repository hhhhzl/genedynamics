"""
MRMFMBD Task Extension Layer.

Unified task registry and factory for co-design tasks.
Extensible: add new task domains without modifying core.
"""

from __future__ import annotations

from .protocols import TaskSpec, TaskDomain
from .registry import (
    TaskExtensionRegistry,
    register_task_domain,
    get_task_spec,
    list_task_ids,
    list_domains,
)
from .softzoo_adapter import SoftZooTaskAdapter

# Auto-register SoftZoo domain when available
try:
    register_task_domain(SoftZooTaskAdapter())
except Exception:
    pass

__all__ = [
    "TaskSpec",
    "TaskDomain",
    "TaskExtensionRegistry",
    "register_task_domain",
    "get_task_spec",
    "list_task_ids",
    "list_domains",
    "SoftZooTaskAdapter",
]
