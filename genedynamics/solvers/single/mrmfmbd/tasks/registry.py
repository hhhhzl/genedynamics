"""
Task Extension Registry: unified registration for co-design task domains.

Enables adding new domains without modifying core.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from .protocols import TaskDomain, TaskSpec

_REGISTRY: Dict[str, TaskDomain] = {}


def register_task_domain(
    domain: TaskDomain,
    *,
    name: Optional[str] = None,
) -> None:
    """
    Register a task domain.

    Args:
        domain: TaskDomain implementation
        name: Override domain name (default: domain.domain_name)
    """
    key = name if name is not None else domain.domain_name
    _REGISTRY[key] = domain


def get_task_spec(
    task_id: str,
    *,
    domain: Optional[str] = None,
) -> TaskSpec:
    """
    Get TaskSpec by task_id. If domain specified, use that domain only.
    Otherwise search all registered domains.

    Raises:
        KeyError: If task_id not found
    """
    if domain is not None:
        if domain not in _REGISTRY:
            raise KeyError(f"Domain '{domain}' not registered. Available: {list(_REGISTRY.keys())}")
        return _REGISTRY[domain].get_task_spec(task_id)

    for dom in _REGISTRY.values():
        if task_id in dom.list_tasks():
            return dom.get_task_spec(task_id)

    raise KeyError(
        f"Task '{task_id}' not found in any domain. "
        f"Domains: {list(_REGISTRY.keys())}"
    )


def list_task_ids(domain: Optional[str] = None) -> List[str]:
    """List task ids. If domain specified, only that domain."""
    if domain is not None:
        if domain not in _REGISTRY:
            return []
        return _REGISTRY[domain].list_tasks()
    seen: set = set()
    out: List[str] = []
    for dom in _REGISTRY.values():
        for tid in dom.list_tasks():
            if tid not in seen:
                seen.add(tid)
                out.append(tid)
    return sorted(out)


def list_domains() -> List[str]:
    """List registered domain names."""
    return list(_REGISTRY.keys())


def has_domain(domain: str) -> bool:
    """Check if domain is registered."""
    return domain in _REGISTRY


class TaskExtensionRegistry:
    """
    Fluent API for task domain registration.

    Usage:
        TaskExtensionRegistry().register(MyTaskAdapter()).freeze()
    """

    def __init__(self) -> None:
        self._pending: Dict[str, TaskDomain] = {}

    def register(self, domain: TaskDomain, *, name: Optional[str] = None) -> "TaskExtensionRegistry":
        key = name if name is not None else domain.domain_name
        self._pending[key] = domain
        return self

    def freeze(self) -> None:
        """Apply pending registrations."""
        for key, dom in self._pending.items():
            register_task_domain(dom, name=key)
        self._pending.clear()
