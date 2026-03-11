"""
Task domain provider: bridges task domains to evaluator creation.

Task-agnostic: any domain (softzoo, 3dgs, quadruped, etc.) implements
this protocol to provide task_spec + evaluator for the baseline platform.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class TaskDomainProvider(Protocol):
    """
    Protocol for task domain providers.

    Implement to add new task domains. The platform uses this to resolve
    task_spec and create evaluator for any baseline.
    """

    @property
    def domain_name(self) -> str:
        """Domain identifier (e.g. 'softzoo', '3dgs')."""
        ...

    def get_task_spec(self, task_id: str) -> Any:
        """Resolve task_id to TaskSpec (or domain-specific spec)."""
        ...

    def create_evaluator(
        self,
        project_root: str,
        *,
        max_workers: int = 0,
        cache_size: int = 64,
        **kwargs: Any,
    ) -> Any:
        """Create evaluator for this domain."""
        ...

    def list_tasks(self) -> List[str]:
        """List available task ids in this domain."""
        ...


_REGISTRY: Dict[str, TaskDomainProvider] = {}


def register_task_domain_provider(
    provider: TaskDomainProvider,
    *,
    name: Optional[str] = None,
) -> None:
    """Register a task domain provider."""
    key = name if name is not None else provider.domain_name
    _REGISTRY[key] = provider


def get_task_domain_provider(domain: str) -> TaskDomainProvider:
    """Get provider by domain name. Raises KeyError if not found."""
    if domain not in _REGISTRY:
        raise KeyError(
            f"Task domain '{domain}' not found. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[domain]


def list_task_domains() -> List[str]:
    """List registered task domain names."""
    return list(_REGISTRY.keys())


def has_task_domain(domain: str) -> bool:
    """Check if domain is registered."""
    return domain in _REGISTRY
