"""
Task domain providers for baseline experiments.

Each domain (softzoo, 3dgs, etc.) provides task_spec resolution and
evaluator creation. Register here to make domains available.
"""

from __future__ import annotations

from .softzoo import SoftZooTaskDomainProvider
from .jax_mpm import JaxMpmTaskDomainProvider

from ...framework.task_domain_provider import register_task_domain_provider

# Auto-register domains
register_task_domain_provider(SoftZooTaskDomainProvider())
register_task_domain_provider(JaxMpmTaskDomainProvider())

__all__ = ["SoftZooTaskDomainProvider", "JaxMpmTaskDomainProvider"]
