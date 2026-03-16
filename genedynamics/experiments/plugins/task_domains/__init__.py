"""
Task domain providers for baseline experiments.

Each domain (softzoo, 3dgs, etc.) provides task_spec resolution and
evaluator creation. Register here to make domains available.
"""

from __future__ import annotations

from .softzoo import SoftZooTaskDomainProvider

from ...framework.task_domain_provider import register_task_domain_provider

# Auto-register SoftZoo
register_task_domain_provider(SoftZooTaskDomainProvider())

__all__ = ["SoftZooTaskDomainProvider"]
