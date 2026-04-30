"""
Task domain providers for baseline experiments.

Each domain provides task_spec resolution and evaluator creation.
Register here to make domains available.
"""

from __future__ import annotations

from .jax_mpm import JaxMpmTaskDomainProvider

from ...framework.task_domain_provider import register_task_domain_provider

register_task_domain_provider(JaxMpmTaskDomainProvider())

__all__ = ["JaxMpmTaskDomainProvider"]
