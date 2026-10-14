"""Task-owned interfaces for explicitly adapted research baselines.

Native paper reproduction and task adaptation have separate qualifications.
No simulator parameter, checkpoint or internal contact label is a policy input.
"""

from .contact_task import CpuTaskAdapter, instantiate_task
from .task_specs import task_spec

__all__ = ["CpuTaskAdapter", "instantiate_task", "task_spec"]
