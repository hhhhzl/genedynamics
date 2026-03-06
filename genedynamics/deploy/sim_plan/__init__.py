"""
Sim/Plan separation via shared memory.

Sim process: runs MuJoCo physics, publishes state, consumes actions.
Plan process: reads state, runs planner, publishes actions.
"""

from genedynamics.deploy.sim_plan.shm_protocol import (
    ShmNames,
    create_sim_shm,
    create_plan_shm,
    attach_sim_shm,
    attach_plan_shm,
)

__all__ = [
    "ShmNames",
    "create_sim_shm",
    "create_plan_shm",
    "attach_sim_shm",
    "attach_plan_shm",
]
