"""quadruped_stepping — batch stepping-walk controller for Go2.

Primary API::

    from genedynamics.deploy.controllers.quadruped_stepping import (
        QuadrupedSteppingController,
        MinimalFollowerConfig,
        load_stepping_plan_from_seed_dir,
    )

    cfg = MinimalFollowerConfig(gait="walk", sim_dt=0.01, phase_steps=20)
    ctrl = QuadrupedSteppingController(cfg=cfg, stepping_scene=scene_dict)
    plan = load_stepping_plan_from_seed_dir(seed_dir, step_width=0.30)
    result = ctrl.rollout(plan["states"])
    # result["qpos"], result["qvel"], result["ctrl"] are (T, n) arrays
"""

from genedynamics.deploy.controllers.quadruped_stepping.controller import (
    QuadrupedSteppingController,
)
from genedynamics.deploy.controllers.quadruped_stepping.stepping_walker import (
    LEG_ORDER,
    MinimalFollowerConfig,
    PhaseState,
    SteppingWalkFollowerMinimal,
    load_stepping_plan_from_seed_dir,
)

__all__ = [
    "QuadrupedSteppingController",
    "MinimalFollowerConfig",
    "PhaseState",
    "SteppingWalkFollowerMinimal",
    "LEG_ORDER",
    "load_stepping_plan_from_seed_dir",
]
