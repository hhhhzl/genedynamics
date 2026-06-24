"""Humanoid domain envs — registered into the environment registry on import.

Backend imports are guarded (brax raises at import on CPU-only installs). Names
backed by a failed import are simply absent, matching the prior lazy make_env
behaviour. The corridor env reuses the shared locomotion energy and exposes the
non-class corridor helpers via its compat shim.
"""

from genedynamics.core.registry.environments import (
    register_environment,
    register_environment_factory,
)

try:
    from . import physics as _phys
    register_environment("humanoid_simplified_physics")(_phys.HumanoidBasePhysicsEnv)
    register_environment("humanoid_g1_physics")(_phys.HumanoidG1PhysicsEnv)
except Exception:
    pass

try:
    from . import mjx as _mjx
    register_environment("humanoid_simplified_mjx")(_mjx.HumanoidSimplifiedMjxEnv)
    register_environment("humanoid_g1_mjx")(_mjx.HumanoidG1MjxEnv)
    register_environment("humanoid_h1_mjx")(_mjx.HumanoidH1MjxEnv)
except Exception:
    pass

try:
    from genedynamics.envs.brax_env import make_brax_humanoid_run
    register_environment_factory("humanoid_run_brax", make_brax_humanoid_run)
except Exception:
    pass
try:
    from . import brax as _brax  # noqa: F401  (class module; crashes on CPU-only -> skipped)
except Exception:
    pass

try:
    from . import corridor as _cor
    register_environment("humanoid_corridor_2d")(_cor.HumanoidCorridor2DEnv)
except Exception:
    pass

# DIAL H1 brax PipelineEnv tasks (faithful port of dial-mpc; stable physics path
# rolled out by the DIAL backend via env.step). Factories bake in the per-task
# env config from dial-mpc's example yamls (gait/default_vx/ramp_up_time);
# kp/kd/dt come from the config dataclass defaults.
try:
    from . import h1_brax as _h1b

    def _make_h1_walk(**_):  # unitree_h1_jog.yaml: gait jog, vx 2.0, ramp 3.0
        return _h1b.UnitreeH1WalkEnv(
            _h1b.UnitreeH1WalkEnvConfig(default_vx=2.0, ramp_up_time=3.0, gait="jog")
        )

    def _make_h1_loco(**_):  # unitree_h1_loco.yaml: gait walk, vx 0.6, ramp 3.0
        return _h1b.UnitreeH1LocoEnv(
            _h1b.UnitreeH1LocoEnvConfig(default_vx=0.6, ramp_up_time=3.0, gait="walk")
        )

    def _make_h1_push_crate(**_):  # unitree_h1_push_crate.yaml: slow_walk, vx 0.8, ramp 2.0
        return _h1b.UnitreeH1PushCrateEnv(
            _h1b.UnitreeH1PushCrateEnvConfig(default_vx=0.8, ramp_up_time=2.0, gait="slow_walk")
        )

    register_environment_factory("humanoid_h1_walk", _make_h1_walk)
    register_environment_factory("humanoid_h1_loco", _make_h1_loco)
    register_environment_factory("humanoid_h1_push_crate", _make_h1_push_crate)
except Exception:
    pass

try:  # humanoid box pushing (contact manifold)
    from . import box_push_brax as _bp

    def _make_humanoid_box_push(**kw):
        return _bp.HumanoidBoxPushEnv(_bp.HumanoidBoxPushConfig(**kw))

    register_environment_factory("humanoid_box_push", _make_humanoid_box_push)
    # task-level aliases (descriptive names): double_support (fixed feet/face),
    # heavy_dr (+ domain randomization), unjam (box-unjamming + contact-face select).
    for _lv in ("double_support", "heavy_dr", "unjam"):
        register_environment_factory(
            f"humanoid_box_push_{_lv}",
            (lambda lv: lambda **kw: _make_humanoid_box_push(level=lv, **kw))(_lv))
except Exception:
    pass
