"""Quadruped domain envs — registered into the environment registry on import.

Backend imports are guarded (brax raises at import on CPU-only installs; mjx
imports but cannot construct without GPU/mjx — both fine here). Names backed by a
failed import are simply absent, matching the prior lazy make_env behaviour.
"""

from genedynamics.core.registry.environments import (
    register_environment,
    register_environment_factory,
)

try:
    from . import physics as _phys
    register_environment("quadruped_flat_physics")(_phys.QuadrupedFlatPhysicsEnv)
    register_environment("quadruped_rough_physics")(_phys.QuadrupedRoughPhysicsEnv)
    register_environment("quadruped_push_physics")(_phys.QuadrupedPushPhysicsEnv)
    register_environment("quadruped_go2_physics")(_phys.QuadrupedGo2PhysicsEnv)
except Exception:
    pass

try:
    from . import mjx as _mjx
    register_environment("quadruped_flat_mjx")(_mjx.QuadrupedAntMjxEnv)
    register_environment("quadruped_go2_mjx")(_mjx.QuadrupedGo2MjxEnv)
except Exception:
    pass

# Brax: register the FACTORY from the (import-safe) brax_env wrapper, matching
# make_env's `quadruped_go2_brax -> make_brax_go2`. The brax env CLASS module is
# imported separately (guarded) and is unused by make_env.
try:
    from genedynamics.envs.brax_env import make_brax_go2
    register_environment_factory("quadruped_go2_brax", make_brax_go2)
except Exception:
    pass
try:
    from . import brax as _brax  # noqa: F401  (class module; crashes on CPU-only -> skipped)
except Exception:
    pass

try:
    from . import stepping_stones as _ss
    register_environment("quadruped_stepping_stones_2d")(_ss.QuadrupedSteppingStones2DEnv)
except Exception:
    pass

# DIAL Go2 brax PipelineEnv tasks (faithful port of dial-mpc). Factories bake in
# the per-task env config from dial-mpc's example yamls.
try:
    import jax.numpy as _jnp
    from . import go2_brax_dial as _g2

    def _make_go2_walk(**_):  # unitree_go2_trot.yaml: gait trot, vx 0.8, ramp 1.0
        return _g2.UnitreeGo2Env(
            _g2.UnitreeGo2EnvConfig(default_vx=0.8, ramp_up_time=1.0, gait="trot")
        )

    def _make_go2_seq_jump(**_):  # unitree_go2_seq_jump.yaml
        return _g2.UnitreeGo2SeqJumpEnv(
            _g2.UnitreeGo2SeqJumpEnvConfig(
                jump_dt=1.0,
                pose_target_sequence=_jnp.array(
                    [[0.0, 0.0, 0.27], [0.4, 0.0, 0.27], [0.8, 0.0, 0.27],
                     [1.2, 0.0, 0.27], [1.6, 0.0, 0.27]]
                ),
                yaw_target_sequence=_jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            )
        )

    def _make_go2_crate_climb(**_):  # unitree_go2_crate_climb.yaml (env defaults)
        return _g2.UnitreeGo2CrateEnv(_g2.UnitreeGo2CrateEnvConfig())

    register_environment_factory("quadruped_go2_walk", _make_go2_walk)
    register_environment_factory("quadruped_go2_seq_jump", _make_go2_seq_jump)
    register_environment_factory("quadruped_go2_crate_climb", _make_go2_crate_climb)
except Exception:
    pass
