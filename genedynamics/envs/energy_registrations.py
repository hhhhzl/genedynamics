"""
Energy functional registrations for environments.

This module registers energy functionals for all environments to the energy registry.
This centralizes energy registration and allows users to extend with custom energy functions.
"""

import numpy as np

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    JAX_AVAILABLE = False

try:
    from genedynamics.core.registry.energy import register_energy
    from genedynamics.core.energy import LegacyEnergyFunctional, EnergyTerm
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    register_energy = None


def _is_jax_array(x):
    """Check if x is a JAX array. Optimized for NumPy backend."""
    if not JAX_AVAILABLE:
        return False
    # Fast path: if it's a NumPy array, it's definitely not JAX
    if isinstance(x, np.ndarray):
        return False
    # Check for JAX array attributes (only for non-NumPy arrays)
    return hasattr(x, '__jax_array__') or (
        hasattr(x, 'dtype') and 
        hasattr(x, 'shape') and
        type(x).__module__ and 'jax' in type(x).__module__
    )


if REGISTRY_AVAILABLE:
    
    def _make_double_integrator_box_energy():
        """Energy functional for double_integrator_box environment."""
        def task_energy(x, u, ctx):
            pos = x[0]
            vel = x[1]
            # Auto-detect backend: use NumPy if input is NumPy, JAX if input is JAX
            if _is_jax_array(x):
                return (pos - 0.0) ** 2 + (vel ** 2)
            else:
                return float((pos - 0.0) ** 2 + (vel ** 2))
        
        def box_energy(x, u, ctx):
            p_max, v_max = 2.0, 2.0
            # Auto-detect backend
            if _is_jax_array(x):
                pos_violate = jnp.maximum(0.0, jnp.abs(x[0]) - p_max)
                vel_violate = jnp.maximum(0.0, jnp.abs(x[1]) - v_max)
                pen = pos_violate ** 2 + vel_violate ** 2
                return pen
            else:
                pos_violate = np.maximum(0.0, np.abs(x[0]) - p_max)
                vel_violate = np.maximum(0.0, np.abs(x[1]) - v_max)
                pen = pos_violate ** 2 + vel_violate ** 2
                return float(pen)
        
        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 1.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    
    register_energy("double_integrator_box", _make_double_integrator_box_energy)
    
    
    def _make_double_integrator_box_2d_energy():
        """Energy functional for double_integrator_box_2d environment."""
        def task_energy(x, u, ctx):
            pos = x[:2]
            vel = x[2:]
            # Auto-detect backend: use NumPy if input is NumPy, JAX if input is JAX
            if _is_jax_array(x):
                return jnp.sum((pos - jnp.array([0.0, 0.0], dtype=jnp.float32)) ** 2) + jnp.sum(vel ** 2)
            else:
                target = np.array([0.0, 0.0], dtype=np.float32)
                return float(np.sum((pos - target) ** 2) + np.sum(vel ** 2))
        
        def box_energy(x, u, ctx):
            p_max, v_max = 2.0, 2.0
            # Auto-detect backend
            if _is_jax_array(x):
                pos_violate = jnp.maximum(0.0, jnp.abs(x[:2]) - p_max)
                vel_violate = jnp.maximum(0.0, jnp.abs(x[2:]) - v_max)
                pen = jnp.sum(pos_violate ** 2 + vel_violate ** 2)
                return pen
            else:
                pos_violate = np.maximum(0.0, np.abs(x[:2]) - p_max)
                vel_violate = np.maximum(0.0, np.abs(x[2:]) - v_max)
                pen = np.sum(pos_violate ** 2 + vel_violate ** 2)
                return float(pen)
        
        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 2.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    
    register_energy("double_integrator_box_2d", _make_double_integrator_box_2d_energy)
    
    
    def _make_single_integrator_box_2d_energy():
        """Energy functional for single_integrator_box_2d environment."""
        def task_energy(x, u, ctx):
            pos = x[:2]
            # Auto-detect backend: use NumPy if input is NumPy, JAX if input is JAX
            if _is_jax_array(x):
                return jnp.sum((pos - jnp.array([0.0, 0.0], dtype=jnp.float32)) ** 2)
            else:
                target = np.array([0.0, 0.0], dtype=np.float32)
                return float(np.sum((pos - target) ** 2))
        
        def box_energy(x, u, ctx):
            p_max = 2.0
            # Auto-detect backend
            if _is_jax_array(x):
                pos_violate = jnp.maximum(0.0, jnp.abs(x[:2]) - p_max)
                pen = jnp.sum(pos_violate ** 2)
                return pen
            else:
                pos_violate = np.maximum(0.0, np.abs(x[:2]) - p_max)
                pen = np.sum(pos_violate ** 2)
                return float(pen)
        
        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 1.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    
    register_energy("single_integrator_box_2d", _make_single_integrator_box_2d_energy)

    def _make_go2_trot_energy():
        """DIAL-MPC go2 trot energy = NEGATED dial-mpc trot reward (a cost).

        Go2 MJX state x = [qpos(19); qvel(18)]: base xyz x[0:3], base quat
        (w,x,y,z) x[3:7], base lin vel (world) x[19:22], base ang vel x[22:25].
        DIAL maximises reward = -energy, so this returns the (>=0) weighted
        tracking cost with the dial-mpc trot weights (vel 1.0, ang_vel 1.0,
        upright 0.5, yaw 0.3, height 1.0). FK-free terms only; the gait z_feet
        term (weight 0.1, needs foot-site forward kinematics) is omitted in this
        self-contained variant and is a planned follow-up for exact parity.
        """
        VX_TAR, Z_NOM = 0.8, 0.3  # go2_trot defaults (default_vx, pos_tar z)

        def task_energy(x, u, ctx):
            jx = _is_jax_array(x)
            B = jnp if jx else np
            z = x[2]
            qw, qx, qy, qz = x[3], x[4], x[5], x[6]
            vwx, vwy = x[19], x[20]
            wz = x[24]  # yaw rate (world ang z)
            yaw = B.arctan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz))
            c, s = B.cos(yaw), B.sin(yaw)
            vbx = c * vwx + s * vwy          # body-frame forward vel
            vby = -s * vwx + c * vwy         # body-frame lateral vel
            cost = (
                1.0 * ((vbx - VX_TAR) ** 2 + vby ** 2)   # forward/lateral vel track
                + 1.0 * (wz ** 2)                        # yaw-rate track (->0)
                + 0.5 * (qx * qx + qy * qy)              # upright (roll/pitch tilt)
                + 0.3 * (yaw ** 2)                       # heading track (->0)
                + 1.0 * ((z - Z_NOM) ** 2)               # height track
            )
            return cost if jx else float(cost)

        return LegacyEnergyFunctional({"task": EnergyTerm(task_energy, 1.0)})

    register_energy("quadruped_go2_mjx_trot", _make_go2_trot_energy)
    register_energy("quadruped_go2_dial", _make_go2_trot_energy)

    # P6: register the legacy (non-toy) energies through the relocated
    # _make_energy_fallback so make_energy dispatches registry-only. The energy
    # bodies are unchanged (relocated verbatim from factories.make_energy). Names
    # NOT listed here (e.g. quadruped_rough/push/go2_physics, humanoid_g1_physics)
    # had no energy in the fallback and remain intentionally unregistered.
    def _legacy_energy_factory(name):
        def _factory():
            from genedynamics.envs.factories import _make_energy_fallback
            return _make_energy_fallback(name)
        return _factory

    for _legacy_name in (
        "drone_box_3d", "drone_full_3d", "drone_full_3d_physics",
        "quadruped_flat_physics", "quadruped_flat_mjx", "quadruped_go2_mjx",
        "quadruped_go2_brax", "quadruped_stepping_stones_2d",
        "humanoid_run_brax", "humanoid_simplified_physics",
        "humanoid_simplified_mjx", "humanoid_g1_mjx", "humanoid_corridor_2d",
    ):
        register_energy(_legacy_name, _legacy_energy_factory(_legacy_name))

