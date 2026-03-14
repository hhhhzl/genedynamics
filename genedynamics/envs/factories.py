"""
Environment and energy factory functions.

This module provides factory functions to create environments and energy
functionals for different environments. This centralizes environment
creation logic.
"""

from typing import Optional, Any

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Create a dummy jnp for type hints
    class DummyJNP:
        @staticmethod
        def array(*args, **kwargs):
            raise RuntimeError("JAX is required but not installed")
        @staticmethod
        def maximum(*args, **kwargs):
            raise RuntimeError("JAX is required but not installed")
        @staticmethod
        def abs(*args, **kwargs):
            raise RuntimeError("JAX is required but not installed")
        @staticmethod
        def sum(*args, **kwargs):
            raise RuntimeError("JAX is required but not installed")
    jnp = DummyJNP()

from genedynamics.core.energy import LegacyEnergyFunctional, EnergyTerm


def make_env(name: str, **kwargs):
    """
    Factory function to create environments using the registry system.
    
    Args:
        name: Environment name (e.g., "double_integrator_box", "double_integrator_box_2d")
        **kwargs: Additional environment parameters
        
    Returns:
        Environment instance
        
    Raises:
        ValueError: If environment name is not recognized
    """
    # Try to use registry first
    try:
        from genedynamics.core.registry.environments import get_environment_registry
        registry = get_environment_registry()
        env_class = registry.get_class(name)
        if env_class is not None:
            return registry.create(name, **kwargs)
    except (ImportError, AttributeError):
        # Registry not available, fall back to hardcoded logic
        pass
    
    # Fallback to hardcoded logic for backward compatibility
    if name == "double_integrator_box":
        from genedynamics.envs.double_integrator_box import DoubleIntegratorBoxEnv
        return DoubleIntegratorBoxEnv(**kwargs)
    elif name == "double_integrator_box_2d":
        from genedynamics.envs.double_integrator_box_2d import DoubleIntegratorBox2DEnv
        return DoubleIntegratorBox2DEnv(**kwargs)
    elif name == "single_integrator_box_2d":
        from genedynamics.envs.single_integrator_box_2d import SingleIntegratorBox2DEnv
        return SingleIntegratorBox2DEnv(**kwargs)
    elif name == "drone_box_3d":
        from genedynamics.envs.drone_box_3d import DroneBox3DEnv
        return DroneBox3DEnv(**kwargs)
    elif name == "drone_full_3d":
        from genedynamics.envs.drone_full_3d import DroneFull3DEnv
        return DroneFull3DEnv(**kwargs)
    elif name == "drone_full_3d_physics":
        from genedynamics.envs.drone_full_3d_physics import DroneFull3DPhysicsEnv
        return DroneFull3DPhysicsEnv(**kwargs)
    elif name == "drone_full_3d_mujoco":
        from genedynamics.envs.drone_full_3d_mujoco import DroneFull3DMujocoEnv
        return DroneFull3DMujocoEnv(**kwargs)
    elif name == "drone_full_3d_mjx":
        from genedynamics.envs.drone_full_3d_mjx import DroneFull3DMjxEnv
        return DroneFull3DMjxEnv(**kwargs)
    elif name == "drone_full_3d_isaac":
        from genedynamics.envs.drone_full_3d_isaac import DroneFull3DIsaacEnv
        return DroneFull3DIsaacEnv(**kwargs)
    elif name == "quadruped_flat_physics":
        from genedynamics.envs.quadruped_base_physics import QuadrupedFlatPhysicsEnv
        return QuadrupedFlatPhysicsEnv(**kwargs)
    elif name == "quadruped_rough_physics":
        from genedynamics.envs.quadruped_base_physics import QuadrupedRoughPhysicsEnv
        return QuadrupedRoughPhysicsEnv(**kwargs)
    elif name == "quadruped_push_physics":
        from genedynamics.envs.quadruped_base_physics import QuadrupedPushPhysicsEnv
        return QuadrupedPushPhysicsEnv(**kwargs)
    elif name == "quadruped_go2_physics":
        from genedynamics.envs.quadruped_base_physics import QuadrupedGo2PhysicsEnv
        return QuadrupedGo2PhysicsEnv(**kwargs)
    elif name == "quadruped_flat_mjx":
        from genedynamics.envs.quadruped_mjx import QuadrupedAntMjxEnv
        return QuadrupedAntMjxEnv(**kwargs)
    elif name == "quadruped_go2_mjx":
        from genedynamics.envs.quadruped_mjx import QuadrupedGo2MjxEnv
        return QuadrupedGo2MjxEnv(**kwargs)
    elif name == "quadruped_go2_brax":
        from genedynamics.envs.brax_env import make_brax_go2
        return make_brax_go2(**kwargs)
    elif name == "humanoid_run_brax":
        from genedynamics.envs.brax_env import make_brax_humanoid_run
        return make_brax_humanoid_run(**kwargs)
    elif name == "humanoid_simplified_physics":
        from genedynamics.envs.humanoid_base_physics import HumanoidBasePhysicsEnv
        return HumanoidBasePhysicsEnv(**kwargs)
    elif name == "humanoid_g1_physics":
        from genedynamics.envs.humanoid_base_physics import HumanoidG1PhysicsEnv
        return HumanoidG1PhysicsEnv(**kwargs)
    elif name == "humanoid_simplified_mjx":
        from genedynamics.envs.humanoid_mjx import HumanoidSimplifiedMjxEnv
        return HumanoidSimplifiedMjxEnv(**kwargs)
    elif name == "humanoid_g1_mjx":
        from genedynamics.envs.humanoid_mjx import HumanoidG1MjxEnv
        return HumanoidG1MjxEnv(**kwargs)
    elif name == "softzoo":
        from genedynamics.experiments.plugins.environments.softzoo import SoftZooEnvironmentPlugin
        return SoftZooEnvironmentPlugin().create_env(kwargs)
    else:
        available = []
        try:
            from genedynamics.core.registry.environments import get_environment_registry
            registry = get_environment_registry()
            available = registry.list_available()
        except (ImportError, AttributeError):
            pass
        if available:
            raise ValueError(
                f"Unknown environment name: {name}. "
                f"Available environments: {available}"
            )
        else:
            raise ValueError(f"Unknown environment name: {name}")


def make_env_adapter(
    env: Any,
    backend: str = "numpy",
    obstacles: Optional[list] = None,
    **kwargs
):
    """
    Factory function to create environment adapter.
    
    Automatically detects environment type and wraps with appropriate adapter.
    
    Args:
        env: Environment to wrap (Gymnasium, Brax, or native genedynamics)
        backend: Computational backend ("numpy", "jax", "torch")
        obstacles: List of obstacles (optional)
        **kwargs: Additional adapter parameters
        
    Returns:
        UnifiedEnvAdapter instance
    """
    from genedynamics.envs.adapters.unified_adapter import UnifiedEnvAdapter
    return UnifiedEnvAdapter(env, backend=backend, obstacles=obstacles, **kwargs)


def make_energy(env_name: str) -> LegacyEnergyFunctional:
    """
    Factory function to create energy functionals for environments using the registry system.
    
    Args:
        env_name: Environment name
        
    Returns:
        LegacyEnergyFunctional instance
        
    Raises:
        ValueError: If environment name is not recognized
    """
    # Try to use energy registry first
    try:
        from genedynamics.core.registry.energy import get_energy_registry
        registry = get_energy_registry()
        # Check if registered (has factory or class)
        if registry.is_registered(env_name):
            # Use registry.create which handles factories automatically
            return registry.create(env_name)
    except (ImportError, AttributeError, TypeError):
        # Registry not available or error, fall back to hardcoded logic
        pass
    except ValueError:
        # Not found in registry, fall back to hardcoded defaults
        pass
    
    # Fallback to hardcoded logic for backward compatibility
    if env_name == "double_integrator_box":
        def task_energy(x, u, ctx):
            pos = x[0]
            vel = x[1]
            return (pos - 0.0) ** 2 + (vel ** 2)

        def box_energy(x, u, ctx):
            p_max, v_max = 2.0, 2.0
            pos_violate = jnp.maximum(0.0, jnp.abs(x[0]) - p_max)
            vel_violate = jnp.maximum(0.0, jnp.abs(x[1]) - v_max)
            pen = pos_violate ** 2 + vel_violate ** 2
            return pen

        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 1.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    elif env_name == "double_integrator_box_2d":
        def task_energy(x, u, ctx):
            pos = x[:2]
            vel = x[2:]
            return jnp.sum((pos - jnp.array([0.0, 0.0], dtype=jnp.float32)) ** 2) + jnp.sum(vel ** 2)

        def box_energy(x, u, ctx):
            p_max, v_max = 2.0, 2.0
            pos_violate = jnp.maximum(0.0, jnp.abs(x[:2]) - p_max)
            vel_violate = jnp.maximum(0.0, jnp.abs(x[2:]) - v_max)
            pen = jnp.sum(pos_violate ** 2 + vel_violate ** 2)
            return pen

        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 2.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    elif env_name == "single_integrator_box_2d":
        def task_energy(x, u, ctx):
            pos = x[:2]
            return jnp.sum((pos - jnp.array([0.0, 0.0], dtype=jnp.float32)) ** 2)

        def box_energy(x, u, ctx):
            p_max = 2.0
            pos_violate = jnp.maximum(0.0, jnp.abs(x[:2]) - p_max)
            pen = jnp.sum(pos_violate ** 2)
            return pen

        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 1.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    elif env_name == "drone_box_3d":
        def task_energy(x, u, ctx):
            pos = x[:3]
            vel = x[3:6]
            return jnp.sum((pos - jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)) ** 2) + jnp.sum(vel ** 2)

        def box_energy(x, u, ctx):
            p_max, v_max = 2.0, 2.0
            pos_violate = jnp.maximum(0.0, jnp.abs(x[:3]) - p_max)
            vel_violate = jnp.maximum(0.0, jnp.abs(x[3:6]) - v_max)
            pen = jnp.sum(pos_violate ** 2 + vel_violate ** 2)
            return pen

        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 2.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    elif env_name == "drone_full_3d":
        def task_energy(x, u, ctx):
            pos = x[:3]
            vel = x[3:6]
            euler = x[6:9]
            target = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
            pos_err = jnp.sum((pos - target) ** 2)
            vel_err = jnp.sum(vel ** 2)
            orientation_err = jnp.sum(euler ** 2)
            return pos_err + 0.1 * vel_err + 0.1 * orientation_err

        def box_energy(x, u, ctx):
            p_max, v_max = 2.0, 2.0
            pos_violate = jnp.maximum(0.0, jnp.abs(x[:3]) - p_max)
            vel_violate = jnp.maximum(0.0, jnp.abs(x[3:6]) - v_max)
            # Orientation bounds
            euler_violate = jnp.maximum(0.0, jnp.abs(x[6:9]) - jnp.pi)
            pen = jnp.sum(pos_violate ** 2 + vel_violate ** 2 + euler_violate ** 2)
            return pen

        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 2.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    elif env_name == "drone_full_3d_physics":
        # Position error + strong overshoot penalty (discourage looping) + velocity + control
        def task_energy(x, u, ctx):
            pos = x[:3]
            vel = x[3:6]
            euler = x[6:9]
            target = ctx.get("target_xy") if ctx else None
            if target is None:
                target = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
            else:
                target = jnp.asarray(target, dtype=jnp.float32).reshape(-1)[:3]
            pos_err = jnp.sum((pos - target) ** 2)
            # Strong overshoot: z > target_z, or |x|,|y| beyond target region (avoid "绕一圈")
            z_overshoot = jnp.maximum(0.0, pos[2] - target[2] - 0.15) ** 2
            xy_overshoot = jnp.maximum(0.0, jnp.abs(pos[0]) - 0.4) ** 2 + jnp.maximum(0.0, jnp.abs(pos[1]) - 0.4) ** 2
            dist = jnp.sqrt(pos_err + 1e-8)
            vel_err = jnp.sum(vel ** 2)
            orientation_err = jnp.sum(euler ** 2)
            vel_weight = 0.4 + 4.0 / (dist + 0.15)
            return pos_err + 15.0 * z_overshoot + 8.0 * xy_overshoot + vel_weight * vel_err + 1.5 * orientation_err + 0.12 * jnp.sum(u ** 2)

        def box_energy(x, u, ctx):
            p_max, v_max = 2.0, 2.0
            pos_violate = jnp.maximum(0.0, jnp.abs(x[:3]) - p_max)
            vel_violate = jnp.maximum(0.0, jnp.abs(x[3:6]) - v_max)
            # Orientation bounds
            euler_violate = jnp.maximum(0.0, jnp.abs(x[6:9]) - jnp.pi)
            pen = jnp.sum(pos_violate ** 2 + vel_violate ** 2 + euler_violate ** 2)
            return pen

        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 2.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    elif env_name in (
        "quadruped_flat_physics", "quadruped_flat_mjx", "quadruped_go2_mjx",
        "quadruped_go2_brax", "humanoid_run_brax",
        "humanoid_simplified_physics", "humanoid_simplified_mjx", "humanoid_g1_mjx",
    ):
        # Quadruped: base position error + velocity penalty + control regularization
        # Target from ctx["target_xy"] or default (2.0, 0.0, 0.5)
        # nq: ant=15, go2=19, humanoid=24, g1=36; infer from state size
        def _nq_from_state(x):
            s = jnp.asarray(x).ravel()
            if s.size >= 72:
                return 36
            if s.size >= 47:
                return 24
            if s.size >= 37:
                return 19
            return 15

        def task_energy(x, u, ctx):
            pos = x[:3]
            nq = _nq_from_state(x)
            vel = x[nq : nq + 3] if jnp.size(x) > nq + 3 else jnp.zeros(3, dtype=jnp.float32)
            ang_vel = x[nq + 3 : nq + 6] if jnp.size(x) > nq + 6 else jnp.zeros(3, dtype=jnp.float32)
            target = ctx.get("target_xy") if ctx else None
            if target is None:
                target = jnp.array([2.0, 0.0, 0.5], dtype=jnp.float32)
            else:
                target = jnp.asarray(target, dtype=jnp.float32).reshape(-1)[:3]
            desired_v = ctx.get("desired_velocity") if ctx else None
            if desired_v is None:
                desired_v = jnp.zeros(3, dtype=jnp.float32)
            else:
                desired_v = jnp.asarray(desired_v, dtype=jnp.float32).reshape(-1)[:3]
            loc_active = jnp.linalg.norm(desired_v[:2]) >= 1e-4
            pos_err = jnp.sum((pos - target) ** 2)
            # Overshoot penalty: discourage looping past target (same as drone)
            z_overshoot = jnp.maximum(0.0, pos[2] - target[2] - 0.2) ** 2
            xy_overshoot = jnp.maximum(0.0, jnp.abs(pos[0] - target[0]) - 0.5) ** 2 + jnp.maximum(0.0, jnp.abs(pos[1] - target[1]) - 0.5) ** 2
            dist = jnp.sqrt(pos_err + 1e-8)
            vel_weight_nominal = 0.3 + 2.0 / (dist + 0.2)
            vel_weight = jnp.where(loc_active, jnp.asarray(0.08, dtype=jnp.float32), vel_weight_nominal)
            vel_err = vel_weight * jnp.sum(vel ** 2)
            # Dial-MPC-style directional shaping: for now prioritize forward x progress and suppress lateral drift.
            lateral_err = (pos[1] - target[1]) ** 2
            # Lightweight progress shaping for point-target:
            # encourage forward motion along target direction and discourage moving away.
            delta_xy = target[:2] - pos[:2]
            dist_xy = jnp.maximum(jnp.linalg.norm(delta_xy), 1e-6)
            dir_xy = delta_xy / dist_xy
            toward_speed = jnp.dot(vel[:2], dir_xy)
            away_penalty = jnp.maximum(0.0, -toward_speed) ** 2
            # Locomotion speed tracking (dial-mpc style idea): encourage steady forward speed.
            desired_speed = 0.90 if nq <= 15 else 0.60
            speed_track_err = (toward_speed - desired_speed) ** 2
            vel_track_err = jnp.sum((vel[:2] - desired_v[:2]) ** 2)
            # Stability terms to avoid "flying"/tumbling trajectories.
            # MuJoCo free-joint quaternion is [w, x, y, z] at qpos[3:7].
            if jnp.size(x) >= 7:
                quat = jnp.asarray(x[3:7], dtype=jnp.float32)
                quat = quat / jnp.maximum(jnp.linalg.norm(quat), 1e-6)
                # Penalize roll/pitch tilt; keep yaw relatively unconstrained.
                upright_err = quat[1] ** 2 + quat[2] ** 2
                # Mild yaw heading regularization for straighter gaits.
                qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
                yaw = jnp.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                yaw_err = yaw ** 2
            else:
                upright_err = jnp.asarray(0.0, dtype=jnp.float32)
                yaw_err = jnp.asarray(0.0, dtype=jnp.float32)
            height_err = (pos[2] - target[2]) ** 2
            vz_err = vel[2] ** 2
            ang_vel_err = jnp.sum(ang_vel ** 2)
            # Joint velocity penalty: discourage erratic leg motion
            joint_vel = x[nq + 6 :] if jnp.size(x) > nq + 6 else jnp.zeros(0, dtype=jnp.float32)
            joint_vel_err = 0.06 * jnp.sum(joint_vel ** 2)
            # Harder penalty once joint speeds exceed a cap (reduces "leg flailing").
            joint_vel_cap = 6.5 if nq == 19 else 8.0
            joint_vel_over = jnp.maximum(0.0, jnp.abs(joint_vel) - joint_vel_cap)
            joint_vel_cap_err = jnp.sum(joint_vel_over ** 2)
            # Ant needs stronger attitude regularization than larger robots.
            upright_w = 5.0 if nq <= 15 else 2.8
            yaw_w = 0.3 if nq <= 15 else 1.4
            lateral_w = 0.8 if nq <= 15 else 3.5
            ang_vel_w = 0.2 if nq <= 15 else 0.35
            joint_cap_w = 0.1 if nq <= 15 else 0.35
            speed_track_w = 0.6 if nq <= 15 else 1.5
            vel_track_w = 0.4 if nq <= 15 else 2.0
            # In locomotion mode (desired velocity provided), weaken point-target attraction.
            pos_w_loc = 0.2 if nq <= 15 else 0.02
            pos_w = jnp.where(loc_active, jnp.asarray(pos_w_loc, dtype=jnp.float32), jnp.asarray(1.0, dtype=jnp.float32))
            return (
                pos_w * pos_err
                + 8.0 * z_overshoot
                + 5.0 * xy_overshoot
                + vel_err
                + lateral_w * lateral_err
                + 2.0 * height_err
                + upright_w * upright_err
                + yaw_w * yaw_err
                + 1.0 * vz_err
                + ang_vel_w * ang_vel_err
                + 0.08 * away_penalty
                + speed_track_w * speed_track_err
                + vel_track_w * vel_track_err
                + joint_vel_err
                + joint_cap_w * joint_vel_cap_err
            )

        def control_energy(x, u, ctx):
            return 0.12 * jnp.sum(u ** 2)  # Stronger penalty for smoother actions

        def box_energy(x, u, ctx):
            p_max, v_max = 5.0, 3.0
            pos = x[:3]
            nq = _nq_from_state(x)
            vel = x[nq : nq + 3] if jnp.size(x) > nq + 3 else jnp.zeros(3)
            pos_violate = jnp.maximum(0.0, jnp.abs(pos) - p_max)
            vel_violate = jnp.maximum(0.0, jnp.abs(vel) - v_max)
            return jnp.sum(pos_violate ** 2 + vel_violate ** 2)

        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 1.0),
            "control": EnergyTerm(control_energy, 1.0),
            "box": EnergyTerm(box_energy, 0.1),
        })
    else:
        raise ValueError(f"Unknown environment name: {env_name}")

