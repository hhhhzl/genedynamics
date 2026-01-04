"""
Isaac Sim USD model file generator with obstacle support.

This module provides utilities to generate USD (Universal Scene Description)
model files for Isaac Sim with obstacles embedded. Isaac Sim supports runtime
obstacle addition, but pre-defining them in USD files is more efficient.
"""

from typing import List, Optional, Dict, Any
import numpy as np
from pathlib import Path

try:
    from pxr import Usd, UsdGeom, Gf, UsdPhysics
    PXR_AVAILABLE = True
except ImportError:
    PXR_AVAILABLE = False
    Usd = None
    UsdGeom = None
    Gf = None
    UsdPhysics = None

from enerdynamics.envs.obstacles.base import ObstacleManager
from enerdynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle


def add_obstacle_to_usd(
    stage: 'Usd.Stage',
    obstacle,
    obstacle_path: str,
    index: int
) -> None:
    """
    Add a single obstacle to USD stage.
    
    High-performance obstacle addition with proper USD prim creation.
    
    Args:
        stage: USD stage to add obstacle to
        obstacle: Obstacle object to add
        obstacle_path: Base path for obstacles (e.g., "/World/obstacles")
        index: Obstacle index for naming
    """
    if not PXR_AVAILABLE:
        raise RuntimeError("PXR (USD) is required for Isaac Sim USD generation")
    
    center = obstacle.center if hasattr(obstacle, 'center') and obstacle.center is not None else np.zeros(3)
    center = np.asarray(center, dtype=np.float32)
    
    prim_path = f"{obstacle_path}/obstacle_{index}"
    obs_prim = stage.DefinePrim(prim_path, "Xform")
    
    # Set position
    translate = obs_prim.CreateAttribute("xformOp:translate", UsdGeom.Tokens.Translate)
    translate.Set(Gf.Vec3f(*center))
    
    if isinstance(obstacle, SphereObstacle):
        radius = float(obstacle.radius)
        geom = UsdGeom.Sphere.Define(stage, f"{prim_path}/geom")
        geom.CreateRadiusAttr(radius)
        geom.CreateColorAttr((0.8, 0.2, 0.2))
        
    elif isinstance(obstacle, BoxObstacle):
        half_extents = obstacle.half_extents
        size = np.asarray(half_extents, dtype=np.float32) * 2.0
        geom = UsdGeom.Cube.Define(stage, f"{prim_path}/geom")
        size_attr = geom.CreateSizeAttr()
        size_attr.Set(Gf.Vec3f(*size))
        geom.CreateColorAttr((0.8, 0.2, 0.2))
    else:
        # Default: sphere
        radius = 0.1
        if hasattr(obstacle, 'radius'):
            radius = float(obstacle.radius)
        geom = UsdGeom.Sphere.Define(stage, f"{prim_path}/geom")
        geom.CreateRadiusAttr(radius)
        geom.CreateColorAttr((0.8, 0.2, 0.2))
    
    # Add physics (static obstacle)
    physics = UsdPhysics.RigidBodyAPI.Apply(obs_prim)
    physics.CreateKinematicEnabledAttr(True)


def create_quadrotor_usd_with_obstacles(
    output_path: str,
    obstacles: Optional[ObstacleManager] = None,
    mass: float = 0.5,
    arm_length: float = 0.17,
    body_size: float = 0.1,
    motor_radius: float = 0.02
) -> str:
    """
    Create quadrotor USD model with obstacles for Isaac Sim.
    
    This function creates a complete USD file with quadrotor and obstacles.
    
    Args:
        output_path: Path to save USD file
        obstacles: Optional ObstacleManager with obstacles to add
        mass: Quadrotor mass (kg)
        arm_length: Distance from center to motor (m)
        body_size: Body cube size (m)
        motor_radius: Motor cylinder radius (m)
        
    Returns:
        Path to created USD file
        
    Raises:
        RuntimeError: If PXR (USD) is not available
    """
    if not PXR_AVAILABLE:
        raise RuntimeError(
            "PXR (USD) is required for Isaac Sim USD generation. "
            "Install with: pip install pxr"
        )
    
    stage = Usd.Stage.CreateNew(output_path)
    
    # Set up units
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    
    # Create root
    root_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(root_prim)
    
    # Create quadrotor
    quadrotor_prim = stage.DefinePrim("/World/quadrotor", "Xform")
    
    # Main body
    body_prim = stage.DefinePrim("/World/quadrotor/body", "Xform")
    body_geom = UsdGeom.Cube.Define(stage, "/World/quadrotor/body/geom")
    body_geom.CreateSizeAttr(body_size)
    body_geom.CreateColorAttr((0.3, 0.3, 0.3))
    
    # Add physics to body
    body_physics = UsdPhysics.RigidBodyAPI.Apply(body_prim)
    body_mass = UsdPhysics.MassAPI.Apply(body_prim)
    body_mass.CreateMassAttr(mass)
    
    # Motors (4 motors at corners)
    motor_positions = [
        (arm_length, arm_length, 0),   # Front-left
        (arm_length, -arm_length, 0),  # Front-right
        (-arm_length, arm_length, 0),  # Back-left
        (-arm_length, -arm_length, 0), # Back-right
    ]
    
    motor_names = ["motor_fl", "motor_fr", "motor_bl", "motor_br"]
    motor_colors = [
        (0.8, 0.2, 0.2),  # Red
        (0.2, 0.8, 0.2),  # Green
        (0.2, 0.2, 0.8),  # Blue
        (0.8, 0.8, 0.2),  # Yellow
    ]
    
    for pos, name, color in zip(motor_positions, motor_names, motor_colors):
        motor_prim = stage.DefinePrim(f"/World/quadrotor/{name}", "Xform")
        translate = motor_prim.CreateAttribute("xformOp:translate", UsdGeom.Tokens.Translate)
        translate.Set(Gf.Vec3f(*pos))
        
        motor_geom = UsdGeom.Cylinder.Define(stage, f"/World/quadrotor/{name}/geom")
        motor_geom.CreateRadiusAttr(motor_radius)
        motor_geom.CreateHeightAttr(motor_radius * 2)
        motor_geom.CreateColorAttr(color)
    
    # Add obstacles if provided
    if obstacles is not None and len(obstacles) > 0:
        obstacles_prim = stage.DefinePrim("/World/obstacles", "Xform")
        for i, obstacle in enumerate(obstacles):
            add_obstacle_to_usd(stage, obstacle, "/World/obstacles", i)
    
    # Save stage
    stage.GetRootLayer().Save()
    
    return output_path


def create_base_quadrotor_usd(
    output_path: str,
    mass: float = 0.5,
    arm_length: float = 0.17,
    body_size: float = 0.1,
    motor_radius: float = 0.02
) -> str:
    """
    Create base quadrotor USD template without obstacles.
    
    Args:
        output_path: Path to save USD file
        mass: Quadrotor mass (kg)
        arm_length: Distance from center to motor (m)
        body_size: Body cube size (m)
        motor_radius: Motor cylinder radius (m)
        
    Returns:
        Path to created USD file
    """
    return create_quadrotor_usd_with_obstacles(
        output_path,
        obstacles=None,
        mass=mass,
        arm_length=arm_length,
        body_size=body_size,
        motor_radius=motor_radius
    )


