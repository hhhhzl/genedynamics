"""
MuJoCo model file generator with obstacle support.

This module provides utilities to generate MuJoCo XML model files
with obstacles embedded. Since MuJoCo doesn't support runtime obstacle
addition, obstacles must be included in the model file at creation time.
"""

from typing import List, Optional, Dict, Any
import numpy as np
from pathlib import Path

from enerdynamics.envs.obstacles.base import ObstacleManager
from enerdynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle


def generate_obstacle_xml(obstacles: ObstacleManager, start_index: int = 0) -> str:
    """
    Generate XML string for obstacles in MuJoCo format.
    
    High-performance generation with proper formatting for MuJoCo.
    
    Args:
        obstacles: ObstacleManager with obstacles to convert
        start_index: Starting index for obstacle naming (default: 0)
        
    Returns:
        XML string containing obstacle geoms
    """
    obstacle_xml = []
    
    for i, obstacle in enumerate(obstacles):
        idx = start_index + i
        center = obstacle.center if hasattr(obstacle, 'center') and obstacle.center is not None else np.zeros(3)
        center = np.asarray(center, dtype=np.float64)
        
        if isinstance(obstacle, SphereObstacle):
            radius = float(obstacle.radius)
            xml = f"""      <geom name="obstacle_{idx}" type="sphere" 
            pos="{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}" 
            size="{radius:.6f}" 
            rgba="0.8 0.2 0.2 0.5" 
            group="1" 
            contype="1" 
            conaffinity="1"/>"""
            
        elif isinstance(obstacle, BoxObstacle):
            half_extents = obstacle.half_extents
            size_str = ' '.join(f"{float(h):.6f}" for h in half_extents)
            xml = f"""      <geom name="obstacle_{idx}" type="box" 
            pos="{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}" 
            size="{size_str}" 
            rgba="0.8 0.2 0.2 0.5" 
            group="1" 
            contype="1" 
            conaffinity="1"/>"""
        else:
            # Default: sphere with default radius
            radius = 0.1
            if hasattr(obstacle, 'radius'):
                radius = float(obstacle.radius)
            xml = f"""      <geom name="obstacle_{idx}" type="sphere" 
            pos="{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}" 
            size="{radius:.6f}" 
            rgba="0.8 0.2 0.2 0.5" 
            group="1" 
            contype="1" 
            conaffinity="1"/>"""
        
        obstacle_xml.append(xml)
    
    return '\n'.join(obstacle_xml)


def generate_mujoco_xml_with_obstacles(
    base_xml_path: str,
    obstacles: ObstacleManager,
    output_path: Optional[str] = None
) -> str:
    """
    Generate MuJoCo XML file with obstacles embedded.
    
    This function reads a base XML file and inserts obstacles before the
    closing </worldbody> tag. MuJoCo requires obstacles to be defined
    in the model file since runtime addition is not supported.
    
    Args:
        base_xml_path: Path to base XML file (without obstacles)
        obstacles: ObstacleManager with obstacles to add
        output_path: Optional path to save generated XML (if None, returns XML string only)
        
    Returns:
        XML string with obstacles embedded
        
    Raises:
        FileNotFoundError: If base_xml_path doesn't exist
        ValueError: If XML structure is invalid
    """
    base_path = Path(base_xml_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base XML file not found: {base_xml_path}")
    
    # Read base XML
    with open(base_path, 'r') as f:
        xml_content = f.read()
    
    # Generate obstacle XML
    obstacle_xml = generate_obstacle_xml(obstacles)
    
    # Find insertion point (before </worldbody>)
    insert_pos = xml_content.rfind('</worldbody>')
    if insert_pos == -1:
        raise ValueError("Could not find </worldbody> tag in XML. Invalid MuJoCo XML structure.")
    
    # Insert obstacles
    new_xml = (
        xml_content[:insert_pos] +
        f"\n    <!-- Generated obstacles -->\n    <body name=\"obstacles\">\n" +
        obstacle_xml +
        "\n    </body>\n" +
        xml_content[insert_pos:]
    )
    
    # Save if output path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, 'w') as f:
            f.write(new_xml)
    
    return new_xml


def create_base_quadrotor_xml(
    output_path: str,
    mass: float = 0.5,
    arm_length: float = 0.17,
    body_size: float = 0.05,
    motor_size: float = 0.02
) -> str:
    """
    Create base quadrotor XML template for MuJoCo.
    
    This creates a minimal quadrotor model that can be extended with obstacles.
    
    Args:
        output_path: Path to save XML file
        mass: Quadrotor mass (kg)
        arm_length: Distance from center to motor (m)
        body_size: Body cube size (m)
        motor_size: Motor cylinder radius (m)
        
    Returns:
        XML string
    """
    xml_template = f"""<mujoco model="quadrotor">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" 
             rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
    <material name="obstacle" rgba="0.8 0.2 0.2 0.5"/>
  </asset>
  
  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" pos="0 0 0" size="10 10 0.1" type="plane" 
          material="grid" condim="3" friction="1 0.005 0.0001"/>
    
    <!-- Quadrotor body -->
    <body name="quadrotor" pos="0 0 1">
      <!-- Main body (box) -->
      <geom name="body" type="box" size="{body_size} {body_size} {body_size/2}" 
            mass="{mass}" rgba="0.3 0.3 0.3 1"/>
      
      <!-- Motors (4 motors at corners) -->
      <body name="motor_fl" pos="{arm_length} {arm_length} 0">
        <geom name="motor_fl_geom" type="cylinder" size="{motor_size} {motor_size/2}" 
              mass="0.01" rgba="0.8 0.2 0.2 1"/>
        <joint name="motor_fl_joint" type="hinge" axis="0 0 1" damping="0.01"/>
      </body>
      
      <body name="motor_fr" pos="{arm_length} -{arm_length} 0">
        <geom name="motor_fr_geom" type="cylinder" size="{motor_size} {motor_size/2}" 
              mass="0.01" rgba="0.2 0.8 0.2 1"/>
        <joint name="motor_fr_joint" type="hinge" axis="0 0 1" damping="0.01"/>
      </body>
      
      <body name="motor_bl" pos="-{arm_length} {arm_length} 0">
        <geom name="motor_bl_geom" type="cylinder" size="{motor_size} {motor_size/2}" 
              mass="0.01" rgba="0.2 0.2 0.8 1"/>
        <joint name="motor_bl_joint" type="hinge" axis="0 0 1" damping="0.01"/>
      </body>
      
      <body name="motor_br" pos="-{arm_length} -{arm_length} 0">
        <geom name="motor_br_geom" type="cylinder" size="{motor_size} {motor_size/2}" 
              mass="0.01" rgba="0.8 0.8 0.2 1"/>
        <joint name="motor_br_joint" type="hinge" axis="0 0 1" damping="0.01"/>
      </body>
      
      <!-- Free joint for 6 DOF motion -->
      <freejoint name="quadrotor_free"/>
    </body>
    
    <!-- Obstacles will be inserted here -->
  </worldbody>
  
  <actuator>
    <!-- Motor actuators (thrust control) -->
    <motor name="motor_fl_act" joint="motor_fl_joint" gear="1" 
           ctrllimited="true" ctrlrange="0 1"/>
    <motor name="motor_fr_act" joint="motor_fr_joint" gear="1" 
           ctrllimited="true" ctrlrange="0 1"/>
    <motor name="motor_bl_act" joint="motor_bl_joint" gear="1" 
           ctrllimited="true" ctrlrange="0 1"/>
    <motor name="motor_br_act" joint="motor_br_joint" gear="1" 
           ctrllimited="true" ctrlrange="0 1"/>
  </actuator>
</mujoco>
"""
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, 'w') as f:
        f.write(xml_template)
    
    return xml_template










