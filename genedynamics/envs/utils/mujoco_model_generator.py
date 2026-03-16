"""
MuJoCo model file generator with obstacle support.

This module provides utilities to generate MuJoCo XML model files
with obstacles embedded. Since MuJoCo doesn't support runtime obstacle
addition, obstacles must be included in the model file at creation time.
"""

from typing import List, Optional, Dict, Any
import numpy as np
from pathlib import Path

from genedynamics.envs.obstacles.base import ObstacleManager
from genedynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle


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


def create_render_xml_with_trajectory(
    output_path: str,
    trajectory_positions: list,
    drone_scale: float = 3.0,
    line_radius: float = 0.008,
    line_rgba: str = "0.2 0.6 1.0 0.7",
) -> str:
    """
    Create quadrotor XML with trajectory line for rendering.

    Args:
        output_path: Path to save XML
        trajectory_positions: List of (x,y,z) positions, shape (N,3)
        drone_scale: Scale drone geometry for visibility
        line_radius: Trajectory line cylinder radius (m)
        line_rgba: Trajectory color "r g b a"

    Returns:
        XML string
    """
    create_base_quadrotor_xml(
        output_path,
        mass=0.5,
        arm_length=0.17 * drone_scale,
        body_size=0.05 * drone_scale,
        motor_size=0.02 * drone_scale,
    )
    with open(output_path) as f:
        xml = f.read()

    # Build trajectory line geoms (cylinder segments between consecutive points)
    # MuJoCo requires fromto points to be sufficiently far apart (min ~1e-4)
    MIN_SEG_LEN = 1e-4
    traj_lines = []
    positions = np.asarray(trajectory_positions, dtype=np.float64)
    if len(positions) < 2:
        traj_xml = ""
    else:
        seg_idx = 0
        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i + 1]
            dist = np.linalg.norm(p2 - p1)
            if dist < MIN_SEG_LEN:
                continue
            fromto = f"{p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f} {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}"
            traj_lines.append(
                f'      <geom name="traj_seg_{seg_idx}" type="cylinder" fromto="{fromto}" '
                f'size="{line_radius}" rgba="{line_rgba}" contype="0" conaffinity="0"/>'
            )
            seg_idx += 1
        if traj_lines:
            traj_xml = (
                '\n    <!-- Trajectory line -->\n    <body name="trajectory" pos="0 0 0">\n'
                + "\n".join(traj_lines)
                + "\n    </body>\n"
            )
        else:
            traj_xml = ""

    insert_pos = xml.rfind("</worldbody>")
    if insert_pos != -1 and traj_xml:
        xml = xml[:insert_pos] + traj_xml + xml[insert_pos:]

    with open(output_path, "w") as f:
        f.write(xml)
    return xml


def create_go2_render_xml_with_trajectory(
    output_path: str,
    trajectory_positions: list,
    go2_xml_path: Optional[str] = None,
    line_radius: float = 0.015,
    line_rgba: str = "0.2 0.6 1.0 0.7",
) -> str:
    """
    Create Go2 XML with trajectory line for rendering.
    Reads base Go2 from mujoco_menagerie, injects trajectory geoms.
    """
    if go2_xml_path is None:
        try:
            from genedynamics.robots.registry import _get_go2_path
            go2_xml_path = _get_go2_path()
        except Exception:
            go2_xml_path = None
    if go2_xml_path is None:
        import os
        menagerie = os.environ.get("MUJOCO_MENAGERIE_PATH")
        if menagerie:
            candidate = Path(menagerie) / "unitree_go2" / "go2.xml"
            if candidate.exists():
                go2_xml_path = str(candidate)
    if go2_xml_path is None:
        proj = Path(__file__).resolve().parents[3]
        for d in (proj / "third_party" / "mujoco_menagerie", proj / "mujoco_menagerie"):
            candidate = d / "unitree_go2" / "go2.xml"
            if candidate.exists():
                go2_xml_path = str(candidate)
                break
    if not go2_xml_path or not Path(go2_xml_path).exists():
        raise FileNotFoundError(
            "Go2 model not found. Set MUJOCO_MENAGERIE_PATH or install mujoco-menagerie."
        )
    go2_xml_path = str(Path(go2_xml_path))
    # Prefer scene XML for rendering (has floor/lights/cameras), fallback to bare go2.xml.
    go2_path_obj = Path(go2_xml_path)
    if go2_path_obj.name in ("go2.xml", "go2_mjx.xml"):
        for scene_name in ("scene.xml", "scene_mjx.xml"):
            scene_candidate = go2_path_obj.parent / scene_name
            if scene_candidate.exists():
                go2_xml_path = str(scene_candidate)
                break

    xml_content = Path(go2_xml_path).read_text()
    MIN_SEG_LEN = 1e-4
    line_radius = max(float(line_radius), 1e-6)
    traj_lines = []
    positions = np.asarray(trajectory_positions, dtype=np.float64)
    if len(positions) >= 2:
        seg_idx = 0
        for i in range(len(positions) - 1):
            p1, p2 = np.asarray(positions[i]), np.asarray(positions[i + 1])
            if np.any(~np.isfinite(p1)) or np.any(~np.isfinite(p2)):
                continue
            if np.linalg.norm(p2 - p1) < MIN_SEG_LEN:
                continue
            fromto = f"{p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f} {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}"
            traj_lines.append(
                f'      <geom name="traj_seg_{seg_idx}" type="cylinder" fromto="{fromto}" '
                f'size="{line_radius:.6f}" rgba="{line_rgba}" contype="0" conaffinity="0"/>'
            )
            seg_idx += 1

    traj_xml = ""
    if traj_lines:
        traj_xml = (
            '\n    <!-- Trajectory line -->\n    <body name="trajectory" pos="0 0 0">\n'
            + "\n".join(traj_lines)
            + "\n    </body>\n"
        )

    insert_pos = xml_content.rfind("</worldbody>")
    if insert_pos != -1 and traj_xml:
        xml_content = xml_content[:insert_pos] + traj_xml + xml_content[insert_pos:]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(xml_content)
    return xml_content


def create_ant_render_xml_with_trajectory(
    output_path: str,
    trajectory_positions: list,
    ant_xml_path: Optional[str] = None,
    line_radius: float = 0.015,
    line_rgba: str = "0.2 0.6 1.0 0.7",
) -> str:
    """Create ant XML with trajectory line for rendering (fallback when Go2 not available)."""
    if ant_xml_path is None:
        try:
            from genedynamics.robots.registry import _get_ant_path
            ant_xml_path = _get_ant_path()
        except Exception:
            ant_xml_path = None
    if not ant_xml_path or not Path(ant_xml_path).exists():
        raise FileNotFoundError("Ant model not found. Install gymnasium: pip install gymnasium")

    xml_content = Path(ant_xml_path).read_text()
    MIN_SEG_LEN = 1e-4
    line_radius = max(float(line_radius), 1e-6)
    traj_lines = []
    positions = np.asarray(trajectory_positions, dtype=np.float64)
    if len(positions) >= 2:
        seg_idx = 0
        for i in range(len(positions) - 1):
            p1, p2 = np.asarray(positions[i]), np.asarray(positions[i + 1])
            if np.any(~np.isfinite(p1)) or np.any(~np.isfinite(p2)):
                continue
            if np.linalg.norm(p2 - p1) < MIN_SEG_LEN:
                continue
            fromto = f"{p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f} {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}"
            traj_lines.append(
                f'      <geom name="traj_seg_{seg_idx}" type="cylinder" fromto="{fromto}" '
                f'size="{line_radius:.6f}" rgba="{line_rgba}" contype="0" conaffinity="0"/>'
            )
            seg_idx += 1

    traj_xml = ""
    if traj_lines:
        traj_xml = (
            '\n    <!-- Trajectory line -->\n    <body name="trajectory" pos="0 0 0">\n'
            + "\n".join(traj_lines)
            + "\n    </body>\n"
        )

    insert_pos = xml_content.rfind("</worldbody>")
    if insert_pos != -1 and traj_xml:
        xml_content = xml_content[:insert_pos] + traj_xml + xml_content[insert_pos:]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(xml_content)
    return xml_content








