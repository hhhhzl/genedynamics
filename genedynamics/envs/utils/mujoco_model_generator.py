"""
MuJoCo model file generator with obstacle support.

This module provides utilities to generate MuJoCo XML model files
with obstacles embedded. Since MuJoCo doesn't support runtime obstacle
addition, obstacles must be included in the model file at creation time.
"""

import re
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
    line_radius: float = 0.006,
    line_rgba: str = "0.2 0.6 1.0 0.7",
    stepping_scene: Optional[Dict[str, Any]] = None,
    swing_trajectories: Optional[Dict[str, np.ndarray]] = None,
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

    scene_xml = ""
    if stepping_scene:
        centers = np.asarray(stepping_scene.get("stones_centers", []), dtype=np.float64)
        radii = np.asarray(stepping_scene.get("stones_radii", []), dtype=np.float64).reshape(-1)
        support_platforms = np.asarray(stepping_scene.get("support_platforms", []), dtype=np.float64).reshape(-1, 4)
        map_y = stepping_scene.get("map_y", [-0.9, 0.9])
        river_x = stepping_scene.get("river_x", [-0.2, 0.2])
        has_river = bool(stepping_scene.get("has_river", True))
        try:
            y0, y1 = float(map_y[0]), float(map_y[1])
            rx0, rx1 = float(river_x[0]), float(river_x[1])
        except Exception:
            y0, y1 = -0.9, 0.9
            rx0, rx1 = -0.2, 0.2

        geoms = []
        if has_river:
            river_half_x = max(1e-4, 0.5 * abs(rx1 - rx0))
            river_half_y = max(1e-4, 0.5 * abs(y1 - y0))
            river_cx = 0.5 * (rx0 + rx1)
            river_cy = 0.5 * (y0 + y1)
            geoms.append(
                f'      <geom name="stepping_river" type="box" pos="{river_cx:.6f} {river_cy:.6f} 0.000200" '
                f'size="{river_half_x:.6f} {river_half_y:.6f} 0.000200" rgba="0.20 0.52 0.84 0.45" '
                f'contype="0" conaffinity="0"/>'
            )
        n = min(len(centers), len(radii))
        for i in range(n):
            c = np.asarray(centers[i], dtype=np.float64).reshape(-1)
            if c.size < 2 or not np.isfinite(c[:2]).all() or not np.isfinite(radii[i]):
                continue
            r = max(1e-4, float(radii[i]))
            geoms.append(
                f'      <geom name="stepping_stone_{i}" type="cylinder" pos="{c[0]:.6f} {c[1]:.6f} 0.003000" '
                f'size="{r:.6f} 0.003000" rgba="0.70 0.72 0.75 0.92" contype="0" conaffinity="0"/>'
            )
        for i, rect in enumerate(support_platforms):
            if rect.size < 4 or not np.isfinite(rect[:4]).all():
                continue
            x0, x1, y0p, y1p = [float(v) for v in rect[:4]]
            hx = max(1e-4, 0.5 * abs(x1 - x0))
            hy = max(1e-4, 0.5 * abs(y1p - y0p))
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0p + y1p)
            geoms.append(
                f'      <geom name="support_platform_{i}" type="box" pos="{cx:.6f} {cy:.6f} 0.003000" '
                f'size="{hx:.6f} {hy:.6f} 0.003000" rgba="0.44 0.49 0.55 0.92" contype="0" conaffinity="0"/>'
            )
        if geoms:
            scene_xml = (
                '\n    <!-- Stepping-stones scene overlay -->\n    <body name="stepping_scene_overlay" pos="0 0 0">\n'
                + "\n".join(geoms)
                + "\n    </body>\n"
            )

    traj_xml = ""
    if traj_lines:
        traj_xml = (
            '\n    <!-- Trajectory line -->\n    <body name="trajectory" pos="0 0 0">\n'
            + "\n".join(traj_lines)
            + "\n    </body>\n"
        )

    swing_xml = ""
    if isinstance(swing_trajectories, dict) and len(swing_trajectories) > 0:
        leg_colors = {
            "FL": "0.90 0.25 0.25 0.85",
            "FR": "0.25 0.55 0.95 0.85",
            "RL": "0.25 0.85 0.35 0.85",
            "RR": "0.90 0.78 0.22 0.85",
        }
        sw_lines = []
        sw_idx = 0
        sw_radius = max(1e-6, 0.45 * line_radius)
        for leg in ("FL", "FR", "RL", "RR"):
            arr = np.asarray(swing_trajectories.get(leg, []), dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] < 2:
                continue
            color = leg_colors.get(leg, "1.0 1.0 1.0 0.85")
            for i in range(arr.shape[0] - 1):
                p1 = arr[i, :3]
                p2 = arr[i + 1, :3]
                if np.any(~np.isfinite(p1)) or np.any(~np.isfinite(p2)):
                    continue
                if np.linalg.norm(p2 - p1) < MIN_SEG_LEN:
                    continue
                fromto = f"{p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f} {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}"
                sw_lines.append(
                    f'      <geom name="swing_{leg}_{sw_idx}" type="cylinder" fromto="{fromto}" '
                    f'size="{sw_radius:.6f}" rgba="{color}" contype="0" conaffinity="0"/>'
                )
                sw_idx += 1
        if sw_lines:
            swing_xml = (
                '\n    <!-- Swing trajectory lines -->\n    <body name="swing_trajectory_overlay" pos="0 0 0">\n'
                + "\n".join(sw_lines)
                + "\n    </body>\n"
            )

    insert_pos = xml_content.rfind("</worldbody>")
    extras = scene_xml + traj_xml + swing_xml
    if insert_pos != -1 and extras:
        xml_content = xml_content[:insert_pos] + extras + xml_content[insert_pos:]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(xml_content)
    return xml_content


def create_go2_sim_xml_with_stepping_scene(
    output_path: str,
    stepping_scene: Dict[str, Any],
    go2_xml_path: Optional[str] = None,
    *,
    bank_top_z: float = 0.0,
    bank_half_thickness: float = 0.04,
    river_depth: float = 0.10,
    river_half_thickness: float = 0.05,
    stone_top_z: float = 0.035,
    stone_half_height: float = 0.0175,
) -> str:
    """
    Create a Go2 MuJoCo scene with physical stepping-stone collision geometry.

    This disables the default infinite floor collision and replaces it with:
    - left/right support banks at z=bank_top_z
    - a lowered river bottom
    - stepping-stone cylinders that can be contacted by the feet
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

    go2_path_obj = Path(str(go2_xml_path))
    if go2_path_obj.name in ("go2.xml", "go2_mjx.xml"):
        for scene_name in ("scene.xml", "scene_mjx.xml"):
            scene_candidate = go2_path_obj.parent / scene_name
            if scene_candidate.exists():
                go2_xml_path = str(scene_candidate)
                break

    xml_content = Path(str(go2_xml_path)).read_text()

    centers = np.asarray(stepping_scene.get("stones_centers", []), dtype=np.float64)
    radii = np.asarray(stepping_scene.get("stones_radii", []), dtype=np.float64).reshape(-1)
    support_platforms = np.asarray(stepping_scene.get("support_platforms", []), dtype=np.float64).reshape(-1, 4)
    map_x = stepping_scene.get("map_x", [-1.6, 1.6])
    map_y = stepping_scene.get("map_y", [-0.9, 0.9])
    river_x = stepping_scene.get("river_x", [-0.2, 0.2])
    has_river = bool(stepping_scene.get("has_river", True))
    try:
        x0, x1 = float(map_x[0]), float(map_x[1])
        y0, y1 = float(map_y[0]), float(map_y[1])
        rx0, rx1 = float(river_x[0]), float(river_x[1])
    except Exception:
        x0, x1 = -1.6, 1.6
        y0, y1 = -0.9, 0.9
        rx0, rx1 = -0.2, 0.2

    if has_river:
        xml_content = re.sub(
            r'<geom([^>]*name="floor"[^>]*)/>',
            r'<geom\1 contype="0" conaffinity="0" rgba="0.16 0.20 0.22 1.0"/>',
            xml_content,
            count=1,
        )
    else:
        xml_content = re.sub(
            r'<geom([^>]*name="floor"[^>]*)/>',
            r'<geom\1 rgba="0.16 0.20 0.22 1.0"/>',
            xml_content,
            count=1,
        )

    geoms = []
    if has_river:
        left_half_x = max(1e-4, 0.5 * max(0.0, rx0 - x0))
        right_half_x = max(1e-4, 0.5 * max(0.0, x1 - rx1))
        half_y = max(1e-4, 0.5 * abs(y1 - y0))
        if left_half_x > 1e-4:
            left_cx = 0.5 * (x0 + rx0)
            geoms.append(
                f'      <geom name="bank_left" type="box" pos="{left_cx:.6f} {0.5 * (y0 + y1):.6f} '
                f'{bank_top_z - bank_half_thickness:.6f}" size="{left_half_x:.6f} {half_y:.6f} '
                f'{bank_half_thickness:.6f}" rgba="0.26 0.28 0.30 1.0" contype="1" conaffinity="1"/>'
            )
        if right_half_x > 1e-4:
            right_cx = 0.5 * (rx1 + x1)
            geoms.append(
                f'      <geom name="bank_right" type="box" pos="{right_cx:.6f} {0.5 * (y0 + y1):.6f} '
                f'{bank_top_z - bank_half_thickness:.6f}" size="{right_half_x:.6f} {half_y:.6f} '
                f'{bank_half_thickness:.6f}" rgba="0.26 0.28 0.30 1.0" contype="1" conaffinity="1"/>'
            )

        river_half_x = max(1e-4, 0.5 * abs(rx1 - rx0))
        river_half_y = max(1e-4, 0.5 * abs(y1 - y0))
        river_cx = 0.5 * (rx0 + rx1)
        river_cy = 0.5 * (y0 + y1)
        geoms.append(
            f'      <geom name="river_bottom" type="box" pos="{river_cx:.6f} {river_cy:.6f} '
            f'{-float(river_depth) - float(river_half_thickness):.6f}" size="{river_half_x:.6f} '
            f'{river_half_y:.6f} {float(river_half_thickness):.6f}" rgba="0.12 0.34 0.58 0.95" '
            f'contype="1" conaffinity="1"/>'
        )

    stone_top_z_local = float(stone_top_z if has_river else min(stone_top_z, 0.006))
    stone_half_height_local = float(stone_half_height if has_river else min(stone_half_height, 0.003))

    n = min(len(centers), len(radii))
    for i in range(n):
        c = np.asarray(centers[i], dtype=np.float64).reshape(-1)
        if c.size < 2 or not np.isfinite(c[:2]).all() or not np.isfinite(radii[i]):
            continue
        r = max(1e-4, float(radii[i]))
        geoms.append(
            f'      <geom name="stepping_stone_{i}" type="cylinder" pos="{c[0]:.6f} {c[1]:.6f} '
            f'{stone_top_z_local - stone_half_height_local:.6f}" size="{r:.6f} {stone_half_height_local:.6f}" '
            f'rgba="0.70 0.72 0.75 1.0" contype="1" conaffinity="1"/>'
        )
    for i, rect in enumerate(support_platforms):
        if rect.size < 4 or not np.isfinite(rect[:4]).all():
            continue
        x0p, x1p, y0p, y1p = [float(v) for v in rect[:4]]
        hx = max(1e-4, 0.5 * abs(x1p - x0p))
        hy = max(1e-4, 0.5 * abs(y1p - y0p))
        cx = 0.5 * (x0p + x1p)
        cy = 0.5 * (y0p + y1p)
        geoms.append(
            f'      <geom name="support_platform_{i}" type="box" pos="{cx:.6f} {cy:.6f} '
            f'{stone_top_z_local - stone_half_height_local:.6f}" size="{hx:.6f} {hy:.6f} {stone_half_height_local:.6f}" '
            f'rgba="0.44 0.49 0.55 1.0" contype="1" conaffinity="1"/>'
        )

    scene_xml = (
        '\n    <!-- Physical stepping-stones scene -->\n    <body name="stepping_scene_collision" pos="0 0 0">\n'
        + "\n".join(geoms)
        + "\n    </body>\n"
    )
    insert_pos = xml_content.rfind("</worldbody>")
    if insert_pos != -1:
        xml_content = xml_content[:insert_pos] + scene_xml + xml_content[insert_pos:]

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


def create_g1_render_xml_with_trajectory(
    output_path: str,
    trajectory_positions: list,
    g1_xml_path: Optional[str] = None,
    line_radius: float = 0.008,
    line_rgba: str = "0.2 0.6 1.0 0.7",
    corridor_scene: Optional[Dict[str, Any]] = None,
) -> str:
    """Create G1 XML with trajectory line for rendering."""
    if g1_xml_path is None:
        try:
            from genedynamics.robots.registry import _get_g1_path
            g1_xml_path = _get_g1_path()
        except Exception:
            g1_xml_path = None
    if g1_xml_path is None:
        import os
        menagerie = os.environ.get("MUJOCO_MENAGERIE_PATH")
        if menagerie:
            candidate = Path(menagerie) / "unitree_g1" / "g1.xml"
            if candidate.exists():
                g1_xml_path = str(candidate)
    if g1_xml_path is None:
        proj = Path(__file__).resolve().parents[3]
        for d in (proj / "third_party" / "mujoco_menagerie", proj / "mujoco_menagerie"):
            candidate = d / "unitree_g1" / "g1.xml"
            if candidate.exists():
                g1_xml_path = str(candidate)
                break
    if not g1_xml_path or not Path(g1_xml_path).exists():
        raise FileNotFoundError(
            "G1 model not found. Set MUJOCO_MENAGERIE_PATH or install mujoco-menagerie."
        )

    g1_path_obj = Path(g1_xml_path)
    if g1_path_obj.name in ("g1.xml", "g1_mjx.xml"):
        for scene_name in ("scene.xml", "scene_mjx.xml"):
            scene_candidate = g1_path_obj.parent / scene_name
            if scene_candidate.exists():
                g1_xml_path = str(scene_candidate)
                break

    xml_content = Path(g1_xml_path).read_text()
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

    corridor_xml = _build_corridor_scene_overlay_xml(corridor_scene)
    insert_pos = xml_content.rfind("</worldbody>")
    extras = corridor_xml + traj_xml
    if insert_pos != -1 and extras:
        xml_content = xml_content[:insert_pos] + extras + xml_content[insert_pos:]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(xml_content)
    return xml_content


def _build_corridor_scene_overlay_xml(corridor_scene: Optional[Dict[str, Any]]) -> str:
    if not isinstance(corridor_scene, dict):
        return ""

    try:
        corridor_width = float(corridor_scene["corridor_width"])
        corridor_length = float(corridor_scene["corridor_length"])
    except Exception:
        return ""

    half_w = max(1e-4, 0.5 * corridor_width)
    half_l = max(1e-4, 0.5 * corridor_length)
    wall_y_min = float(corridor_scene.get("wall_y_min", -half_w))
    wall_y_max = float(corridor_scene.get("wall_y_max", half_w))
    wall_thickness = 0.025
    wall_height = 1.10
    geoms = []
    if not corridor_scene.get("hide_floor_patch", False):
        geoms.append(
            f'      <geom name="corridor_floor_patch" type="box" pos="{half_l:.6f} 0.000000 0.000500" '
            f'size="{half_l:.6f} {half_w:.6f} 0.000500" rgba="0.14 0.16 0.19 0.35" contype="0" conaffinity="0"/>'
        )
    geoms.extend([
        f'      <geom name="corridor_wall_left" type="box" pos="{half_l:.6f} {wall_y_max + 0.5 * wall_thickness:.6f} {0.5 * wall_height:.6f}" '
        f'size="{half_l:.6f} {0.5 * wall_thickness:.6f} {0.5 * wall_height:.6f}" rgba="0.72 0.75 0.80 0.28" contype="0" conaffinity="0"/>',
        f'      <geom name="corridor_wall_right" type="box" pos="{half_l:.6f} {wall_y_min - 0.5 * wall_thickness:.6f} {0.5 * wall_height:.6f}" '
        f'size="{half_l:.6f} {0.5 * wall_thickness:.6f} {0.5 * wall_height:.6f}" rgba="0.72 0.75 0.80 0.28" contype="0" conaffinity="0"/>',
    ])

    obstacles = corridor_scene.get("obstacles", [])
    # Unified obstacle color across ALL zones/shapes: a muted teal — distinct from
    # the gray-blue corridor walls (0.72 0.75 0.80), see-through (alpha 0.42 lets
    # the robot + walls behind show), harmonious with the cool MuJoCo scene.
    OBSTACLE_RGBA = "0.33 0.58 0.60 0.42"
    obs_zc = 0.5 * wall_height      # obstacles span the same height as the walls
    obs_zh = 0.5 * wall_height
    for i, obs in enumerate(obstacles):
        if not isinstance(obs, dict):
            continue
        shape = str(obs.get("shape", "box")).lower()

        if shape == "sphere":
            try:
                cx = float(obs["cx"]); cy = float(obs["cy"])
                radius = max(1e-4, float(obs["radius"]))
            except Exception:
                continue
            z_center = 0.5 * (float(obs.get("z_min", 0.0)) + float(obs.get("z_max", 2.0)))
            geoms.append(
                f'      <geom name="corridor_obstacle_{i}" type="sphere" pos="{cx:.6f} {cy:.6f} {z_center:.6f}" '
                f'size="{radius:.6f}" rgba="{OBSTACLE_RGBA}" contype="0" conaffinity="0"/>'
            )
            continue

        if shape == "qc":
            # Quarter-disk (rounded corner): render the real 90deg pie slice as a fan
            # of thin rotated boxes, NOT the bbox square. Quadrant from clip_sign (x
            # side) and sign(cy) (y side), matching _qc_sdf.
            try:
                cx = float(obs["cx"]); cy = float(obs["cy"])
                r = max(1e-4, float(obs["radius"]))
                clip = float(obs.get("qc_clip_sign", 1.0))
            except Exception:
                continue
            left = clip > 0.0       # region x <= cx
            below = cy > 0.0        # region y <= cy
            a0 = {(True, True): 180.0, (True, False): 90.0,
                  (False, True): 270.0, (False, False): 0.0}[(left, below)]
            N = 10
            step = 90.0 / N
            hw = max(1e-3, r * float(np.sin(np.deg2rad(step) / 2.0)) * 1.25)
            for k in range(N):
                th = np.deg2rad(a0 + (k + 0.5) * step)
                bx = cx + 0.5 * r * float(np.cos(th))
                by = cy + 0.5 * r * float(np.sin(th))
                qw = float(np.cos(th / 2.0)); qz = float(np.sin(th / 2.0))
                geoms.append(
                    f'      <geom name="corridor_obstacle_{i}_{k}" type="box" '
                    f'pos="{bx:.6f} {by:.6f} {obs_zc:.6f}" '
                    f'size="{0.5 * r:.6f} {hw:.6f} {obs_zh:.6f}" '
                    f'quat="{qw:.6f} 0 0 {qz:.6f}" rgba="{OBSTACLE_RGBA}" '
                    f'contype="0" conaffinity="0"/>'
                )
            continue

        # box (corridor walls/pillars), rendered at wall height, unified color.
        try:
            x_min = float(obs["x_min"]); x_max = float(obs["x_max"])
            y_min = float(obs["y_min"]); y_max = float(obs["y_max"])
        except Exception:
            continue
        x_center = 0.5 * (x_min + x_max); y_center = 0.5 * (y_min + y_max)
        x_half = max(1e-4, 0.5 * abs(x_max - x_min))
        y_half = max(1e-4, 0.5 * abs(y_max - y_min))
        geoms.append(
            f'      <geom name="corridor_obstacle_{i}" type="box" '
            f'pos="{x_center:.6f} {y_center:.6f} {obs_zc:.6f}" '
            f'size="{x_half:.6f} {y_half:.6f} {obs_zh:.6f}" rgba="{OBSTACLE_RGBA}" '
            f'contype="0" conaffinity="0"/>'
        )

    start_pos = corridor_scene.get("start_pos")
    goal_pos = corridor_scene.get("goal_pos")
    if isinstance(start_pos, (list, tuple)) and len(start_pos) >= 2:
        geoms.append(
            f'      <geom name="corridor_start_marker" type="cylinder" pos="{float(start_pos[0]):.6f} {float(start_pos[1]):.6f} 0.005000" '
            f'size="0.100000 0.005000" rgba="0.18 0.78 0.42 0.80" contype="0" conaffinity="0"/>'
        )
    if isinstance(goal_pos, (list, tuple)) and len(goal_pos) >= 2:
        geoms.append(
            f'      <geom name="corridor_goal_marker" type="cylinder" pos="{float(goal_pos[0]):.6f} {float(goal_pos[1]):.6f} 0.005000" '
            f'size="0.100000 0.005000" rgba="0.18 0.58 0.96 0.80" contype="0" conaffinity="0"/>'
        )

    if not geoms:
        return ""
    return (
        '\n    <!-- Corridor scene overlay -->\n    <body name="corridor_scene_overlay" pos="0 0 0">\n'
        + "\n".join(geoms)
        + "\n    </body>\n"
    )






