"""Small construction-time MJCF composer for robot/tool/task assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class SphericalToolSpec:
    tool_id: str
    mount_role: str = "tool_mount"
    geom_name: str = "probe"
    radius: float = 0.02
    rgba: str = "0.9 0.3 0.1 1"


@dataclass(frozen=True)
class PegToolSpec:
    """Task tool welded to a robot's semantic tool mount.

    The peg axis is local ``+z``.  A dedicated tip site is added so insertion
    tasks can control and measure the physical tool tip without changing the
    robot-only :class:`RobotProfile`.
    """

    tool_id: str = "rectangular_peg"
    mount_role: str = "tool_mount"
    body_name: str = "peg_tool"
    geom_name: str = "peg"
    root_site_name: str = "peg_root"
    tip_site_name: str = "peg_tip"
    half_size_x: float = 0.010
    half_size_y: float = 0.006
    length: float = 0.050
    mass: float = 0.08
    rgba: str = "0.85 0.25 0.08 1"


@dataclass(frozen=True)
class SocketSpec:
    """Fixed rectangular socket represented only by convex box primitives.

    A hole cannot be one concave collision mesh in MuJoCo.  The four walls and
    bottom are separate boxes; the lead-in is a small staircase of boxes whose
    inner aperture contracts from ``chamfer_width`` to the nominal clearance.
    """

    fixture_id: str = "rectangular_socket"
    body_name: str = "socket"
    entrance_site_name: str = "socket_entrance"
    bottom_site_name: str = "socket_bottom"
    hole_half_size_x: float = 0.011
    hole_half_size_y: float = 0.007
    depth: float = 0.040
    wall_thickness: float = 0.010
    bottom_thickness: float = 0.006
    chamfer_depth: float = 0.006
    chamfer_width: float = 0.002
    chamfer_steps: int = 2
    rgba: str = "0.25 0.35 0.55 1"


class MjcfSceneComposer:
    """Compose static MJCF topology before MuJoCo/MJX compilation."""

    def __init__(self, robot_profile):
        self.profile = robot_profile
        self.model_path = Path(robot_profile.model_path()).resolve()
        self.root = ET.parse(self.model_path).getroot()
        self._make_asset_paths_absolute()

    def _make_asset_paths_absolute(self):
        compiler = self.root.find("compiler")
        if compiler is None:
            return
        for key in ("meshdir", "texturedir", "assetdir"):
            value = compiler.get(key)
            if value and not Path(value).is_absolute():
                compiler.set(key, str((self.model_path.parent / value).resolve()))

    def _top_level(self, tag: str, *, before: Optional[str] = None):
        element = self.root.find(tag)
        if element is not None:
            return element
        element = ET.Element(tag)
        if before is not None:
            children = list(self.root)
            for index, child in enumerate(children):
                if child.tag == before:
                    self.root.insert(index, element)
                    break
            else:
                self.root.append(element)
        else:
            self.root.append(element)
        return element

    def add_spherical_tool(
        self,
        tool: SphericalToolSpec,
        *,
        friction: float,
        collidable: bool,
    ) -> None:
        mount_name = self.profile.elements[tool.mount_role].name
        parent = None
        mount = None
        for candidate in self.root.iter():
            for child in candidate:
                if child.tag == "site" and child.get("name") == mount_name:
                    parent, mount = candidate, child
                    break
            if mount is not None:
                break
        if parent is None or mount is None:
            raise KeyError(
                f"robot '{self.profile.model_id}' has no mount site '{mount_name}'"
            )
        if any(g.get("name") == tool.geom_name for g in self.root.iter("geom")):
            raise ValueError(f"tool geom already exists in robot MJCF: {tool.geom_name}")
        attrs = {
            "name": tool.geom_name,
            "type": "sphere",
            "size": f"{tool.radius:.9g}",
            "pos": mount.get("pos", "0 0 0"),
            "rgba": tool.rgba,
            "contype": "1" if collidable else "0",
            "conaffinity": "1" if collidable else "0",
            "friction": f"{friction:.6g} 0.01 0.001",
        }
        children = list(parent)
        parent.insert(children.index(mount) + 1, ET.Element("geom", attrs))

    def _mount(self, role: str):
        mount_name = self.profile.elements[role].name
        for parent in self.root.iter():
            for child in parent:
                if child.tag == "site" and child.get("name") == mount_name:
                    return parent, child
        raise KeyError(
            f"robot '{self.profile.model_id}' has no mount site '{mount_name}'"
        )

    def add_peg_tool(
        self,
        tool: PegToolSpec,
        *,
        friction: Sequence[float] = (0.6, 0.005, 0.0001),
        solref: str = "0.01 1",
        solimp: str = "0.9 0.95 0.001",
    ) -> None:
        """Attach a collidable keyed peg while preserving the robot MJCF."""
        parent, mount = self._mount(tool.mount_role)
        if any(g.get("name") == tool.geom_name for g in self.root.iter("geom")):
            raise ValueError(f"tool geom already exists in robot MJCF: {tool.geom_name}")
        body = ET.Element("body", {
            "name": tool.body_name,
            "pos": mount.get("pos", "0 0 0"),
            "quat": mount.get("quat", "1 0 0 0"),
        })
        ET.SubElement(body, "site", {
            "name": tool.root_site_name,
            "pos": "0 0 0",
            "size": "0.002",
        })
        ET.SubElement(body, "geom", {
            "name": tool.geom_name,
            "type": "box",
            "size": (
                f"{tool.half_size_x:.9g} {tool.half_size_y:.9g} "
                f"{0.5 * tool.length:.9g}"
            ),
            "pos": f"0 0 {0.5 * tool.length:.9g}",
            "mass": f"{tool.mass:.9g}",
            "rgba": tool.rgba,
            "contype": "1",
            "conaffinity": "2",
            "condim": "3",
            "friction": " ".join(f"{float(v):.7g}" for v in friction),
            "solref": str(solref),
            "solimp": str(solimp),
        })
        ET.SubElement(body, "site", {
            "name": tool.tip_site_name,
            "pos": f"0 0 {tool.length:.9g}",
            "size": "0.002",
            "rgba": "1 0.8 0 1",
        })
        children = list(parent)
        parent.insert(children.index(mount) + 1, body)

    def add_rectangular_socket(
        self,
        socket: SocketSpec,
        *,
        position,
        quaternion,
        friction: Sequence[float] = (0.6, 0.005, 0.0001),
        solref: str = "0.01 1",
        solimp: str = "0.9 0.95 0.001",
    ) -> None:
        """Add a fixed socket with a convex-only collision decomposition."""
        if socket.depth <= 0.0 or socket.wall_thickness <= 0.0:
            raise ValueError("socket depth and wall thickness must be positive")
        if socket.hole_half_size_x <= 0.0 or socket.hole_half_size_y <= 0.0:
            raise ValueError("socket aperture must be positive")
        if socket.chamfer_steps < 0:
            raise ValueError("socket chamfer_steps cannot be negative")

        worldbody = self._top_level("worldbody")
        body = ET.SubElement(worldbody, "body", {
            "name": socket.body_name,
            "pos": " ".join(f"{float(v):.9g}" for v in position),
            "quat": " ".join(f"{float(v):.9g}" for v in quaternion),
        })
        ET.SubElement(body, "site", {
            "name": socket.entrance_site_name,
            "pos": "0 0 0",
            "size": "0.002",
            "rgba": "0 1 0 1",
        })
        ET.SubElement(body, "site", {
            "name": socket.bottom_site_name,
            "pos": f"0 0 {socket.depth:.9g}",
            "size": "0.002",
            "rgba": "0 0.5 1 1",
        })

        friction_text = " ".join(f"{float(v):.7g}" for v in friction)
        common = {
            "type": "box",
            "rgba": socket.rgba,
            "contype": "2",
            "conaffinity": "1",
            "condim": "3",
            "friction": friction_text,
            "solref": str(solref),
            "solimp": str(solimp),
        }

        def wall_ring(prefix: str, hx: float, hy: float, z: float, half_z: float):
            wt = socket.wall_thickness
            definitions = (
                ("x_pos", (hx + 0.5 * wt, 0.0, z), (0.5 * wt, hy + wt, half_z)),
                ("x_neg", (-hx - 0.5 * wt, 0.0, z), (0.5 * wt, hy + wt, half_z)),
                ("y_pos", (0.0, hy + 0.5 * wt, z), (hx, 0.5 * wt, half_z)),
                ("y_neg", (0.0, -hy - 0.5 * wt, z), (hx, 0.5 * wt, half_z)),
            )
            for suffix, pos, size in definitions:
                ET.SubElement(body, "geom", {
                    **common,
                    "name": f"{prefix}_{suffix}",
                    "pos": " ".join(f"{v:.9g}" for v in pos),
                    "size": " ".join(f"{v:.9g}" for v in size),
                })

        chamfer_depth = min(max(socket.chamfer_depth, 0.0), socket.depth)
        if socket.chamfer_steps and chamfer_depth > 0.0:
            layer = chamfer_depth / socket.chamfer_steps
            for index in range(socket.chamfer_steps):
                # Aperture contracts monotonically towards the straight bore.
                fraction = 1.0 - (index + 0.5) / socket.chamfer_steps
                wall_ring(
                    f"socket_leadin_{index}",
                    socket.hole_half_size_x + fraction * socket.chamfer_width,
                    socket.hole_half_size_y + fraction * socket.chamfer_width,
                    (index + 0.5) * layer,
                    0.51 * layer,
                )
        straight_depth = socket.depth - chamfer_depth
        if straight_depth > 0.0:
            wall_ring(
                "socket_wall",
                socket.hole_half_size_x,
                socket.hole_half_size_y,
                chamfer_depth + 0.5 * straight_depth,
                0.5 * straight_depth,
            )
        ET.SubElement(body, "geom", {
            **common,
            "name": "socket_bottom_geom",
            "pos": f"0 0 {socket.depth + 0.5 * socket.bottom_thickness:.9g}",
            "size": (
                f"{socket.hole_half_size_x + socket.wall_thickness:.9g} "
                f"{socket.hole_half_size_y + socket.wall_thickness:.9g} "
                f"{0.5 * socket.bottom_thickness:.9g}"
            ),
        })

    def strip_mesh_geoms(self) -> None:
        """Remove visual/link meshes for lightweight contact-only CPU scenes."""
        for parent in self.root.iter():
            for child in list(parent):
                if child.tag == "geom" and child.get("mesh") is not None:
                    parent.remove(child)
        referenced = {
            geom.get("mesh") for geom in self.root.iter("geom") if geom.get("mesh")
        }
        asset = self.root.find("asset")
        if asset is not None:
            for child in list(asset):
                if child.tag == "mesh" and child.get("name") not in referenced:
                    asset.remove(child)

    def add_hfield_surface(
        self,
        *,
        name: str,
        nrow: int,
        ncol: int,
        size,
        position,
        friction: float,
        solref: str,
    ) -> None:
        asset = self._top_level("asset", before="worldbody")
        ET.SubElement(asset, "hfield", {
            "name": name,
            "nrow": str(nrow),
            "ncol": str(ncol),
            # Preserve the original scan-scene quantization. Contact-local
            # controllability is measured at O(1e-4 m), so silently changing
            # this grid during refactoring can change the measured direction.
            "size": " ".join(f"{float(v):.5f}" for v in size),
        })
        worldbody = self._top_level("worldbody")
        ET.SubElement(worldbody, "geom", {
            "name": name,
            "type": "hfield",
            "hfield": name,
            "pos": " ".join(f"{float(v):.5f}" for v in position),
            "contype": "1",
            "conaffinity": "1",
            "friction": f"{friction:.6g} 0.01 0.001",
            "solref": str(solref),
        })

    def replace_actuators_with_motors(self, torque_limits) -> None:
        actuator = self._top_level("actuator")
        old = list(actuator)
        if len(old) != len(self.profile.actuated_joints):
            raise ValueError(
                f"{self.profile.model_id} actuator count {len(old)} does not match "
                f"profile count {self.profile.num_actuated}"
            )
        actuator.clear()
        names = self.profile.actuator_names or self.profile.actuated_joints
        for name, joint, limit in zip(names, self.profile.actuated_joints, torque_limits):
            ET.SubElement(actuator, "motor", {
                "name": name,
                "joint": joint,
                "ctrlrange": f"{-float(limit):.7g} {float(limit):.7g}",
            })

    @property
    def worldbody(self):
        return self._top_level("worldbody")

    @property
    def contact(self):
        return self._top_level("contact")

    def set_home_keyframe(self, qpos) -> None:
        keyframe = self._top_level("keyframe")
        for key in list(keyframe):
            if key.get("name") == self.profile.home_keyframe:
                keyframe.remove(key)
        ET.SubElement(keyframe, "key", {
            "name": self.profile.home_keyframe,
            "qpos": " ".join(f"{float(v):.9g}" for v in qpos),
        })

    def compile(self):
        import mujoco

        xml = ET.tostring(self.root, encoding="unicode")
        return mujoco.MjModel.from_xml_string(xml)
