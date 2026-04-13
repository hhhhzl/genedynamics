"""
pyrender mesh backend (Tier 3, true 3D).

Loads G1 / Go2 STL meshes from
``third_party/mujoco_menagerie/{unitree_g1,unitree_go2}`` via a small
mujoco-xml kinematics parser, applies pose driven by ``RobotPoseIR``,
builds a procedural "indoor corridor" / "stepping stones river" scene
with brick + tile textures, and renders offline through pyrender's
``OffscreenRenderer``. The resulting RGB image is drawn into the host
matplotlib axes via ``ax.imshow``.

Only the joints relevant to our 14D state are driven:
  * waist_yaw_joint        ← torso_yaw
  * left/right_shoulder_yaw_joint, left/right_elbow_joint  ← arm_tuck_L/R
All other joints stay at default angle = 0.

Body height (crouch) is rendered as a root-z translation; legs do not
bend. This was an explicit design choice — see backends/__init__.py.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

from ..scene_ir import RobotPoseIR, SceneIR


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
# backends -> visualizations -> plugins -> experiments -> genedynamics -> repo
REPO_ROOT = _THIS.parents[5]
THIRD_PARTY = REPO_ROOT / "third_party" / "mujoco_menagerie"
G1_DIR = THIRD_PARTY / "unitree_g1"
GO2_DIR = THIRD_PARTY / "unitree_go2"
G1_XML = G1_DIR / "g1.xml"
GO2_XML = GO2_DIR / "go2.xml"

# Joints we drive from a RobotPoseIR (G1).
# Each entry maps joint_name -> (pose_field, scale).
G1_DRIVEN: Dict[str, Tuple[str, float]] = {
    "waist_yaw_joint":         ("torso_yaw", 1.0),
    "left_shoulder_yaw_joint": ("arm_tuck_L", -1.30),
    "right_shoulder_yaw_joint":("arm_tuck_R",  1.30),
    "left_elbow_joint":        ("arm_tuck_L",  1.65),
    "right_elbow_joint":       ("arm_tuck_R",  1.65),
}
G1_NOMINAL_PELVIS_Z = 0.793  # default standing pelvis height
G1_NOMINAL_BODY_H   = 0.75   # H_NOMINAL from humanoid_corridor_2d


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def _quat_xyzw_from_wxyz(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert mujoco-style (w,x,y,z) quaternion to 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _axis_angle_to_mat(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    x, y, z = a
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _parse_floats(s: Optional[str], n: int, default) -> np.ndarray:
    if s is None:
        return np.asarray(default, dtype=np.float64)
    parts = [float(p) for p in s.split()]
    if len(parts) != n:
        return np.asarray(default, dtype=np.float64)
    return np.asarray(parts, dtype=np.float64)


# ---------------------------------------------------------------------------
# MuJoCo XML kinematics parser
# ---------------------------------------------------------------------------
class MJBody:
    __slots__ = ("name", "parent", "pos", "quat", "joint_axis",
                 "joint_name", "geoms")

    def __init__(self, name, parent, pos, quat):
        self.name = name
        self.parent = parent  # parent body name or None
        self.pos = pos        # (3,)
        self.quat = quat      # (4,) wxyz
        self.joint_axis: Optional[np.ndarray] = None
        self.joint_name: Optional[str] = None
        # geoms: list of dicts {mesh, pos, quat, material}
        self.geoms: List[Dict[str, Any]] = []


class MJModel:
    """Lightweight reading of mujoco XML for offline rendering only."""
    def __init__(self, xml_path: Path):
        self.xml_path = Path(xml_path)
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # mesh dir
        compiler = root.find("compiler")
        meshdir = "."
        if compiler is not None and compiler.get("meshdir"):
            meshdir = compiler.get("meshdir")
        self.mesh_dir = (self.xml_path.parent / meshdir).resolve()

        # mesh table: name -> file
        self.meshes: Dict[str, Path] = {}
        asset = root.find("asset")
        if asset is not None:
            for m in asset.findall("mesh"):
                file = m.get("file")
                if file is None:
                    continue
                name = m.get("name") or Path(file).stem
                self.meshes[name] = (self.mesh_dir / file).resolve()
            # materials
            self.materials: Dict[str, Tuple[float, float, float, float]] = {}
            for mat in asset.findall("material"):
                rgba = _parse_floats(mat.get("rgba"), 4, [0.7, 0.7, 0.7, 1.0])
                self.materials[mat.get("name", "")] = tuple(rgba.tolist())
        else:
            self.materials = {}

        # body tree
        self.bodies: Dict[str, MJBody] = {}
        self.body_order: List[str] = []
        wb = root.find("worldbody")
        if wb is None:
            raise ValueError(f"{xml_path} has no <worldbody>")
        for child in wb.findall("body"):
            self._walk_body(child, parent_name=None)

    def _walk_body(self, elem, parent_name):
        name = elem.get("name") or f"body_{len(self.body_order)}"
        pos = _parse_floats(elem.get("pos"), 3, [0, 0, 0])
        quat = _parse_floats(elem.get("quat"), 4, [1, 0, 0, 0])
        body = MJBody(name=name, parent=parent_name, pos=pos, quat=quat)

        # joint (hinge only — freejoint and ball ignored / treated as fixed)
        for j in elem.findall("joint"):
            jaxis = _parse_floats(j.get("axis"), 3, [0, 0, 1])
            body.joint_axis = jaxis
            body.joint_name = j.get("name")
            break  # only first hinge per body
        # geoms (visual)
        for g in elem.findall("geom"):
            mesh = g.get("mesh")
            if mesh is None:
                continue  # skip non-mesh geoms (sphere/box collision)
            cls = g.get("class") or ""
            # Skip explicit collision geoms; class "visual" or unset is fine
            if cls == "collision":
                continue
            gpos = _parse_floats(g.get("pos"), 3, [0, 0, 0])
            gquat = _parse_floats(g.get("quat"), 4, [1, 0, 0, 0])
            mat = g.get("material")
            body.geoms.append({
                "mesh": mesh,
                "pos": gpos,
                "quat": gquat,
                "material": mat,
            })
        self.bodies[name] = body
        self.body_order.append(name)
        for child in elem.findall("body"):
            self._walk_body(child, parent_name=name)


# Cache parsed models
_MODEL_CACHE: Dict[str, MJModel] = {}
_MESH_CACHE: Dict[str, trimesh.Trimesh] = {}


def _get_model(path: Path) -> MJModel:
    key = str(path)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = MJModel(path)
    return _MODEL_CACHE[key]


def _get_mesh(path: Path) -> trimesh.Trimesh:
    key = str(path)
    if key not in _MESH_CACHE:
        m = trimesh.load(str(path), force="mesh")
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate([
                g for g in m.geometry.values()
            ])
        _MESH_CACHE[key] = m
    return _MESH_CACHE[key]


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------
def _compute_world_transforms(
    model: MJModel,
    joint_angles: Dict[str, float],
    root_T: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Walk the body tree and produce {body_name: 4x4 world transform}."""
    out: Dict[str, np.ndarray] = {}

    def _visit(name: str, parent_T: np.ndarray):
        body = model.bodies[name]
        R_local = _quat_to_mat(body.quat)
        T_static = _make_T(R_local, body.pos)

        # Apply joint angle (rotation about axis at body origin)
        if body.joint_axis is not None and body.joint_name:
            theta = float(joint_angles.get(body.joint_name, 0.0))
            R_j = _axis_angle_to_mat(body.joint_axis, theta)
            T_joint = np.eye(4)
            T_joint[:3, :3] = R_j
        else:
            T_joint = np.eye(4)

        T_world = parent_T @ T_static @ T_joint
        out[name] = T_world
        # Recurse into children
        for child_name in model.body_order:
            if model.bodies[child_name].parent == name:
                _visit(child_name, T_world)

    # Root bodies have parent=None
    for name in model.body_order:
        if model.bodies[name].parent is None:
            _visit(name, root_T)
    return out


# ---------------------------------------------------------------------------
# Robot scene builder
# ---------------------------------------------------------------------------
def _tint_rgba(
    rgba: Tuple[float, float, float, float],
    tint: Optional[Tuple[float, float, float]],
    alpha_mul: float = 1.0,
    tint_strength: float = 0.75,
) -> Tuple[float, float, float, float]:
    """Blend the base color with the tint, biasing strongly toward the tint
    so dark base materials (e.g. Go2 black) still read as colored ghosts.

    Also lifts the result toward `tint` if the base is very dark, so the
    user can actually see the rainbow palette.
    """
    if tint is None:
        return (rgba[0], rgba[1], rgba[2],
                float(np.clip(rgba[3] * alpha_mul, 0.0, 1.0)))
    base_lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    # Brighter base → lean on tint less; darker base → lean on tint more.
    s = float(np.clip(tint_strength + (1.0 - base_lum) * 0.20, 0.0, 0.95))
    return (
        float(np.clip(rgba[0] * (1 - s) + tint[0] * s, 0.0, 1.0)),
        float(np.clip(rgba[1] * (1 - s) + tint[1] * s, 0.0, 1.0)),
        float(np.clip(rgba[2] * (1 - s) + tint[2] * s, 0.0, 1.0)),
        float(np.clip(rgba[3] * alpha_mul, 0.0, 1.0)),
    )


def _ghost_palette(n: int) -> List[Tuple[float, float, float]]:
    """Rainbow palette using matplotlib's `viridis` for N ghosts."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap("turbo")
    return [tuple(cmap(i / max(n - 1, 1))[:3]) for i in range(n)]


def _build_g1_nodes(
    pose: RobotPoseIR,
    base_color_override: Optional[Tuple[float, float, float, float]] = None,
    tint: Optional[Tuple[float, float, float]] = None,
) -> List[Tuple[trimesh.Trimesh, np.ndarray, Tuple[float, float, float, float]]]:
    """Build a list of (mesh, world_transform, rgba) for one G1 pose."""
    model = _get_model(G1_XML)

    # Joint angles from pose
    pose_dict = {
        "torso_yaw": float(pose.torso_yaw),
        "arm_tuck_L": float(pose.arm_tuck_L),
        "arm_tuck_R": float(pose.arm_tuck_R),
    }
    joint_angles: Dict[str, float] = {}
    for jname, (field, scale) in G1_DRIVEN.items():
        joint_angles[jname] = pose_dict[field] * scale

    # Root pose: place pelvis so feet rest on z=0 even when crouching.
    # Pelvis nominal z = 0.793. Crouch lowers pelvis by (H_NOMINAL - body_h).
    crouch = max(0.0, G1_NOMINAL_BODY_H - float(pose.body_height))
    pelvis_z = G1_NOMINAL_PELVIS_Z - crouch * 1.0
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    R_root = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    # The G1 model places pelvis at world (0,0,0.793) by default; we want
    # to translate the whole robot to (x, y, pelvis_z - 0.793) and rotate
    # about z by yaw. Build root_T.
    root_T = _make_T(R_root, np.array([pose.x, pose.y, pelvis_z - G1_NOMINAL_PELVIS_Z],
                                      dtype=np.float64))

    world_T = _compute_world_transforms(model, joint_angles, root_T)

    out = []
    default_rgba = (0.85, 0.85, 0.88, 1.0)
    for body_name in model.body_order:
        body = model.bodies[body_name]
        T_b = world_T[body_name]
        for g in body.geoms:
            mesh_name = g["mesh"]
            mesh_path = model.meshes.get(mesh_name)
            if mesh_path is None or not mesh_path.exists():
                continue
            mesh = _get_mesh(mesh_path).copy()
            R_g = _quat_to_mat(g["quat"])
            T_geom = _make_T(R_g, g["pos"])
            T_world = T_b @ T_geom
            mat_name = g.get("material") or ""
            rgba = model.materials.get(mat_name, default_rgba)
            if base_color_override is not None:
                rgba = base_color_override
            elif tint is not None:
                rgba = _tint_rgba(rgba, tint)
            out.append((mesh, T_world, rgba))
    return out


def _build_go2_nodes(
    pose: RobotPoseIR,
    tint: Optional[Tuple[float, float, float]] = None,
) -> List[Tuple[trimesh.Trimesh, np.ndarray, Tuple[float, float, float, float]]]:
    """Build a Go2 quadruped at the given (x,y,yaw=0). Default pose, no leg drive."""
    model = _get_model(GO2_XML)
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    R_root = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    # Go2 base nominal z = 0.445
    root_T = _make_T(R_root, np.array([pose.x, pose.y, 0.0], dtype=np.float64))
    world_T = _compute_world_transforms(model, {}, root_T)

    out = []
    default_rgba = (0.18, 0.18, 0.18, 1.0)
    for name in model.body_order:
        body = model.bodies[name]
        T_b = world_T[name]
        for g in body.geoms:
            mesh_name = g["mesh"]
            mesh_path = model.meshes.get(mesh_name)
            if mesh_path is None or not mesh_path.exists():
                continue
            mesh = _get_mesh(mesh_path).copy()
            R_g = _quat_to_mat(g["quat"])
            T_geom = _make_T(R_g, g["pos"])
            T_world = T_b @ T_geom
            mat_name = g.get("material") or ""
            rgba = model.materials.get(mat_name, default_rgba)
            if tint is not None:
                rgba = _tint_rgba(rgba, tint)
            out.append((mesh, T_world, rgba))
    return out


# ---------------------------------------------------------------------------
# Procedural textures
# ---------------------------------------------------------------------------
def _make_brick_texture(
    width: int = 512, height: int = 512,
    brick_w: int = 96, brick_h: int = 36,
    mortar: int = 5,
    base=(218, 198, 170), grout=(110, 100, 88),
) -> Image.Image:
    img = Image.new("RGB", (width, height), grout)
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(7)
    rows = height // brick_h + 2
    cols = width // brick_w + 2
    for r in range(rows):
        offset = (brick_w // 2) if r % 2 == 0 else 0
        y0 = r * brick_h + mortar
        y1 = y0 + brick_h - 2 * mortar
        for c in range(-1, cols):
            x0 = c * brick_w + offset + mortar
            x1 = x0 + brick_w - 2 * mortar
            jitter = rng.integers(-15, 15, size=3)
            color = tuple(int(np.clip(base[i] + jitter[i], 0, 255)) for i in range(3))
            draw.rectangle([x0, y0, x1, y1], fill=color)
    img = img.filter(ImageFilter.SMOOTH)
    return img


def _make_tile_texture(
    width: int = 512, height: int = 512,
    tile: int = 64, grout: int = 4,
    base=(180, 180, 195), grout_color=(70, 70, 80),
) -> Image.Image:
    img = Image.new("RGB", (width, height), grout_color)
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(11)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            jitter = rng.integers(-12, 12, size=3)
            color = tuple(int(np.clip(base[i] + jitter[i], 0, 255)) for i in range(3))
            draw.rectangle(
                [x + grout, y + grout, x + tile - grout, y + tile - grout],
                fill=color,
            )
    return img


def _make_concrete_texture(
    width: int = 512, height: int = 512,
    base=(190, 190, 195),
) -> Image.Image:
    rng = np.random.default_rng(3)
    arr = np.full((height, width, 3), base, dtype=np.int16)
    noise = rng.integers(-25, 25, size=(height, width, 1))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _texture_material(img: Image.Image) -> trimesh.visual.material.PBRMaterial:
    return trimesh.visual.material.PBRMaterial(
        baseColorTexture=img,
        metallicFactor=0.0,
        roughnessFactor=0.85,
    )


# ---------------------------------------------------------------------------
# Scene builders (corridor + stepping)
# ---------------------------------------------------------------------------
def _box_with_uvs(extents, transform=None) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=extents, transform=transform)
    # Compute simple planar UVs based on xy of vertices for top texturing
    v = box.vertices
    uvs = np.zeros((v.shape[0], 2), dtype=np.float32)
    uvs[:, 0] = (v[:, 0] - v[:, 0].min()) / max(v[:, 0].ptp(), 1e-6)
    uvs[:, 1] = (v[:, 1] - v[:, 1].min()) / max(v[:, 1].ptp(), 1e-6)
    box.visual = trimesh.visual.TextureVisuals(uv=uvs)
    return box


def _wall_with_uvs(p0, p1, height, thickness, repeat=(2, 1)) -> trimesh.Trimesh:
    """Build a thin wall slab between two xy points (z = 0..height)."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return trimesh.Trimesh()
    direction /= length
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    half_t = 0.5 * thickness
    base = []
    for sign in (-1, 1):
        for end_pt in (p0, p1):
            base.append(end_pt + sign * half_t * normal)
    # Vertices: 0,1 bottom front; 2,3 bottom back; then top
    v = []
    for z in (0.0, height):
        for bp in base:
            v.append([bp[0], bp[1], z])
    v = np.asarray(v, dtype=np.float64)
    # Faces (12)
    f = np.array([
        [0, 1, 5], [0, 5, 4],   # -normal side (front)
        [1, 3, 7], [1, 7, 5],   # +y end
        [3, 2, 6], [3, 6, 7],   # +normal side (back)
        [2, 0, 4], [2, 4, 6],   # -y end
        [4, 5, 7], [4, 7, 6],   # top
        [0, 2, 3], [0, 3, 1],   # bottom
    ], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    # UV: tile by length on the long faces
    uvs = np.zeros((v.shape[0], 2), dtype=np.float32)
    for i in range(v.shape[0]):
        u = np.dot(v[i, :2] - p0, direction) / max(length, 1e-6) * repeat[0]
        w = (v[i, 2] / max(height, 1e-6)) * repeat[1]
        uvs[i] = (u, w)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    return mesh


def _floor_with_uvs(xmin, xmax, ymin, ymax, repeat=(4, 2), z=0.0) -> trimesh.Trimesh:
    v = np.array([
        [xmin, ymin, z], [xmax, ymin, z],
        [xmax, ymax, z], [xmin, ymax, z],
    ], dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    uvs = np.array([
        [0.0, 0.0], [repeat[0], 0.0],
        [repeat[0], repeat[1]], [0.0, repeat[1]],
    ], dtype=np.float32)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    return mesh


# ---------------------------------------------------------------------------
# Lazy pyrender import (heavy)
# ---------------------------------------------------------------------------
def _import_pyrender():
    import pyrender  # noqa
    return pyrender


# ---------------------------------------------------------------------------
# Render entry points
# ---------------------------------------------------------------------------
def _make_pyrender_mesh(tm: trimesh.Trimesh, rgba=None, smooth=True):
    pyrender = _import_pyrender()
    if rgba is not None and not isinstance(tm.visual, trimesh.visual.TextureVisuals):
        tm = tm.copy()
        tm.visual = trimesh.visual.ColorVisuals(
            mesh=tm,
            face_colors=np.tile(np.array(rgba, dtype=np.float64), (len(tm.faces), 1)),
        )
    return pyrender.Mesh.from_trimesh(tm, smooth=smooth)


def _frame_camera(
    bounds: Tuple[float, float, float, float],
    *,
    pitch_deg: float = 58.0,
    yfov_deg: float = 38.0,
    aspect: float = 16 / 9,
    margin: float = 1.20,
    z_target: float = 0.4,
    z_room: float = 2.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (eye, target) for a high-oblique top-center camera that
    frames the entire xy bounds plus some headroom for tall obstacles.

    pitch_deg = 90 means straight-down. 0 means horizontal. We aim for ~58.
    """
    xmin, xmax, ymin, ymax = bounds
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    length_x = xmax - xmin
    length_y = ymax - ymin

    pitch = math.radians(pitch_deg)
    yfov = math.radians(yfov_deg)
    xfov = 2.0 * math.atan(math.tan(yfov / 2.0) * aspect)

    # Required distance so that the half-extent fits in each FOV cone.
    half_x = 0.5 * length_x * margin
    half_y = 0.5 * length_y * margin
    half_z = 0.5 * z_room

    # When looking at the floor at pitch p, the floor projects with
    # foreshortening factor sin(p) along the camera-up axis (~y in world).
    # To be safe, use the larger of length_x / xfov and length_y / yfov / sin(p).
    d_x = half_x / max(math.tan(xfov / 2.0), 1e-3)
    d_y = (half_y * math.sin(pitch) + half_z * math.cos(pitch)) \
          / max(math.tan(yfov / 2.0), 1e-3)
    d = max(d_x, d_y) + 0.5

    eye = np.array([
        cx,
        cy - d * math.cos(pitch),
        z_target + d * math.sin(pitch),
    ], dtype=np.float64)
    target = np.array([cx, cy, z_target], dtype=np.float64)
    return eye, target


def _add_lights_and_camera(
    pyr_scene,
    cam_eye: np.ndarray,
    cam_target: np.ndarray,
    yfov_deg: float = 35.0,
    aspect: float = 16 / 9,
):
    pyrender = _import_pyrender()
    # Camera pose: look-at
    forward = (cam_target - cam_eye)
    forward /= max(np.linalg.norm(forward), 1e-9)
    up = np.array([0, 0, 1], dtype=np.float64)
    right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, forward)
    R = np.column_stack([right, up, -forward])
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = R
    cam_pose[:3, 3] = cam_eye

    cam = pyrender.PerspectiveCamera(yfov=math.radians(yfov_deg), aspectRatio=aspect)
    pyr_scene.add(cam, pose=cam_pose)

    # Key directional light from upper right behind camera
    key = pyrender.DirectionalLight(color=np.ones(3), intensity=4.5)
    key_pose = np.eye(4)
    key_pose[:3, 3] = cam_eye + np.array([0, 0, 4.0])
    # rotate to look down
    key_pose[:3, :3] = _axis_angle_to_mat([1, 0, 0], -math.radians(45))
    pyr_scene.add(key, pose=key_pose)

    # Soft ambient via a second weaker dir light from opposite
    fill = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    fill_pose = np.eye(4)
    fill_pose[:3, 3] = cam_target + np.array([0, -4.0, 4.0])
    fill_pose[:3, :3] = _axis_angle_to_mat([1, 0, 0], -math.radians(60))
    pyr_scene.add(fill, pose=fill_pose)


def _render_offscreen(pyr_scene, width: int, height: int) -> np.ndarray:
    pyrender = _import_pyrender()
    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, _ = r.render(pyr_scene)
    finally:
        r.delete()
    return color


def _build_ribbon(
    points: np.ndarray,
    *,
    radius: float = 0.025,
    sections: int = 12,
    arrow_len: float = 0.18,
    arrow_radius: float = 0.07,
) -> trimesh.Trimesh:
    """Build a tube along the given (N,3) polyline + a cone arrow at the tip."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    parts: List[trimesh.Trimesh] = []
    for i in range(pts.shape[0] - 1):
        a, b = pts[i], pts[i + 1]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=L, sections=sections)
        # cylinder is along +z; rotate to align with d
        z = np.array([0, 0, 1.0])
        axis = np.cross(z, d / L)
        sin_a = float(np.linalg.norm(axis))
        cos_a = float(np.dot(z, d / L))
        if sin_a < 1e-9:
            R = np.eye(3) if cos_a > 0 else _axis_angle_to_mat([1, 0, 0], math.pi)
        else:
            R = _axis_angle_to_mat(axis / sin_a, math.atan2(sin_a, cos_a))
        T = _make_T(R, 0.5 * (a + b))
        cyl.apply_transform(T)
        parts.append(cyl)
    # Arrow head
    if pts.shape[0] >= 2:
        a, b = pts[-2], pts[-1]
        d = b - a
        L = float(np.linalg.norm(d))
        if L > 1e-6:
            tip_dir = d / L
            cone = trimesh.creation.cone(radius=arrow_radius, height=arrow_len,
                                         sections=sections)
            z = np.array([0, 0, 1.0])
            axis = np.cross(z, tip_dir)
            sin_a = float(np.linalg.norm(axis))
            cos_a = float(np.dot(z, tip_dir))
            if sin_a < 1e-9:
                R = np.eye(3) if cos_a > 0 else _axis_angle_to_mat([1, 0, 0], math.pi)
            else:
                R = _axis_angle_to_mat(axis / sin_a, math.atan2(sin_a, cos_a))
            # Place cone base at b, tip extending forward by arrow_len
            T = _make_T(R, b)
            cone.apply_transform(T)
            parts.append(cone)
    if not parts:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(parts)


def _add_trimesh_to_pyrender(pyr_scene, tm: trimesh.Trimesh,
                             T: Optional[np.ndarray] = None,
                             rgba=None, smooth=True):
    pyrender = _import_pyrender()
    use_smooth = smooth
    if rgba is not None:
        # Force-replace visual so face_colors actually apply, even for
        # meshes loaded with TextureVisuals (.obj with materials).
        tm = tm.copy()
        col = np.tile(np.array(rgba, dtype=np.float64), (len(tm.faces), 1))
        tm.visual = trimesh.visual.ColorVisuals(mesh=tm, face_colors=col)
        use_smooth = False  # face colors require flat shading
    mesh = pyrender.Mesh.from_trimesh(tm, smooth=use_smooth)
    pyr_scene.add(mesh, pose=T if T is not None else np.eye(4))


# ---------------------------------------------------------------------------
# Public renderers
# ---------------------------------------------------------------------------
def render_corridor_pyrender(
    fig: Any,
    ax,
    scene: SceneIR,
    poses: List[RobotPoseIR],
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    extras = extras or {}
    pyrender = _import_pyrender()

    width = int(extras.get("render_width", 1280))
    height = int(extras.get("render_height", 720))
    n_ghosts = int(extras.get("n_ghosts", 6))

    xmin, xmax, ymin, ymax = scene.bounds
    # Wall height: keep low enough that the front wall doesn't dominate the
    # high-oblique frame. 1.6 m is taller than every G1 ghost (~0.8) but well
    # below the top of the floor in screen space.
    z_room = float(extras.get("room_height", 1.6))

    pyr_scene = pyrender.Scene(
        bg_color=np.array([0.92, 0.93, 0.96, 1.0]),
        ambient_light=np.array([0.35, 0.35, 0.38]),
    )

    # Floor (concrete)
    floor_tex = _make_concrete_texture()
    floor_mesh = _floor_with_uvs(
        xmin - 0.5, xmax + 0.5, ymin - 0.5, ymax + 0.5,
        repeat=((xmax - xmin) / 1.5, (ymax - ymin) / 1.5),
    )
    floor_mesh.visual.material = _texture_material(floor_tex)
    _add_trimesh_to_pyrender(pyr_scene, floor_mesh)

    # Walls (brick). All 4 walls are drawn so the room reads as fully
    # enclosed, but the camera-side front wall (y=ymin) is rendered as a
    # low guardrail so it does not occlude the floor in a top-mid view.
    brick_tex = _make_brick_texture()
    wall_thick = 0.10
    wall_h = z_room
    front_h = float(extras.get("front_wall_height", 0.40))
    wall_specs = [
        # (p0, p1, height) — front wall is the low one
        ((xmin, ymin), (xmax, ymin), front_h),
        ((xmin, ymax), (xmax, ymax), wall_h),
        ((xmin, ymin), (xmin, ymax), wall_h),
        ((xmax, ymin), (xmax, ymax), wall_h),
    ]
    for p0, p1, h in wall_specs:
        seg_len = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        w = _wall_with_uvs(p0, p1, h, wall_thick,
                           repeat=(max(seg_len / 1.0, 0.5),
                                   max(h / 1.5, 0.5)))
        w.visual.material = _texture_material(brick_tex)
        _add_trimesh_to_pyrender(pyr_scene, w)

    # Obstacle styling — tall (full-height) walls get heavy alpha drop so
    # G1 ghosts behind them remain visible.
    tall_alpha = float(extras.get("tall_obstacle_alpha", 0.35))
    short_alpha = float(extras.get("short_obstacle_alpha", 0.85))

    def _obstacle_rgba(z_lo: float, z_hi: float) -> Tuple[float, float, float, float]:
        if z_lo <= 0.05 and z_hi >= 1.8:
            return (0.30, 0.34, 0.42, tall_alpha)
        if z_lo > 0.4:
            return (0.85, 0.30, 0.30, short_alpha)
        if z_hi < 0.5:
            return (0.85, 0.50, 0.20, short_alpha)
        return (0.55, 0.40, 0.75, short_alpha)

    # Box obstacles
    for b in scene.boxes:
        if b.shape != "box":
            continue
        cx = 0.5 * (b.x_min + b.x_max)
        cy = 0.5 * (b.y_min + b.y_max)
        cz = 0.5 * (b.z_min + b.z_max)
        ext = (b.x_max - b.x_min, b.y_max - b.y_min, b.z_max - b.z_min)
        T = _make_T(np.eye(3), np.array([cx, cy, cz], dtype=np.float64))
        bm = trimesh.creation.box(extents=ext)
        _add_trimesh_to_pyrender(pyr_scene, bm, T=T,
                                 rgba=_obstacle_rgba(b.z_min, b.z_max))

    # Sphere obstacles
    for s in scene.spheres:
        cz = 0.5 * (s.z_min + s.z_max)
        sm = trimesh.creation.uv_sphere(radius=s.radius, count=[24, 16])
        T = _make_T(np.eye(3), np.array([s.cx, s.cy, cz], dtype=np.float64))
        _add_trimesh_to_pyrender(pyr_scene, sm, T=T,
                                 rgba=_obstacle_rgba(s.z_min, s.z_max))

    # Quarter-circle obstacles → approximate as a box (visual is small)
    for qc in scene.quarter_circles:
        x_min = qc.cx - qc.radius if qc.clip_sign > 0 else qc.cx
        x_max = qc.cx if qc.clip_sign > 0 else qc.cx + qc.radius
        y_min = qc.cy - qc.radius if qc.cy > 0 else qc.cy
        y_max = qc.cy if qc.cy > 0 else qc.cy + qc.radius
        ext = (x_max - x_min, y_max - y_min, qc.z_max - qc.z_min)
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        cz = 0.5 * (qc.z_min + qc.z_max)
        T = _make_T(np.eye(3), np.array([cx, cy, cz], dtype=np.float64))
        bm = trimesh.creation.box(extents=ext)
        _add_trimesh_to_pyrender(pyr_scene, bm, T=T,
                                 rgba=_obstacle_rgba(qc.z_min, qc.z_max))

    # Trajectory ribbon (tube + arrow tip)
    if poses and len(poses) >= 2:
        traj_pts = np.array([[p.x, p.y, 0.04] for p in poses])
        ribbon = _build_ribbon(
            traj_pts,
            radius=float(extras.get("ribbon_radius", 0.035)),
            arrow_len=float(extras.get("ribbon_arrow_len", 0.20)),
            arrow_radius=float(extras.get("ribbon_arrow_radius", 0.09)),
        )
        if len(ribbon.faces) > 0:
            _add_trimesh_to_pyrender(pyr_scene, ribbon,
                                     rgba=(0.18, 0.50, 0.95, 1.0))

    # G1 ghosts with rainbow tint along the trajectory
    T_total = len(poses)
    if T_total > 0 and n_ghosts > 0:
        step = max(1, T_total // n_ghosts)
        indices = list(range(0, T_total, step))
        if indices[-1] != T_total - 1:
            indices.append(T_total - 1)
        palette = _ghost_palette(len(indices))
        for k, t in enumerate(indices):
            for tm, T_w, rgba in _build_g1_nodes(poses[t], tint=palette[k]):
                _add_trimesh_to_pyrender(pyr_scene, tm, T=T_w, rgba=rgba)

    # Camera: high oblique top-center frame that covers the whole corridor
    # plus enough perspective to read 3D depth. Default 62° pitch leans
    # noticeably top-down while still showing side walls. yaml-overridable.
    pitch_deg = float(extras.get("camera_pitch_deg", 62.0))
    yfov_deg = float(extras.get("camera_yfov_deg", 36.0))
    aspect = width / max(height, 1)
    if "camera_eye" in extras and "camera_target" in extras:
        eye = np.asarray(extras["camera_eye"], dtype=np.float64)
        target = np.asarray(extras["camera_target"], dtype=np.float64)
    else:
        eye, target = _frame_camera(
            scene.bounds,
            pitch_deg=pitch_deg,
            yfov_deg=yfov_deg,
            aspect=aspect,
            margin=1.15,
            z_target=0.5,
            z_room=z_room,
        )
    _add_lights_and_camera(pyr_scene, eye, target, yfov_deg=yfov_deg, aspect=aspect)

    color = _render_offscreen(pyr_scene, width, height)

    ax.imshow(color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Corridor Trajectory (pyrender 3D)", fontsize=10)


def render_stepping_pyrender(
    fig: Any,
    ax,
    scene: SceneIR,
    poses: List[RobotPoseIR],
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    extras = extras or {}
    pyrender = _import_pyrender()

    width = int(extras.get("render_width", 1280))
    height = int(extras.get("render_height", 720))

    xmin, xmax, ymin, ymax = scene.bounds

    pyr_scene = pyrender.Scene(
        bg_color=np.array([0.78, 0.86, 0.94, 1.0]),
        ambient_light=np.array([0.42, 0.42, 0.45]),
    )

    # Tiled stone floor
    tile_tex = _make_tile_texture()
    floor_mesh = _floor_with_uvs(
        xmin - 0.5, xmax + 0.5, ymin - 0.6, ymax + 0.6,
        repeat=((xmax - xmin) / 0.8, (ymax - ymin) / 0.8),
        z=-0.02,
    )
    floor_mesh.visual.material = _texture_material(tile_tex)
    _add_trimesh_to_pyrender(pyr_scene, floor_mesh)

    # River (semi-transparent flat polygon)
    if scene.river is not None:
        rx0, rx1, ry0, ry1 = scene.river
        river_mesh = trimesh.creation.box(extents=(rx1 - rx0, ry1 - ry0, 0.02))
        T = _make_T(np.eye(3), np.array([0.5 * (rx0 + rx1), 0.5 * (ry0 + ry1), 0.0]))
        _add_trimesh_to_pyrender(pyr_scene, river_mesh, T=T,
                                 rgba=(0.45, 0.65, 0.85, 0.85))

    # Platforms as low concrete pads
    for p in scene.platforms:
        ext = (p.x_max - p.x_min, p.y_max - p.y_min, 0.10)
        T = _make_T(np.eye(3),
                    np.array([0.5 * (p.x_min + p.x_max),
                              0.5 * (p.y_min + p.y_max),
                              0.05]))
        bm = trimesh.creation.box(extents=ext)
        _add_trimesh_to_pyrender(pyr_scene, bm, T=T, rgba=(0.55, 0.58, 0.60, 1.0))

    # Stones as short cylinders
    for st in scene.stones:
        cyl = trimesh.creation.cylinder(radius=st.radius, height=0.10, sections=24)
        T = _make_T(np.eye(3), np.array([st.cx, st.cy, 0.05]))
        _add_trimesh_to_pyrender(pyr_scene, cyl, T=T, rgba=(0.55, 0.55, 0.58, 1.0))

    # Trajectory ribbon (tube + arrow)
    if poses and len(poses) >= 2:
        traj_pts = np.array([[p.x, p.y, 0.16] for p in poses])
        ribbon = _build_ribbon(
            traj_pts,
            radius=float(extras.get("ribbon_radius", 0.030)),
            arrow_len=float(extras.get("ribbon_arrow_len", 0.18)),
            arrow_radius=float(extras.get("ribbon_arrow_radius", 0.085)),
        )
        if len(ribbon.faces) > 0:
            _add_trimesh_to_pyrender(pyr_scene, ribbon,
                                     rgba=(0.18, 0.50, 0.95, 1.0))

    # Go2 ghosts along the trajectory, with rainbow tint.
    n_ghosts = int(extras.get("n_ghosts", 5))
    T_total = len(poses)
    if T_total > 0 and n_ghosts > 0:
        step = max(1, T_total // n_ghosts)
        indices = list(range(0, T_total, step))
        if indices[-1] != T_total - 1:
            indices.append(T_total - 1)
        palette = _ghost_palette(len(indices))
        for k, t in enumerate(indices):
            # Yaw from forward direction (next-prev body delta) so the dog
            # actually faces along the path. Fall back to 0.
            if t < T_total - 1:
                dx = poses[t + 1].x - poses[t].x
                dy = poses[t + 1].y - poses[t].y
            elif t > 0:
                dx = poses[t].x - poses[t - 1].x
                dy = poses[t].y - poses[t - 1].y
            else:
                dx, dy = 1.0, 0.0
            ghost_pose = RobotPoseIR(
                x=poses[t].x, y=poses[t].y,
                yaw=math.atan2(dy, dx),
            )
            for tm, T_w, rgba in _build_go2_nodes(ghost_pose, tint=palette[k]):
                _add_trimesh_to_pyrender(pyr_scene, tm, T=T_w, rgba=rgba)

    # Camera: high oblique top-center. Default 65° pitch leans further
    # top-down for a more map-like view; yaml-overridable.
    pitch_deg = float(extras.get("camera_pitch_deg", 65.0))
    yfov_deg = float(extras.get("camera_yfov_deg", 38.0))
    aspect = width / max(height, 1)
    if "camera_eye" in extras and "camera_target" in extras:
        eye = np.asarray(extras["camera_eye"], dtype=np.float64)
        target = np.asarray(extras["camera_target"], dtype=np.float64)
    else:
        eye, target = _frame_camera(
            scene.bounds,
            pitch_deg=pitch_deg,
            yfov_deg=yfov_deg,
            aspect=aspect,
            margin=1.20,
            z_target=0.15,
            z_room=0.6,
        )
    _add_lights_and_camera(pyr_scene, eye, target, yfov_deg=yfov_deg, aspect=aspect)

    color = _render_offscreen(pyr_scene, width, height)

    ax.imshow(color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Stepping Stones Trajectory (pyrender 3D)", fontsize=10)
