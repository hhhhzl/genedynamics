"""Render the panel (e) Pose-OOD closeups from a horizontal camera.

The published frames look down on the socket. This keeps the same saved
execution states and only changes the camera to eye level.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from genedynamics.robots.panda.profile import panda_profile

_composition_path = ROOT / "genedynamics/envs/composition.py"
_comp_spec = importlib.util.spec_from_file_location("mga_composition", _composition_path)
_composition = importlib.util.module_from_spec(_comp_spec)
sys.modules["mga_composition"] = _composition
_comp_spec.loader.exec_module(_composition)
MjcfSceneComposer = _composition.MjcfSceneComposer
PegToolSpec = _composition.PegToolSpec
SocketSpec = _composition.SocketSpec

_scene_path = Path(__file__).resolve().parent / "render_mechanism_scenes.py"
_spec = importlib.util.spec_from_file_location("mga_render_mechanism_scenes", _scene_path)
_scenes = importlib.util.module_from_spec(_spec)
sys.modules["mga_render_mechanism_scenes"] = _scenes
_spec.loader.exec_module(_scenes)
_manuscript_sky = _scenes._manuscript_sky
_restore_panda_visuals = _scenes._restore_panda_visuals

import mujoco


OUT = Path(__file__).resolve().parent / "output" / "e_eyelevel"
TRAJ = (
    ROOT
    / "peg_insert_completed_20260926/results/arm/peg_insert/main/mga"
    / "level_ood_pose/seed_0/trajectory/trajectory.json"
)


def _rotation_xyz(angles):
    rx, ry, rz = (float(x) for x in angles)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rxm = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float64)
    rym = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float64)
    rzm = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float64)
    return rzm @ rym @ rxm


def _quat(rotation):
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, np.float64).reshape(-1))
    return quat


def _physics():
    """Match the saved ood_pose seed-0 execution environment."""
    rng = np.random.default_rng(0 + 4109)
    clearance = float(rng.uniform(0.0015, 0.0015))
    friction = float(rng.uniform(0.6, 0.6))
    position = rng.uniform(-0.001, 0.001, size=2)
    angles = rng.uniform(-0.0122173, 0.0122173, size=3)
    time_constant = float(rng.uniform(0.03, 0.03))
    return {
        "clearance": clearance,
        "friction": friction,
        "position": position,
        "angles": angles,
        "solref": f"{time_constant:.7g} 1",
    }


def _home_tool_pose(profile):
    model = mujoco.MjModel.from_xml_path(profile.model_path())
    data = mujoco.MjData(model)
    home = np.asarray(profile.controller_defaults["home_qpos"], np.float64)
    data.qpos[: home.size] = home
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    return np.asarray(data.site_xpos[sid]), np.asarray(data.site_xmat[sid]).reshape(3, 3)


def build_model():
    profile = panda_profile()
    physics = _physics()
    root_pos, root_rot = _home_tool_pose(profile)
    peg = PegToolSpec(tool_id="rectangular_peg", half_size_x=0.010, half_size_y=0.006, length=0.050, mass=0.08)
    clearance = float(physics["clearance"])
    socket = SocketSpec(
        hole_half_size_x=peg.half_size_x + clearance,
        hole_half_size_y=peg.half_size_y + clearance,
        depth=0.040,
        wall_thickness=0.010,
        bottom_thickness=0.006,
        chamfer_depth=0.006,
        chamfer_width=0.002,
        chamfer_steps=2,
        # Figure 2 (b) socket mint, RGB 176, 214, 210.
        rgba="0.690196 0.839216 0.823529 1",
    )
    hole_rot = root_rot @ _rotation_xyz(physics["angles"])
    approach_gap = 0.004
    entrance = (
        root_pos
        + root_rot[:, 2] * peg.length
        + root_rot @ np.asarray([physics["position"][0], physics["position"][1], approach_gap], np.float64)
    )
    friction = (float(physics["friction"]), 0.005, 0.0001)
    composer = MjcfSceneComposer(profile)
    composer.add_peg_tool(peg, friction=friction, solref=physics["solref"], solimp="0.9 0.95 0.001")
    composer.add_rectangular_socket(
        socket,
        position=entrance,
        quaternion=_quat(hole_rot),
        friction=friction,
        solref=physics["solref"],
        solimp="0.9 0.95 0.001",
    )
    model = composer.compile()
    return model


def event_indices(trajectory):
    del trajectory
    # Same clock times as the published captions: 0.00 s, 0.20 s, 0.94 s.
    return [("initial", 0), ("contact", 10), ("complete", 47)]


def render_frames(model, states, indices, azimuth, elevation, distance, lookat):
    import imageio.v3 as imageio

    width, height = 870, 580
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    data = mujoco.MjData(model)
    frames = {}
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for label, index in indices:
            q = np.asarray(states[index]["q"], np.float64)
            qd = np.asarray(states[index]["qd"], np.float64)
            data.qpos[: q.size] = q
            data.qvel[: qd.size] = qd
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            image = renderer.render()
            path = OUT / f"{label}.png"
            imageio.imwrite(path, image)
            frames[label] = path
    return frames


def manuscript_light(model):
    """Match panel (a): a bright white arm on the blue checker floor."""
    model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
    model.vis.headlight.diffuse[:] = [0.70, 0.70, 0.70]
    model.vis.headlight.specular[:] = [0.04, 0.04, 0.04]
    model.mat_reflectance[:] = 0
    model.mat_specular[:] = 0.12


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trajectory = json.loads(TRAJ.read_text())
    indices = event_indices(trajectory)
    print("events", [(name, index, round(index * 0.02, 2)) for name, index in indices])
    model = build_model()
    visual, _checks = _restore_panda_visuals(model, trajectory["states"], [index for _, index in indices])
    _manuscript_sky(visual)
    manuscript_light(visual)
    for gid in range(visual.ngeom):
        name = mujoco.mj_id2name(visual, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("socket_"):
            visual.geom_rgba[gid] = [176 / 255, 214 / 255, 210 / 255, 1]
    data = mujoco.MjData(visual)
    data.qpos[:7] = np.asarray(trajectory["states"][indices[1][1]]["q"], np.float64)[:7]
    mujoco.mj_forward(visual, data)
    entrance = data.site_xpos[mujoco.mj_name2id(visual, mujoco.mjtObj.mjOBJ_SITE, "socket_entrance")].copy()
    # Horizontal gaze through the socket mouth, so the hole is seen from the side.
    lookat = entrance + np.array([0.0, 0.0, -0.01])
    print("lookat", lookat.tolist(), "nq", visual.nq, "q", len(trajectory["states"][0]["q"]))
    # A little below horizontal, so the checker floor is visible without a top view.
    render_frames(visual, trajectory["states"], indices, 78.0, -16.0, 0.42, lookat)
    print("rendered final")


if __name__ == "__main__":
    main()
