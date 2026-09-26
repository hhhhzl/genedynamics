#!/usr/bin/env python3
"""Render saved Panda reset poses with a light manuscript-only presentation.

Requires native MuJoCo, numpy and Pillow, but not Brax/JAX. Task geometry is
constructed through the repository's unchanged MJCF composer. Scanning uses
the exact rigid-plane setup; PegInsert uses the exact nominal ID setup. No
simulation steps, fabricated trajectories, or raster recoloring are used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def profile():
    return SimpleNamespace(
        model_path=lambda: ROOT / "genedynamics/envs/assets/franka_panda/panda_arm.xml",
        model_id="panda", elements={"tool_mount": SimpleNamespace(name="ee")},
    )


def home_pose(robot):
    model = mujoco.MjModel.from_xml_path(str(robot.model_path()))
    data = mujoco.MjData(model)
    data.qpos[:] = [0, -.5, 0, -2, 0, 1.5, .78]
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()


def native_scene(kind, compose):
    robot = profile()
    point, rotation = home_pose(robot)
    builder = compose.MjcfSceneComposer(robot)
    # Native MuJoCo 3.3 requires a positive explicit inertial once zero-mass
    # visuals are attached to the welded base. Its inertia has no moving DOFs;
    # this render-only compatibility declaration changes no task geometry/FK.
    ET.SubElement(builder.root.find('.//body[@name="link0"]'), "inertial",
                  mass="1", pos="0 0 0", diaginertia="1 1 1")
    if kind == "scanning":
        # surface_geometry.make_plane: u=(.3,0,0), v=(0,.3,0),
        # translated so (xi,eta)=(.1,.5) is .0193 m below the home EE.
        builder.add_spherical_tool(compose.SphericalToolSpec("scan_probe", radius=.02),
                                   friction=.1, collidable=True)
        center = point + np.array([.12, 0, -.0193])
        builder.add_hfield_surface(name="surf", nrow=36, ncol=36,
                                  size=(.15, .15, .001, .05), position=center,
                                  friction=.1, solref="0.02 1")
        model = builder.compile()
        model.hfield_data[:] = 0
        source = "results/arm/surface_scan/main/mga/level_rigid_plane/seed_0/trajectory/trajectory.json"
    else:
        peg = compose.PegToolSpec()
        socket = compose.SocketSpec(hole_half_size_x=.0115, hole_half_size_y=.0075)
        entrance = point + rotation[:, 2] * (.05 + .004)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
        builder.add_peg_tool(peg, solref="0.03 1")
        builder.add_rectangular_socket(socket, position=entrance, quaternion=quat,
                                       solref="0.03 1")
        model = builder.compile()
        source = "results/arm/peg_insert/main/mga/level_id_wide/seed_0/trajectory/trajectory.json"
    states = json.loads((ROOT / source).read_text())["states"]
    return model, states, source


def appearance(model):
    # Replace only texture pixels and material colors before native rendering.
    for tid in range(model.ntex):
        start = int(model.tex_adr[tid])
        w, h, n = map(int, (model.tex_width[tid], model.tex_height[tid], model.tex_nchannel[tid]))
        pixels = model.tex_data[start:start+w*h*n].reshape(h, w, n)
        if model.tex_type[tid] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            pixels[:, :, :3] = [245, 247, 246]
        else:
            yy, xx = np.indices((h, w))
            grid = ((xx // max(1, w//2)) + (yy // max(1, h//2))) % 2
            pixels[:, :, :3] = np.where(grid[:, :, None], [230, 236, 233], [243, 246, 242])
    model.mat_reflectance[:] = 0
    model.mat_specular[:] = .1
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name == "surf":
            model.geom_rgba[gid] = [214/255, 237/255, 232/255, 1]
        elif name == "peg":
            model.geom_rgba[gid] = [31/255, 122/255, 120/255, 1]
        elif name == "probe":
            model.geom_rgba[gid] = [217/255, 102/255, 14/255, 1]
        elif name.startswith("socket_"):
            model.geom_rgba[gid] = [176/255, 214/255, 210/255, 1]
    model.vis.headlight.ambient[:] = [.55, .55, .55]
    model.vis.headlight.diffuse[:] = [.65, .65, .65]
    model.vis.headlight.specular[:] = [.05, .05, .05]
    model.vis.rgba.haze[:] = [.96, .97, .96, 1]
    model.vis.global_.offwidth, model.vis.global_.offheight = 1200, 800


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "scripts/paper/mga/output/env_overview_v2/assets")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    compose = load_module("mga_native_composition", ROOT / "genedynamics/envs/composition.py")
    scene_tools = load_module("mga_native_visuals", ROOT / "scripts/paper/mga/render_mechanism_scenes.py")
    metadata = {}
    for kind in ("scanning", "peg"):
        native, states, source = native_scene(kind, compose)
        model, checks = scene_tools._restore_panda_visuals(native, states, [0])
        appearance(model)
        data = mujoco.MjData(model)
        data.qpos[:] = states[0]["q"]
        data.qvel[:] = states[0]["qd"]
        mujoco.mj_forward(model, data)
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation = 128, -19
        if kind == "peg":
            ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
            camera.lookat[:] = data.site_xpos[ee_id] + [-.06, 0, -.06]
            camera.distance = .88
        else:
            camera.lookat[:] = [.25, 0, .43]
            camera.distance = 1.25
        opt = mujoco.MjvOption()
        opt.sitegroup[:] = 0
        with mujoco.Renderer(model, 800, 1200) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=opt)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            Image.fromarray(renderer.render()).save(args.output / f"{kind}_hero.png")
        metadata[kind] = {"trajectory": source, "state_index": 0,
                          "native_geometry": "rigid_plane" if kind == "scanning" else "id_wide",
                          "visual_restoration": checks, "simulation_steps": 0,
                          "fixed_base_render_inertial": "MuJoCo 3.3 compatibility; no moving DOFs or geometry changed",
                          "camera": {"lookat": camera.lookat.tolist(), "distance": float(camera.distance),
                                     "azimuth": float(camera.azimuth), "elevation": float(camera.elevation)},
                          "size": [1200, 800], "style": "manuscript-light-teal-indigo-orange"}
    (args.output / "panda_render_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
