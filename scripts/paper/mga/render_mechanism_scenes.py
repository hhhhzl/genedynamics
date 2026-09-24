#!/usr/bin/env python3
"""Render native simulation scenes from saved MGA execution states, never rollouts.

The scene additions are genuine earlier-pose robot/object geometry, saved paths
and true initial/final box outlines. The hybrid surface is not painted with physical
material stripes: its stiffness map is evaluated in the task's chart coordinates.
Run with the project's CPU simulation dependencies (MuJoCo, Brax, JAX, imageio).

H1 uses the genuine Brax Web/Three.js renderer, not brax.io.image (which wraps
MuJoCo). First export saved-state geometry with ``--task humanoid``, then run
``--capture-brax`` where headless Chromium is available. A disposable Linux
browser can be used without installing anything into the experiment image::

    docker run --rm --cpus=2 --memory=2g -v "$PWD:/workspace" -w /workspace \
      public.ecr.aws/docker/library/python:3.10.20-slim-bookworm sh -c \
      'apt-get update -qq && apt-get install -y --no-install-recommends chromium \
       && python scripts/paper/mga/render_mechanism_scenes.py --capture-brax'

Only presentation assets are written. No dynamics, controllers or training are
run; image, HTML, payload and formal trajectory hashes are saved for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RENDER_STYLE = "surface-blue-gif-reference-thin-skirt"
HUMANOID_RENDER_STYLE = "brax-web-first-last-contrast-silhouettes"
FILMSTRIP_SCENE = "surface_rigid_bumpy"

SCENES = {
    "surface_hybrid_stripes": ("surface", "arm/surface_scan", "hybrid_stripes"),
    "surface_soft_convex": ("surface", "arm/surface_scan", "soft_convex"),
    "surface_rigid_bumpy": ("surface", "arm/surface_scan", "rigid_bumpy"),
    "humanoid_force_regulation": ("humanoid", "humanoid/push_to_line", "p1_force_15n"),
    "humanoid_fixed_stance_push": ("humanoid", "humanoid/push_to_line", "p2_push_nominal"),
    "humanoid_unjamming": ("humanoid", "humanoid/push_to_line", "p3_unjam"),
}


def _read_json(path):
    with path.open() as handle:
        return json.load(handle)


def _persist_scene_metadata(path, updates):
    """Merge per-scene updates so Web capture and native export do not erase each other."""
    import fcntl
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = _read_json(path) if path.exists() else {}
        current.update(updates)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".json.tmp", delete=False) as handle:
            json.dump(current, handle, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(path)
        fcntl.flock(lock, fcntl.LOCK_UN)


def _manuscript_sky(model):
    """Preserve the native gradient while matching the manuscript's blue horizon."""
    import mujoco
    import numpy as np
    for texture in range(model.ntex):
        if int(model.tex_type[texture]) != int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
            continue
        channels = int(model.tex_nchannel[texture])
        start = int(model.tex_adr[texture])
        size = int(model.tex_width[texture] * model.tex_height[texture] * channels)
        pixels = model.tex_data[start:start + size].reshape(-1, channels)
        pixels[:, :3] = np.maximum(pixels[:, :3], np.array([38, 64, 89], dtype=np.uint8))


def _execution_env(result):
    from genedynamics.experiments.plugins.environments import (
        HumanoidBoxPushPlugin, ManipulatorSurfaceScanPlugin, ManipulatorPegInsertPlugin,
    )

    config = result["config_snapshot"]
    plugin_class = {
        "manipulator_surface_scan": ManipulatorSurfaceScanPlugin,
        "humanoid_box_push": HumanoidBoxPushPlugin,
        "manipulator_peg_insert": ManipulatorPegInsertPlugin,
    }[config["env_name"]]
    plugin = plugin_class()
    env = plugin.create_env({
        **config.get("env_params", {}),
        "_experiment_seed": int(result["seed"]),
        "_controller_method": config.get("method_params", {}).get(
            "controller_method", "mga_controllable_gate"),
        "_execution_env_params": config.get("execution_env_params") or {},
    })
    return plugin.execution_env(env)


def _native_model(env):
    """Keep native render geometry consistent with the executed MJX model."""
    import numpy as np

    model = env.sys.mj_model
    for field in ("body_pos", "body_quat", "geom_pos", "geom_quat", "geom_size",
                  "geom_rgba", "site_pos", "site_quat"):
        source = getattr(env.sys, field, None)
        target = getattr(model, field, None)
        if source is not None and target is not None:
            value = np.asarray(source)
            if value.shape == target.shape:
                target[:] = value
    model.vis.rgba.haze[:] = [0.15, 0.25, 0.35, 1]
    model.vis.headlight.ambient[:] = [0.3, 0.3, 0.3]
    model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.specular[:] = [0, 0, 0]
    return model


def _add_reference_backdrop(tree):
    assets = tree.find("asset")
    if assets is None:
        assets = ET.SubElement(tree, "asset")
    for texture in list(assets.findall("texture")):
        if texture.get("type") == "skybox":
            assets.remove(texture)
    ET.SubElement(assets, "texture", type="skybox", builtin="gradient",
                  rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
    ET.SubElement(assets, "texture", type="2d", name="paper_ground",
                  builtin="checker", mark="edge", rgb1="0.2 0.3 0.4",
                  rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8", width="300", height="300")
    ET.SubElement(assets, "material", name="paper_ground", texture="paper_ground",
                  texuniform="true", texrepeat="5 5", reflectance="0.2")
    floor = tree.find('.//geom[@name="floor"]')
    floor.set("material", "paper_ground")
    floor.set("rgba", "1 1 1 1")


def _restore_humanoid_backdrop(model, states, indices):
    """Restore texture assets removed by the execution model's scene composer."""
    import mujoco
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="mga-paper-h1-render-") as temporary:
        native_xml = Path(temporary) / "native.xml"
        mujoco.mj_saveLastXML(str(native_xml), model)
        tree = ET.parse(native_xml).getroot()
    _add_reference_backdrop(tree)
    # Native exporter retains mesh paths; resolve any portable relative path
    # against the actual vendored H1 asset directory, not a substitute model.
    for mesh in tree.findall("./asset/mesh"):
        source = mesh.get("file")
        if source and not Path(source).is_absolute():
            candidate = ROOT / "genedynamics/envs/assets/unitree_h1/assets" / Path(source).name
            if candidate.exists():
                mesh.set("file", str(candidate))
    visual = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    old_data, new_data = mujoco.MjData(model), mujoco.MjData(visual)
    largest_error = 0.0
    for index in indices:
        for data, native in ((old_data, model), (new_data, visual)):
            data.qpos[:] = states[index]["q"]
            data.qvel[:] = states[index]["qd"]
            mujoco.mj_forward(native, data)
        largest_error = max(largest_error, float(np.max(np.abs(old_data.xpos-new_data.xpos))),
                            float(np.max(np.abs(old_data.xmat-new_data.xmat))))
    if largest_error > 1e-5:
        raise ValueError(f"Backdrop restoration changed robot FK by {largest_error}")
    visual.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
    visual.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
    return visual, {"source": "genedynamics/envs/assets/unitree_h1/mjx_scene_h1_box_push.xml",
                    "max_saved_pose_fk_error": largest_error, "physics_modified": False,
                    "note": "Restore native blue checker/sky render assets only; robot/task unchanged"}


def _restore_panda_visuals(model, states, indices):
    """Add vendored official link meshes to the intentionally mesh-free arm.

    Only link0..link7 visuals are restored; no hand/fingers or fictitious probe
    adapter is added. Native robot topology and the recorded probe are unchanged.
    """
    import copy
    import mujoco
    import numpy as np

    asset_dir = ROOT / "third_party/mujoco_menagerie/franka_emika_panda"
    donor = ET.parse(asset_dir / "panda.xml").getroot()
    with tempfile.TemporaryDirectory(prefix="mga-paper-render-") as temporary:
        native_xml = Path(temporary) / "native.xml"
        mujoco.mj_saveLastXML(str(native_xml), model)
        tree = ET.parse(native_xml).getroot()
    assets = tree.find("asset")
    if assets is None:
        assets = ET.SubElement(tree, "asset")
    donor_bodies = {b.get("name"): b for b in donor.findall(".//body")}
    needed_meshes, needed_materials = set(), set()
    for body in tree.findall(".//body"):
        name = body.get("name")
        if name not in {f"link{i}" for i in range(8)}:
            continue
        for source in donor_bodies[name].findall("geom"):
            if source.get("class") != "visual":
                continue
            geom = copy.deepcopy(source)
            geom.attrib.pop("class", None)
            geom.attrib.update(type="mesh", contype="0", conaffinity="0", group="2", mass="0")
            body.append(geom)
            needed_meshes.add(geom.get("mesh"))
            needed_materials.add(geom.get("material"))
    for source in donor.find("asset"):
        if source.tag == "mesh":
            name = source.get("name", Path(source.get("file", "")).stem)
            if name not in needed_meshes:
                continue
            elem = copy.deepcopy(source)
            elem.set("file", str(asset_dir / "assets" / source.get("file")))
        elif source.tag == "material" and source.get("name") in needed_materials:
            elem = copy.deepcopy(source)
            elem.attrib.pop("class", None)
        else:
            continue
        assets.append(elem)
    # Match the explicitly requested legacy Surface GIF, not the independent
    # bright-gray Brax H1 style.  This changes only presentation materials.
    _add_reference_backdrop(tree)
    surface = tree.find('.//geom[@name="surf"]')
    if surface is not None:
        surface.set("rgba", "0.8 0.8 0.2 0.6")
    visual = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    visual.hfield_data[:] = model.hfield_data
    # Remove the 50 mm display pedestal; height samples, vertical scale and
    # top/contact world coordinates are identical to the execution model.
    source_base = model.hfield_size[:, 3].copy()
    visual.hfield_size[:, 3] = 0.001
    if (not np.array_equal(visual.hfield_data, model.hfield_data)
            or not np.array_equal(visual.hfield_size[:, :3], model.hfield_size[:, :3])):
        raise ValueError("Presentation skirt adjustment changed the true top surface")
    if (visual.nq, visual.nv) != (model.nq, model.nv):
        raise ValueError("Visual restoration changed robot state dimensions")
    old_data, new_data = mujoco.MjData(model), mujoco.MjData(visual)
    largest_error = 0.0
    for index in indices:
        for data, native in ((old_data, model), (new_data, visual)):
            data.qpos[:] = states[index]["q"]
            data.qvel[:] = states[index]["qd"]
            mujoco.mj_forward(native, data)
        for name in ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7"):
            old_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            new_id = mujoco.mj_name2id(visual, mujoco.mjtObj.mjOBJ_BODY, name)
            largest_error = max(largest_error,
                float(np.max(np.abs(old_data.xpos[old_id] - new_data.xpos[new_id]))),
                float(np.max(np.abs(old_data.xmat[old_id] - new_data.xmat[new_id]))))
        old_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        new_id = mujoco.mj_name2id(visual, mujoco.mjtObj.mjOBJ_SITE, "ee")
        largest_error = max(largest_error,
            float(np.max(np.abs(old_data.site_xpos[old_id] - new_data.site_xpos[new_id]))))
    if largest_error > 1e-5:
        raise ValueError(f"Visual restoration changed FK by {largest_error}")
    return visual, {"source": str((asset_dir / "panda.xml").relative_to(ROOT)),
                    "links": [f"link{i}" for i in range(8)],
                    "max_saved_pose_fk_error": largest_error,
                    "physics_modified": False,
                    "surface_presentation_override": {
                        "source_downward_base_m": source_base.tolist(),
                        "display_downward_base_m": visual.hfield_size[:, 3].tolist(),
                        "top_height_samples_scale_and_world_pose_unchanged": True,
                        "note": "Rendering-model skirt only; original execution model/results untouched; no dynamics rerun"},
                    "note": "Visual-only official meshes; no gripper or probe adapter added"}


def _state_indices(trajectory, name, model):
    """Exclude post-success padding and select genuine chronological poses."""
    import numpy as np

    states = trajectory["states"]
    valid = [i for i, state in enumerate(states)
             if state.get("q") is not None and state.get("qd") is not None]
    padding = trajectory.get("task_signals", {}).get("success_padding", [])
    first_padding = next((i for i, value in enumerate(padding) if value), len(padding))
    # state i+1 is the state after transition i; include the first success state,
    # but not duplicate transitions subsequently emitted as success padding.
    last = min(valid[-1], first_padding) if padding else valid[-1]
    if name.startswith("surface"):
        return sorted(set(int(i) for i in np.linspace(0, last, 3))), last
    # Honest start-to-finish comparison: no intermediate pose selection and no
    # enlarged motion, even when the physical displacement is small.
    return sorted(set([0, last])), last


def _geom_snapshot(geom):
    import numpy as np
    return {field: (value.copy() if isinstance(value, np.ndarray) else value)
            for field in ("type", "dataid", "matid", "objtype", "objid", "category",
                          "size", "pos", "mat", "rgba", "emission", "specular",
                          "shininess", "reflectance", "modelrbound", "texcoord")
            for value in [getattr(geom, field)]}


def _append_ghost(scene, snapshot, alpha):
    import mujoco
    import numpy as np

    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("Scene geometry capacity exhausted")
    geom = scene.geoms[scene.ngeom]
    rgba = snapshot["rgba"].copy()
    rgba[3] *= alpha
    mujoco.mjv_initGeom(geom, snapshot["type"],
                       np.asarray(snapshot["size"], dtype=np.float64),
                       np.asarray(snapshot["pos"], dtype=np.float64),
                       np.asarray(snapshot["mat"], dtype=np.float64).ravel(), rgba)
    for field, value in snapshot.items():
        if field == "rgba":
            continue
        target = getattr(geom, field)
        if isinstance(target, np.ndarray):
            target[:] = value
        else:
            setattr(geom, field, value)
    geom.transparent = True
    geom.segid = int(scene.ngeom)
    scene.ngeom += 1


def _append_path(scene, points, radius=0.0010, rgba=(0.98, 0.76, 0.20, 0.90)):
    import mujoco
    import numpy as np

    for start, end in zip(points[:-1], points[1:]):
        if np.linalg.norm(end - start) < 1e-7:
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                           np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                           np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end)
        geom.segid = int(scene.ngeom)
        scene.ngeom += 1


def _append_box_motion(scene, model, states, last):
    """Actual initial/final top-face outlines and an XY-projected center trace."""
    import mujoco
    import numpy as np

    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_box")
    if box_id < 0:
        return None
    data = mujoco.MjData(model)
    half = model.geom_size[box_id].copy()
    corners = np.array([[-half[0], -half[1], half[2]],
                        [half[0], -half[1], half[2]],
                        [half[0], half[1], half[2]],
                        [-half[0], half[1], half[2]],
                        [-half[0], -half[1], half[2]]])
    centers = []
    for index in range(last + 1):
        data.qpos[:] = states[index]["q"]
        data.qvel[:] = states[index]["qd"]
        mujoco.mj_forward(model, data)
        position = data.geom_xpos[box_id].copy()
        centers.append(position)
        if index in (0, last):
            points = corners @ data.geom_xmat[box_id].reshape(3, 3).T + position
            points[:, 2] += 0.006
            _append_path(scene, points, radius=0.004,
                         rgba=(0.95, 0.98, 1.0, 0.85) if index == 0 else (1.0, 0.58, 0.03, 1.0))
    centers = np.asarray(centers)
    # Draw the center's true XY trace on the top face, not through the solid box.
    centers[:, 2] += half[2] + 0.010
    _append_path(scene, centers, radius=0.0035, rgba=(0.96, 0.25, 0.18, 0.85))
    return {"outline_states": [0, last], "center_path_states": [0, last],
            "xy_displacement_m": (centers[-1, :2] - centers[0, :2]).tolist(),
            "outline_annotation_height_offset_m": 0.006,
            "center_path_note": "True XY positions projected to box top face, not scaled"}


def _camera(model, name, detail=False, view=None, focus=None):
    import mujoco

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if name.startswith("surface"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "surf")
        center = model.geom_pos[sid].copy()
        if view == "side_contact":
            camera.lookat[:] = focus + [0.0, 0.0, 0.02]
            camera.distance, camera.azimuth, camera.elevation = 0.55, 90, -6
        elif view == "top_contact":
            camera.lookat[:] = focus
            camera.distance, camera.azimuth, camera.elevation = 0.58, 90, -90
        elif view == "oblique_contact":
            camera.lookat[:] = focus + [0.0, 0.0, 0.035]
            camera.distance, camera.azimuth, camera.elevation = 0.60, 125, -24
        elif detail:
            camera.lookat[:] = [center[0] - 0.11, center[1], center[2] + 0.055]
            camera.distance, camera.azimuth, camera.elevation = 0.56, 132, -26
        else:
            camera.lookat[:] = [0.26, 0.0, 0.45]
            camera.distance, camera.azimuth, camera.elevation = 1.35, 128, -19
    else:
        camera.lookat[:] = [0.52, 0.0, 0.88]
        camera.distance, camera.azimuth, camera.elevation = 3.30, 90, 0
        if name == "humanoid_force_regulation":
            # Same lateral scale as fixed-stance push, with blank upper-right
            # sky reserved for the force inset; no robot or object is cropped.
            camera.lookat[2] = 1.08
        if name == "humanoid_unjamming":
            camera.lookat[:] = [0.78, 0.0, 0.70]
            camera.distance, camera.azimuth, camera.elevation = 3.40, 90, -90
    return camera


def _render_pose_scene(model, states, indices, path, name, output, *, detail=False,
                       size=None, view=None, box_last=None):
    import imageio.v3 as imageio
    import mujoco
    import numpy as np

    is_surface = name.startswith("surface")
    width, height = size or ((1500, 640) if is_surface and not detail else
                            ((1100, 900) if is_surface else (1100, 1000)))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    data.qpos[:] = states[indices[-1]]["q"]
    data.qvel[:] = states[indices[-1]]["qd"]
    mujoco.mj_forward(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    focus = data.site_xpos[ee_id].copy() if ee_id >= 0 else None
    camera = _camera(model, name, detail, view, focus)
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    ghosts = []
    crop = [0, 0, width, height]
    box_motion = None
    with mujoco.Renderer(model, height=height, width=width, max_geom=10000) as renderer:
        for index in indices:
            data.qpos[:] = states[index]["q"]
            data.qvel[:] = states[index]["qd"]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=opt)
            if index != indices[-1]:
                geometry = []
                for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                    if int(geom.objtype) != int(mujoco.mjtObj.mjOBJ_GEOM):
                        continue
                    geom_id = int(geom.objid)
                    if geom_id < 0 or int(model.geom_bodyid[geom_id]) == 0:
                        continue
                    if float(geom.rgba[3]) <= 0.01:
                        continue
                    geometry.append(_geom_snapshot(geom))
                ghosts.append(geometry)
        for pose_id, geometry in enumerate(ghosts):
            alpha = 0.24 + 0.16 * pose_id
            for snapshot in geometry:
                _append_ghost(renderer.scene, snapshot, alpha)
        if is_surface and path is not None:
            _append_path(renderer.scene, path)
        if not is_surface and box_last is not None:
            box_motion = _append_box_motion(renderer.scene, model, states, box_last)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
        # Native reference scenes use haze, not MuJoCo's default black fog.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        image = renderer.render()
    # No image compositing of independently rendered poses: all ghosts share
    # the same native depth buffer and camera.
    imageio.imwrite(output, image)
    return {"image_size": [int(image.shape[1]), int(image.shape[0])],
        "native_render_size": [width, height], "crop_xyxy": crop, "camera": {
        "lookat": camera.lookat.tolist(), "distance": float(camera.distance),
        "azimuth": float(camera.azimuth), "elevation": float(camera.elevation)},
        "ghost_opacity": [0.24 + 0.16 * i for i in range(len(ghosts))],
        "box_motion_annotation": box_motion,
        "view": view or ("top" if name == "humanoid_unjamming" else
                          "side" if not is_surface else "overview"),
        "saved_ee_path": bool(is_surface and path is not None)}


def _export_brax_web(env, model, trajectory, indices, last, name, output_dir):
    """Export genuine Brax transforms/meshes for Brax's official Web renderer.

    This route never calls mujoco.Renderer.  Installed brax.io.image is itself a
    MuJoCo wrapper, so the independent Brax Web/Three.js scene builder is used.
    """
    import shutil
    from types import SimpleNamespace
    import brax
    from brax import kinematics
    from brax.io import json as brax_json
    import jax.numpy as jnp
    import mujoco
    import numpy as np

    brax_states = []
    native_data = mujoco.MjData(model)
    error = 0.0
    for index in indices:
        saved = trajectory["states"][index]
        q, qd = jnp.asarray(saved["q"]), jnp.asarray(saved["qd"])
        x, xd = kinematics.forward(env.sys, q, qd)
        brax_states.append(SimpleNamespace(x=x))
        native_data.qpos[:] = saved["q"]
        native_data.qvel[:] = saved["qd"]
        mujoco.mj_forward(model, native_data)
        error = max(error, float(np.max(np.abs(np.asarray(x.pos)-native_data.xpos[1:]))))
    if error > 2e-5:
        raise ValueError(f"Brax saved-state FK differs from execution model by {error}")
    payload = json.loads(brax_json.dumps(env.sys, brax_states))
    # Brax serializes every collider. Match the physical scene's visual layers,
    # honoring task-owned invisible walls and native material colors.
    link_names = [n or f"link {i}" for i, n in enumerate(env.sys.link_names)] + ["world"]
    counters = {key: 0 for key in payload["geoms"]}
    selected_geoms = {key: [] for key in payload["geoms"]}
    for geom_id in range(model.ngeom):
        link_index = int(np.asarray(env.sys.geom_bodyid)[geom_id]) - 1
        key = link_names[link_index]
        geom = payload["geoms"][key][counters[key]]
        counters[key] += 1
        if int(model.geom_group[geom_id]) == 3 or float(model.geom_rgba[geom_id, 3]) <= 0.001:
            continue
        material = int(model.geom_matid[geom_id])
        rgba = (model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom_id]).copy()
        # Explicit geom colors override the default material tint in MJCF.
        if not np.allclose(model.geom_rgba[geom_id], [0.5, 0.5, 0.5, 1.0]):
            rgba = model.geom_rgba[geom_id].copy()
        geom["rgba"] = rgba.tolist()
        selected_geoms[key].append(geom)
    payload = {"geoms": {key: value for key, value in selected_geoms.items() if value},
               "states": payload["states"], "opt": {"timestep": 0.02}}
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_box")
    box_half = np.asarray(model.geom_size[box_id])
    box_poses = []
    for index in range(last + 1):
        native_data.qpos[:] = trajectory["states"][index]["q"]
        native_data.qvel[:] = trajectory["states"][index]["qd"]
        mujoco.mj_forward(model, native_data)
        box_poses.append({"position": native_data.geom_xpos[box_id].tolist(),
                          "rotation": native_data.geom_xmat[box_id].reshape(3, 3).tolist()})
    goal_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal_line")
    goal = None
    if goal_site >= 0 and float(model.site_rgba[goal_site, 3]) > 0:
        goal = {"position": model.site_pos[goal_site].tolist(),
                "half_size": model.site_size[goal_site].tolist()}
    width, height = (1700, 500) if name == "humanoid_force_regulation" else (1320, 1000)
    camera = ({"position": [-0.23, -3.8, 1.65], "target": [0.67, 0, 0.82], "up": [0, 0, 1]}
              if name != "humanoid_unjamming" else
              {"position": [0.78, 0, 4.2], "target": [0.78, 0, 0.55], "up": [0, 1, 0]})
    payload["paper"] = {"width": width, "height": height, "name": name,
                         "camera": camera, "pose_indices": indices,
                         "box_half_size": box_half.tolist(), "box_poses": box_poses,
                         "goal": goal}
    json_path = output_dir / f"{name}.brax.json"
    json_path.write_text(json.dumps(payload, separators=(",", ":")))
    asset_dir = output_dir / "brax_web"
    asset_dir.mkdir(exist_ok=True)
    source = Path(brax.__file__).parent / "visualizer/js/system.js"
    shutil.copyfile(source, asset_dir / "system.js")
    html_path = output_dir / f"{name}.brax.html"
    html_path.write_text(_BRAX_WEB_HTML.replace("__PAYLOAD__", json_path.name))
    return {"image_size": [width, height], "native_render_size": [width, height],
            "crop_xyxy": [0, 0, width, height], "camera": camera,
            "view": "top" if name == "humanoid_unjamming" else "side-oblique",
            "rendering": "Brax kinematics.forward -> brax.io.json -> official Brax Web createScene -> Three.js/Chrome",
            "brax_version": brax.__version__, "brax_web_scene_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "brax_fk_max_position_error_m": error, "web_html": html_path.name,
            "web_payload": json_path.name, "capture_pending": True,
            "visual_alpha_override": {"final_box": 1.0, "source_box": 0.6,
                "initial_box_fill": False,
                "note": "Presentation only; initial box shown by true outline, final box opaque; geometry/FK/physical parameters unchanged"},
            "ghost_opacity": [0.16 for _ in indices[:-1]],
            "robot_pose_annotation": _robot_pose_annotation(),
            "box_motion_annotation": {"outline_states": [0, last],
                "outline_colors": {"initial": "#4B2E83", "final": "#D9660E"},
                "outline_style": "solid tubes", "outline_radius_m": 0.008,
                "center_path_states": [0, last], "center_path_note": "True XY path projected onto box top; no displacement scaling"}}


def _robot_pose_annotation():
    return {"colors": {"initial": "#4B2E83", "final": "#D9660E"},
            "outline_width_px": {"initial": 5.0, "final": 3.5},
            "initial_fill_alpha": 0.16, "final_fill": "native opaque material",
            "style": "solid image-space contours of exact saved-pose robot masks",
            "xray_initial_outline": True,
            "note": "True first/last poses; silhouettes are overlaid for visibility, not enlarged geometry or shifted joints"}


_BRAX_WEB_HTML = r'''<!doctype html><html><head><meta charset="utf-8">
<title>MGA_Brax_loading</title><style>html,body{margin:0;overflow:hidden;background:#f3f3f1}canvas{display:block}</style>
<script type="importmap">{"imports":{"three":"./brax_web/three.module.js"}}</script></head><body>
<script type="module">
import * as THREE from 'three';
import {createScene} from './brax_web/system.js';
try {
 const system = await (await fetch('./__PAYLOAD__')).json();
 const p=system.paper;
 THREE.Object3D.DEFAULT_UP.set(0,0,1);
 const scene=createScene(system);
 const maskScenes=[new THREE.Scene(),new THREE.Scene()];
 scene.background=new THREE.Color(0xf3f3f1);
 scene.fog=new THREE.Fog(0xf3f3f1,14,30);
 function pose(group,state,index) {
   group.position.fromArray(state.pos[index]);
   const q=state.rot[index]; group.quaternion.set(q[1],q[2],q[3],q[0]);
 }
 for (const [name,geoms] of Object.entries(system.geoms)) {
   const group=scene.getObjectByName(name.replaceAll('/','_'));
   const index=geoms[0].link_idx;
   for(let j=0;j<geoms.length;j++) {
     const geom=geoms[j],child=group.children[j];
     child.traverse(o=>{if(o.isMesh){
       const opacity=(name==='box_body' && geom.name==='Box') ? 1.0 : geom.rgba[3];
       o.material.opacity=opacity;o.material.transparent=opacity<.999;
       o.material.shininess=25;o.receiveShadow=true;
       if(geom.name==='Plane'&&o.material.map){
         const d=o.material.map.image.data;
         for(let k=0;k<4;k++){const v=[105,150,150,105][k];d[k*4]=v;d[k*4+1]=v;d[k*4+2]=v;d[k*4+3]=255;}
         o.material.map.repeat.set(5000,5000);o.material.map.needsUpdate=true;
       }
     }});
   }
   if(index>=0){
     pose(group,system.states.x.at(-1),index);
     if(name!=='box_body'){
       for(let k=0;k<2;k++){
         const mask=group.clone(true);pose(mask,k===0?system.states.x[0]:system.states.x.at(-1),index);
         mask.traverse(o=>{if(o.isMesh){o.material=new THREE.MeshBasicMaterial({color:0xffffff});o.castShadow=false;o.receiveShadow=false;}});
         maskScenes[k].add(mask);
       }
     }
     for(let i=0;i<system.states.x.length-1;i++){
       if(name==='box_body')continue; // initial box is represented by its exact outline only
       const ghost=group.clone(true);pose(ghost,system.states.x[i],index);
       ghost.traverse(o=>{if(o.isMesh){o.material=o.material.clone();o.material.color.setHex(0x4b2e83);o.material.opacity*=.16;o.material.transparent=true;o.material.depthWrite=false;o.castShadow=false;}});
       scene.add(ghost);
     }
   }
 }
 scene.add(new THREE.HemisphereLight(0xffffff,0x9a9b9d,.70));
 const light=new THREE.DirectionalLight(0xffffff,.75);light.position.set(-2,-4,7);light.castShadow=true;
 light.shadow.camera.left=-3;light.shadow.camera.right=3;light.shadow.camera.top=3;light.shadow.camera.bottom=-3;
 light.shadow.mapSize.set(4096,4096);light.shadow.bias=-.0002;light.shadow.normalBias=.003;scene.add(light);
 const fill=new THREE.DirectionalLight(0xffffff,.15);fill.position.set(4,3,4);scene.add(fill);
 function line(points,color,opacity=1){
   const geometry=new THREE.BufferGeometry().setFromPoints(points.map(v=>new THREE.Vector3(...v)));
   scene.add(new THREE.Line(geometry,new THREE.LineBasicMaterial({color:new THREE.Color(color).convertSRGBToLinear(),transparent:opacity<1,opacity})));
 }
 function thickOutline(points,color){
   const curve=new THREE.CurvePath();
   for(let i=1;i<points.length;i++)curve.add(new THREE.LineCurve3(new THREE.Vector3(...points[i-1]),new THREE.Vector3(...points[i])));
   const mesh=new THREE.Mesh(new THREE.TubeGeometry(curve,32,.008,8,false),new THREE.MeshBasicMaterial({color:new THREE.Color(color).convertSRGBToLinear(),transparent:true,opacity:1,depthTest:false,depthWrite:false}));
   mesh.renderOrder=5;scene.add(mesh);
 }
 function topPoints(pose){const h=p.box_half_size, r=pose.rotation, c=pose.position;
   return [[-h[0],-h[1],h[2]],[h[0],-h[1],h[2]],[h[0],h[1],h[2]],[-h[0],h[1],h[2]],[-h[0],-h[1],h[2]]].map(v=>r.map((row,i)=>row.reduce((s,x,j)=>s+x*v[j],0)+c[i]+(i===2?.006:0)));
 }
 if(p.name!=='humanoid_force_regulation'){
   thickOutline(topPoints(p.box_poses[0]),0x4b2e83);thickOutline(topPoints(p.box_poses.at(-1)),0xd9660e);
   line(p.box_poses.map(b=>[b.position[0],b.position[1],b.position[2]+p.box_half_size[2]+.010]),0xa94f0b,.95);
 }
 if(p.goal){const g=p.goal;const mesh=new THREE.Mesh(new THREE.BoxGeometry(...g.half_size.map(v=>v*2)),new THREE.MeshPhongMaterial({color:0x2b9868,transparent:true,opacity:.6}));mesh.position.fromArray(g.position);scene.add(mesh);}
 const camera=new THREE.PerspectiveCamera(35,p.width/p.height,.01,100);
 camera.up.fromArray(p.camera.up);camera.position.fromArray(p.camera.position);camera.lookAt(...p.camera.target);
 const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
 renderer.setSize(p.width,p.height);renderer.setPixelRatio(1);renderer.outputEncoding=THREE.sRGBEncoding;
 renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
 document.body.appendChild(renderer.domElement);renderer.render(scene,camera);
 // Render exact first/last robot silhouettes, then draw only their contours.
 // This is a visibility overlay: all saved body transforms remain unchanged.
 const targets=maskScenes.map(maskScene=>{
   const target=new THREE.WebGLRenderTarget(p.width,p.height,{depthBuffer:true,stencilBuffer:false});
   renderer.setRenderTarget(target);renderer.setClearColor(0x000000,0);renderer.clear();renderer.render(maskScene,camera);return target;
 });
 const overlayScene=new THREE.Scene();
 const overlayCamera=new THREE.OrthographicCamera(-1,1,1,-1,0,1);
 const overlayMaterial=new THREE.ShaderMaterial({transparent:true,depthTest:false,depthWrite:false,
   uniforms:{initialMask:{value:targets[0].texture},finalMask:{value:targets[1].texture},pixel:{value:new THREE.Vector2(1/p.width,1/p.height)}},
   vertexShader:`varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}`,
   fragmentShader:`uniform sampler2D initialMask;uniform sampler2D finalMask;uniform vec2 pixel;varying vec2 vUv;
   float contour(sampler2D mask,vec2 uv,float radius){float center=texture2D(mask,uv).r;float neighbor=0.;
     for(int k=0;k<16;k++){float angle=6.28318530718*float(k)/16.;vec2 shift=vec2(cos(angle),sin(angle))*pixel*radius;
       neighbor=max(neighbor,texture2D(mask,uv+shift).r);}
     return max(neighbor-center,0.);}
   void main(){float a=contour(initialMask,vUv,5.);float b=contour(finalMask,vUv,3.5);
     vec3 initial=vec3(75.,46.,131.)/255.;vec3 final=vec3(217.,102.,14.)/255.;
     gl_FragColor=vec4(b>.05?final:initial,max(a,b)*.98);}`});
 overlayScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),overlayMaterial));
 renderer.setRenderTarget(null);renderer.autoClear=false;renderer.render(overlayScene,overlayCamera);renderer.autoClear=true;
 document.title='MGA_Brax_READY';window.mgaBrax={renderer,scene,camera,system};
} catch(error){document.title='MGA_Brax_ERROR';document.body.textContent=error.stack;console.error(error);}
</script></body></html>'''


def _capture_brax_web(output_dir, force=False):
    """Capture exported Brax Web pages with an isolated local headless Chrome."""
    import functools
    import http.server
    import shutil
    import subprocess
    import threading
    import urllib.request

    chrome = (shutil.which("google-chrome") or shutil.which("chromium") or
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not Path(chrome).is_file():
        raise RuntimeError("Headless Chrome is required for genuine Brax Web capture")
    asset = output_dir / "brax_web/three.module.js"
    if not asset.exists():
        with urllib.request.urlopen("https://unpkg.com/three@0.150.1/build/three.module.js", timeout=30) as response:
            asset.write_bytes(response.read())
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass
    handler = functools.partial(QuietHandler, directory=str(output_dir.resolve()))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    metadata_path = output_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    try:
        for name, item in metadata.items():
            if not name.startswith("humanoid") or not item.get("web_html"):
                continue
            screenshot = (output_dir / f"{name}.png").resolve()
            html = _BRAX_WEB_HTML.replace("__PAYLOAD__", item["web_payload"])
            html_hash = hashlib.sha256(html.encode()).hexdigest()
            payload_hash = hashlib.sha256((output_dir / item["web_payload"]).read_bytes()).hexdigest()
            if (not force and not item.get("capture_pending", True)
                    and screenshot.exists()
                    and item.get("web_html_sha256") == html_hash
                    and item.get("web_payload_sha256") == payload_hash
                    and item.get("image_sha256") == hashlib.sha256(screenshot.read_bytes()).hexdigest()):
                continue
            # Rebuild the presentation layer so a lighting-only adjustment does
            # not require recomputing already exported saved-state transforms.
            (output_dir / item["web_html"]).write_text(html)
            width, height = item["image_size"]
            with tempfile.TemporaryDirectory(prefix="mga-brax-chrome-") as profile:
                command = [chrome, "--headless", "--hide-scrollbars", "--no-first-run", "--disable-dev-shm-usage",
                           "--no-default-browser-check", "--use-gl=angle", "--use-angle=swiftshader",
                           "--enable-unsafe-swiftshader", "--force-device-scale-factor=1",
                           f"--user-data-dir={profile}", f"--window-size={width},{height}",
                           "--virtual-time-budget=12000", "--run-all-compositor-stages-before-draw",
                           f"--screenshot={screenshot}", "--dump-dom",
                           f"http://127.0.0.1:{server.server_port}/{item['web_html']}"]
                if hasattr(os, "geteuid") and os.geteuid() == 0:
                    command.insert(1, "--no-sandbox")  # isolated disposable Linux capture container
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if result.returncode or "<title>MGA_Brax_READY</title>" not in result.stdout:
                raise RuntimeError(f"Brax Web capture failed for {name}: {result.stdout[-1600:]} {result.stderr[-1000:]}")
            item["capture_pending"] = False
            item["visual_alpha_override"] = {"final_box": 1.0, "source_box": 0.6,
                "initial_box_fill": False,
                "note": "Presentation only; initial box shown by true outline, final box opaque; geometry/FK/physical parameters unchanged"}
            item["box_motion_annotation"]["outline_colors"] = {
                "initial": "#4B2E83", "final": "#D9660E"}
            item["box_motion_annotation"]["outline_style"] = "solid tubes; true projected box-top corners"
            item["box_motion_annotation"]["outline_radius_m"] = 0.008
            item["box_motion_annotation"]["xray_outline"] = True
            item["box_motion_annotation"]["center_path_color"] = "#A94F0B"
            item["box_motion_annotation"]["visible"] = name != "humanoid_force_regulation"
            item["robot_pose_annotation"] = _robot_pose_annotation()
            item["ghost_opacity"] = [0.16]
            item["render_style"] = HUMANOID_RENDER_STYLE
            item["three_module_sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()
            item["web_html_sha256"] = hashlib.sha256(
                (output_dir / item["web_html"]).read_bytes()).hexdigest()
            item["web_payload_sha256"] = payload_hash
            item["image_sha256"] = hashlib.sha256(screenshot.read_bytes()).hexdigest()
            _persist_scene_metadata(metadata_path, {name: item})
            print(f"Captured genuine Brax Web scene: {screenshot}", flush=True)
    finally:
        server.shutdown()
        server.server_close()
    return 0


APPENDIX_GROUPS = {
    "surface_hard_rollouts": ("Surface scanning: rigid", [f"surface_rigid_{s}" for s in ("plane", "cylinder", "convex", "bumpy")]),
    "surface_soft_rollouts": ("Surface scanning: compliant", [f"surface_soft_{s}" for s in ("plane", "cylinder", "convex", "bumpy")]),
    "surface_hybrid_rollouts": ("Surface scanning: spatially hybrid", [f"surface_hybrid_{s}" for s in ("stripes", "center_hard", "center_soft")]),
    "surface_unseen_rollouts": ("Surface scanning: unseen", ["surface_rigid_unseen", "surface_soft_unseen"]),
    "peg_rollouts": ("Peg insertion", ["peg_id_wide", "peg_ood_pose", "peg_ood_sensing"]),
    "humanoid_rollouts": ("Humanoid contact tasks", ["humanoid_force_regulation", "humanoid_fixed_stance_push", "humanoid_unjamming"]),
}


def _appendix_scenes():
    scenes = {f"surface_{medium}_{shape}": ("surface", "arm/surface_scan", f"{medium}_{shape}")
              for medium in ("rigid", "soft")
              for shape in ("plane", "cylinder", "convex", "bumpy", "unseen")}
    scenes.update({f"surface_hybrid_{shape}": ("surface", "arm/surface_scan", f"hybrid_{shape}")
                   for shape in ("stripes", "center_hard", "center_soft")})
    scenes.update({f"peg_{suite}": ("peg", "arm/peg_insert", suite)
                   for suite in ("id_wide", "ood_pose", "ood_sensing")})
    scenes.update({name: value for name, value in SCENES.items() if value[0] == "humanoid"})
    return scenes


def _appendix_indices(trajectory):
    """Exclude explicitly marked held copies, not merely done=true states.

    PegInsert continues physically evolving after first success/done, whereas
    the H1 runner explicitly marks its repeated success-padding transitions.
    """
    import numpy as np
    states = trajectory["states"]
    valid = [i for i, state in enumerate(states)
             if state.get("q") is not None and state.get("qd") is not None]
    last = valid[-1]
    padding = trajectory.get("task_signals", {}).get("success_padding", [])
    first_padding = next((i for i, value in enumerate(padding) if value), None)
    if first_padding is not None:
        last = min(last, first_padding)
    indices = np.rint(np.linspace(0, last, 5)).astype(int).tolist()
    if len(set(indices)) != 5:
        raise ValueError(f"Only {last + 1} recorded physical states; cannot invent five distinct frames")
    return indices, last


def _appendix_arm_camera(model, states, task, closeup=False):
    import mujoco
    import numpy as np
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation = 128, -19
    if task == "surface":
        camera.lookat[:] = [0.26, 0.0, 0.45]
        camera.distance = 1.35
    else:
        data = mujoco.MjData(model)
        data.qpos[:] = states[0]["q"]
        data.qvel[:] = states[0]["qd"]
        mujoco.mj_forward(model, data)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "socket_entrance")
        if site < 0:
            candidates = [g for g in range(model.ngeom)
                          if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("socket_")]
            center = np.mean(data.geom_xpos[candidates], axis=0)
        else:
            center = data.site_xpos[site].copy()
        camera.lookat[:] = center + ([0, 0, 0.01] if closeup else [-.055, 0, .045])
        camera.distance = .27 if closeup else .95
        if closeup:
            camera.azimuth, camera.elevation = 128, -25
    return camera


def _render_appendix_arm(model, states, indices, camera, output, *, name):
    import imageio.v3 as imageio
    import mujoco
    width, height = 810, 540
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    data = mujoco.MjData(model)
    frames = []
    with mujoco.Renderer(model, height=height, width=width, max_geom=10000) as renderer:
        for number, index in enumerate(indices):
            data.qpos[:] = states[index]["q"]
            data.qvel[:] = states[index]["qd"]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            path = output / f"{name}_frame_{number}.png"
            imageio.imwrite(path, renderer.render())
            frames.append({"image": path.name, "state_index": index,
                           "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"frames": frames, "image_size": [width, height], "camera": {
        "lookat": camera.lookat.tolist(), "distance": float(camera.distance),
        "azimuth": float(camera.azimuth), "elevation": float(camera.elevation)}}


def _render_appendix(args):
    metadata_path = args.output_dir / "metadata.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    for name, (task, domain, suite) in _appendix_scenes().items():
        if args.task not in ("all", task) or (args.suite and args.suite != suite):
            continue
        result_path = ROOT / "results" / domain / "main/mga" / f"level_{suite}" / f"seed_{args.seed}" / "results.json"
        trajectory_path = result_path.parent / "trajectory/trajectory.json"
        result, trajectory = _read_json(result_path), _read_json(trajectory_path)
        trajectory_hash = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
        result_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
        indices, last = _appendix_indices(trajectory)
        previous = metadata.get(name, {})
        if (not args.force and previous.get("trajectory_sha256") == trajectory_hash
                and previous.get("result_sha256") == result_hash
                and previous.get("style") == "appendix-native-five-frames-v1"
                and previous.get("state_indices") == indices
                and (task == "humanoid" or all((args.output_dir / f["image"]).exists() for f in previous.get("frames", [])))
                and previous.get("state_indices")):
            print(f"Source-verified appendix scene present: {name}", flush=True)
            continue
        print(f"Appendix saved-state rendering: {name}", flush=True)
        dt = float(result["config_snapshot"].get("env_params", {}).get("dt", .02))
        env = _execution_env(result)
        model = _native_model(env)
        record = {"result": str(result_path.relative_to(ROOT)), "result_sha256": result_hash,
            "trajectory": str(trajectory_path.relative_to(ROOT)), "trajectory_sha256": trajectory_hash,
            "seed": args.seed, "suite": suite, "task": task, "method": "MGA",
            "state_indices": indices, "time_seconds": [i * dt for i in indices],
            "last_unpadded_state": last, "saved_state_count": len(trajectory["states"]),
            "terminal_state_done": bool(trajectory["states"][last].get("done", False)),
            "selection": "Five uniformly spaced distinct saved physical states through the last executed state; exclude only explicit success_padding transitions, not done=true states",
            "time_convention": "State i follows i control transitions; t=i*dt; initial state t=0",
            "dt": dt, "simulation_steps_executed": 0, "controller_rollouts_executed": 0,
            "style": "appendix-native-five-frames-v1",
            "execution_env_params": result["config_snapshot"].get("execution_env_params", {}),
            "geometry_source": "Saved result config_snapshot, including execution_env_params and recorded seed",
            "hybrid_note": "Stiffness is a task-chart field, not visible painted material; no invented stripes or deformation" if "hybrid" in suite else None}
        if task == "humanoid":
            record.update(_export_brax_web(env, model, trajectory, indices, last, name, args.output_dir))
            for unused in ("ghost_opacity", "robot_pose_annotation", "box_motion_annotation", "visual_alpha_override"):
                record.pop(unused, None)
            record["rendering"] = "Brax forward kinematics from saved q/qd; native Web scene builder; manuscript-v1-box appearance"
        else:
            model, checks = _restore_panda_visuals(model, trajectory["states"], indices)
            _manuscript_sky(model)
            camera = _appendix_arm_camera(model, trajectory["states"], task)
            record.update(_render_appendix_arm(model, trajectory["states"], indices, camera, args.output_dir, name=name))
            record["visual_restoration"] = checks
            record["rendering"] = "Native MuJoCo forward kinematics of saved q/qd, official Panda link visuals, manuscript blue-checker native materials"
            if task == "peg":
                close_camera = _appendix_arm_camera(model, trajectory["states"], task, closeup=True)
                closeup = _render_appendix_arm(model, trajectory["states"], indices, close_camera, args.output_dir, name=f"{name}_closeup")
                record["closeup"] = closeup
                if suite == "ood_sensing":
                    contact = next((i for i, state in enumerate(trajectory["states"][:last+1])
                                    if state.get("info", {}).get("contact_count", 0) > 0), None)
                    completion = next((i for i, state in enumerate(trajectory["states"][:last+1])
                                       if i > 0 and state.get("done", 0)), None)
                    event_indices = [0] + ([contact] if contact is not None else []) + ([completion] if completion is not None else []) + [last]
                    event_labels = ["initial"] + (["first_contact"] if contact is not None else []) + (["first_completion"] if completion is not None else []) + ["last_recorded"]
                    events = _render_appendix_arm(model, trajectory["states"], event_indices, close_camera, args.output_dir, name=f"{name}_events")
                    for frame, label in zip(events["frames"], event_labels):
                        source = args.output_dir / frame["image"]
                        target = args.output_dir / f"{name}_closeup_{label}.png"
                        source.replace(target)
                        frame.update(image=target.name, event=label, time_seconds=frame["state_index"] * dt)
                        if label == "first_completion":
                            import shutil
                            shutil.copyfile(target, args.output_dir / f"{name}_closeup_terminal.png")
                            frame["compatibility_alias"] = f"{name}_closeup_terminal.png"
                            frame["note"] = "First done/success, not final recorded state; later states physically evolve"
                    record["event_closeups"] = events
        for frame, seconds in zip(record.get("frames", []), record["time_seconds"]):
            frame["time_seconds"] = seconds
        for frame in record.get("closeup", {}).get("frames", []):
            frame["time_seconds"] = frame["state_index"] * dt
        _persist_scene_metadata(metadata_path, {name: record})
        print(f"Recorded {name}: indices={indices}, seconds={record['time_seconds']}", flush=True)
        # Environment construction is used only for geometry. Release JAX's
        # construction caches before the next suite; never call reset/step.
        del env, model
        import gc
        import jax
        jax.clear_caches()
        gc.collect()
    return 0


def _capture_appendix_brax(output_dir, force=False):
    import functools
    import http.server
    import shutil
    import subprocess
    import threading
    import time
    import render_env_humanoid as presentation
    chrome = (shutil.which("google-chrome") or shutil.which("chromium") or
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not Path(chrome).is_file():
        raise RuntimeError("Local Chrome/Chromium required for saved-state Brax capture")
    # Reuse the exact vetted, offline Three.js build used by the environment image.
    for asset in ("system.js", "three.module.js"):
        path = output_dir / "brax_web" / asset
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / "reports/mga/paper_figures/scenes/brax_web" / asset, path)
    ready = set()
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            if self.path.startswith("/__capture_ready__/"):
                ready.add(self.path.rsplit("/", 1)[-1])
                self.send_response(200)
                self.end_headers()
            else:
                super().do_GET()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(ROOT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    metadata_path = output_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    try:
        for name, item in metadata.items():
            if item.get("task") != "humanoid":
                continue
            frames = []
            payload = output_dir / item["web_payload"]
            payload_hash = hashlib.sha256(payload.read_bytes()).hexdigest()
            previous_frames = {f["image"]: f for f in item.get("frames", [])}
            for number, index in enumerate(item["state_indices"]):
                key = f"{name}_frame_{number}"
                image = (output_dir / f"{key}.png").resolve()
                settings = {"payload": "/" + str(payload.relative_to(ROOT)), "name": key,
                    "state": number, "width": 810, "height": 540,
                    "high": name == "humanoid_unjamming", "palette": presentation.PALETTES["manuscript-v1-box"]}
                html = output_dir / f"{key}.html"
                text = presentation.HTML.replace("__SETTINGS__", json.dumps(settings)).replace(
                    "/reports/mga/paper_figures/scenes/brax_web/", "/" + str((output_dir / "brax_web").relative_to(ROOT)) + "/")
                html.write_text(text)
                previous = previous_frames.get(image.name, {})
                cache_valid = (not force and image.exists()
                    and item.get("web_payload_sha256") == payload_hash
                    and previous.get("image_sha256") == hashlib.sha256(image.read_bytes()).hexdigest()
                    and previous.get("web_html_sha256") == hashlib.sha256(text.encode()).hexdigest())
                if not cache_valid:
                    with tempfile.TemporaryDirectory(prefix="mga-appendix-chrome-") as profile:
                        command = [chrome, "--headless=new", "--hide-scrollbars", "--no-first-run",
                            "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                            "--no-default-browser-check", "--disable-background-networking", "--disable-sync",
                            "--force-device-scale-factor=1", f"--user-data-dir={profile}", "--window-size=810,540",
                            "--virtual-time-budget=10000", "--run-all-compositor-stages-before-draw",
                            f"--screenshot={image}", "--dump-dom",
                            f"http://127.0.0.1:{server.server_port}/{html.relative_to(ROOT)}"]
                        if hasattr(os, "geteuid") and os.geteuid() == 0:
                            command.insert(1, "--no-sandbox")
                        with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(mode="w+") as stderr:
                            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
                            deadline = time.monotonic() + 40
                            captured = False
                            try:
                                while time.monotonic() < deadline:
                                    if key in ready and image.exists() and image.stat().st_size > 1000:
                                        captured = True
                                        break
                                    if process.poll() is not None:
                                        break
                                    time.sleep(.1)
                            finally:
                                if process.poll() is None:
                                    process.terminate()
                                try:
                                    process.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                    process.wait(timeout=5)
                            if not captured:
                                stdout.seek(0); stderr.seek(0)
                                raise RuntimeError(f"Brax capture failed: {stdout.read()[-1000:]} {stderr.read()[-1000:]}")
                frames.append({"image": image.name, "state_index": index,
                    "time_seconds": item["time_seconds"][number],
                    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "web_html": html.name,
                    "web_html_sha256": hashlib.sha256(html.read_bytes()).hexdigest()})
                print(f"Captured {image.name}", flush=True)
            item.update(frames=frames, image_size=[810, 540], capture_pending=False,
                palette=presentation.PALETTES["manuscript-v1-box"],
                camera={"position": [-.35,-3.8,5.6] if name == "humanoid_unjamming" else [-.55,-4.5,2.4],
                        "lookat": [.70,0,.64] if name == "humanoid_unjamming" else [.60,0,.88],
                        "projection": "orthographic", "fixed_across_row": True},
                web_payload_sha256=payload_hash)
            _persist_scene_metadata(metadata_path, {name: item})
    finally:
        server.shutdown()
        server.server_close()
    return 0


def _appendix_stiffness_maps(output_dir):
    """Export model-owned chart fields, never color native physical surfaces.

    Only the soft unseen constructor is materialized to verify the exact
    contact parameter and solref; no environment reset or step is called.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from genedynamics.core.contact.elastic_foundation import stiffness_field
    from genedynamics.envs.domains.manipulation.panda_brax import SurfaceScanConfig

    scene_metadata = _read_json(output_dir / "metadata.json")
    field_source = ROOT / "genedynamics/core/contact/elastic_foundation.py"
    env_source = ROOT / "genedynamics/envs/domains/manipulation/panda_brax.py"
    maps = {}
    coordinates = np.linspace(0., 1., 101)
    xx, yy = np.meshgrid(coordinates, coordinates)
    for name, item in scene_metadata.items():
        if item["task"] != "surface":
            continue
        result = _read_json(ROOT / item["result"])
        trajectory = _read_json(ROOT / item["trajectory"])
        snapshot = result["config_snapshot"]
        params = {**snapshot["env_params"], **snapshot.get("execution_env_params", {})}
        params.setdefault("surface_seed", int(item["seed"]))
        cfg = SurfaceScanConfig(**params)
        signals = trajectory["task_signals"]
        xi_start = float(np.asarray(signals["scan_start"]).ravel()[0])
        xi_target = float(np.asarray(signals["scan_target"]).ravel()[0])
        medium = str(cfg.medium).lower()
        origin = float(cfg.stiffness_map_xi_origin) if medium == "hybrid" else xi_start
        span = float(cfg.stiffness_map_xi_span) if medium == "hybrid" else xi_target - xi_start
        command_xi = np.asarray([trajectory["states"][i]["info"]["xi"] for i in item["state_indices"]])
        command_eta = np.asarray([trajectory["states"][i]["info"]["eta"] for i in item["state_indices"]])
        for index, xi in zip(item["state_indices"], command_xi):
            if index and not np.isclose(xi, signals["scan_xi"][index - 1], atol=1e-8, rtol=0):
                raise ValueError(f"Saved command-coordinate alignment differs: {name}, state {index}")
        entry = {"medium": medium, "source_result": item["result"], "result_sha256": item["result_sha256"],
            "source_trajectory": item["trajectory"], "trajectory_sha256": item["trajectory_sha256"],
            "seed": item["seed"], "state_indices": item["state_indices"], "time_seconds": item["time_seconds"],
            "chart_domain": {"tilde_xi": [0., 1.], "eta": [0., 1.]},
            "chart_transform": {"xi_origin": origin, "xi_span": span,
                "formula": "tilde_xi=clip((command_xi-xi_origin)/max(xi_span,1e-6),0,1)",
                "role": "task-owned material chart" if medium == "hybrid" else "display normalization of reference scan segment; field is uniform/categorical"},
            "shown_command_xi": command_xi.tolist(), "shown_command_eta": command_eta.tolist(),
            "shown_chart_xi": np.clip((command_xi-origin)/max(span, 1e-6), 0, 1).tolist(),
            "map_source": str(field_source.relative_to(ROOT)), "map_source_sha256": hashlib.sha256(field_source.read_bytes()).hexdigest(),
            "environment_source": str(env_source.relative_to(ROOT)), "environment_source_sha256": hashlib.sha256(env_source.read_bytes()).hexdigest(),
            "color_scale_N_per_m": [0., 8000.], "units": "N/m", "native_frames_modified": False,
            "map_interpretation": "Separate nominal reference-chart map; not EE world position, physical surface texture, or measured deformation",
            "saved_nonhybrid_k_surf_warning": "The extractor calls _k_surf_fn even outside hybrid; that diagnostic is not the soft MuJoCo contact parameter" if medium != "hybrid" else None}
        if medium == "rigid":
            entry.update(kind="categorical_rigid", values_N_per_m=None,
                         display_line="Rigid contact", display_note="No finite stiffness assigned to this map")
        elif medium == "soft":
            value = float(cfg.soft_stiffness)
            solref = f"{0.02 * (1.0e4 / max(value, 1.0)):.4f} 1"
            if str(cfg.level).lower() == "unseen":
                # Same task-owned seed/key split, independently checked against
                # the exact execution environment's constructed contact model.
                stiffness_key, _ = jax.random.split(jax.random.PRNGKey(int(cfg.surface_seed) + 9973))
                lo, hi = cfg.s4_stiffness_range
                sampled = float(jax.random.uniform(stiffness_key, minval=lo, maxval=hi))
                env = _execution_env(result)
                value = float(env._contact_stiffness_draw)
                solref = str(env._contact_solref)
                if not np.isclose(value, sampled, atol=1e-4, rtol=0):
                    raise ValueError("Soft unseen constructor and independent key-split reconstruction differ")
                if solref != f"{0.02 * (1.0e4 / max(value, 1.0)):.4f} 1":
                    raise ValueError("Soft unseen contact solref does not match the reconstructed parameter")
                entry["randomization"] = {"range_N_per_m": [float(lo), float(hi)],
                    "source": "SurfaceScanEnv constructor: first key of jax.random.split(PRNGKey(surface_seed+9973))",
                    "surface_seed": int(cfg.surface_seed), "constructor_draw_N_per_m": value,
                    "independent_draw_N_per_m": sampled, "constructor_verified": True,
                    "no_reset_or_step": True}
                del env
            entry.update(kind="uniform_contact_parameter", value_N_per_m=value, values_N_per_m=[[value]],
                contact_solref=solref, display_line=f"Uniform contact parameter: {value/1000:.2f} kN/m",
                display_note=("Unseen draw; parameter range 0.8–2.0 kN/m" if str(cfg.level).lower() == "unseen"
                              else "Uniform nominal contact parameter"))
        elif medium == "hybrid":
            def local_stiffness(xi, eta):
                transformed = jnp.clip((jnp.asarray(xi)-origin)/max(span, 1e-6), 0., 1.)
                return stiffness_field(cfg.stiffness_map, transformed, eta, cfg.k_hard, cfg.k_soft,
                                       cfg.stiffness_transition_width)
            values = np.asarray(local_stiffness(origin + span * xx, jnp.asarray(yy)))
            # Center maps depend on eta as well as xi. The command can drift
            # slightly from the reference eta=.5; use the saved command state,
            # exactly as arm_surface_scan_signals.per_step does.
            eta_signal = np.asarray([state["info"]["eta"]
                for state in trajectory["states"][1:len(signals["scan_xi"])+1]])
            expected = np.asarray(local_stiffness(jnp.asarray(signals["scan_xi"]), jnp.asarray(eta_signal)))
            saved = np.asarray(signals["k_surf"])
            error = float(np.max(np.abs(expected-saved)))
            if not np.allclose(expected, saved, atol=.01, rtol=0):
                raise ValueError(f"Task-owned stiffness field differs from saved command-coordinate k_surf: {name}: {error}")
            entry.update(kind="hybrid_task_field", values_N_per_m=values.tolist(),
                stiffness_map=str(cfg.stiffness_map), k_soft_N_per_m=float(cfg.k_soft), k_hard_N_per_m=float(cfg.k_hard),
                transition_width=float(cfg.stiffness_transition_width), grid_min_N_per_m=float(values.min()),
                grid_max_N_per_m=float(values.max()), verification={"signal": "k_surf", "coordinate": "saved scan_xi and post-action state.info.eta",
                    "eta_range": [float(eta_signal.min()), float(eta_signal.max())],
                    "samples": int(saved.size), "maximum_absolute_error_N_per_m": error, "absolute_tolerance_N_per_m": .01},
                display_line=f"Nominal regions: {cfg.k_soft/1000:g}–{cfg.k_hard/1000:g} kN/m",
                display_note="Smoothed field in the transformed reference chart")
        else:
            raise ValueError(f"Unsupported surface medium: {medium}")
        maps[name] = entry
        print(f"Verified stiffness map {name}: {entry['display_line']}", flush=True)
    path = output_dir.parent / "stiffness_maps.json"
    path.write_text(json.dumps(maps, indent=2) + "\n")
    return 0


def _draw_appendix_stiffness_header(fig, map_item, label, row_top, frame_top, compact=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    height = fig.get_figheight()
    def text(x, y, value, fontsize, **kwargs):
        return fig.text(x/12, y/height, value, fontsize=fontsize, color="#3A4655", **kwargs)
    text(.14, row_top-.10, label, 16, va="top", fontweight="bold")
    text(.14, row_top-.49, map_item["display_line"], 15, va="top")
    if not compact:
        text(.14, row_top-.84, map_item["display_note"], 14, va="top")
    text(7.04, row_top-(.49 if compact else .62), "Nominal chart\n" + r"$(\tilde\xi,\eta)\in[0,1]^2$", 14,
         ha="right", va="center", linespacing=1.4)
    map_size = .8 if compact else 1.06
    ax = fig.add_axes([7.26/12, (frame_top+(.12 if compact else .14))/height,
                       map_size/12, map_size/height])
    cmap = LinearSegmentedColormap.from_list("appendix_stiffness_cool", [
        (0., "#E7F3F0"), (.25, "#70B7AF"), (1., "#4B2E83")])
    norm = Normalize(0, 8)
    if map_item["kind"] == "categorical_rigid":
        ax.set_facecolor("#E4E7EA")
        text(8.75, frame_top+(.68 if compact else .82), "Rigid contact", 15, va="center")
        text(8.75, frame_top+(.40 if compact else .49), "Categorical; no finite K", 14, va="center")
    else:
        ax.imshow(np.asarray(map_item["values_N_per_m"])/1000, origin="lower", extent=[0,1,0,1],
                  cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
        cax = fig.add_axes([8.75/12, (frame_top+(.50 if compact else .62))/height,
                           2.92/12, (.10 if compact else .14)/height])
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                     ticks=[0, 2, 4, 8])
        cax.tick_params(labelsize=13.5, pad=1.5, length=2)
        text(10.21, frame_top+(.84 if compact else .99), r"Contact parameter $K$ [kN/m]", 15,
             ha="center", va="center")
    ax.plot([0,1], [.5,.5], color="white", linewidth=1.2, linestyle="--", alpha=.95)
    ax.scatter(map_item["shown_chart_xi"], map_item["shown_command_eta"], s=23,
               c="white", edgecolors="#233F55", linewidths=.8, zorder=4, clip_on=False)
    ax.set(xlim=(0,1), ylim=(0,1), xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_color("#7B8B95"); spine.set_linewidth(.6)
    text(8.75, frame_top+(.13 if compact else .17), "Dots: shown command locations", 13.5, va="center")


def _assemble_appendix(output_dir, selected_groups=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metadata = _read_json(output_dir / "metadata.json")
    maps_path = output_dir.parent / "stiffness_maps.json"
    if not maps_path.exists():
        raise ValueError("Export verified fields first: --appendix --stiffness-maps-only")
    stiffness_maps = _read_json(maps_path)
    verified = {}
    for name, item in metadata.items():
        trajectory_path, result_path = ROOT / item["trajectory"], ROOT / item["result"]
        if (hashlib.sha256(trajectory_path.read_bytes()).hexdigest() != item["trajectory_sha256"]
                or hashlib.sha256(result_path.read_bytes()).hexdigest() != item["result_sha256"]):
            raise ValueError(f"Source changed after scene rendering: {name}")
        trajectory = _read_json(trajectory_path)
        expected_indices, expected_last = _appendix_indices(trajectory)
        if item["state_indices"] != expected_indices:
            print(f"Awaiting refreshed actual-episode frame selection: {name}", flush=True)
            continue
        if len(item.get("frames", [])) != 5:
            continue
        for frame in item["frames"] + item.get("closeup", {}).get("frames", []):
            path = output_dir / frame["image"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != frame["image_sha256"]:
                raise ValueError(f"Frame changed after rendering: {path}")
        item["selection"] = "Five uniformly spaced saved physical states through last executed state; exclude explicit success_padding only, not done=true states"
        item["first_done_state"] = next((i for i, state in enumerate(trajectory["states"])
                                         if i > 0 and state.get("done", 0)), None)
        item["last_unpadded_state"] = expected_last
        item["excluded_padding_states"] = len(trajectory["states"]) - expected_last - 1
        item["fixed_camera_across_row"] = True
        if item["task"] == "surface":
            material = stiffness_maps[name]
            for key in ("result_sha256", "trajectory_sha256"):
                if material[key] != item[key]:
                    raise ValueError(f"Stiffness-map source changed: {name}")
            for path_key, hash_key in (("map_source", "map_source_sha256"),
                                       ("environment_source", "environment_source_sha256")):
                if hashlib.sha256((ROOT / material[path_key]).read_bytes()).hexdigest() != material[hash_key]:
                    raise ValueError(f"Task-owned field implementation changed: {name}")
        for frame in item.get("closeup", {}).get("frames", []):
            frame["time_seconds"] = frame["state_index"] * item["dt"]
        if item["task"] == "humanoid":
            for unused in ("ghost_opacity", "robot_pose_annotation", "box_motion_annotation", "visual_alpha_override"):
                item.pop(unused, None)
            item["frame_annotations"] = "No ghosts, pose contours, or invented motion; one actual saved pose per frame"
            if "web_html" in item:
                item["unused_export_preview_html"] = item.pop("web_html")
            for frame in item["frames"]:
                frame["web_html"] = Path(frame["image"]).with_suffix(".html").name
        verified[name] = item
    _persist_scene_metadata(output_dir / "metadata.json", verified)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    labels = {"peg_id_wide": "ID", "peg_ood_pose": "Pose-OOD", "peg_ood_sensing": "Sensing-OOD",
        "humanoid_force_regulation": "Force regulation · 15 N", "humanoid_fixed_stance_push": "Fixed-stance push · nominal",
        "humanoid_unjamming": "Unjamming"}
    grids_metadata_path = output_dir.parent / "rollout_grids_metadata.json"
    assembled = _read_json(grids_metadata_path) if selected_groups and grids_metadata_path.exists() else {}
    for filename, (title, names) in APPENDIX_GROUPS.items():
        if selected_groups and filename not in selected_groups:
            continue
        if any(name not in verified for name in names):
            print(f"Not yet complete, skip grid {filename}", flush=True)
            continue
        rows = len(names)
        with_maps = verified[names[0]]["task"] == "surface"
        compact_maps = filename in ("surface_hybrid_rollouts", "surface_unseen_rollouts")
        row_height = 2.76 if compact_maps else 3.18
        header_height = .98 if compact_maps else 1.35
        figure_height = rows * row_height + .75 if with_maps else rows * 2.08 + 1.03
        fig = plt.figure(figsize=(12, figure_height), facecolor="white")
        if not with_maps:
            grid = fig.add_gridspec(rows, 5, left=.012, right=.988, bottom=.35/figure_height, top=1-.88/figure_height,
                                   wspace=.018, hspace=.40)
        for row, name in enumerate(names):
            item = verified[name]
            label = labels.get(name, name.removeprefix("surface_").replace("_", " ").capitalize())
            if with_maps:
                row_top = figure_height - .67 - row * row_height
                frame_top = row_top - header_height
                _draw_appendix_stiffness_header(fig, stiffness_maps[name], label, row_top, frame_top,
                                                 compact=compact_maps)
            row_frames = item["closeup"]["frames"] if item["task"] == "peg" else item["frames"]
            for column, frame in enumerate(row_frames):
                if with_maps:
                    image_width = (12 - .28 - 4*.06) / 5
                    image_height = image_width * 2/3
                    ax = fig.add_axes([(.14 + column*(image_width+.06))/12,
                        (frame_top-image_height)/figure_height, image_width/12, image_height/figure_height])
                else:
                    ax = fig.add_subplot(grid[row, column])
                ax.imshow(plt.imread(output_dir / frame["image"]))
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xlabel(f"t = {frame['time_seconds']:.2f} s", fontsize=15, labelpad=3, color="#3A4655")
                if column == 0 and not with_maps:
                    ax.text(0, 1.055, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=16,
                            color="#3A4655", fontweight="bold")
        fig.text(.012, 1-.10/figure_height, f"{title}  |  MGA", ha="left", va="top", fontsize=18,
                 color="#3A4655", fontweight="bold")
        for suffix in ("png", "pdf"):
            path = output_dir.parent / f"{filename}.{suffix}"
            fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        assembled[filename] = {"scene_rows": names, "frames_per_row": 5,
            "seed": 0, "sources": str((output_dir / "metadata.json").relative_to(ROOT)),
            "typography_pt": {"title": 18, "row_label": 16, "timestamp": 15},
            "stiffness_maps": {"source": str(maps_path.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(maps_path.read_bytes()).hexdigest(),
                "scene_rows": names, "display": "Separate transformed reference-chart panel; native frames unchanged",
                "color_scale_kN_per_m": [0, 8], "rigid_scale": "categorical; no finite stiffness",
                "caption": "Separate nominal reference-chart maps accompany each native rollout. Rigid contact is categorical; compliant values are uniform contact-model parameters, not measured moduli. Hybrid fields use the task-owned smoothed map at transformed command coordinates, validated against every saved k_surf sample. White dots locate the five displayed command states, not actual end-effector world positions. No physical texture or deformation is painted onto the native scenes."} if with_maps else None,
            "row_views": {name: "closeup" if verified[name]["task"] == "peg" else "overview" for name in names},
            "row_frame_images": {name: [f["image"] for f in (verified[name]["closeup"]["frames"]
                if verified[name]["task"] == "peg" else verified[name]["frames"])] for name in names},
            "selection": "Uniform chronological samples from each actual unpadded episode; terminal outcomes unchanged"}
        if compact_maps:
            assembled[filename]["layout"] = {"compact_chart_header": True,
                "canvas_width_inches": 12, "canvas_height_inches": figure_height,
                "row_height_inches": row_height, "header_height_inches": header_height,
                "chart_width_inches": .8, "display_note_in_visual_header": False,
                "native_frame_width_inches": image_width, "native_frame_height_inches": image_height}
        print(f"Assembled {filename}.pdf/.png", flush=True)
    grids_metadata_path.write_text(json.dumps(assembled, indent=2) + "\n")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "surface", "peg", "humanoid"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/mga/paper_figures/scenes")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--capture-brax", action="store_true",
                        help="Host-only: capture the exported genuine Brax Web humanoid pages")
    parser.add_argument("--appendix", action="store_true",
                        help="Render five true chronological saved poses for each appendix suite")
    parser.add_argument("--assemble-only", action="store_true",
                        help="Assemble appendix grids from already rendered frames")
    parser.add_argument("--grid", action="append", choices=list(APPENDIX_GROUPS),
                        help="Appendix assemble-only: update only this grid (repeatable), preserving others")
    parser.add_argument("--stiffness-maps-only", action="store_true",
                        help="Export exact task-owned scanning chart maps and verify saved stiffness signals")
    parser.add_argument("--suite", help="Appendix: render one exact suite name")
    args = parser.parse_args(argv)
    if args.appendix:
        if args.output_dir == ROOT / "reports/mga/paper_figures/scenes":
            args.output_dir = ROOT / "reports/mga/paper_figures/appendix/scenes"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.stiffness_maps_only:
            return _appendix_stiffness_maps(args.output_dir)
        if args.assemble_only:
            return _assemble_appendix(args.output_dir, args.grid)
        if args.capture_brax:
            return _capture_appendix_brax(args.output_dir, args.force)
        return _render_appendix(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.capture_brax:
        return _capture_brax_web(args.output_dir, args.force)
    import numpy as np
    metadata_path = args.output_dir / "metadata.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    for name, (task, domain, suite) in SCENES.items():
        if args.task not in ("all", task):
            continue
        output = args.output_dir / f"{name}.png"
        result_path = ROOT / "results" / domain / "main/mga" / f"level_{suite}" / f"seed_{args.seed}" / "results.json"
        trajectory_path = result_path.parent / "trajectory/trajectory.json"
        trajectory_hash = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
        result_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
        previous = metadata.get(name, {})
        render_style = HUMANOID_RENDER_STYLE if task == "humanoid" else RENDER_STYLE
        detail_ready = (name != "surface_hybrid_stripes" or (
            (args.output_dir / f"{name}_detail.png").exists()
            and metadata.get(f"{name}_detail", {}).get("trajectory_sha256") == trajectory_hash
            and metadata.get(f"{name}_detail", {}).get("seed") == args.seed))
        filmstrip_ready = (name != FILMSTRIP_SCENE or
            all((args.output_dir / f"surface_keyframe_{i}.png").exists()
                    and metadata.get(f"surface_keyframe_{i}", {}).get("render_style") == RENDER_STYLE
                    and metadata.get(f"surface_keyframe_{i}", {}).get("trajectory_sha256") == trajectory_hash
                    and metadata.get(f"surface_keyframe_{i}", {}).get("seed") == args.seed
                    for i in range(4)))
        if (output.exists() and not args.force and detail_ready and filmstrip_ready
                and previous.get("render_style") == render_style
                and previous.get("seed") == args.seed
                and previous.get("trajectory") == str(trajectory_path.relative_to(ROOT))
                and previous.get("result_sha256") == result_hash
                and previous.get("trajectory_sha256") == trajectory_hash):
            print(f"Present, source verified: {output}", flush=True)
            continue
        print(f"Rendering {name} from {result_path.relative_to(ROOT)}", flush=True)
        result, trajectory = _read_json(result_path), _read_json(trajectory_path)
        env = _execution_env(result)
        model = _native_model(env)
        indices, last = _state_indices(trajectory, name, model)
        visual_restoration = None
        if task == "surface":
            model, visual_restoration = _restore_panda_visuals(model, trajectory["states"], indices)
        if task == "surface":
            _manuscript_sky(model)
        positions = trajectory.get("task_signals", {}).get("positions")
        path = np.asarray(positions) if task == "surface" else None
        render_meta = (_export_brax_web(env, model, trajectory, indices, last, name, args.output_dir)
                       if task == "humanoid" else _render_pose_scene(model, trajectory["states"], indices, path,
                                        name, output,
                                        box_last=None))
        metadata[name] = {
            "result": str(result_path.relative_to(ROOT)),
            "result_sha256": result_hash,
            "trajectory": str(trajectory_path.relative_to(ROOT)),
            "trajectory_sha256": trajectory_hash,
            "seed": args.seed, "state_indices": indices, "last_unpadded_state": last,
            "render_style": render_style,
            "style_reference": (["results/arm/impedence/rigid/bumpy/arm_render_bumpy.gif"]
                if task == "surface" else ["accepted bright-gray Brax Web H1 scenes"]),
            "pose_selection": "true initial state 0 and last unpadded executed state only" if task == "humanoid" else "uniform chronological poses",
            "visual_restoration": visual_restoration,
            "surface_presentation_override": (visual_restoration or {}).get("surface_presentation_override"),
            "time_seconds": [i * float(result["config_snapshot"]["env_params"].get("dt", 0.02)) for i in indices],
            "rendering": "native MuJoCo saved q/qd forward kinematics; no physics or controller rerun",
            "hybrid_material_note": "No physical stripes added: hybrid stiffness is chart-dependent" if "stripes" in name else None,
            **render_meta,
        }
        if name == "surface_hybrid_stripes":
            detail_output = args.output_dir / f"{name}_detail.png"
            metadata[f"{name}_detail"] = {
                **metadata[name], **_render_pose_scene(model, trajectory["states"], indices,
                                                       path, name, detail_output, detail=True)}
        if name == FILMSTRIP_SCENE:
            views = ["overview", "side_contact", "top_contact", "oblique_contact"]
            keyframes = np.rint(np.linspace(0, last, 4)).astype(int).tolist()
            dt = float(result["config_snapshot"]["env_params"].get("dt", 0.02))
            for number, (index, view) in enumerate(zip(keyframes, views)):
                key = f"surface_keyframe_{number}"
                frame_output = args.output_dir / f"{key}.png"
                frame_meta = _render_pose_scene(model, trajectory["states"], [index],
                    path[:index], name, frame_output, size=(960, 600), view=view)
                metadata[key] = {**metadata[name], **frame_meta,
                    "state_indices": [index], "time_seconds": [index * dt],
                    "camera_label": view.replace("_", " "),
                    "pose_selection": "four time-ordered bins of the same executed MGA episode; varying labeled views"}
        updated_keys = [name]
        if name == "surface_hybrid_stripes":
            updated_keys.append(f"{name}_detail")
        if name == FILMSTRIP_SCENE:
            updated_keys.extend(f"surface_keyframe_{i}" for i in range(4))
        _persist_scene_metadata(metadata_path, {key: metadata[key] for key in updated_keys})
        print(f"Exported Brax Web scene: {args.output_dir / render_meta['web_html']}" if task == "humanoid"
              else f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
