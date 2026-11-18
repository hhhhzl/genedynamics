"""Evidence-backed visualization for the four transfer tasks.

This module belongs to the task layer because it interprets the task-owned
episode evidence schema.  Paper scripts only parse command-line arguments.
The renderer never advances physics: each frame restores the recorded public
joint state and calls ``mj_forward``.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .episode import load_episode_evidence
from .specs import mechanism_spec


def replay_frame_indices(state_count: int, maximum_frames: int) -> np.ndarray:
    """Return monotone indices including the first and last recorded states."""
    if state_count <= 0:
        raise ValueError("episode has no recorded states")
    if maximum_frames <= 0:
        raise ValueError("maximum_frames must be positive")
    count = min(state_count, maximum_frames)
    return np.unique(np.rint(np.linspace(0, state_count - 1, count)).astype(int))


def _scene_for(spec):
    """Reconstruct the execution geometry from the evidence's exact task config."""
    mechanism = mechanism_spec(spec.alias)
    if spec.alias == "T1":
        from genedynamics.envs.domains.manipulation.transfer_workcell.workcell import (
            build_workcell,
        )

        return build_workcell(spec.config, mechanism, validate_scene=False)
    if spec.alias in {"T2", "T3"}:
        from genedynamics.envs.domains.manipulation.transfer_workcell.contact_workcell import (
            build_contact_workcell,
        )

        return build_contact_workcell(spec.config, mechanism)
    if spec.alias == "T4":
        from genedynamics.envs.domains.manipulation.transfer_workcell.floating_workcell import (
            build_go2_z1,
        )

        scene, _ = build_go2_z1(spec.config, mechanism)
        return scene
    raise ValueError(f"unsupported transfer task alias: {spec.alias}")


def _recorded_states(record, model):
    initial = record.reset_evidence.get("physical_state", {})
    states = []
    if len(initial.get("qpos", ())) == model.nq:
        states.append(
            {
                "qpos": np.asarray(initial["qpos"], dtype=float),
                "qvel": np.asarray(initial.get("qvel", np.zeros(model.nv)), dtype=float),
                "normal_force": 0.0,
                "control_index": 0,
            }
        )
    for trace_index, trace in enumerate(record.traces):
        if not trace.get("valid", False):
            continue
        qpos = np.asarray(trace.get("joint_position", ()), dtype=float)
        qvel = np.asarray(trace.get("joint_velocity", ()), dtype=float)
        if qpos.shape != (model.nq,) or qvel.shape != (model.nv,):
            raise ValueError(
                f"recorded state shape {(qpos.shape, qvel.shape)} differs from "
                f"replay model {((model.nq,), (model.nv,))}"
            )
        states.append(
            {
                "qpos": qpos,
                "qvel": qvel,
                "normal_force": float(trace.get("normal_force", 0.0)),
                "control_index": trace_index + 1,
            }
        )
    if not states:
        raise ValueError("selected record contains no valid physical states")
    return states


def _absolute_assets(donor_path: Path, donor_root, *, name_prefix: str = ""):
    mesh_dir = donor_root.find("compiler").get("meshdir", "")
    result = []
    for source in donor_root.find("asset") or ():
        element = copy.deepcopy(source)
        file_name = element.get("file")
        if name_prefix:
            source_name = element.get("name") or (
                Path(file_name).stem if file_name else None
            )
            if source_name:
                element.set("name", name_prefix + source_name)
        if file_name:
            element.set("file", str((donor_path.parent / mesh_dir / file_name).resolve()))
        result.append(element)
    return result


def _copy_mesh_geoms(target_root, donor_root, *, body_prefix="", mesh_prefix=""):
    target_bodies = {body.get("name"): body for body in target_root.findall(".//body")}
    copied = 0
    source_bodies = donor_root.findall(".//body")
    for source_body in source_bodies:
        source_name = source_body.get("name")
        target = target_bodies.get(body_prefix + source_name)
        if target is None:
            continue
        # Collision proxies remain in the replay model but are hidden during
        # presentation.  They continue to define the geometry provenance.
        for existing in target.findall("geom"):
            if existing.get("mesh") is None:
                existing.set("group", "3")
        for source in source_body.findall("geom"):
            if source.get("mesh") is None:
                continue
            geom = copy.deepcopy(source)
            geom.attrib.pop("class", None)
            geom.set("mesh", mesh_prefix + geom.get("mesh"))
            if geom.get("material") and mesh_prefix:
                geom.set("material", mesh_prefix + geom.get("material"))
            geom.attrib.update(
                type="mesh", contype="0", conaffinity="0", group="2", mass="0"
            )
            target.append(geom)
            copied += 1
    return copied


def _presentation_model(scene, spec, states, *, frame_opacity=1.0):
    """Restore vendored visual-only meshes and prove recorded-pose FK parity."""
    import mujoco

    root = ET.fromstring(scene.xml)
    visual_settings = root.find("visual")
    if visual_settings is None:
        visual_settings = ET.SubElement(root, "visual")
    headlight = visual_settings.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual_settings, "headlight")
    headlight.attrib.update(
        ambient="0.45 0.45 0.45",
        diffuse="0.75 0.75 0.75",
        specular="0.15 0.15 0.15",
    )
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "light",
        name="paper_key_light",
        pos="1 -1 2.2",
        dir="-0.25 0.15 -1",
        diffuse="0.8 0.8 0.8",
        specular="0.1 0.1 0.1",
        directional="true",
    )
    for geom in world.findall("geom"):
        name = geom.get("name", "")
        if name == "floor":
            geom.set("rgba", "0.14 0.16 0.20 1")
        elif name.startswith(("guide_", "socket_", "keyway_")):
            geom.set("rgba", "0.52 0.58 0.66 1")
        elif name.startswith(("aperture_", "aperture2_")) and frame_opacity < 1.0:
            color = "0.33 0.68 0.88" if name.startswith("aperture_") else "0.66 0.48 0.85"
            geom.set("rgba", f"{color} {frame_opacity}")
    assets = root.find("asset")
    if assets is None:
        assets = ET.Element("asset")
        root.insert(1, assets)
    sources = []
    if spec.config.robot == "go2_z1":
        repository = Path(__file__).resolve().parents[3]
        donors = (
            (repository / "third_party/mujoco_menagerie/unitree_go2/go2.xml", "", ""),
            (repository / "third_party/mujoco_menagerie/unitree_z1/z1.xml", "z1_", "z1_"),
        )
    else:
        from genedynamics.robots.panda.profile import panda_profile
        from genedynamics.robots.ur5e.profile import ur5e_profile
        from genedynamics.robots.xarm7.profile import xarm7_profile

        profile = {
            "panda": panda_profile,
            "xarm7": xarm7_profile,
            "ur5e": ur5e_profile,
        }[spec.config.robot]()
        donors = ((Path(profile.model_path()), "", ""),)
    copied = 0
    for donor_path, body_prefix, asset_prefix in donors:
        donor_root = ET.parse(donor_path).getroot()
        for element in _absolute_assets(
            donor_path, donor_root, name_prefix=asset_prefix
        ):
            assets.append(element)
        copied += _copy_mesh_geoms(
            root,
            donor_root,
            body_prefix=body_prefix,
            mesh_prefix=asset_prefix,
        )
        sources.append(str(donor_path))
    if copied == 0:
        raise ValueError(f"no vendored visual meshes restored for {spec.config.robot}")
    visual = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if (visual.nq, visual.nv) != (scene.model.nq, scene.model.nv):
        raise ValueError("visual mesh restoration changed state dimensions")
    old_data, new_data = mujoco.MjData(scene.model), mujoco.MjData(visual)
    maximum_error = 0.0
    check_indices = replay_frame_indices(len(states), min(16, len(states)))
    for state_index in check_indices:
        for model, data in ((scene.model, old_data), (visual, new_data)):
            data.qpos[:] = states[int(state_index)]["qpos"]
            data.qvel[:] = states[int(state_index)]["qvel"]
            mujoco.mj_forward(model, data)
        for body_id in range(scene.model.nbody):
            name = scene.model.body(body_id).name
            visual_id = mujoco.mj_name2id(visual, mujoco.mjtObj.mjOBJ_BODY, name)
            if visual_id < 0:
                raise ValueError(f"visual replay model lost body {name}")
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(old_data.xpos[body_id] - new_data.xpos[visual_id]))),
                float(np.max(np.abs(old_data.xmat[body_id] - new_data.xmat[visual_id]))),
            )
    if maximum_error > 1e-8:
        raise ValueError(f"visual mesh restoration changed saved-pose FK by {maximum_error}")
    return visual, {
        "sources": sources,
        "mesh_geoms": copied,
        "physics_modified": False,
        "maximum_saved_pose_fk_error": maximum_error,
        "aperture_frame_opacity": frame_opacity,
        "opacity_scope": "presentation only; original collision geometry retained",
    }


def _camera(spec, states, model):
    import mujoco

    data = mujoco.MjData(model)
    points = []
    site_name = "mobile_tcp" if spec.alias == "T4" else (
        "attachment_site" if spec.config.robot == "ur5e" else "attachment_site"
    )
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        # Scene manifests bind the tool site under robot-specific names.  The
        # recorded TCP path is used below when a presentation site is absent.
        points = []
    else:
        for state in states[:: max(1, len(states) // 32)]:
            data.qpos[:] = state["qpos"]
            data.qvel[:] = state["qvel"]
            mujoco.mj_forward(model, data)
            points.append(data.site_xpos[site_id].copy())
    center = np.mean(points, axis=0) if points else np.array([.42, -.10, .48])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center
    camera.azimuth = 135.0 if spec.alias != "T4" else 150.0
    camera.elevation = -24.0
    camera.distance = 1.05 if spec.alias != "T4" else 1.6
    return camera


def _annotate(frame, lines):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(12, round(image.width / 60)))
    except OSError:
        font = ImageFont.load_default()
    line_height = max(19, round(image.width / 40))
    height = line_height * len(lines) + 12
    text_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
    draw.rounded_rectangle((8, 8, min(image.width - 8, text_width + 30), 8 + height),
                           radius=6, fill=(12, 18, 28, 190))
    for index, line in enumerate(lines):
        draw.text((17, 14 + line_height * index), line, font=font, fill=(245, 247, 250, 255))
    return image


def recorded_event_at_time(episode_evaluation, elapsed, physics_dt):
    """Display only evaluator-recorded events already reached in replay time."""
    events = episode_evaluation.get("event_physics_steps", {})
    reached = [(int(step), name) for name, step in events.items()
               if step is not None and int(step) * physics_dt <= elapsed + 1e-9]
    if not reached:
        return "approach / carry"
    return max(reached)[1].replace("_", " ")


def render_episode_evidence(
    evidence_path,
    output_path,
    *,
    record_index: int = 0,
    maximum_frames: int = 120,
    fps: float = 12.0,
    width: int = 640,
    height: int = 480,
    title: str | None = None,
    subtitle: str | None = None,
    frame_opacity: float = 1.0,
    camera_azimuth: float | None = None,
    camera_elevation: float | None = None,
    camera_distance: float | None = None,
):
    """Render one recorded episode to GIF and write a provenance manifest."""
    import mujoco

    evidence_path = Path(evidence_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.suffix.lower() != ".gif":
        raise ValueError("output path must end in .gif")
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("fps and image dimensions must be positive")
    if not np.isfinite(frame_opacity) or not 0.0 < frame_opacity <= 1.0:
        raise ValueError("frame_opacity must be in (0, 1]")
    for value in (camera_azimuth, camera_elevation, camera_distance):
        if value is not None and not np.isfinite(value):
            raise ValueError("camera overrides must be finite")
    if camera_distance is not None and camera_distance <= 0:
        raise ValueError("camera_distance must be positive")
    records, spec, evaluation = load_episode_evidence(evidence_path)
    if not 0 <= record_index < len(records):
        raise IndexError("record_index is outside the evidence record list")
    record = records[record_index]
    scene = _scene_for(spec)
    states = _recorded_states(record, scene.model)
    indices = replay_frame_indices(len(states), maximum_frames)
    model, visual_manifest = _presentation_model(scene, spec, states, frame_opacity=frame_opacity)
    camera = _camera(spec, states, model)
    for name, value in (("azimuth", camera_azimuth), ("elevation", camera_elevation),
                        ("distance", camera_distance)):
        if value is not None:
            setattr(camera, name, value)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[3] = 0
    frames = []
    # GIF encodes centiseconds, so record and use the actual quantized cadence.
    frame_duration_ms = max(10, round(100.0 / fps) * 10)
    elapsed_seconds = states[-1]["control_index"] * float(spec.config.control_dt)
    playback_seconds = len(indices) * frame_duration_ms / 1000.0
    playback_speed = elapsed_seconds / playback_seconds
    episodes = evaluation.get("episodes", [])
    episode_evaluation = episodes[record_index] if len(episodes) == len(records) else {}
    try:
        for ordinal, state_index in enumerate(indices):
            state = states[int(state_index)]
            data.qpos[:] = state["qpos"]
            data.qvel[:] = state["qvel"]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            frame = renderer.render().copy()
            elapsed = state["control_index"] * float(spec.config.control_dt)
            lines = [title or spec.name.replace("_", " "),
                     subtitle or f"{spec.config.robot} | mode {record.requested_mode} | recorded replay",
                     f"t = {elapsed:5.2f} s | playback {playback_speed:.2f}x",
                     recorded_event_at_time(episode_evaluation, elapsed, float(spec.config.physics_dt))]
            if frame_opacity < 1.0:
                lines.append("Aperture frames translucent for visibility only")
            frames.append(
                _annotate(
                    frame,
                    lines,
                )
            )
    finally:
        renderer.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )
    evidence_bytes = evidence_path.read_bytes()
    manifest = {
        "schema_version": "transfer-evidence-visualization-v1",
        "gif": str(output_path),
        "gif_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "evidence": str(evidence_path),
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "task": spec.name,
        "legacy_task_alias": spec.alias,
        "presentation_title": title,
        "presentation_subtitle": subtitle,
        "robot": spec.config.robot,
        "task_spec": asdict(spec),
        "record_index": record_index,
        "sample_id": record.sample_id,
        "seed": record.seed,
        "requested_mode": record.requested_mode,
        "controller": record.controller,
        "record_state": record.state,
        "recorded_evaluation": evaluation,
        "recorded_state_count": len(states),
        "rendered_state_indices": indices.tolist(),
        "fps": fps,
        "frame_duration_ms": frame_duration_ms,
        "playback_seconds": playback_seconds,
        "recorded_duration_seconds": elapsed_seconds,
        "playback_speed": playback_speed,
        "width": width,
        "height": height,
        "camera": {
            "lookat": camera.lookat.tolist(),
            "azimuth": camera.azimuth,
            "elevation": camera.elevation,
            "distance": camera.distance,
        },
        "replay_contract": {
            "physics_steps": 0,
            "state_source": "recorded qpos/qvel; no interpolation",
            "geometry": "execution geometry reconstructed from exact task config",
            "mechanism_presentation": "task-default; mechanism changes mass/material, not recorded pose",
            "visual_meshes": visual_manifest,
        },
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest

