"""Render verified frozen CPU episodes; native meshes are presentation only.

Usage (CPU container, MUJOCO_GL=osmesa):
  python -m scripts.cross_embodiment.render_mechanism_gifs

Original evaluation bundles remain immutable. Replays must match every saved
trace channel before any joint recording can be used for a visualization.
"""
from __future__ import annotations

import argparse
import copy
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from genedynamics.learning.cross_embodiment import MechanismSpec, RandomRequest
from genedynamics.learning.cross_embodiment.contact import ContactConfig, ContactRunner
from genedynamics.learning.cross_embodiment.contact_behavior import behavior_reset, PhaseGaussianPrior
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.reference_transport import ReferenceTransport
from scripts.cross_embodiment.cpu_mechanism_validation import policy_step

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tmp/cross_embodiment/mechanism_revision"
PHASES = ("Load socket lip", "Withdraw", "Align", "Insert + hold")
STATUS = {0: "RUNNING", 1: "SUCCESS", 2: "FORBIDDEN CONTACT", 3: "JOINT LIMIT",
          4: "VELOCITY LIMIT", 5: "TIMEOUT", 6: "NONFINITE", 8: "CONTACT LOST", 9: "FORCE LIMIT"}
BG = (240, 244, 248)
INK = (25, 42, 62)


def replay(path, prior, transport, output):
    saved, metadata = load_bundle(path)
    config = ContactConfig(**metadata["config"])
    runner = ContactRunner(config=config, mechanism=MechanismSpec(**metadata["diagnostic_mechanism"]))
    seed, mode = metadata["seed"], metadata["mode"]
    state = runner.reset_batch(RandomRequest(seed, (seed * 2 + mode,)), initial="jammed")
    memory = behavior_reset(1, [mode])
    qpos = [state.physics[0].qpos.copy()]
    traces, phases, chunk_phases, references = [], [], [], []
    for _ in range(config.timeout_steps // 10):
        command, memory, reference = policy_step(runner, state, memory, prior, seed)
        if transport is not None:
            command = np.array(transport.apply(command, phase=memory.phase), copy=True)
        chunk_phases.append(memory.phase.copy())
        references.append(reference)
        # Retain the original 10-step policy clock, but record all joint states.
        for step in range(command.shape[1]):
            result = runner.collect_chunk(state, command[:, step:step + 1])
            state = result.state
            traces.append(result.trace)
            phases.append(int(memory.phase[0]))
            qpos.append(state.physics[0].qpos.copy())
        if np.all(state.status != 0):
            break
    actual = {n: np.concatenate([getattr(t, n) for t in traces], axis=1)
              for n in traces[0]._fields}
    actual.update(phase=np.asarray(chunk_phases), reference=np.asarray(references))
    errors = {}
    for name, expected in saved.items():
        value = actual[name]
        if value.shape != expected.shape:
            raise ValueError(f"Replay shape mismatch: {path}/{name}")
        error = float(np.max(np.abs(value.astype(float) - expected.astype(float))))
        if not np.allclose(value, expected, rtol=0, atol=1e-9):
            raise ValueError(f"Replay mismatch: {path}/{name}: {error}")
        errors[name] = error
    valid_count = int(saved["valid"][0].sum())
    positions = np.asarray(qpos)[:valid_count + 1]
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "joint_replay.npz", qpos=positions, phase=np.asarray(phases)[:valid_count])
    provenance = {"original_bundle": str(path.relative_to(ROOT)), "tensor_sha256": metadata["tensor_sha256"],
                  "trace_channels_max_abs_error": errors, "all_channels_match_atol": 1e-9,
                  "status": int(state.status[0]), "source_prior_sha256": prior.regressor.metadata["tensor_sha256"],
                  "transport_sha256": transport.regressor.metadata.get("tensor_sha256") if transport else None,
                  "transport_parameters": {k: transport.regressor.metadata.get(k) for k in
                      ("force_scale", "insertion_bias", "phase_conditioned")} if transport else None,
                  "role": "visualization replay; not an additional evaluation trial"}
    (output / "verification.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Verified {path.relative_to(BASE)}: {valid_count} steps; max error {max(errors.values()):.3g}", flush=True)
    return dict(runner=runner, qpos=positions, phase=np.asarray(phases)[:valid_count],
                saved=saved, metadata=metadata, steps=valid_count, provenance=provenance)


def display_model(episode):
    """Restore native robot link visuals without stepping this display model."""
    physics = episode["runner"]._impl.model
    tree = ET.fromstring(episode["runner"]._impl.scene.xml)
    robot = episode["metadata"]["config"]["robot"]
    folder = ROOT / "third_party/mujoco_menagerie" / ("franka_emika_panda" if robot == "panda" else "ufactory_xarm7")
    donor = ET.parse(folder / ("panda.xml" if robot == "panda" else "xarm7_nohand.xml")).getroot()
    bodies = {b.get("name"): b for b in donor.findall(".//body")}
    meshes, materials = set(), set()
    for body in tree.findall(".//body"):
        name = body.get("name")
        if name not in bodies or name == "transfer_tool":
            continue
        for source in bodies[name].findall("geom"):
            if robot == "panda" and source.get("class") != "visual":
                continue
            geom = copy.deepcopy(source)
            geom.attrib.pop("class", None)
            geom.attrib.update(type="mesh", contype="0", conaffinity="0", mass="0", group="2")
            if robot == "xarm7":
                geom.set("material", source.get("material", "white"))
            body.append(geom)
            meshes.add(geom.get("mesh"))
            materials.add(geom.get("material"))
    assets = tree.find("asset")
    if assets is None:
        assets = ET.SubElement(tree, "asset")
    for source in donor.find("asset"):
        name = source.get("name", Path(source.get("file", "")).stem)
        if source.tag == "mesh" and name in meshes:
            element = copy.deepcopy(source)
            element.set("file", str(folder / "assets" / source.get("file")))
        elif source.tag == "material" and name in materials:
            element = copy.deepcopy(source)
            element.attrib.pop("class", None)
            element.set("specular", ".35")
            element.set("shininess", ".4")
        else:
            continue
        assets.append(element)
    ET.SubElement(assets, "texture", name="render_sky", type="skybox", builtin="gradient",
                  rgb1=".88 .93 .98", rgb2=".98 .99 1", width="256", height="256")
    for geom in tree.iter("geom"):
        name = geom.get("name", "")
        if name.startswith("collision_"):
            geom.set("group", "3")
        elif name.startswith("contact_"):
            geom.set("rgba", ".95 .51 .15 1")
        elif name.startswith("socket_"):
            geom.set("rgba", ".23 .63 .64 1")
        elif name == "floor":
            geom.set("rgba", ".9 .93 .96 1")
    # The simulation rigidly attaches the peg at an offset from the TCP. A
    # display-only stem makes that existing rigid attachment visible; its
    # shape is illustrative and is never part of the contact/dynamics model.
    tool = tree.find(".//body[@name='transfer_tool']")
    ET.SubElement(tool, "geom", name="display_tool_stem", type="cylinder", pos="0 0 .0025",
                  size=".005 .0425", rgba=".36 .43 .51 1", mass="0", contype="0", conaffinity="0", group="2")
    # The near walls are translucent in the close-up only. This is clearly
    # labelled, and changes neither the replay nor the actual contact model.
    world = tree.find("worldbody")
    ET.SubElement(world, "geom", name="display_fixture_support", type="box", pos=".44 -.2 .14",
                  size=".06 .06 .29", rgba=".70 .77 .83 1", contype="0", conaffinity="0", group="2")
    ET.SubElement(world, "light", name="render_key", pos=".6 -.8 1.7", dir="-.2 .4 -1",
                  diffuse=".8 .8 .8", castshadow="true")
    visual = tree.find("visual")
    if visual is None:
        visual = ET.SubElement(tree, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    if (model.nq, model.nv) != (physics.nq, physics.nv):
        raise ValueError("Display mesh import changed state dimensions")
    model.vis.headlight.diffuse[:] = [.7, .7, .7]
    model.vis.headlight.ambient[:] = [.35, .35, .35]
    data, original = mujoco.MjData(model), mujoco.MjData(physics)
    sid = model.site(episode["runner"]._impl.scene.binding.profile.elements["tool_mount"].name).id
    max_error = 0.
    for qpos in episode["qpos"]:
        data.qpos[:] = original.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        mujoco.mj_forward(physics, original)
        max_error = max(max_error, float(np.max(np.abs(data.site_xpos[sid] - original.site_xpos[episode["runner"]._impl.sid]))))
    if max_error > 1e-10:
        raise ValueError("Visual restoration changed recorded TCP pose")
    episode["provenance"]["visual_fk_max_error"] = max_error
    episode["provenance"]["presentation"] = "native link meshes, illustrative rigid tool stem, decorative support, translucent near socket walls; display model never stepped"
    return model, data


def font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{filename}", size)


def camera(lookat, distance, azimuth, elevation):
    result = mujoco.MjvCamera()
    result.lookat[:] = lookat
    result.distance, result.azimuth, result.elevation = distance, azimuth, elevation
    return result


def render_panel(episode, title, color, index, clock, renderer, model, data):
    width, height = 400, 496
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 5), fill=color)
    draw.text((16, 16), title, font=font(21, True), fill=INK)
    mode = "straight withdrawal" if episode["metadata"]["mode"] == 0 else "lateral withdrawal"
    draw.text((16, 45), mode, font=font(15), fill=(80, 98, 118))
    index = min(index, episode["steps"])
    data.qpos[:] = episode["qpos"][index]
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.sitegroup[:] = 0
    for name in ("socket_front", "socket_right"):
        model.geom_rgba[model.geom(name).id, 3] = .18
    renderer.update_scene(data, camera((.444, -.2, .541), .29, 115, -20), scene_option=option)
    canvas.paste(Image.fromarray(renderer.render()).resize((width, 286), Image.Resampling.LANCZOS), (0, 72))
    for name in ("socket_front", "socket_right"):
        model.geom_rgba[model.geom(name).id, 3] = 1.
    renderer.update_scene(data, camera((.18, -.08, .40), 1.25, 135, -22), scene_option=option)
    inset = Image.fromarray(renderer.render()).resize((140, 105), Image.Resampling.LANCZOS)
    canvas.paste(inset, (width - 148, 80))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((8, 80, 179, 102), radius=5, fill=(248, 251, 253))
    draw.text((14, 83), "Socket cutaway view", font=font(12), fill=INK)
    t = max(0, index - 1)
    status = int(episode["saved"]["status"][0, t]) if index else 0
    phase = int(episode["phase"][t])
    load = float(episode["saved"]["normal_force"][0, t]) if index else 0.
    pos_error = float(np.linalg.norm(episode["saved"]["tcp_position"][0, t] - episode["runner"].config.goal) * 1000)
    draw.text((16, 366), f"t = {index * .02:4.2f} s   |   {PHASES[phase]}", font=font(16, True), fill=INK)
    status_color = (22, 133, 97) if status == 1 else ((185, 67, 58) if status else color)
    draw.text((16, 394), STATUS.get(status, str(status)), font=font(17, True), fill=status_color)
    draw.text((16, 422), f"Axial load  {load:4.1f} N   (target 5 N)", font=font(15), fill=INK)
    draw.text((16, 445), f"Goal error  {pos_error:4.1f} mm   (limit 2 mm)", font=font(15),
              fill=(22, 133, 97) if pos_error <= 2 else INK)
    draw.line((16, 477, width - 16, 477), fill=(208, 220, 232), width=5)
    draw.line((16, 477, 16 + (width - 32) * min(clock / 15., 1.), 477), fill=color, width=5)
    return canvas


def render_comparison(episodes, labels, title, subtitle, output, fps=10, speed=2.):
    colors = [(57, 107, 165), (25, 145, 111), (192, 118, 50)]
    models = [display_model(e) for e in episodes]
    renderers = [mujoco.Renderer(m, height=300, width=400) for m, _ in models]
    end = max(e["steps"] * .02 for e in episodes)
    times = np.arange(0, end, speed / fps).tolist() + [end] * fps
    frames = []
    try:
        for i, clock in enumerate(times):
            canvas = Image.new("RGB", (400 * len(episodes), 594), BG)
            draw = ImageDraw.Draw(canvas)
            draw.text((18, 12), title, font=font(24, True), fill=INK)
            draw.text((18, 45), subtitle, font=font(14), fill=(74, 92, 112))
            for j, (episode, label, (model, data), renderer) in enumerate(zip(episodes, labels, models, renderers)):
                panel = render_panel(episode, label, colors[j], round(clock / .02), clock, renderer, model, data)
                canvas.paste(panel, (400 * j, 74))
            draw = ImageDraw.Draw(canvas)
            draw.text((18, 576), f"Verified frozen CPU simulation  |  {speed:g}x playback  |  native meshes; proxy collision physics", font=font(12), fill=(90, 105, 120))
            frames.append(canvas)
            if i % 25 == 0:
                print(f"Rendering {output.name}: {i + 1}/{len(times)}", flush=True)
        palette = Image.new("RGB", (400 * len(episodes), 594 * 4))
        for i, n in enumerate(np.linspace(0, len(frames) - 1, 4, dtype=int)):
            palette.paste(frames[n], (0, 594 * i))
        palette = palette.quantize(colors=192)
        encoded = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        encoded[0].save(output, save_all=True, append_images=encoded[1:], duration=round(1000 / fps), loop=0, optimize=True)
        frames[-1].save(output.with_suffix(".png"))
    finally:
        for renderer in renderers:
            renderer.close()
    record = dict(file=output.name, frames=len(frames), simulation_seconds=end, fps=fps, playback_speed=speed,
                  episodes=[e["provenance"] for e in episodes])
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def write_gallery(out, manifest):
    captions = {
        "panda_recovery_modes.gif": ("Panda：两种恢复模式", "同一个冻结源 prior，分别执行直线撤回和侧向撤回，再对齐并插入。1 倍速。"),
        "xarm7_mechanism_comparison.gif": ("Panda → xArm7：机制对比", "相同 reset，gain 0.65、延迟 60 ms。左：直接迁移；中：校准 transport；右：仅补偿力。2 倍速。"),
        "xarm7_stress_example.gif": ("更强延迟：一个失败对照", "gain 0.60、延迟 80 ms，seed 12403、侧向撤回。左：nominal response 超时；右：校准 transport 成功。这是选定的失败例，不代表统计显著性。2 倍速。"),
    }
    cards = []
    for item in manifest:
        filename = item["file"]
        heading, caption = captions[filename]
        cards.append(f'<section><h2>{heading}</h2><p>{caption}</p><img src="{html.escape(filename)}" alt="{heading}"><p><a href="{filename}">打开 GIF</a> · <a href="{Path(filename).with_suffix(".json")}">重放核对记录</a></p></section>')
    (out / "index.html").write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cross-embodiment CPU 回放</title><style>body{margin:32px auto;max-width:1200px;padding:0 20px;background:#eff4f8;color:#192a3e;font:16px/1.65 system-ui}h1{font-size:30px}h2{font-size:23px;margin-bottom:8px}section{margin:30px 0;background:white;border-radius:14px;padding:20px;box-shadow:0 2px 12px #1625360c}img{display:block;max-width:100%;height:auto;margin:auto;border-radius:8px}a{color:#2877ab}p{margin:8px 0 16px}</style>
<h1>Cross-embodiment：CPU 机制回放</h1><p>冻结评估的真实 MuJoCo 重放。每条回放逐通道核对原始轨迹，误差为 0；画面显示恢复阶段、轴向接触载荷和 TCP 目标误差。成功判据还包含姿态、力限值和保持时间。</p><p>整臂使用原生外观网格；局部孔壁半透明，工具连接杆和支撑台用于显示。物理仍为原评估的碰撞代理模型，显示模型不参与仿真。</p>''' + "\n".join(cards) + '</html>\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=BASE / "gifs")
    parser.add_argument("--preview", action="store_true", help="Only render a camera check after the first verified episode")
    parser.add_argument("--only", choices=("panda", "comparison", "stress"), help="Render one panel set")
    parser.add_argument("--gallery-only", action="store_true", help="Refresh the HTML gallery from the completed manifest")
    args = parser.parse_args()
    base, out = args.base.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.gallery_only:
        write_gallery(out, json.loads((out / "manifest.json").read_text()))
        return
    prior = PhaseGaussianPrior(FrozenRegressor.load(base / "strict_phase_prior"))
    full = ReferenceTransport.load(base / "strict_chain_derated_v2/transport")
    stress = ReferenceTransport.load(base / "strict_chain_stress_v2/transport")
    nominal = ReferenceTransport.load(base / "strict_chain_derated/nominal_response_baseline/transport")
    report = json.loads((base / "strict_chain_derated_v2/report.json").read_text())
    force_only = ReferenceTransport.from_parameters(report["transport_scale"])

    def episode(relative, transport=None):
        path = base / relative
        return replay(path, prior, transport, out / "replays" / relative)

    if args.preview:
        e = episode("frozen_derated/calibrated_transport/s12000_m1", full)
        m, d = display_model(e)
        renderer = mujoco.Renderer(m, height=300, width=400)
        image = render_panel(e, "Calibrated transport", (25, 145, 111), 0, 0, renderer, m, d)
        image.save(out / "camera_preview.png")
        renderer.close()
        return

    items = []
    if args.only in (None, "panda"):
        items.append(render_comparison(
        [episode(f"strict_source_heldout/s8100_m{mode}") for mode in (0, 1)],
        ["Panda / mode 0", "Panda / mode 1"],
        "Source skill: two recovery modes", "Frozen source prior  |  seed 8100  |  keyed insertion recovery",
        out / "panda_recovery_modes.gif", speed=1.))
    if args.only in (None, "comparison"):
        items.append(render_comparison(
        [episode("frozen_derated/direct/s12000_m1"),
         episode("frozen_derated/calibrated_transport/s12000_m1", full),
         episode("frozen_derated/force_only_transport/s12000_m1", force_only)],
        ["Direct transfer", "Calibrated transport", "Force-only transport"],
        "Panda to xArm7: mechanism adaptation", "Same reset: seed 12000, mode 1  |  target gain 0.65, delay 60 ms  |  position tolerance 2 mm",
        out / "xarm7_mechanism_comparison.gif"))
    if args.only in (None, "stress"):
        items.append(render_comparison(
        [episode("frozen_stress/nominal_response_transport/s12403_m1", nominal),
         episode("frozen_stress/calibrated_transport/s12403_m1", stress)],
        ["Nominal response", "Calibrated transport"],
        "Stress mechanism: one failure example", "Seed 12403, mode 1  |  gain 0.60, delay 80 ms  |  selected example; not an aggregate claim",
        out / "xarm7_stress_example.gif"))
    manifest_path = out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if args.only and manifest_path.exists() else []
    previous = [p for p in previous if p["file"] not in {i["file"] for i in items}]
    manifest = previous + items
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_gallery(out, manifest)
    print(json.dumps([dict(file=i["file"], bytes=(out / i["file"]).stat().st_size) for i in items]), flush=True)


if __name__ == "__main__":
    main()
