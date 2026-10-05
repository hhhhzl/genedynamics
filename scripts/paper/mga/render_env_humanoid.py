#!/usr/bin/env python3
"""Render clean H1 environment views from already exported genuine Brax states.

Only camera, lighting, and material appearance change. Robot/object transforms
and corridor geometry are read unchanged from the verified saved-state payloads.
No simulation, policy rollout, or manufactured motion is used.
"""

from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import http.server
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "reports/mga/paper_figures/scenes"
OUTPUT = ROOT / "scripts/paper/mga/output/env_overview_v4/assets"
APPENDIX_SOURCE = ROOT / "reports/mga/paper_figures/appendix/scenes"
GIF_OUTPUT = ROOT / "reports/mga/paper_figures/gifs"

PALETTES = {
    "manuscript": {
        "robot": "#3A4655", "box": "#E4F1D3", "corridor": "#9FB8B5",
        "goal": "#5C8F2E", "background": "#F7F8F6",
        "checker": [175, 205, 205, 175], "srgb_materials": True,
    },
    "brax-reference": {
        "robot": "#35342F", "box": "#C7B384", "corridor": "#9BA6AF",
        "goal": "#5C9B6F", "background": "#F5F5F3",
        "checker": [175, 205, 205, 175], "srgb_materials": False,
    },
}
PALETTES["manuscript-v1-box"] = {
    **PALETTES["manuscript"],
    "box": PALETTES["brax-reference"]["box"],
    "box_srgb_materials": PALETTES["brax-reference"]["srgb_materials"],
}

VIEWS = {
    "humanoid_hero": ("humanoid_fixed_stance_push", -1, 1200, 800, False),
    "humanoid_force_regulation": ("humanoid_force_regulation", -1, 600, 400, False),
    "humanoid_fixed_stance_push": ("humanoid_fixed_stance_push", -1, 600, 400, False),
    "humanoid_unjamming": ("humanoid_unjamming", 0, 600, 400, True),
}

HTML = r'''<!doctype html><html><head><meta charset="utf-8">
<title>MGA_ENV_LOADING</title>
<style>html,body{margin:0;overflow:hidden;background:#f5f5f3}canvas{display:block}</style>
<script type="importmap">{"imports":{"three":"/reports/mga/paper_figures/scenes/brax_web/three.module.js"}}</script>
</head><body><script type="module">
import * as THREE from 'three';
import {createScene} from '/reports/mga/paper_figures/scenes/brax_web/system.js';
try {
 const settings=__SETTINGS__;
 const system=await (await fetch(settings.payload)).json();
 const scene=createScene(system);
 const state=system.states.x.at(settings.state);
 const palette=settings.palette;
 function materialColor(material,value,srgb=palette.srgb_materials){
   material.color.set(value);
   if(srgb)material.color.convertSRGBToLinear();
 }
 scene.background=new THREE.Color(palette.background);
 scene.fog=new THREE.Fog(palette.background,8,20);
 for(const [name,geoms] of Object.entries(system.geoms)){
   const group=scene.getObjectByName(name.replaceAll('/','_'));
   const index=geoms[0].link_idx;
   if(index>=0){
     group.position.fromArray(state.pos[index]);
     const q=state.rot[index];group.quaternion.set(q[1],q[2],q[3],q[0]);
   }
   geoms.forEach((geom,j)=>group.children[j].traverse(o=>{
     if(!o.isMesh)return;
     o.material.opacity=1;o.material.transparent=false;
     o.material.depthWrite=true;o.material.shininess=8;
     o.material.specular.set(0x101010);
     o.castShadow=geom.name!=='Plane';o.receiveShadow=true;
     if(geom.name==='Plane'){
       const d=o.material.map.image.data;
       palette.checker.forEach((v,k)=>{d[k*4]=v;d[k*4+1]=v;d[k*4+2]=v;d[k*4+3]=255;});
       o.material.map.repeat.set(3800,3800);o.material.map.needsUpdate=true;
       o.material.color.set(0xffffff);
     }else if(name==='box_body'){
       materialColor(o.material,palette.box,palette.box_srgb_materials??palette.srgb_materials);
     }else if(name==='world'){
       materialColor(o.material,palette.corridor);
     }else{
       materialColor(o.material,palette.robot);
     }
   }));
 }
 if(system.paper.goal){
   const g=system.paper.goal;
   const marker=new THREE.Mesh(new THREE.BoxGeometry(...g.half_size.map(v=>2*v)),
       new THREE.MeshPhongMaterial({transparent:true,opacity:.70}));
   materialColor(marker.material,palette.goal);
   marker.position.fromArray(g.position);scene.add(marker);
 }
 const hemi=new THREE.HemisphereLight(0xffffff,0xa7a7a0,.72);scene.add(hemi);
 const key=new THREE.DirectionalLight(0xffffff,.78);
 key.position.set(-3,-4,7);key.castShadow=true;
 Object.assign(key.shadow.camera,{left:-3,right:3,top:3,bottom:-3,near:.1,far:20});
 key.shadow.mapSize.set(2048,2048);key.shadow.bias=-.00015;key.shadow.normalBias=.002;
 key.shadow.radius=4;scene.add(key);
 const fill=new THREE.DirectionalLight(0xffffff,.12);fill.position.set(3,2,4);scene.add(fill);
 const aspect=settings.width/settings.height;
 const halfHeight=settings.high?1.22:1.08;
 const camera=new THREE.OrthographicCamera(-halfHeight*aspect,halfHeight*aspect,halfHeight,-halfHeight,.01,100);
 camera.up.set(0,0,1);
 camera.position.fromArray(settings.camera?.position??(settings.high?[-.35,-3.8,5.6]:[-.55,-4.5,2.4]));
 camera.lookAt(...(settings.camera?.lookat??(settings.high?[.70,0,.64]:[.60,0,.88])));
 const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
 renderer.setPixelRatio(1);renderer.setSize(settings.width,settings.height);
 renderer.outputEncoding=THREE.sRGBEncoding;
 renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
 document.body.appendChild(renderer.domElement);
 renderer.render(scene,camera);
 window.mgaEnvironment={scene,camera,renderer,system,settings};
 window.mgaEnvironment.setState=index=>{
   const next=system.states.x[index];
   for(const [name,geoms] of Object.entries(system.geoms)){
     const link=geoms[0].link_idx;
     if(link<0)continue;
     const group=scene.getObjectByName(name.replaceAll('/','_'));
     group.position.fromArray(next.pos[link]);
     const q=next.rot[link];group.quaternion.set(q[1],q[2],q[3],q[0]);
   }
   renderer.render(scene,camera);
 };
 document.title='MGA_ENV_READY';
 await fetch('/__capture_ready__/'+settings.name);
}catch(error){document.title='MGA_ENV_ERROR';document.body.textContent=error.stack;}
</script></body></html>'''


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _export_gif_payloads(output_dir, scene_metadata=APPENDIX_SOURCE / "metadata.json"):
    """Export saved-state FK, with the same source verification as Visualizations."""
    from render_mechanism_scenes import (
        _appendix_indices, _execution_env, _export_brax_web, _native_model,
    )

    from genedynamics.experiments.plugins.metrics.extractors import _humanoid_evaluation_view

    source = json.loads(scene_metadata.read_text())
    visualization = json.loads((APPENDIX_SOURCE / "metadata.json").read_text())
    payload_dir = output_dir / "humanoid_saved_states"
    payload_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, item in source.items():
        if item.get("task") != "humanoid":
            continue
        result_path, trajectory_path = ROOT / item["result"], ROOT / item["trajectory"]
        if (_sha256(result_path) != item["result_sha256"] or
                _sha256(trajectory_path) != item["trajectory_sha256"]):
            raise ValueError(f"Source changed since Visualizations: {name}")
        result = json.loads(result_path.read_text())
        trajectory = json.loads(trajectory_path.read_text())
        if int(result["seed"]) != int(item["seed"]):
            raise ValueError(f"Result seed does not match selected scene: {name}")
        if (item["camera"] != visualization[name]["camera"] or
                item["palette"] != visualization[name]["palette"]):
            raise ValueError(f"GIF selection must preserve the Visualizations camera/palette: {name}")
        signals = _humanoid_evaluation_view(trajectory["task_signals"])
        verified_success = (bool(signals["force_step_pass"][0])
                            if signals.get("force_step_applicable", False)
                            else bool(any(signals["task_success"])
                                      and signals["safe_success_goal_error"] <= signals["goal_tolerance"]
                                      and signals["safe_success_violation"] <= 0.0))
        if item.get("selection", {}).get("success") and not verified_success:
            raise ValueError(f"Selected successful run fails the task-owned evaluation contract: {name}")
        _, last = _appendix_indices(trajectory)
        if last != item["last_unpadded_state"]:
            raise ValueError(f"Executed endpoint changed: {name}")
        # Sample real saved states at 25 fps. Include odd-indexed final states.
        indices = list(range(0, last + 1, 2))
        if indices[-1] != last:
            indices.append(last)
        env = _execution_env(result)
        recorded_dt = float(trajectory["task_signals"]["dt"])
        if abs(float(env.dt) - recorded_dt) > 1e-8 or abs(recorded_dt - float(item["dt"])) > 1e-8:
            raise ValueError(f"Saved, configured, and selected control clocks disagree: {name}")
        model = _native_model(env)
        exported = _export_brax_web(env, model, trajectory, indices, last, name, payload_dir)
        payload = payload_dir / exported["web_payload"]
        saved_payload = json.loads(payload.read_text())
        # Exact old-payload comparison applies only to replaying the same seed.
        # A selected successful seed may have different task-owned geometry;
        # reconstruct it from its own verified result and validate saved-state FK.
        if (item["trajectory_sha256"] == visualization[name]["trajectory_sha256"] and
                item["result_sha256"] == visualization[name]["result_sha256"]):
            original = json.loads((APPENDIX_SOURCE / visualization[name]["web_payload"]).read_text())
            if saved_payload["geoms"] != original["geoms"]:
                raise ValueError(f"Render geometry changed since Visualizations: {name}")
        records[name] = {
            "method": "MGA", "suite": item["suite"], "seed": item["seed"],
            "source_result": item["result"], "result_sha256": item["result_sha256"],
            "source_trajectory": item["trajectory"], "trajectory_sha256": item["trajectory_sha256"],
            "state_indices": indices, "last_unpadded_state": last,
            "excluded_padding_states": len(trajectory["states"]) - last - 1,
            "dt": recorded_dt, "time_seconds": [index * recorded_dt for index in indices],
            "payload": str(payload.relative_to(output_dir)), "payload_sha256": _sha256(payload),
            "camera": item["camera"], "palette": item["palette"],
            "dimensions": [810, 540], "simulation_steps_executed": 0,
            "controller_rollouts_executed": 0, "interpolated_states": 0,
            "brax_fk_max_position_error_m": exported["brax_fk_max_position_error_m"],
            "rendering": "Saved q/qd -> Brax forward kinematics -> native Brax Web scene -> Three.js",
            "source_selection_metadata": str(scene_metadata.relative_to(ROOT)),
            "source_selection_metadata_sha256": _sha256(scene_metadata),
            "selection": item.get("selection"), "task_success_independently_verified": verified_success,
            "first_done_state": next((i for i, state in enumerate(trajectory["states"])
                                      if state.get("done", False)), None),
            "geometry_source": "Selected saved result config_snapshot and seed; no state interpolation or geometry substitution",
        }
        print(f"Exported {name}: {len(indices)} real saved states, t=0..{last * item['dt']:.2f} s", flush=True)
    (output_dir / "humanoid_gifs.json").write_text(json.dumps(records, indent=2) + "\n")


def _capture_gifs(output_dir):
    """Capture every exported pose from one fixed camera, without interpolating."""
    from PIL import Image
    from render_mechanism_scenes import (
        ANNOTATION_STYLE, _annotate_force_frame, _saved_force_annotations,
    )

    metadata_path = output_dir / "humanoid_gifs.json"
    records = json.loads(metadata_path.read_text())
    output_dir = output_dir.resolve()
    chrome = (shutil.which("google-chrome") or shutil.which("chromium") or
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    completed = set()
    errors = {}
    received = {}
    frame_dir = output_dir / "humanoid_saved_states"

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            pieces = self.path.strip("/").split("/")
            content = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if len(pieces) == 3 and pieces[0] == "__gif_frame__" and pieces[1] in records:
                name, index = pieces[1], int(pieces[2])
                if not 0 <= index < len(records[name]["state_indices"]):
                    self.send_error(400)
                    return
                if not content.startswith(b"data:image/png;base64,"):
                    self.send_error(400)
                    return
                target = frame_dir / f"{name}_{index:04d}.png"
                target.write_bytes(base64.b64decode(content.partition(b",")[2], validate=True))
                received.setdefault(name, set()).add(index)
            elif len(pieces) == 2 and pieces[0] == "__gif_done__" and pieces[1] in records:
                completed.add(pieces[1])
            elif len(pieces) == 2 and pieces[0] == "__gif_error__" and pieces[1] in records:
                errors[pieces[1]] = content.decode()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(ROOT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    capture = r'''
 try {
   for(let index=0;index<system.states.x.length;index++){
     window.mgaEnvironment.setState(index);
     const response=await fetch('/__gif_frame__/'+settings.name+'/'+index,
       {method:'POST',body:renderer.domElement.toDataURL('image/png')});
     if(!response.ok)throw Error('Frame upload failed: '+index);
   }
   await fetch('/__gif_done__/'+settings.name,{method:'POST',body:'done'});
 }catch(error){
   await fetch('/__gif_error__/'+settings.name,{method:'POST',body:error.stack});
 }
'''
    try:
        for name, item in records.items():
            payload = output_dir / item["payload"]
            if _sha256(payload) != item["payload_sha256"]:
                raise ValueError(f"Saved-state payload changed: {name}")
            trajectory_path = ROOT / item["source_trajectory"]
            if _sha256(trajectory_path) != item["trajectory_sha256"]:
                raise ValueError(f"Saved trajectory changed before force annotation: {name}")
            trajectory = json.loads(trajectory_path.read_text())
            force_annotations = _saved_force_annotations(
                trajectory, "humanoid", item["state_indices"], item["dt"])
            annotation_records = force_annotations["records"]
            if [record["state_index"] for record in annotation_records] != item["state_indices"]:
                raise ValueError(f"Force annotation state alignment changed: {name}")
            settings = {"payload": "/" + str(payload.relative_to(ROOT)), "name": name,
                        "state": 0, "width": 810, "height": 540,
                        "high": name == "humanoid_unjamming", "palette": item["palette"],
                        "camera": item["camera"]}
            html = frame_dir / f"{name}.gif.html"
            html.write_text(HTML.replace("__SETTINGS__", json.dumps(settings)).replace(
                " await fetch('/__capture_ready__/'+settings.name);", capture))
            with tempfile.TemporaryDirectory(prefix="mga-humanoid-gif-chrome-") as profile:
                command = [chrome, "--headless=new", "--hide-scrollbars", "--no-first-run",
                           "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                           "--no-default-browser-check", "--disable-background-networking", "--disable-sync",
                           "--force-device-scale-factor=1", f"--user-data-dir={profile}", "--window-size=810,540",
                           f"http://127.0.0.1:{server.server_port}/{html.relative_to(ROOT)}"]
                with tempfile.TemporaryFile(mode="w+") as diagnostic:
                    process = subprocess.Popen(command, stdout=diagnostic, stderr=diagnostic)
                    deadline = time.monotonic() + 180
                    try:
                        while name not in completed and name not in errors:
                            if process.poll() is not None or time.monotonic() > deadline:
                                diagnostic.seek(0)
                                raise RuntimeError(f"Capture stopped {name}: {diagnostic.read()[-2000:]}")
                            time.sleep(.1)
                    finally:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
            if name in errors:
                raise RuntimeError(errors[name])
            if received[name] != set(range(len(item["state_indices"]))):
                raise ValueError(f"Incomplete frame sequence: {name}")
            paths = [frame_dir / f"{name}_{i:04d}.png" for i in range(len(item["state_indices"]))]
            frames = []
            for path, annotation in zip(paths, annotation_records, strict=True):
                with Image.open(path) as raw_frame:
                    if raw_frame.size != (810, 540):
                        raise ValueError(f"Unexpected native frame dimensions: {path}")
                    frames.append(_annotate_force_frame(raw_frame, annotation))
            if any(frame.size != (810, 588) for frame in frames):
                raise ValueError(f"Force footer must add exactly 48 pixels: {name}")
            # A shared palette avoids frame-to-frame color flicker. Preserve real
            # timing, including a possible final 20-ms step, then hold 800 ms.
            swatches = Image.new("RGB", (270 * len(frames), 196))
            for i, frame in enumerate(frames):
                swatches.paste(frame.resize((270, 196)), (i * 270, 0))
            palette = swatches.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
            durations = [round((b - a) * item["dt"] * 1000)
                         for a, b in zip(item["state_indices"][:-1], item["state_indices"][1:])] + [800]
            gif = output_dir / f"mga_{name}.gif"
            frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations,
                           loop=0, disposal=2, optimize=False)
            with Image.open(gif) as check:
                count = check.n_frames
                if count != len(paths) or check.size != (810, 588):
                    raise ValueError(f"GIF frame validation failed: {name}")
                saved_durations = []
                for i in range(count):
                    check.seek(i)
                    saved_durations.append(check.info["duration"])
                if saved_durations != durations:
                    raise ValueError(f"GIF timing changed: {name}")
            item.update(gif=gif.name, gif_sha256=_sha256(gif), frame_count=count,
                        native_image_size=[810, 540], image_size=[810, 588],
                        dimensions=[810, 588], force_annotations=force_annotations,
                        annotation_style=ANNOTATION_STYLE,
                        frame_duration_ms=durations, final_pause_ms=800, playback_speed=1.0,
                        image_source_sha256=[_sha256(path) for path in paths],
                        total_gif_duration_seconds=sum(durations) / 1000,
                        dynamic_pose_frames_verified=True,
                        script_sha256=_sha256(Path(__file__)))
            metadata_path.write_text(json.dumps(records, indent=2) + "\n")
            print(f"Rendered {gif}: {count} frames; {sum(durations) / 1000:.2f} s", flush=True)
    finally:
        server.shutdown()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--only", choices=tuple(VIEWS))
    parser.add_argument("--palette", choices=tuple(PALETTES), default="manuscript-v1-box",
                        help="Manuscript colors, or the original warm Brax reference palette")
    parser.add_argument("--export-gif-payloads", action="store_true",
                        help="Export genuine 25-fps saved-state Brax FK; requires the experiment dependencies")
    parser.add_argument("--scene-metadata", type=Path, default=APPENDIX_SOURCE / "metadata.json",
                        help="Optional verified source-selection metadata; default replays the current Visualizations seeds")
    parser.add_argument("--gifs", action="store_true",
                        help="Capture exported saved-state GIFs using the Visualizations camera and appearance")
    args = parser.parse_args()
    if args.export_gif_payloads or args.gifs:
        if args.output_dir == OUTPUT:
            args.output_dir = GIF_OUTPUT
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.export_gif_payloads:
            _export_gif_payloads(args.output_dir.resolve(), args.scene_metadata.resolve())
        if args.gifs:
            _capture_gifs(args.output_dir)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chrome = (shutil.which("google-chrome") or shutil.which("chromium") or
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not Path(chrome).is_file():
        raise RuntimeError("Google Chrome or Chromium is required")

    ready = set()

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path.startswith('/__capture_ready__/'):
                ready.add(self.path.rsplit('/', 1)[-1])
                self.send_response(200)
                self.end_headers()
            else:
                super().do_GET()

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(ROOT)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    original = json.loads((SOURCE / "metadata.json").read_text())
    metadata_path = args.output_dir / "humanoid_render_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    try:
        for name, (source, state_index, width, height, high) in VIEWS.items():
            if args.only and name != args.only:
                continue
            payload = SOURCE / f"{source}.brax.json"
            settings = {
                "payload": "/" + str(payload.relative_to(ROOT)),
                "name": name,
                "state": state_index, "width": width, "height": height,
                "high": high,
                "palette": PALETTES[args.palette],
            }
            html = args.output_dir / f"{name}.html"
            html.write_text(HTML.replace("__SETTINGS__", json.dumps(settings)))
            image = (args.output_dir / f"{name}.png").resolve()
            with tempfile.TemporaryDirectory(prefix="mga-env-chrome-") as profile:
                command = [chrome, "--headless=new", "--hide-scrollbars", "--no-first-run",
                           "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                           "--no-default-browser-check", "--disable-background-networking",
                           "--disable-component-update", "--disable-sync",
                           "--force-device-scale-factor=1", f"--user-data-dir={profile}",
                           f"--window-size={width},{height}", "--virtual-time-budget=10000",
                           "--run-all-compositor-stages-before-draw", f"--screenshot={image}",
                           "--dump-dom", f"http://127.0.0.1:{server.server_port}/{html.relative_to(ROOT)}"]
                # Some macOS Chrome builds remain alive after writing the image.
                # Stop only this isolated process after an explicit render-ready
                # handshake and a fresh screenshot, rather than waiting on exit.
                with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(mode="w+") as stderr:
                    before = image.stat().st_mtime_ns if image.exists() else -1
                    process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
                    deadline = time.monotonic() + 40
                    captured = False
                    try:
                        while time.monotonic() < deadline:
                            fresh = image.exists() and image.stat().st_mtime_ns != before
                            if name in ready and fresh and image.stat().st_size > 1000:
                                time.sleep(.25)
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
                    stdout.seek(0)
                    stderr.seek(0)
                    html_result, diagnostic = stdout.read(), stderr.read()
            if not captured:
                raise RuntimeError(f"Capture failed: {html_result[-1000:]} {diagnostic[-1000:]}")
            saved = json.loads(payload.read_text())
            metadata[name] = {
                "source_payload": str(payload.relative_to(ROOT)),
                "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "source_result": original[source]["result"],
                "source_trajectory": original[source]["trajectory"],
                "trajectory_sha256": original[source]["trajectory_sha256"],
                "saved_state_index": saved["paper"]["pose_indices"][state_index],
                "payload_state_index": state_index,
                "dimensions": [width, height],
                "view": "high oblique" if high else "side oblique",
                "geometry_or_pose_modified": False,
                "appearance": "opaque materials, bright checker ground, matte H1, soft shadows",
                "palette_name": args.palette,
                "palette": PALETTES[args.palette],
                "goal_marker": saved["paper"].get("goal"),
                "renderer_dependencies": [
                    "reports/mga/paper_figures/scenes/brax_web/system.js",
                    "reports/mga/paper_figures/scenes/brax_web/three.module.js",
                    "local Google Chrome/Chromium",
                ],
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(image, flush=True)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
