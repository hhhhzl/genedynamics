#!/usr/bin/env python3
"""Render clean H1 environment views from already exported genuine Brax states.

Only camera, lighting, and material appearance change. Robot/object transforms
and corridor geometry are read unchanged from the verified saved-state payloads.
No simulation, policy rollout, or manufactured motion is used.
"""

from __future__ import annotations

import argparse
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
    "humanoid_p1": ("humanoid_force_regulation", -1, 600, 400, False),
    "humanoid_p2": ("humanoid_fixed_stance_push", -1, 600, 400, False),
    "humanoid_p3": ("humanoid_unjamming", 0, 600, 400, True),
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
 camera.position.fromArray(settings.high?[-.35,-3.8,5.6]:[-.55,-4.5,2.4]);
 camera.lookAt(...(settings.high?[.70,0,.64]:[.60,0,.88]));
 const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
 renderer.setPixelRatio(1);renderer.setSize(settings.width,settings.height);
 renderer.outputEncoding=THREE.sRGBEncoding;
 renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
 document.body.appendChild(renderer.domElement);
 renderer.render(scene,camera);
 window.mgaEnvironment={scene,camera,renderer,system,settings};
 document.title='MGA_ENV_READY';
 await fetch('/__capture_ready__/'+settings.name);
}catch(error){document.title='MGA_ENV_ERROR';document.body.textContent=error.stack;}
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--only", choices=tuple(VIEWS))
    parser.add_argument("--palette", choices=tuple(PALETTES), default="manuscript-v1-box",
                        help="Manuscript colors, or the original warm Brax reference palette")
    args = parser.parse_args()
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
