"""TripoSG image->mesh batch adapter (Table-3 image arm).

Loads the pipeline once and runs every image in a folder -> .glb meshes, which the
MAIN env then robotizes (`build_bank_from_raw.py`). TripoSG is image-conditioned, so
it sits OFF the text axis (image arm) — see morphology_priors/README.md.

Run in the isolated triposg venv (numpy<2):
  TRIPOSG_REPO=... python triposg_batch_infer.py --img-dir <dir> --out-dir <dir> [--steps 50]

Like the other adapters, this lives in the PARENT repo and injects the pristine
git submodule (third_party/morphology_priors/triposg/repo) onto sys.path + chdir's
into it (its `triposg` package + `pretrained_weights/` are repo-relative) — the
submodule is never edited. Override the repo location via TRIPOSG_REPO.
"""
import argparse
import glob
import importlib.machinery
import os
import sys
import types

# TripoSG lives in a git submodule; add it to the path + chdir (its `triposg`
# package + pretrained_weights/ are relative). Override via TRIPOSG_REPO.
_REPO = os.environ.get("TRIPOSG_REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../../../third_party/morphology_priors/triposg/repo"))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

# torchaudio stub: the system .so is ABI-incompatible and TripoSG never uses it,
# but `transformers` imports it at load. Register an empty module tree first.
_ta = types.ModuleType("torchaudio")
_ta.__version__ = "2.2.0"
_ta.__path__ = []
_ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None, is_package=True)
_ta.__spec__.submodule_search_locations = []
sys.modules["torchaudio"] = _ta
for _sub in ("transforms", "functional", "io", "backend", "sox_effects",
             "compliance", "datasets", "models", "pipelines", "utils", "_extension"):
    _m = types.ModuleType(f"torchaudio.{_sub}")
    _m.__spec__ = importlib.machinery.ModuleSpec(f"torchaudio.{_sub}", loader=None)
    sys.modules[f"torchaudio.{_sub}"] = _m
    setattr(_ta, _sub, _m)

import numpy as np
import torch
import trimesh
from PIL import Image
from triposg.pipelines.pipeline_triposg import TripoSGPipeline


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img-dir", default="assets/example_data", help="folder of .png/.jpg inputs")
    ap.add_argument("--out-dir", default="gen_meshes", help="output folder for .glb meshes")
    ap.add_argument("--weights", default="pretrained_weights/TripoSG",
                    help="TripoSG weights dir (repo-relative or absolute)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"
    print("loading pipeline (fp16) ...", flush=True)
    pipe = TripoSGPipeline.from_pretrained(args.weights).to(dev, torch.float16)

    imgs = sorted(glob.glob(os.path.join(args.img_dir, "*.png"))
                  + glob.glob(os.path.join(args.img_dir, "*.jpg")))
    print(f"{len(imgs)} images -> {out_dir}", flush=True)
    ok = 0
    for i, f in enumerate(imgs):
        name = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(out_dir, f"{name}.glb")
        try:
            img = Image.open(f).convert("RGB")
            res = pipe(image=img, generator=torch.Generator(device=dev).manual_seed(args.seed),
                       num_inference_steps=args.steps, guidance_scale=args.guidance).samples[0]
            mesh = trimesh.Trimesh(np.asarray(res[0]).astype(np.float32),
                                   np.ascontiguousarray(res[1]), process=False)
            mesh.export(out)
            ok += 1
            print(f"  [{i+1}/{len(imgs)}] {name}: verts={len(mesh.vertices)} "
                  f"watertight={mesh.is_watertight} -> {out}", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(imgs)}] {name}: FAILED {type(e).__name__}: {e}", flush=True)
    print(f"DONE: {ok}/{len(imgs)} meshes generated into {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
