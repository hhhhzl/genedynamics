#!/usr/bin/env python
"""TRELLIS-text text->mesh batch inference (structured-voxel / SLAT prior).

Open-weights (microsoft/TRELLIS-text-xlarge, MIT). We decode ONLY the mesh format
(skip gaussian/radiance-field) and pull raw vertices/faces from the
MeshExtractResult — flexicubes is vendored (pure torch) and mesh geometry needs
neither nvdiffrast nor to_glb (those are texture/render only). Sparse attention
uses xformers (ATTN_BACKEND), spconv is a pip wheel: no CUDA compiles.

Run (in the trellis-equipped venv):
  ATTN_BACKEND=xformers SPCONV_ALGO=native HF_HOME=/workspace/morph_cache/hf \
  python batch_infer.py --prompts-file <prompts> --out-dir data/asset_banks/trellis/raw \
      --n-per-prompt 3 --seed 0
"""
import os
import sys
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "xformers")

# TRELLIS lives in a git submodule (third_party/morphology_priors/trellis_repo);
# add it to the path. Override via TRELLIS_REPO. (flexicubes is its own nested
# submodule — run `git submodule update --init --recursive` first.)
_REPO = os.environ.get("TRELLIS_REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../../../third_party/morphology_priors/trellis_repo"))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

import argparse
import json
import time

import numpy as np
import torch
import trimesh

from trellis.pipelines import TrellisTextTo3DPipeline


def _read_prompts(path):
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-prompt", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="microsoft/TRELLIS-text-xlarge")
    args = ap.parse_args()

    print(f"[trellis] loading {args.ckpt} ...", flush=True)
    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.ckpt)
    pipeline.cuda()

    os.makedirs(args.out_dir, exist_ok=True)
    prompts = _read_prompts(args.prompts_file)
    print(f"[trellis] {len(prompts)} prompts x {args.n_per_prompt} samples", flush=True)
    n_done = 0
    gen_times = []
    for pi, prompt in enumerate(prompts):
        for si in range(args.n_per_prompt):
            _t0 = time.perf_counter()
            outputs = pipeline.run(prompt, seed=args.seed + 1000 * pi + si,
                                   formats=["mesh"])
            mesh = outputs["mesh"][0]                       # MeshExtractResult
            verts = mesh.vertices.detach().cpu().numpy().astype(np.float32)
            faces = mesh.faces.detach().cpu().numpy().astype(np.int64)
            torch.cuda.synchronize()
            dt = time.perf_counter() - _t0
            gen_times.append(dt)
            out = os.path.join(args.out_dir, f"trellis_{pi:02d}_{si:02d}.obj")
            trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(out)
            n_done += 1
            print(f"[trellis] {n_done}: {prompt!r} sample {si} -> {out} "
                  f"({len(verts)} v, {len(faces)} f, {dt:.1f}s)", flush=True)
    _gt = {"prior": "trellis", "n": len(gen_times),
           "mean_s_per_asset": float(np.mean(gen_times)), "std_s": float(np.std(gen_times))}
    json.dump(_gt, open(os.path.join(args.out_dir, "..", "gen_time.json"), "w"), indent=2)
    print(f"[trellis] DONE: {n_done} meshes -> {args.out_dir} "
          f"(mean {_gt['mean_s_per_asset']:.1f}s/asset)", flush=True)


if __name__ == "__main__":
    main()
