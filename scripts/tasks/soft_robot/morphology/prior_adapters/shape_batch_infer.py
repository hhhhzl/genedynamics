#!/usr/bin/env python
"""Shap-E text->mesh batch inference (runs in the isolated shape venv).

Text-native implicit prior (OpenAI Shap-E): a text-conditioned latent diffusion
whose latents decode to a watertight implicit mesh. Outputs one .obj per
(prompt, sample); the MAIN env robotizes via `robotize_mesh`.

Run (in venv):
  /workspace/morph_venvs/shape/bin/python batch_infer.py \
      --prompts-file configs/soft_robot/co_design/table3/prompts.txt \
      --out-dir data/asset_banks/shape/raw --n-per-prompt 3 --seed 0
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh


def _read_prompts(path):
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _save_obj(tm, path):
    import trimesh
    verts = tm.verts if hasattr(tm, "verts") else tm.vertices
    faces = tm.faces
    import numpy as np
    trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces),
                    process=False).export(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-prompt", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guidance-scale", type=float, default=15.0)
    ap.add_argument("--karras-steps", type=int, default=64)
    # Weights cache under data/ (gitignored). Shap-E's default_cache_dir() ==
    # CWD/shap_e_model_cache; chdir into the cache parent so every download lands
    # under data/ instead of the repo root.
    ap.add_argument("--cache-dir", default="data/model_caches/shap_e_model_cache")
    args = ap.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    args.prompts_file = os.path.abspath(args.prompts_file)
    cache = os.path.abspath(args.cache_dir)
    os.makedirs(cache, exist_ok=True)
    if os.path.basename(cache) == "shap_e_model_cache":
        os.chdir(os.path.dirname(cache))
    args.cache_dir = cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[shape] device={device}; loading transmitter + text300M...", flush=True)
    xm = load_model("transmitter", device=device, cache_dir=args.cache_dir)
    model = load_model("text300M", device=device, cache_dir=args.cache_dir)
    diffusion = diffusion_from_config(load_config("diffusion", cache_dir=args.cache_dir))

    os.makedirs(args.out_dir, exist_ok=True)
    prompts = _read_prompts(args.prompts_file)
    print(f"[shape] {len(prompts)} prompts x {args.n_per_prompt} samples", flush=True)
    n_done = 0
    gen_times = []
    for pi, prompt in enumerate(prompts):
        for si in range(args.n_per_prompt):
            torch.manual_seed(args.seed + 1000 * pi + si)
            _t0 = time.perf_counter()
            latents = sample_latents(
                batch_size=1, model=model, diffusion=diffusion,
                guidance_scale=args.guidance_scale,
                model_kwargs=dict(texts=[prompt]),
                progress=False, clip_denoised=True, use_fp16=True,
                use_karras=True, karras_steps=args.karras_steps,
                sigma_min=1e-3, sigma_max=160, s_churn=0,
            )
            tm = decode_latent_mesh(xm, latents[0]).tri_mesh()
            torch.cuda.synchronize()
            dt = time.perf_counter() - _t0
            gen_times.append(dt)
            out = os.path.join(args.out_dir, f"shape_{pi:02d}_{si:02d}.obj")
            _save_obj(tm, out)
            n_done += 1
            print(f"[shape] {n_done}: {prompt!r} sample {si} -> {out} ({dt:.1f}s)", flush=True)
    _gt = {"prior": "shape_e", "n": len(gen_times),
           "mean_s_per_asset": float(np.mean(gen_times)), "std_s": float(np.std(gen_times))}
    json.dump(_gt, open(os.path.join(args.out_dir, "..", "gen_time.json"), "w"), indent=2)
    print(f"[shape] DONE: {n_done} meshes -> {args.out_dir} "
          f"(mean {_gt['mean_s_per_asset']:.1f}s/asset)", flush=True)


if __name__ == "__main__":
    main()
