#!/usr/bin/env python
"""Point-E text->point-cloud batch inference (runs in the isolated pointe venv).

DiffuseBot's default prior is text-conditioned Point-E (base40M-textvec), so this
is the apples-to-apples anchor row for Table 3. Outputs one .npz per (prompt,
sample) holding `points` (N,3) — the MAIN env then robotizes via `pc_robotize`
(point-cloud -> voxel occupancy -> SoftBodySpec). Heavy prior never shares a
process with the JAX/MPM rollouts.

Run (in venv):
  /workspace/morph_venvs/pointe/bin/python batch_infer.py \
      --prompts-file configs/soft_robot/co_design/table3/prompts.txt \
      --out-dir data/asset_banks/pointe/raw --n-per-prompt 3 --seed 0
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config


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
    ap.add_argument("--num-points", type=int, default=4096)
    # Weights cache MUST live under data/ (gitignored). Point-E's
    # default_cache_dir() == CWD/point_e_model_cache is used by BOTH load_checkpoint
    # AND the CLIP text encoder (ViT-L-14), so we chdir into the cache parent to
    # redirect every download there instead of polluting the repo root.
    ap.add_argument("--cache-dir", default="data/model_caches/point_e_model_cache")
    args = ap.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    args.prompts_file = os.path.abspath(args.prompts_file)
    cache = os.path.abspath(args.cache_dir)
    os.makedirs(cache, exist_ok=True)
    if os.path.basename(cache) == "point_e_model_cache":
        os.chdir(os.path.dirname(cache))   # CWD/point_e_model_cache == cache
    args.cache_dir = cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pointe] device={device}; loading base40M-textvec + upsampler...", flush=True)
    base_name = "base40M-textvec"
    base_model = model_from_config(MODEL_CONFIGS[base_name], device); base_model.eval()
    base_diff = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    up_model = model_from_config(MODEL_CONFIGS["upsample"], device); up_model.eval()
    up_diff = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])
    base_model.load_state_dict(load_checkpoint(base_name, device, cache_dir=args.cache_dir))
    up_model.load_state_dict(load_checkpoint("upsample", device, cache_dir=args.cache_dir))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, up_model],
        diffusions=[base_diff, up_diff],
        num_points=[1024, args.num_points - 1024],
        aux_channels=["R", "G", "B"],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=("texts", ""),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    prompts = _read_prompts(args.prompts_file)
    print(f"[pointe] {len(prompts)} prompts x {args.n_per_prompt} samples", flush=True)
    n_done = 0
    gen_times = []
    for pi, prompt in enumerate(prompts):
        for si in range(args.n_per_prompt):
            torch.manual_seed(args.seed + 1000 * pi + si)
            _t0 = time.perf_counter()
            samples = None
            for x in sampler.sample_batch_progressive(
                batch_size=1, model_kwargs=dict(texts=[prompt])
            ):
                samples = x
            pc = sampler.output_to_point_clouds(samples)[0]
            coords = np.asarray(pc.coords, dtype=np.float32)  # (num_points, 3)
            # Keep Point-E's RGB per point so the gallery renders like DiffuseBot's
            # plot_point_cloud (RGB-colored scatter).
            try:
                rgb = np.stack([pc.channels["R"], pc.channels["G"], pc.channels["B"]],
                               axis=-1).astype(np.float32)
            except Exception:
                rgb = np.zeros((coords.shape[0], 3), np.float32)
            torch.cuda.synchronize()
            dt = time.perf_counter() - _t0
            gen_times.append(dt)
            out = os.path.join(args.out_dir, f"pointe_{pi:02d}_{si:02d}.npz")
            np.savez(out, points=coords, rgb=rgb, prompt=prompt, prior="point_e")
            n_done += 1
            print(f"[pointe] {n_done}: {prompt!r} seed-variant {si} -> {out} "
                  f"({coords.shape[0]} pts, {dt:.1f}s)", flush=True)
    _gt = {"prior": "point_e", "n": len(gen_times),
           "mean_s_per_asset": float(np.mean(gen_times)),
           "std_s": float(np.std(gen_times))}
    json.dump(_gt, open(os.path.join(args.out_dir, "..", "gen_time.json"), "w"), indent=2)
    print(f"[pointe] DONE: {n_done} point clouds -> {args.out_dir} "
          f"(mean {_gt['mean_s_per_asset']:.1f}s/asset)", flush=True)


if __name__ == "__main__":
    main()
