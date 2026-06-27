#!/usr/bin/env python
"""SplatFlow text->3D-Gaussian batch inference (3DGS prior for Table 3).

Runs ONLY SplatFlow's step-1 (multi-view rectified-flow -> GS decoder -> initial
3DGS); skips the step-2 SDS++ refiner (needs another diffusion model + the CUDA
rasterizer we don't need). Memory budget for a 24 GB GPU: `sd3_guidance=False`
(drops the 4 GB SD3 transformer) and `cfg=False`. Extracts the Gaussian CENTERS
(means) from the written gaussian.ply -> .npz {centers, scales, opacities} for the
MAIN env's `robotize_gaussians` (3DGS shell -> solid-fill -> occupancy).

Run (in the splatflow venv, from the repo dir):
  HF_HOME=/root/.cache/hf python batch_infer.py \
      --prompts-file <prompts> --out-dir data/asset_banks/splatflow/raw \
      --ckpt-dir /workspace/genedynamics/data/splatflow_ckpts --n-per-prompt 3
"""
import argparse
import json
import os
import sys
import time

# SplatFlow lives in a git submodule (third_party/morphology_priors/splatflow_repo);
# add it to the path + chdir (its hydra config + `model`/`inference` packages are
# relative). Override via SPLATFLOW_REPO. The fp16 / from_config loader fixes live
# in prior_adapters/splatflow_fp16_fromconfig.patch — apply it to the submodule
# once: `git -C splatflow_repo apply <this dir>/splatflow_fp16_fromconfig.patch`.
_REPO = os.environ.get("SPLATFLOW_REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../../../third_party/morphology_priors/splatflow_repo"))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

import numpy as np
import torch

from hydra import compose, initialize_config_dir
from diffusers import FlowMatchEulerDiscreteScheduler
from plyfile import PlyData

from model.gsdecoder.load_gsdecoder import create_gsdecoder
from model.multiview_rf.load_mv_sd3 import create_sd_multiview_rf_model
from model.util import create_vae
from inference.generate import generate_sampling


def _read_prompts(path):
    with open(path) as f:
        return [s.strip() for s in f if s.strip() and not s.startswith("#")]


def _load_gaussian_ply(path):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], -1).astype(np.float32)
    sc = [k for k in ("scale_0", "scale_1", "scale_2") if k in v.data.dtype.names]
    scales = (np.exp(np.stack([np.asarray(v[k]) for k in sc], -1)).astype(np.float32)
              if len(sc) == 3 else None)
    opac = (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float32)))
            if "opacity" in v.data.dtype.names else None)
    return xyz, scales, opac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--n-per-prompt", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-steps", type=int, default=50)
    args = ap.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    args.prompts_file = os.path.abspath(args.prompts_file)
    os.makedirs(args.out_dir, exist_ok=True)
    tmp_save = os.path.abspath(os.path.join(args.out_dir, "..", "_sf_tmp"))

    mv_ckpt = os.path.join(os.path.abspath(args.ckpt_dir), "mv_rf_ema.pt")
    gs_ckpt = os.path.join(os.path.abspath(args.ckpt_dir), "gs_decoder.pt")

    with initialize_config_dir(config_dir=os.path.abspath("config"), version_base="1.1"):
        cfg = compose(config_name="base_config", overrides=[
            "+experiments=generation",
            f"inference.mv_rf_ckpt={mv_ckpt}",
            f"inference.gsdecoder_ckpt={gs_ckpt}",
            "inference.sample.sd3_guidance=false",
            "inference.sample.cfg=true",   # CFG path assigns clean_ray_latent (required)
            f"inference.sample.num_steps={args.num_steps}",
            f"inference.generate.save_path={tmp_save}",
        ])

    device = torch.device("cuda")
    dtype = torch.float16
    print("[splatflow] loading VAE + GS decoder + MV-RF + text encoders ...", flush=True)
    vae = create_vae(cfg).to(device=device, dtype=dtype).eval()
    decoder = create_gsdecoder(cfg).to(device=device)
    decoder.load_state_dict(torch.load(gs_ckpt, map_location="cpu", weights_only=False))
    decoder.eval()
    model, tokenizer, text_encoders = create_sd_multiview_rf_model()
    model = model.to(device=device, dtype=dtype)
    model.load_state_dict(torch.load(mv_ckpt, map_location="cpu", weights_only=False))
    model.eval()
    text_encoders = [t.requires_grad_(False).to(device, dtype=dtype) for t in text_encoders]
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.mv_rf_model.hf_path, subfolder="scheduler", shift=1.0)

    prompts = _read_prompts(args.prompts_file)
    print(f"[splatflow] {len(prompts)} prompts x {args.n_per_prompt}", flush=True)
    n_done, gen_times = 0, []
    for pi, prompt in enumerate(prompts):
        for si in range(args.n_per_prompt):
            torch.manual_seed(args.seed + 1000 * pi + si)
            cfg.inference.generate.prompt = prompt
            t0 = time.perf_counter()
            with torch.no_grad():
                generate_sampling(prompt, text_encoders, tokenizer, model, decoder,
                                  vae, scheduler, None, cfg, device, dtype)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            ply = os.path.join(tmp_save, prompt, "gaussian.ply")
            xyz, scales, opac = _load_gaussian_ply(ply)
            out = os.path.join(args.out_dir, f"splatflow_{pi:02d}_{si:02d}.npz")
            np.savez(out, centers=xyz, scales=scales, opacities=opac,
                     prompt=prompt, prior="splatflow")
            gen_times.append(dt)
            n_done += 1
            print(f"[splatflow] {n_done}: {prompt!r} sample {si} -> {out} "
                  f"({len(xyz)} gaussians, {dt:.1f}s)", flush=True)
    gt = {"prior": "splatflow", "n": len(gen_times),
          "mean_s_per_asset": float(np.mean(gen_times)), "std_s": float(np.std(gen_times))}
    json.dump(gt, open(os.path.join(args.out_dir, "..", "gen_time.json"), "w"), indent=2)
    print(f"[splatflow] DONE: {n_done} -> {args.out_dir} (mean {gt['mean_s_per_asset']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
