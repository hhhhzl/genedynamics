"""Morphology-latent system for MRMFMBD (A2: joint model-free / model-based).

Provides the learned morphology prior for the joint MF/MB sampler:
    - MorphDecoderConfig : frozen architecture/output-range config
    - VAE train + decode : g(w) → per-voxel occupancy (x_morph) on the fixed grid
    - MorphDecoder       : jitted runtime wrapper (hot-loop ready)
    - build_occupancy_dataset : offline dataset from a robotized asset bank

Mirrors the fidelity_system / mode_system subpackage layout
(specs.py + logic module + __init__ exports).
"""

from __future__ import annotations

from .specs import MorphDecoderConfig, default_morph_decoder_config
from .decoder import (
    MorphDecoder,
    init_params,
    encode,
    decode,
    decode_occ01,
    vae_loss,
    train_vae,
    save_params,
    load_params,
)
from .dataset import build_occupancy_dataset

__all__ = [
    "MorphDecoderConfig",
    "default_morph_decoder_config",
    "MorphDecoder",
    "init_params",
    "encode",
    "decode",
    "decode_occ01",
    "vae_loss",
    "train_vae",
    "save_params",
    "load_params",
    "build_occupancy_dataset",
]
