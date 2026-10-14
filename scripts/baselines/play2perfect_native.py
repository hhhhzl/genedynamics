"""Inspect actual published Play2Perfect actor checkpoints using original code.

Network loading/forward profiling is separate from native Isaac Sim closed-loop
qualification. The in-hand mechanism is never replaced by a fixed grasp here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

from scripts.baselines.sources import LOCK, ROOT, verify


def _numpy1_scalar(*args):
    core = getattr(np, "_core", None)
    if core is None:
        core = np.core
    return core.multiarray.scalar(*args)


_numpy1_scalar.__module__ = "numpy.core.multiarray"
_numpy1_scalar.__name__ = "scalar"
_numpy1_scalar.__qualname__ = "scalar"


def load_published(path):
    # The checkpoint's NumPy1 scalar global needs an explicit alias on NumPy2;
    # keep the restricted Tensor/primitive loader instead of arbitrary pickle.
    allowed = [_numpy1_scalar, np.dtype, type(np.dtype("float64")), type(np.dtype("float32"))]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    # The native torch_runner selects rank0 from distributed checkpoints.
    return payload[0] if 0 in payload else payload


def native_model(payload, *, device="cuda:0", batch_size=1):
    spec = next(s for s in json.loads(LOCK.read_text())["sources"] if s["method"] == "play2perfect")
    verify(spec)
    source = ROOT / "baselines/Play2Perfect"
    sys.path.insert(0, str(source / "rl_games"))
    from rl_games.algos_torch.model_builder import ModelBuilder

    recipe = source / "isaacsimenvs/cfg/train/PreciseAssemblySAPG.yaml"
    params = yaml.safe_load(recipe.read_text())["params"]
    state = payload["model"]
    obs_dim = state["running_mean_std.running_mean"].numel()
    action_dim = state["a2c_network.mu.bias"].numel()
    builder = ModelBuilder().load(params)
    model = builder.build({
        "actions_num": action_dim, "input_shape": (obs_dim + 1,),
        "num_seqs": batch_size, "value_size": 1,
        "normalize_value": True, "normalize_input": True,
        "type": "extra_param", "coef_ids": torch.linspace(50., 0., 6).to(device),
        "coef_id_idx": obs_dim, "param_size": 32,
    }).to(device).eval()
    model.load_state_dict(state, strict=True)
    return model, obs_dim, action_dim, recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from scripts.baselines import require_non_paper_output
    args.output = require_non_paper_output(args.output)
    payload = load_published(args.checkpoint)
    profiles = []
    for batch in (1, 8):
        model, obs_dim, action_dim, recipe = native_model(payload, device=args.device, batch_size=batch)
        obs = torch.zeros((batch, obs_dim + 1), device=args.device)
        # Native exploitation block has coefficient0; preserve the coefficient
        # condition and learned32D embedding rather than dropping SAPG context.
        states = [state.to(args.device) for state in model.get_default_rnn_state()]
        durations = []
        with torch.inference_mode():
            for step in range(11):
                if torch.device(args.device).type == "cuda": torch.cuda.synchronize()
                start = time.perf_counter()
                result = model({"is_train": False, "prev_actions": None, "obs": obs.clone(), "rnn_states": states})
                states = result["rnn_states"]
                if torch.device(args.device).type == "cuda": torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
                assert result["mus"].shape == (batch, action_dim)
                assert torch.isfinite(result["mus"]).all()
        profiles.append({"batch": batch, "device": str(next(model.parameters()).device),
                         "cold_seconds": durations[0], "p50_seconds": float(np.median(durations[1:])),
                         "p95_seconds": float(np.percentile(durations[1:], 95))})
        del model
    report = {
        "status": "published_checkpoint_network_loaded",
        "checkpoint_sha256": hashlib.file_digest(args.checkpoint.open("rb"), "sha256").hexdigest(),
        "native_epoch": int(payload["epoch"]), "native_frames": int(payload["frame"]),
        "observation_dim": obs_dim, "action_dim": action_dim,
        "source_config": str(recipe.relative_to(ROOT)), "profiles": profiles,
        "scope": "actual native actor/checkpoint/schema/forward profile; synthetic inputs, no physical episode",
        "native_closed_loop_qualified": False,
        "pending": ["Python3.11 IsaacLab2.3.2.post1/IsaacSim environment qualification",
                    "native in-hand play and assembly closed-loop", "full-denominator native evaluation"],
        "native_evaluation_note": "upstream eval_offline excludes unstable short first episodes; any shared comparison must also retain all requested trials",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoint_network.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
