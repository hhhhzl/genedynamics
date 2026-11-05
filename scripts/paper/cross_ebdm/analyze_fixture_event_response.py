#!/usr/bin/env python3
"""Recompute event/quiet response errors for two frozen model objectives."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from genedynamics.solvers.single.cross_ebdm.learning.artifact_bundle import load_bundle
from genedynamics.solvers.single.cross_ebdm.learning.fixture_response import (
    CALIBRATED_EVENT_VERSION, EVENT_VERSION, FixtureResponseEnsemble,
)
from genedynamics.solvers.single.cross_ebdm.learning.response import FIXTURE_SEGMENT_SCALE


ROOT = Path(__file__).resolve().parents[3]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _predict(path: Path, *, version: str, tensor_sha: str,
             features: torch.Tensor, commands: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["version"] != version or payload["data_sha256"] != tensor_sha:
        raise ValueError("fixture event checkpoint identity differs")
    model = FixtureResponseEnsemble.from_checkpoint(payload, device="cpu").eval()
    with torch.inference_mode():
        inputs = torch.cat((features, commands.flatten(start_dim=1)), dim=1)
        values = torch.stack([member(inputs) for member in model.members])
    width = len(model.spec.segment_steps) * 27
    raw = values[..., :width].reshape(model.spec.members, len(features),
                                      len(model.spec.segment_steps), 27).numpy()
    probability = torch.sigmoid(values[..., width:-2]).numpy()
    scale = float(FIXTURE_SEGMENT_SCALE[18])
    conditional_x = (raw[..., 18] * scale).mean(axis=0)
    gated_x = (raw[..., 18] * probability * scale).mean(axis=0)
    return conditional_x, gated_x, probability.mean(axis=0)


def _metrics(truth: np.ndarray, conditional: np.ndarray, gated: np.ndarray,
             probability: np.ndarray, threshold: float,
             steps: tuple[int, ...]) -> list[dict]:
    rows = []
    for column, step in enumerate(steps):
        target = truth[:, column]
        event = np.abs(target) > threshold
        quiet = ~event
        if not event.any() or not quiet.any():
            raise ValueError("event/quiet validation group absent")
        prediction = gated[:, column]
        rows.append({
            "step": step, "event_windows": int(event.sum()),
            "quiet_windows": int(quiet.sum()),
            "all_mae_um": float(1e6 * np.abs(prediction - target).mean()),
            "event_mae_um": float(1e6 * np.abs(prediction[event] - target[event]).mean()),
            "quiet_mae_um": float(1e6 * np.abs(prediction[quiet] - target[quiet]).mean()),
            "zero_event_mae_um": float(1e6 * np.abs(target[event]).mean()),
            "conditional_event_mae_um": float(
                1e6 * np.abs(conditional[event, column] - target[event]).mean()),
            "event_probability_mean": float(probability[event, column].mean()),
            "quiet_probability_mean": float(probability[quiet, column].mean()),
            "event_brier": float(np.mean((probability[:, column] - event) ** 2)),
        })
    return rows


def analyze(config_paths: tuple[Path, Path], output: Path) -> dict:
    configs = [json.loads(path.read_text()) for path in config_paths]
    old, new = configs
    if (old["version"] != "compliant-socket-segmented-event-response-training-v2"
            or new["version"] != "compliant-socket-segmented-calibrated-event-response-training-v3"
            or old["data"] != new["data"] or old["model"] != new["model"]):
        raise ValueError("fixture objective comparison must share data and model")
    bundle = ROOT / old["data"]["path"]
    if _sha(bundle / "metadata.json") != old["data"]["metadata_sha256"]:
        raise ValueError("fixture response metadata changed")
    arrays, metadata = load_bundle(bundle)
    if metadata["tensor_sha256"] != old["data"]["tensor_sha256"]:
        raise ValueError("fixture response tensor changed")
    for row in new["source_bindings"]:
        snapshot = ROOT / new["output"] / "source_snapshot" / row["path"]
        if _sha(snapshot) != row["sha256"]:
            raise ValueError("calibrated event source snapshot differs")
    if json.loads((ROOT / new["output"] / "recipe.json").read_text()) != new:
        raise ValueError("calibrated event training recipe differs")
    index = np.flatnonzero(arrays["split"] == 1)
    if len(index) != metadata["validation_windows"]:
        raise ValueError("fixture validation split differs")
    features = torch.from_numpy(arrays["features"][index])
    commands = torch.from_numpy(arrays["commands"][index])
    truth = arrays["segment_targets"][index, :, 18] * float(FIXTURE_SEGMENT_SCALE[18])
    torch.set_num_threads(4)
    checkpoints = [ROOT / config["output"] / "model.pt" for config in configs]
    results = {}
    for label, checkpoint, version in zip(("conditional_v2", "calibrated_v3"), checkpoints,
                                          (EVENT_VERSION, CALIBRATED_EVENT_VERSION), strict=True):
        conditional, gated, probability = _predict(
            checkpoint, version=version, tensor_sha=old["data"]["tensor_sha256"],
            features=features, commands=commands,
        )
        results[label] = {
            "checkpoint_sha256": _sha(checkpoint),
            "segments": _metrics(
                truth, conditional, gated, probability,
                new["model"]["event_threshold_m"],
                tuple(new["model"]["segment_steps"]),
            ),
        }
    result = {
        "schema_version": "fixture-event-objective-comparison-v1",
        "status": "complete_development_diagnostic", "physical_calls": 0, "gpu_calls": 0,
        "formal_performance_evidence": False,
        "scope": "same previously inspected episode-disjoint validation split; no independent task or safety claim",
        "source_sha256": _sha(Path(__file__)),
        "config_sha256": {label: _sha(path) for label, path in zip(
            ("conditional_v2", "calibrated_v3"), config_paths, strict=True)},
        "data_metadata_sha256": old["data"]["metadata_sha256"],
        "validation_windows": len(index), "models": results,
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-config", type=Path, required=True)
    parser.add_argument("--new-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze((args.old_config, args.new_config), args.output)
    print(json.dumps(result["models"], indent=2))


if __name__ == "__main__":
    main()
