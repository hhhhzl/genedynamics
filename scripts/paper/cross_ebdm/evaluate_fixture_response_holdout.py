#!/usr/bin/env python3
"""Evaluate two frozen fixture-response objectives on new public episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analyze_fixture_event_response import _metrics, _predict
from genedynamics.solvers.single.cross_ebdm.learning.artifact_bundle import load_bundle
from genedynamics.solvers.single.cross_ebdm.learning.fixture_response import (
    CALIBRATED_EVENT_VERSION, EVENT_VERSION,
)
from genedynamics.solvers.single.cross_ebdm.learning.response import FIXTURE_SEGMENT_SCALE
from genedynamics.tasks.transfer.episode import audit_episode_evidence


ROOT = Path(__file__).resolve().parents[3]
VERSION = "fixture-response-frozen-model-holdout-v1"
LABELS = ("conditional_v2", "calibrated_v3")
VERSIONS = (EVENT_VERSION, CALIBRATED_EVENT_VERSION)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer(pointer: dict) -> Path:
    if set(pointer) != {"path", "sha256"}:
        raise ValueError("holdout pointer fields differ")
    path = ROOT / pointer["path"]
    if not path.is_file() or _sha(path) != pointer["sha256"]:
        raise ValueError(f"holdout pointer changed: {path}")
    return path


def evaluate(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    if (config["version"] != VERSION or tuple(config["models"]) != LABELS
            or len(config["collection"]) != 2
            or config["event_threshold_m"] != 1e-5
            or config["formal_performance_evidence"] is not False):
        raise ValueError("frozen fixture holdout identity differs")
    for source in config["source_bindings"]:
        _pointer(source)
    data_recipe = json.loads(_pointer(config["data_recipe"]).read_text())
    if data_recipe["version"] != "fixture-response-data-preparation-v1":
        raise ValueError("fixture holdout data recipe differs")
    arrays, metadata = load_bundle(ROOT / data_recipe["output"])
    if (_sha(ROOT / data_recipe["output"] / "metadata.json") != config["data_metadata_sha256"]
            or metadata["tensor_sha256"] != config["data_tensor_sha256"]
            or len(metadata["episodes"]) != 8):
        raise ValueError("fixture holdout bundle changed")
    collected = {}
    audits = []
    for entry in config["collection"]:
        recipe = json.loads(_pointer(entry["recipe"]).read_text())
        report = json.loads(_pointer(entry["report"]).read_text())
        if (not report["passed"] or report["plan_sha256"] != entry["recipe"]["sha256"]
                or len(report["episodes"]) != 4 or recipe["output_dir"] != entry["output_dir"]
                or recipe["failure_replacement"] is not False):
            raise ValueError("fixture collection report differs")
        for row in report["episodes"]:
            if not row["evidence_audit_passed"]:
                raise ValueError("fixture evidence was not audited at collection")
            checked = audit_episode_evidence(ROOT / row["evidence_path"])
            if not checked["passed"]:
                raise ValueError("fixture holdout raw episode audit failed")
            collected[row["seed"]] = (str(ROOT / row["public_path"]), row["public_sha256"],
                                       row["mode"], entry["name"])
            audits.append({"seed": row["seed"], "evidence_sha256": checked["sha256"]})
    if len(collected) != 8 or set(config["mechanisms"]) != {item[3] for item in collected.values()}:
        raise ValueError("fixture holdout episode roster differs")
    for episode in metadata["episodes"]:
        if (episode["path"], episode["public_sha256"], episode["mode"]) != collected[episode["seed"]][:3]:
            raise ValueError("fixture holdout public trajectory differs")

    model_configs = [json.loads(_pointer(config["models"][label]["recipe"]).read_text())
                     for label in LABELS]
    if (model_configs[0]["data"] != model_configs[1]["data"]
            or model_configs[0]["model"] != model_configs[1]["model"]
            or model_configs[0]["model"]["event_threshold_m"] != config["event_threshold_m"]):
        raise ValueError("fixture model comparison differs")
    train_bundle = ROOT / model_configs[0]["data"]["path"]
    _, train_metadata = load_bundle(train_bundle)
    if (_sha(train_bundle / "metadata.json") != model_configs[0]["data"]["metadata_sha256"]
            or train_metadata["tensor_sha256"] != model_configs[0]["data"]["tensor_sha256"]):
        raise ValueError("fixture training bundle changed")
    train_seeds = {episode["seed"] for episode in train_metadata["episodes"]}
    train_sha = {episode["public_sha256"] for episode in train_metadata["episodes"]}
    if train_seeds & set(collected) or train_sha & {item[1] for item in collected.values()}:
        raise ValueError("fixture holdout overlaps either model's training bundle")

    features = torch.from_numpy(arrays["features"])
    commands = torch.from_numpy(arrays["commands"])
    truth = arrays["segment_targets"][:, :, 18] * float(FIXTURE_SEGMENT_SCALE[18])
    steps = tuple(metadata["segment_steps"])
    if steps != (3, 6, 10):
        raise ValueError("fixture holdout segment roster differs")
    torch.set_num_threads(4)
    results = {}
    episode_ids = arrays["episode_id"]
    for label, version, model_config in zip(LABELS, VERSIONS, model_configs, strict=True):
        checkpoint = _pointer(config["models"][label]["checkpoint"])
        if checkpoint != ROOT / model_config["output"] / "model.pt":
            raise ValueError("fixture model checkpoint path differs")
        conditional, gated, probability = _predict(
            checkpoint, version=version,
            tensor_sha=model_config["data"]["tensor_sha256"],
            features=features, commands=commands,
        )
        groups = {"all": np.ones(len(features), dtype=bool)}
        for name in config["mechanisms"]:
            ids = [episode["episode_id"] for episode in metadata["episodes"]
                   if collected[episode["seed"]][3] == name]
            if len(ids) != 4:
                raise ValueError("fixture mechanism episode roster differs")
            groups[name] = np.isin(episode_ids, ids)
        results[label] = {
            "checkpoint_sha256": _sha(checkpoint),
            "groups": {name: {"windows": int(mask.sum()), "segments": _metrics(
                truth[mask], conditional[mask], gated[mask], probability[mask],
                config["event_threshold_m"], steps,
            )} for name, mask in groups.items()},
        }
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "version": VERSION, "status": "complete_independent_prediction_holdout",
        "formal_performance_evidence": False,
        "scope": "eight new public episodes, same two fixture mechanisms, frozen checkpoints; prediction only",
        "recipe_sha256": _sha(config_path), "data_metadata_sha256": config["data_metadata_sha256"],
        "data_tensor_sha256": config["data_tensor_sha256"],
        "new_physical_episodes": 8, "model_updates": 0, "gpu_calls": 0,
        "episode_audits": audits, "models": results,
    }
    (output / "recipe.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.config)
    print(json.dumps({label: result["models"][label]["groups"]["all"]
                      for label in LABELS}, indent=2))


if __name__ == "__main__":
    main()
