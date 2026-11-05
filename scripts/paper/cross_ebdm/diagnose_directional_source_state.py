"""Compare the archived T3 directional gate with a same-target-state proxy.

This is a read-only, public-validation-model diagnostic. It does not change
the trained policy or the frozen physical comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from genedynamics.core.artifacts import atomic_json
from genedynamics.solvers.single.cross_ebdm.learning.recovery_transport import recovery_task_proxy
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import task_proxy
from scripts.paper.cross_ebdm.train_t4_completion import (
    ROOT, ensemble_task_spec, load_context, load_instance,
    load_training_config, response_normalization_spec, select, sha,
)


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha(path)}


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    old = np.array([row["old_advantage"] for row in rows], dtype=float)
    same = np.array([row["same_state_advantage"] for row in rows], dtype=float)
    old_mask = np.array([row["old_accept"] for row in rows], dtype=bool)
    same_mask = np.array([row["same_state_accept"] for row in rows], dtype=bool)
    return {
        "states": len(rows),
        "old_accept_fraction": float(old_mask.mean()),
        "same_state_accept_fraction": float(same_mask.mean()),
        "old_only_states": int(np.sum(old_mask & ~same_mask)),
        "same_state_only_states": int(np.sum(same_mask & ~old_mask)),
        "mean_old_advantage": float(old.mean()),
        "mean_same_state_advantage": float(same.mean()),
        "mean_advantage_shift": float((same - old).mean()),
    }


def run(config_path: Path, output: Path) -> dict[str, object]:
    if torch.cuda.is_initialized():
        raise RuntimeError("diagnostic must not initialize CUDA")
    config_path = config_path.resolve()
    config = load_training_config(config_path)
    if ensemble_task_spec(config) is not None:
        raise ValueError("this archived directional gate must use the mean task proxy")
    response_spec = response_normalization_spec(config)
    threshold = float(response_spec.minimum_source_task_advantage)
    original_actor_sha = config["actor_manifest"]["sha256"]
    actor_path = ROOT / config["actor_manifest"]["path"]
    current_actor_sha = sha(actor_path)
    # The actor binding changed after this training. It does not construct
    # validation states, but load_context checks its pointer. Override only
    # this pointer in memory and retain both hashes in the diagnostic report.
    config["actor_manifest"]["sha256"] = current_actor_sha
    device = torch.device("cpu")
    _, validation, _, data, feedback, _, _ = load_context(config, device)
    all_rows: list[dict[str, object]] = []
    instance_reports: dict[str, object] = {}
    for name, instance in sorted(config["instances"].items()):
        response, task_only, _, _, _, checkpoints = load_instance(
            name, instance, device, feedback,
        )
        rows: list[dict[str, object]] = []
        with torch.no_grad():
            for start in range(0, len(validation["target_features"]), 64):
                batch = select(validation, slice(start, start + 64))
                task_only_proxy = recovery_task_proxy(task_only, response, feedback, batch)
                source_command = batch["source_commands"]
                old_source = response(batch["source_features"], source_command)
                matched_source = response(batch["target_features"], source_command)
                old_proxy = task_proxy(
                    old_source, batch["current"], batch["task_reference"],
                ).mean(0)
                matched_proxy = task_proxy(
                    matched_source, batch["current"], batch["task_reference"],
                ).mean(0)
                old_advantage = (task_only_proxy - old_proxy).cpu().numpy()
                same_advantage = (task_only_proxy - matched_proxy).cpu().numpy()
                modes = batch["requested_mode"].reshape(-1).cpu().numpy()
                enabled = batch["recovery_enabled"].reshape(-1).cpu().numpy()
                if not np.isfinite(old_advantage).all() or not np.isfinite(same_advantage).all():
                    raise ValueError(f"nonfinite directional proxy in {name}")
                for index in range(len(old_advantage)):
                    row = {
                        "instance": name,
                        "validation_index": start + index,
                        "requested_mode": int(modes[index]),
                        "recovery_enabled": bool(enabled[index]),
                        "old_advantage": float(old_advantage[index]),
                        "same_state_advantage": float(same_advantage[index]),
                        "old_accept": bool(old_advantage[index] >= threshold),
                        "same_state_accept": bool(same_advantage[index] >= threshold),
                    }
                    rows.append(row)
        instance_reports[name] = {
            "all": summarize(rows),
            "recovery_enabled": summarize([row for row in rows if row["recovery_enabled"]]),
            "by_mode": {
                str(mode): summarize([row for row in rows if row["requested_mode"] == mode and row["recovery_enabled"]])
                for mode in (0, 1)
            },
            "checkpoints": checkpoints,
        }
        all_rows.extend(rows)
    report = {
        "version": "t3-directional-source-same-state-diagnostic-v1",
        "status": "complete_cpu_diagnostic",
        "scope": "archived public validation states and frozen model checkpoints; no physical task evidence",
        "threshold": threshold,
        "old_proxy": "response(source_features, source_commands) evaluated against target current/reference",
        "same_state_proxy": "response(target_features, source_commands) evaluated against the same target current/reference",
        "all": summarize(all_rows),
        "recovery_enabled": summarize([row for row in all_rows if row["recovery_enabled"]]),
        "by_mode": {
            str(mode): summarize([row for row in all_rows if row["requested_mode"] == mode and row["recovery_enabled"]])
            for mode in (0, 1)
        },
        "instances": instance_reports,
        "rows": all_rows,
        "provenance": {
            "config": pointer(config_path),
            "actor_manifest_original_sha256": original_actor_sha,
            "actor_manifest_current": pointer(actor_path),
            "actor_manifest_pointer_adjusted_in_memory": original_actor_sha != current_actor_sha,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "public_data": data,
            "cuda_initialized": torch.cuda.is_initialized(),
            "new_physics_calls": 0,
            "target_truth_opened": False,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault(
        "CROSS_EBDM_NATIVE_LIBRARY",
        str(ROOT / ".runtime/cross_ebdm/native/libcross_ebdm.so"),
    )
    report = run(args.config, args.output)
    print(json.dumps({
        "status": report["status"],
        "all": report["all"],
        "recovery_enabled": report["recovery_enabled"],
        "by_mode": report["by_mode"],
        "output": str(args.output / "report.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
