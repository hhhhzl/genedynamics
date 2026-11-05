#!/usr/bin/env python3
"""Uniform CPU predeployment package qualification for all four-task rows.

This gate performs no reset and no physics.  It validates the common task
contract, immutable artifacts, actor construction, and resettable controller
state.  A predeclared rejection remains a scored roster row and is never
silently omitted or replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.bindings import (
    create_transfer_actor,
    resolve_transfer_binding,
    validate_transfer_actor_manifest,
)
from genedynamics.tasks.transfer.specs import mechanism_spec


ROOT = Path(__file__).resolve().parents[3]
NATIVE_LIBRARY = ROOT / ".runtime/cross_ebdm/native/libcross_ebdm.so"
GPU_UUIDS = {
    0: "GPU-253f0813-4432-2d51-a149-3e73b38bf77a",
    1: "GPU-58b54684-3577-11d3-f039-0e8469190739",
}
TASKS = {
    "T1": ROOT / "configs/arm/obstacle_transfer",
    "T2": ROOT / "configs/arm/guided_contact_alignment",
    "T3": ROOT / "configs/arm/keyed_insertion_recovery",
    "T4": ROOT / "configs/quadruped/mobile_keyed_insertion_recovery",
}
MAIN_CONFIG = {
    "cross_ebdm": "main/cross_ebdm.yaml",
    "task_only": "baseline/task_only.yaml",
    "fada": "baseline/fada.yaml",
    "acp": "baseline/acp.yaml",
    "human2sim2robot": "baseline/human2sim2robot.yaml",
    "dial_transfer": "baseline/dial.yaml",
    "response_mpc": "baseline/response_mpc.yaml",
    "public_geometry": "baseline/public_geometry.yaml",
    "public_source_prior": "baseline/public_source_prior.yaml",
}
ABLATION_CONFIG = {
    "frozen_warm": "ablation/frozen_warm.yaml",
    "task_only": "baseline/task_only.yaml",
    "matching_no_band": "ablation/matching_no_band.yaml",
    "band_only": "ablation/band_only.yaml",
    "ungated_full": "ablation/ungated_full.yaml",
    "gated_full": "main/cross_ebdm.yaml",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def torch_devices(value, *, depth=0, seen=None) -> set[str]:
    """Find Torch parameter/buffer devices without descending into task physics."""
    import torch

    seen = set() if seen is None else seen
    if value is None or id(value) in seen or depth > 10:
        return set()
    seen.add(id(value))
    if isinstance(value, torch.nn.Module):
        return {
            str(tensor.device)
            for tensor in list(value.parameters()) + list(value.buffers())
        }
    if isinstance(value, dict):
        return set().union(*(
            torch_devices(child, depth=depth + 1, seen=seen)
            for child in value.values()
        ), set())
    if isinstance(value, (list, tuple)):
        return set().union(*(
            torch_devices(child, depth=depth + 1, seen=seen)
            for child in value
        ), set())
    if hasattr(value, "__dict__") and not value.__class__.__module__.startswith(
        ("genedynamics.envs", "genedynamics.tasks.transfer.episode")
    ):
        return torch_devices(vars(value), depth=depth + 1, seen=seen)
    return set()


def build_task(config: ExperimentConfig):
    values = dict(config.env_params)
    robot = values.pop("robot", None)
    history_len = int(values.pop("history_len", 4))
    values.pop("mode", None)
    shared_execution = values.pop("shared_execution", None)
    mechanism_values = dict(values.pop("mechanism", {}))
    body_support = dict(values.pop("body_support", {}))
    mechanism = mechanism_spec(
        config.env_name, mechanism_values,
        body_support_parameters=body_support,
    )
    task = instantiate_task(
        config.env_name, robot=robot, mechanism_spec=mechanism,
        history_len=history_len, config_overrides=values or None,
        shared_execution=shared_execution,
    )
    task._experiment_mode = int(config.env_params.get("mode", 0))
    return task


def qualify_row(alias: str, family: str, key: str, path: Path, *, device: str) -> dict:
    config = ExperimentConfig.from_yaml(path)
    selection = dict(config.method_params["actor_binding"])
    selection["family"] = family
    selection["key"] = key
    # Instance-ready rows need an explicit slot even for status resolution.
    # Deterministic and rejected rows ignore this field by contract.
    selection["instance"] = "instance_01"
    parameters = dict(config.method_params)
    parameters["device"] = device
    parameters["actor_binding"] = selection
    binding = resolve_transfer_binding(parameters, task_alias=alias)
    base = {
        "task": alias, "family": family, "key": key,
        "config": str(path.relative_to(ROOT)),
        "method": config.method, "method_identity": binding["method_identity"],
        "actor_device": device,
        "input_status": binding["status"],
        "common_contract_inherited": alias == "T1" or bool(
            config.env_params.get("shared_execution")
        ),
    }
    if binding["status"] == "predeployment_rejection":
        scoring = binding["scoring"]
        if (
            binding.get("target_physical_execution") != 0
            or scoring.get("predeployment_rejection") != 1.0
            or scoring.get("constrained_success") != 0.0
        ):
            raise ValueError("predeployment rejection has non-frozen scoring")
        return {
            **base, "outcome": "retained_predeployment_rejection",
            "reason": binding["reason"], "actor_instances_constructed": 0,
            "physics_steps": 0, "passed": True,
        }

    instances = (
        ["fixed"] if binding["status"] == "deterministic_single_identity"
        else [f"instance_{index:02d}" for index in range(1, 9)]
    )
    classes = set()
    devices = set()
    constructed = 0
    for instance in instances:
        modes = (0, 1) if alias == "T3" and key in {"cross_ebdm", "task_only"} else (0,)
        for mode in modes:
            task = build_task(config)
            task._experiment_mode = mode
            run_parameters = dict(parameters)
            run_parameters["mode"] = mode
            run_selection = dict(selection)
            if instance != "fixed":
                run_selection["instance"] = instance
            run_parameters["actor_binding"] = run_selection
            run_binding = resolve_transfer_binding(
                run_parameters, task_alias=alias,
            )
            actor = create_transfer_actor(task, config.method, run_parameters)
            if not callable(getattr(actor, "act", None)):
                raise TypeError("qualified actor has no public act method")
            reset = getattr(actor, "reset", None)
            if callable(reset):
                reset()
            if any(task.accounting.values()) or any(task.lifetime_accounting.values()):
                raise RuntimeError("package qualification executed task physics")
            classes.add(f"{type(actor).__module__}.{type(actor).__qualname__}")
            actor_devices = torch_devices(actor)
            learned = any(
                label in run_binding
                for label in ("checkpoint", "policy_checkpoint", "response_checkpoint")
            )
            if learned and actor_devices != ({"cuda:0"} if device == "cuda" else {"cpu"}):
                raise RuntimeError(
                    f"learned actor tensors are on {sorted(actor_devices)}, expected {device}"
                )
            devices.update(actor_devices)
            constructed += 1
    return {
        **base, "outcome": "package_qualified",
        "actor_instances_constructed": constructed,
        "actor_classes": sorted(classes), "torch_devices": sorted(devices),
        "physics_steps": 0, "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--physical-gpu-index", type=int, choices=tuple(GPU_UUIDS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        folder = "_qualification" if args.device == "cpu" else "_calibration"
        args.output = ROOT / f"results/{folder}/four_task_transfer/method_packages_20261021/report.json"
    if args.device == "cuda":
        import torch
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical_index = (
            args.physical_gpu_index
            if args.physical_gpu_index is not None
            else int(visible) if visible in {"0", "1"} else None
        )
        if physical_index not in GPU_UUIDS or visible != str(physical_index):
            raise RuntimeError(
                "GPU calibration requires exactly the declared physical GPU"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("the declared physical GPU must be the only visible CUDA device")
        uuid = subprocess.check_output([
            "nvidia-smi", "-i", str(physical_index),
            "--query-gpu=uuid", "--format=csv,noheader",
        ], text=True).strip()
        if uuid != GPU_UUIDS[physical_index]:
            raise RuntimeError(f"physical GPU{physical_index} UUID changed: {uuid}")
    if NATIVE_LIBRARY.is_file():
        os.environ.setdefault("CROSS_EBDM_NATIVE_LIBRARY", str(NATIVE_LIBRARY))
    rows = []
    for alias, root in TASKS.items():
        base = ExperimentConfig.from_yaml(root / "_base.yaml")
        validate_transfer_actor_manifest(
            base.method_params["actor_binding"]["manifest"]
        )
        for family, mapping in (("main", MAIN_CONFIG), ("ablation", ABLATION_CONFIG)):
            for key, relative in mapping.items():
                try:
                    rows.append(qualify_row(
                        alias, family, key, root / relative, device=args.device,
                    ))
                except Exception as error:
                    rows.append({
                        "task": alias, "family": family, "key": key,
                        "config": str((root / relative).relative_to(ROOT)),
                        "outcome": "qualification_failure", "passed": False,
                        "reason": str(error), "error_type": type(error).__name__,
                        "traceback": traceback.format_exc(), "physics_steps": 0,
                    })
    counts = {}
    for row in rows:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    report = {
        "schema_version": "cross-ebdm-method-package-qualification-v1",
        "status": "passed" if all(row["passed"] for row in rows) else "failed",
        "passed": all(row["passed"] for row in rows),
        "device": args.device, "gpu_used": args.device == "cuda", "physics_steps": 0,
        "performance_evidence": False,
        "row_count": len(rows), "counts": counts, "rows": rows,
        "policy": {
            "same_gate_for_every_row": True,
            "retained_rejection_is_valid_outcome": True,
            "old_checkpoint_performance_reused": False,
            "qualified_artifacts_are_interface_compatibility_only": True,
            "fresh_training_or_calibration_required_before_physical_admission": True,
        },
        "gpu_binding": (
            {
                "physical_index": physical_index, "logical_index": 0,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "name": __import__("torch").cuda.get_device_name(0),
                "uuid": GPU_UUIDS[physical_index],
            }
            if args.device == "cuda" else None
        ),
        "native_library": (
            {"path": str(NATIVE_LIBRARY.relative_to(ROOT)), "sha256": sha(NATIVE_LIBRARY)}
            if NATIVE_LIBRARY.is_file() else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "output": str(args.output),
        "sha256": sha(args.output), "counts": counts,
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
