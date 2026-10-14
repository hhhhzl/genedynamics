"""Measure FADA history + policy + real nominal Panda physics at batch1 and4.

Eleven actual batched control calls per workload consume55 source-world control
steps in total. These short trajectories qualify workload execution, not task
success. No target environment, training or synthetic repeated inputs are used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import zipfile

import numpy as np
import torch

from genedynamics.experiments.artifacts import atomic_json
from genedynamics.experiments.baselines import instantiate_task
from genedynamics.experiments.framework.config import ExperimentConfig
from genedynamics.learning.baselines.fada.task_training import PORTABLE_FIELDS, bounded_task_action, portable_history
from genedynamics.solvers.single.fada import FADASolver


def _json(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def _resources(device):
    status = Path("/proc/self/status").read_text()
    rss = next(int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith("VmRSS:"))
    result = {"cpu_rss_bytes": rss, "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if torch.device(device).type == "cuda":
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                      cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        try:
            query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
            rows = [line.split(",") for line in query.stdout.splitlines() if line.split(",")[0].strip() == str(os.getpid())]
            result["gpu_process_memory"] = [{"gpu_uuid": row[1].strip(), "used_memory_MiB": int(row[2].strip())} for row in rows]
        except (OSError, subprocess.SubprocessError, ValueError):
            result["gpu_process_memory"] = None
    return result


def execute_workload(task, policy, *, sample_ids, seed, device, iterations=11, progress=None):
    """Step an already-reset real task; reset only ended slots before their next call."""
    baseline = dict(task.lifetime_accounting)
    durations, actions, statuses, history_counts, resets, memory = [], [], [], [], [], []
    modes = np.arange(len(sample_ids)) % 2
    done = np.asarray(task.state.status) != 0
    for index in range(iterations):
        if torch.device(device).type == "cuda": torch.cuda.synchronize(device)
        started = time.perf_counter()
        if done.any():
            task.reset(sample_ids, seed=seed + index + 1, mask=done)
            task.set_modes(modes); policy.reset()
            resets.append({"before_iteration": index, "mask": done.tolist(), "seed": seed + index + 1})
        observation = portable_history(task)
        history_counts.append(observation["history_valid_mask"].sum(-1).tolist())
        command = bounded_task_action(task, policy.act(observation))
        result = task.step(command)
        if torch.device(device).type == "cuda": torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
        actions.append(command.copy()); statuses.append(result.status.copy())
        done = result.done.copy()
        memory.append(_resources(device))
        if progress:
            progress({"completed_iterations": index + 1, "lifetime_accounting": dict(task.lifetime_accounting)})
    delta = {key: task.lifetime_accounting[key] - baseline[key] for key in baseline}
    if delta["control_steps"] != iterations * len(sample_ids):
        raise RuntimeError("workload lost an ended slot or exceeded its physical control budget")
    return {"iterations": iterations, "batch": len(sample_ids), "cold_seconds": durations[0],
            "warm_p50_seconds": float(np.median(durations[1:])), "warm_p95_seconds": float(np.percentile(durations[1:], 95)),
            "all_pipeline_seconds": durations, "history_valid_counts_before_step": history_counts,
            "partial_resets": resets, "physical_accounting_delta_after_initial_reset": delta,
            "total_physical_accounting_including_initial_reset": dict(task.lifetime_accounting),
            "resource_samples_after_each_step": memory,
            "sampled_max_cpu_rss_bytes": max(row["cpu_rss_bytes"] for row in memory),
            "task_evaluator": task.evaluate()}, np.asarray(actions), np.asarray(statuses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    origin = args.qualification.resolve()
    context = json.loads((origin.parent / "source-context.json").read_text())
    cfg = ExperimentConfig.from_yaml(context["argv"]["config"])
    checkpoint_path = origin.parent / "source_policy.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["cfg"]["obs_dim"] != 35 or tuple(checkpoint["genedynamics_task_adapter"]["portable_fields"]) != PORTABLE_FIELDS:
        raise ValueError("real task workloads require the35D measured physical state chart")
    if cfg.env_name not in {"obstacle_transfer", "keyed_insertion_recovery"}:
        raise ValueError("only the qualified T1/T3 nominal Panda source tasks are allowed")
    output = origin.parent / "workloads_batch_1_4"
    output.mkdir(exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    paths = set((root / "genedynamics").rglob("*.py")) | {Path(__file__).resolve(), Path(context["argv"]["config"]).resolve()}
    hashes = {}
    with zipfile.ZipFile(output / "source-overlay.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            data = path.read_bytes(); name = str(path.relative_to(root))
            hashes[name] = hashlib.sha256(data).hexdigest(); archive.writestr(name, data)
    atomic_json(output / "provenance.json", {"source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "parent_qualification_sha256": hashlib.sha256(origin.read_bytes()).hexdigest(), "source_sha256": hashes,
        "source_overlay_sha256": hashlib.sha256((output / "source-overlay.zip").read_bytes()).hexdigest(),
        "seed": args.seed, "source_only": True, "target_calls": 0})
    atomic_json(output / "attempt.json", {"status": "running", "target_calls": 0, "maximum_source_control_steps": 55})
    if torch.device(args.device).type == "cuda": torch.cuda.init()
    profiles, arrays = [], {}
    for batch in (1, 4):
        setup_started = time.perf_counter()
        policy = FADASolver.from_checkpoint(checkpoint, device=args.device)
        task = instantiate_task(cfg.env_name, robot="panda", seed=args.seed + batch, history_len=checkpoint["cfg"]["history_len"])
        identities = tuple(args.seed + batch * 100 + index for index in range(batch))
        task.reset(identities, seed=args.seed + batch); task.set_modes(np.arange(batch) % 2); policy.reset()
        setup_seconds = time.perf_counter() - setup_started
        if torch.device(args.device).type == "cuda": torch.cuda.reset_peak_memory_stats(args.device)
        profile, commands, statuses = execute_workload(task, policy, sample_ids=identities, seed=args.seed + batch,
            device=args.device, progress=lambda row: atomic_json(output / "progress.json", {"batch": batch, **row}))
        profile.update(setup_seconds=setup_seconds, parameter_count=sum(value.numel() for value in policy.model.parameters()),
                       parameter_device=str(next(policy.model.parameters()).device), physics_device="cpu", history_len=checkpoint["cfg"]["history_len"],
                       action_horizon=checkpoint["cfg"]["pred_horizon"], task_contract=task.describe())
        profiles.append(profile)
        arrays[f"batch_{batch}.actions"], arrays[f"batch_{batch}.statuses"] = commands, statuses
        (output / f"batch_{batch}_episode_records.json").write_text(json.dumps([asdict(row) for row in task.records], default=_json, indent=2) + "\n")
        atomic_json(output / f"batch_{batch}_workload.json", json.loads(json.dumps(profile, default=_json)))
        del task, policy
    np.savez_compressed(output / "workload_controls.npz", **arrays)
    total = sum(row["total_physical_accounting_including_initial_reset"]["control_steps"] for row in profiles)
    if total != 55: raise RuntimeError("unexpected source workload charge")
    combined = {key: sum(row["total_physical_accounting_including_initial_reset"][key] for row in profiles)
                for key in profiles[0]["total_physical_accounting_including_initial_reset"]}
    summary = {"status": "real_source_pipeline_workloads_verified", "source_qualification": str(origin), "target_calls": 0,
        "source_only_nominal_Panda": True, "additional_source_control_steps": total,
        "additional_source_physics_steps": combined["physics_steps"], "additional_source_accounting": combined,
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), "profiles": profiles,
        "timing_scope": "portable_history + bounded FADA policy + actual CPU physics step; any partial reset is included in the following timed call",
        "cold_scope": "first actual control call per batch after source task/policy setup; both workloads share one process and CUDA context",
        "resource_scope": "per-step RSS and nvidia-smi samples; PyTorch per-workload allocator peaks; process peak RSS includes earlier workloads",
        "performance_scope": "short workload qualification; evaluator retains all incomplete rows, no full-task success or strong-baseline qualification",
        "controls_sha256": hashlib.sha256((output / "workload_controls.npz").read_bytes()).hexdigest()}
    atomic_json(output / "workloads.json", json.loads(json.dumps(summary, default=_json)))
    atomic_json(output / "attempt.json", {"status": "complete", "target_calls": 0, "additional_source_control_steps": total})
    print(json.dumps({"status": summary["status"], "source_controls": total, "target_calls": 0}))


if __name__ == "__main__":
    main()
