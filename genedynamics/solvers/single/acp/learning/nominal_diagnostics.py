"""Bounded nominal checks at the original supervised ACP reset states.

Teacher replay and action reconstruction do not establish that the learned
policy can execute the task.  This diagnostic reuses one fit and one held-out
demonstration reset without changing the controller or task. A separate,
explicit recipe permits one fixed-budget single-demonstration fitting check.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from genedynamics.core.artifacts import atomic_json
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.learning.datasets.transfer.target import PublicTargetTrajectory
from genedynamics.tasks.transfer.controllers.acp import ACPFourTaskActor
from genedynamics.tasks.transfer.demonstrators.acp_collection import (
    _task_from_config, evidence_bytes,
)
from genedynamics.tasks.transfer.episode import audit_episode_evidence, episode_evidence
from genedynamics.tasks.transfer.evaluation import evaluate_records


VERSION = "acp-nominal-learning-diagnostic-v1"
OVERFIT_VERSION = "acp-single-demonstration-fit-diagnostic-v1"


def _checked(root: Path, binding: dict) -> Path:
    path = root / binding["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
        raise ValueError(f"diagnostic input SHA differs: {path}")
    return path


def learning_summary(training: dict, demonstrations: dict) -> dict:
    validations = training["validations"]
    return {
        "completed_updates": training["completed_updates"],
        "validation_curve": [{"update": r["update"], "loss": r["loss"]} for r in validations],
        "last_interval_relative_loss_change": (
            (validations[-1]["loss"] - validations[-2]["loss"]) / validations[-2]["loss"]
        ),
        "independent_fit_episodes": demonstrations["splits"]["fit"]["episodes"],
        "independent_validation_episodes": demonstrations["splits"]["validation"]["episodes"],
        "fit_windows": demonstrations["splits"]["fit"]["windows"],
        "windows_are_independent_episodes": False,
        "convergence_proven": False,
        "interpretation": "Finite denoising loss and teacher success do not establish learned closed-loop capability or convergence.",
    }


def run_nominal_diagnostic(config: dict, root: Path) -> dict:
    if config.get("version") != VERSION or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU nominal diagnostic requires its explicit config and CUDA_VISIBLE_DEVICES=''")
    if config["budget"]["episodes"] != 2 or len(config["requests"]) != 2:
        raise ValueError("nominal diagnostic is bounded to two original demonstration resets")
    if {r["purpose"] for r in config["requests"]} != {"fit", "validation"}:
        raise ValueError("one fit and one held-out reset are required")
    torch.set_num_threads(1)
    checkpoint = _checked(root, config["checkpoint"])
    training_path = _checked(root, config["training_report"])
    demonstration_path = _checked(root, config["demonstration_report"])
    training = json.loads(training_path.read_text())
    demonstrations = json.loads(demonstration_path.read_text())
    teacher_config_path = _checked(root, config["teacher_collection_config"])
    teacher_config = json.loads(teacher_config_path.read_text())
    experiment = ExperimentConfig.from_yaml(root / teacher_config["teacher_config"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", config)
    started = time.monotonic()
    results = []
    for request in config["requests"]:
        if time.monotonic() - started >= config["budget"]["maximum_wall_seconds"]:
            raise TimeoutError("CPU nominal diagnostic wall-time budget exhausted")
        original = next(r for r in teacher_config["episodes"] if r["seed"] == request["seed"])
        if original["mode"] != request["mode"] or original["purpose"] != request["purpose"]:
            raise ValueError("requested diagnostic reset differs from original demonstration")
        receipt = next(r for r in demonstrations["splits"][request["purpose"]]["receipts"]
                       if r["seed"] == request["seed"])
        data = PublicTargetTrajectory.load(receipt["path"])
        if data.digest != receipt["trajectory_sha256"]:
            raise ValueError("demonstration arrays changed")
        directory = output / request["purpose"]
        directory.mkdir()
        task = _task_from_config(experiment, deepcopy(original), task_alias=config["task"])
        public = task.reset((original["sample_id"],), seed=original["seed"],
                            initial="jammed" if config["task"] in {"T3", "T4"} else "clear",
                            modes=[original["mode"]])
        actor = ACPFourTaskActor(task, payload, original["mode"], config["sampling_seed"], device="cpu")
        for _ in range(int(teacher_config["n_steps"])):
            if time.monotonic() - started >= config["budget"]["maximum_wall_seconds"]:
                raise TimeoutError("CPU nominal diagnostic wall-time budget exhausted")
            if bool(np.asarray(task.state.status)[0]):
                break
            action = np.asarray(actor.act(public), np.float32)
            intent = task.public_body_intent(public)
            public = (task.step(action) if intent is None else task.step_contact(action, body_twist=intent)).observation
        evaluation = evaluate_records(task.records, task.spec)
        evidence_path = directory / "evidence.json.gz"
        evidence_path.write_bytes(evidence_bytes(episode_evidence(task.records, task.spec, evaluation)))
        audit = audit_episode_evidence(evidence_path)
        row = {
            "purpose": original["purpose"], "mode": original["mode"],
            "reset_seed": original["seed"], "sample_id": original["sample_id"],
            "evaluation": evaluation["episodes"][0],
            "evidence_audit_passed": bool(audit["passed"]),
            "evidence": {"path": str(evidence_path.relative_to(root)),
                         "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest()},
            "network_calls": actor.irreversible_network_calls,
            "accounting": dict(task.accounting),
        }
        results.append(row)
        atomic_json(directory / "report.json", row)
    report = {
        "version": VERSION, "status": "complete", "task": config["task"],
        "scope": "original supervised fit/held-out resets; no new shifted target conditions; not paper performance",
        "learning": learning_summary(training, demonstrations), "results": results,
        "checkpoint": config["checkpoint"], "new_training_updates": 0,
        "gpu_used": False, "wall_seconds": time.monotonic() - started,
        "nominal_fit_success": bool(results[0]["evaluation"]["constrained_success"]),
        "interpretation": "Failures here concern this learned task adaptation before any new dynamics shift; they do not establish an original-method SOTA limitation.",
    }
    atomic_json(output / "report.json", report)
    return report


def overfit_plan(config: dict, root: Path) -> dict:
    if config.get("version") != OVERFIT_VERSION:
        raise ValueError("explicit single-demonstration diagnostic config required")
    execution = config["execution"]
    cpu = execution == {"device": "cpu", "cuda_visible_devices": ""}
    expected_uuid = execution.get("physical_uuid")
    gpu = (
        isinstance(expected_uuid, str)
        and re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", expected_uuid) is not None
        and execution.get("cuda_visible_devices") in {"0", "1"}
        and execution == {"device": "cuda:0", "cuda_visible_devices": execution["cuda_visible_devices"], "physical_uuid": expected_uuid}
    )
    if not (cpu or gpu):
        raise ValueError("single-demonstration diagnostic requires CPU or one explicitly bound physical GPU")
    expected_updates = [0, 32, 64, 128] if cpu else [0, 128, 256, 512]
    if config["task"] != "T1" or config["budget"]["max_updates"] != expected_updates[-1]:
        raise ValueError("single-demo diagnostic update budget changed")
    if config["monitor_updates"] != expected_updates:
        raise ValueError("single-demo monitor budget changed")
    for name in ("checkpoint", "training_report", "demonstration_report", "teacher_collection_config"):
        _checked(root, config[name])
    if (root / config["output"]).exists():
        raise FileExistsError("single-demo diagnostic output is immutable")
    return {**config, "status": "ready_not_started", "gpu_started": False,
            "new_demonstrations": 0, "nominal_closed_loop_episodes": 4,
            "scope": "fixed existing supervised fit episode only; no independent target performance claim"}


def run_overfit_diagnostic(config: dict, root: Path) -> dict:
    """One fixed diagnostic opportunity; no retries or automatic budget growth."""
    from .target_task_training import PublicACPTaskDataset
    from .task_training import build_policy, model_configuration, sampling_profile
    from .trainer import ACPTrainer

    plan = overfit_plan(config, root)
    device = torch.device(config["execution"]["device"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["execution"]["cuda_visible_devices"]:
        raise RuntimeError("diagnostic visible-device binding differs")
    if device.type == "cuda":
        if torch.cuda.device_count() != 1:
            raise RuntimeError("single-demonstration diagnostic must expose exactly one physical GPU")
        properties = torch.cuda.get_device_properties(0)
        expected_uuid = config["execution"]["physical_uuid"]
        if str(properties.uuid).removeprefix("GPU-").lower() != expected_uuid.removeprefix("GPU-").lower():
            raise RuntimeError("visible GPU UUID differs from configured physical GPU")
        torch.cuda.set_per_process_memory_fraction(config["budget"]["maximum_gpu_gib"] * 1024**3 / properties.total_memory, 0)
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"]); np.random.seed(config["seed"])
    demo_report = json.loads(_checked(root, config["demonstration_report"]).read_text())
    receipt = next(r for r in demo_report["splits"]["fit"]["receipts"] if r["seed"] == config["fit_seed"])
    data = PublicTargetTrajectory.load(receipt["path"])
    if data.digest != receipt["trajectory_sha256"]:
        raise ValueError("single demonstration changed")
    teacher_config = json.loads(_checked(root, config["teacher_collection_config"]).read_text())
    original = next(r for r in teacher_config["episodes"] if r["seed"] == config["fit_seed"])
    experiment = ExperimentConfig.from_yaml(root / teacher_config["teacher_config"])
    payload = torch.load(_checked(root, config["checkpoint"]), map_location="cpu", weights_only=True)
    dataset = PublicACPTaskDataset([data], purpose="fit", history_len=payload["history_len"],
        horizon=16, action_chart=payload["action_chart"], action_down_sample_steps=payload["action_down_sample_steps"])
    policy = build_policy(payload["shape_meta"], payload["history_len"], action_chart=payload["action_chart"])
    policy.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
    _, cfg = model_configuration(payload["history_len"], 16, config["seed"], shared_context=True,
                                  action_down_sample_steps=payload["action_down_sample_steps"])
    cfg.optimizer.lr = config["learning_rate"]
    cfg.training.lr_scheduler = "constant"
    cfg.training.lr_warmup_steps = 0
    maximum_updates = config["budget"]["max_updates"]
    trainer = ACPTrainer(policy, cfg, device=str(device), total_steps=maximum_updates, optimizer_mode="all_trainable")
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", plan)
    indices = np.unique(np.linspace(0, len(dataset)-1, 8, dtype=int))
    batch = dataset.batch(indices)
    fixed_batch = {name: {group: ({k: v.to(device) for k, v in value.items()} if isinstance(value, dict) else value.to(device))
                         for group, value in groups.items()} for name, groups in batch.items()}
    rng = np.random.default_rng(config["seed"])
    started, curve, losses = time.monotonic(), [], []

    def monitor(update):
        if time.monotonic() - started > config["budget"]["maximum_wall_seconds"]:
            raise TimeoutError("single-demo diagnostic wall budget exhausted")
        torch.manual_seed(config["seed"] + 50000)
        trainer.ema_model.eval()
        with torch.inference_mode():
            loss = float(trainer.ema_model.compute_loss(fixed_batch, None).detach().cpu())
        reconstruction = sampling_profile(trainer.ema_model, dataset, indices, seed_base=config["seed"] + 60000)
        current = {**payload, "state_dicts": {"ema_model": deepcopy(trainer.ema_model.state_dict())}}
        directory = output / f"update_{update}"
        directory.mkdir()
        torch.save(current, directory / "policy.pt")
        task = _task_from_config(experiment, deepcopy(original), task_alias=config["task"])
        public = task.reset((original["sample_id"],), seed=original["seed"], initial="clear", modes=[original["mode"]])
        actor = ACPFourTaskActor(task, current, original["mode"], config["sampling_seed"], device=str(device))
        for _ in range(int(teacher_config["n_steps"])):
            if time.monotonic() - started > config["budget"]["maximum_wall_seconds"]:
                raise TimeoutError("single-demo diagnostic wall budget exhausted")
            if bool(np.asarray(task.state.status)[0]): break
            public = task.step(np.asarray(actor.act(public), np.float32)).observation
        evaluated = evaluate_records(task.records, task.spec)
        evidence = directory / "evidence.json.gz"
        evidence.write_bytes(evidence_bytes(episode_evidence(task.records, task.spec, evaluated)))
        audit = audit_episode_evidence(evidence)
        row = {"update": update, "fixed_noise_denoising_loss": loss,
               "sampled_reconstruction": reconstruction["aggregate"],
               "nominal_fit_rollout": evaluated["episodes"][0],
               "evidence_audit_passed": bool(audit["passed"]),
               "wall_seconds": time.monotonic() - started}
        atomic_json(directory / "report.json", row)
        curve.append(row)
        atomic_json(output / "curve.json", curve)

    try:
        monitor(0)
        for update in range(1, maximum_updates + 1):
            if time.monotonic() - started > config["budget"]["maximum_wall_seconds"]:
                raise TimeoutError("single-demo diagnostic wall budget exhausted")
            indices_batch = rng.integers(0, len(dataset), size=config["budget"]["batch_size"])
            raw = dataset.batch(indices_batch)
            gpu_batch = {name: {group: ({k: v.to(device) for k, v in value.items()} if isinstance(value, dict) else value.to(device))
                               for group, value in groups.items()} for name, groups in raw.items()}
            losses.append(trainer.train_step(gpu_batch))
            if update in config["monitor_updates"]:
                monitor(update)
        report = {"version": OVERFIT_VERSION, "status": "complete", "task": config["task"],
            "curve": curve, "updates": len(losses), "losses": losses,
            "wall_seconds": time.monotonic() - started, "gpu_started": device.type == "cuda",
            "scope": plan["scope"], "optimizer": "warm EMA weights; new AdamW; constant LR; unchanged native objective",
            "interpretation": "Only nominal single-demonstration learning capacity; no SOTA or shifted-target comparison."}
    except BaseException as error:
        atomic_json(output / "report.json", {"status": "failed", "error": repr(error),
            "curve": curve, "updates": len(losses), "wall_seconds": time.monotonic()-started})
        raise
    atomic_json(output / "report.json", report)
    return report
