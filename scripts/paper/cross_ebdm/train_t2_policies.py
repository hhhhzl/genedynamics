#!/usr/bin/env python3
"""Train the eight T2 policy/ablation instances on the admitted GPU."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from genedynamics.solvers.single.cross_ebdm.learning.gated_transport import (
    PublicAnchoringGateSpec,
    public_condition_distance,
    response_disagreement,
    select_task_compatible_checkpoint,
)
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import (
    PublicTransportSpec,
    task_proxy,
)
from genedynamics.solvers.single.cross_ebdm.learning.skill_bank import PublicSkillBank
from genedynamics.solvers.single.cross_ebdm.learning.task_response import (
    PublicTaskIssuedResponseEnsemble,
)
from genedynamics.solvers.single.cross_ebdm.learning.task_transport import (
    PublicTaskAdditiveBandTransportTrainer,
    PublicTaskGatedTransportTrainer,
    PublicTaskResidualTransport,
    PublicTaskTransportTrainer,
)
from genedynamics.tasks.transfer.controllers.cross_ebdm.guide_corridor_v2 import (
    PublicCorridorGuideDenseActor,
)
from genedynamics.tasks.transfer.specs import task_spec


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": sha(path),
    }


def checked(value: dict[str, str], label: str) -> Path:
    path = ROOT / value["path"]
    if not path.is_file() or sha(path) != value["sha256"]:
        raise ValueError(f"{label} changed")
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def concatenate(rows: list[Any]) -> Any:
    if not rows:
        raise ValueError("empty training-state collection")
    if isinstance(rows[0], dict):
        return {key: concatenate([row[key] for row in rows]) for key in rows[0]}
    return np.concatenate(rows)


def tensorize(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: tensorize(child, device) for key, child in value.items()}
    tensor = torch.as_tensor(value, device=device)
    return tensor.float() if tensor.is_floating_point() else tensor


def select(value: Any, indices: np.ndarray) -> Any:
    if isinstance(value, dict):
        return {key: select(child, indices) for key, child in value.items()}
    return value[torch.as_tensor(indices, device=value.device)]


def policy_values(policy, batch):
    return policy(
        batch["target_features"], batch["source_features"],
        batch["source_commands"], batch["base_proto"], batch["phase"],
        batch["own_context"],
    )


def proxy_values(policy, response, batch):
    normalized = policy_values(policy, batch)
    predicted = response(batch["target_features"], normalized)
    return task_proxy(predicted, batch["current"], batch["task_reference"]).mean(0)


def validation_proxy(policy, response, validation) -> float:
    with torch.no_grad():
        values = [
            proxy_values(policy, response, select(
                validation, np.arange(start, min(start + 256, len(validation["phase"])))
            )).cpu()
            for start in range(0, len(validation["phase"]), 256)
        ]
    return float(torch.cat(values).mean())


def _legacy_transport(config: dict[str, Any]) -> bool:
    return config.get("version") == "t2-policy-instances-training-v1"


def _execution_scale(config: dict[str, Any]) -> float:
    return .25 if config.get("version") == "t2-policy-instances-training-v3" else 1.


def prepare_states(config: dict[str, Any], response_spec, bank: PublicSkillBank):
    frozen_path = checked(config["frozen_public_inputs"], "frozen public inputs")
    frozen = json.loads(frozen_path.read_text())
    dummy = PublicTaskResidualTransport(
        response_spec, PublicTransportSpec(residual_scale=.025, hidden=(8,)),
        legacy_full_request=_legacy_transport(config),
        t2_execution_scale=_execution_scale(config),
    )
    groups: dict[str, list[Any]] = {"fit": [], "validation": []}
    inputs = {}
    counts = {}
    for relative, metadata in frozen["training_validation_episodes"].items():
        if metadata.get("deduplicated_within_split"):
            continue
        path = ROOT / relative
        actual = sha(path)
        if frozen["sha256"].get(relative) != actual:
            raise ValueError(f"frozen public episode changed: {relative}")
        with np.load(path, allow_pickle=False) as archive:
            public = {
                key[7:]: archive[key].copy()
                for key in archive.files if key.startswith("public.")
            }
        actor = PublicCorridorGuideDenseActor(
            task_spec("T2", robot=metadata["body"]), dummy, bank,
            mode=int(metadata["mode"]), device="cpu",
        )
        rows = []
        for tick in range(len(public["tcp_position"])):
            batch, _ = actor.prepare({key: value[tick:tick + 1] for key, value in public.items()})
            if tick % int(config["training"]["state_stride"]):
                continue
            batch.pop("source_query_distance")
            rows.append(batch)
        groups[metadata["purpose"]].append(concatenate(rows))
        inputs[relative] = actual
        counts[relative] = len(rows)
    return concatenate(groups["fit"]), concatenate(groups["validation"]), {
        "frozen_inputs": pointer(frozen_path), "episode_sha256": inputs,
        "state_counts": counts, "source_bank_sha256": bank.digest,
    }


def calibrate_gate(task_only, response, validation, training: dict[str, Any]):
    distances, disagreements = [], []
    with torch.no_grad():
        for start in range(0, len(validation["phase"]), 256):
            batch = select(validation, np.arange(start, min(start + 256, len(validation["phase"]))))
            normalized = policy_values(task_only, batch)
            predicted = response(batch["target_features"], normalized)
            source = response(batch["source_features"], batch["source_commands"])
            distances.append(public_condition_distance(batch, response.spec).cpu().numpy())
            disagreements.append(response_disagreement(predicted, source).cpu().numpy())
    method = training["quantile_method"]
    quantile = float(training["condition_quantile"])
    return PublicAnchoringGateSpec(
        task_tolerance=float(training["task_tolerance"]),
        maximum_condition_distance=float(np.quantile(np.concatenate(distances), quantile, method=method)),
        maximum_response_disagreement=float(np.quantile(np.concatenate(disagreements), quantile, method=method)),
        band_penalty=float(training["band_penalty"]),
    )


def train_instance(name: str, row: dict[str, Any], config: dict[str, Any],
                   fit, validation, device: torch.device, output: Path):
    seed = int(row["seed"])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    response_path = checked(row["response_checkpoint"], f"{name} response checkpoint")
    response = PublicTaskIssuedResponseEnsemble.from_checkpoint(
        torch.load(response_path, map_location="cpu", weights_only=True), device
    ).eval()
    for parameter in response.parameters():
        parameter.requires_grad_(False)
    spec_values = dict(config["training"]["transport_spec"])
    spec_values["hidden"] = tuple(spec_values["hidden"])
    policy = PublicTaskResidualTransport(
        response.spec, PublicTransportSpec(**spec_values),
        legacy_full_request=_legacy_transport(config),
        t2_execution_scale=_execution_scale(config),
    ).to(device)
    trainer = PublicTaskTransportTrainer(policy, response, None)
    batch_size = int(config["training"]["batch_size"])
    warm_rng = np.random.default_rng(seed + 1)
    stream_rng = np.random.default_rng(seed + 2)
    warm_ids = [warm_rng.integers(len(fit["phase"]), size=batch_size) for _ in range(int(config["training"]["warmup_updates"]))]
    branch_ids = [stream_rng.integers(len(fit["phase"]), size=batch_size) for _ in range(int(config["training"]["branch_updates"]))]

    def run_updates(active, ids, kwargs, folder):
        curve = []
        for index, indices in enumerate(ids, 1):
            batch = select(fit, indices)
            metrics = active.step(batch, **kwargs(batch))
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"{name}/{folder.name} nonfinite objective")
            if index % int(config["training"]["log_interval"]) == 0:
                curve.append({"update": index, "metrics": metrics})
        save_torch(folder / "checkpoint.pt", active.policy.checkpoint())
        save_torch(folder / "continuation.pt", active.state())
        return curve

    warm_folder = output / "frozen_warm"; warm_folder.mkdir(parents=True)
    warm_curve = run_updates(trainer, warm_ids, lambda _: {"response_weight": 0.0}, warm_folder)
    warm_state = deepcopy(trainer.state())
    warm_proxy = validation_proxy(policy, response, validation)

    task_folder = output / "task_only"; task_folder.mkdir()
    task_policy = PublicTaskResidualTransport.from_checkpoint(warm_state["checkpoint"], device)
    task_trainer = PublicTaskTransportTrainer(task_policy, response, None)
    task_trainer.restore(deepcopy(warm_state))
    task_curve = run_updates(task_trainer, branch_ids, lambda _: {"response_weight": 0.0}, task_folder)
    task_proxy_value = validation_proxy(task_policy, response, validation)
    gate = calibrate_gate(task_policy, response, validation, config["training"])
    task_only_pointer = pointer(task_folder / "checkpoint.pt")
    reports = {
        "frozen_warm": {"curve": warm_curve, "validation_task_proxy": warm_proxy,
                        "selected_checkpoint": pointer(warm_folder / "checkpoint.pt")},
        "task_only": {"curve": task_curve, "validation_task_proxy": task_proxy_value,
                      "selected_checkpoint": task_only_pointer},
    }
    selected = {"frozen_warm": reports["frozen_warm"]["selected_checkpoint"],
                "task_only": task_only_pointer}
    definitions = {
        "matching_no_band": (PublicTaskTransportTrainer, 1.0, False),
        "band_only": (PublicTaskAdditiveBandTransportTrainer, 0.0, True),
        "ungated_full": (PublicTaskAdditiveBandTransportTrainer, 1.0, True),
        "gated_full": (PublicTaskGatedTransportTrainer, 1.0, True),
    }
    stream = hashlib.sha256(b"".join(np.asarray(ids, np.int64).tobytes() for ids in branch_ids)).hexdigest()
    for branch, (trainer_type, weight, uses_band) in definitions.items():
        folder = output / branch; folder.mkdir()
        candidate = PublicTaskResidualTransport.from_checkpoint(warm_state["checkpoint"], device)
        active = trainer_type(candidate, response, None); active.restore(deepcopy(warm_state))
        def arguments(batch, weight=weight, uses_band=uses_band):
            if not uses_band:
                return {"response_weight": weight}
            with torch.no_grad():
                reference = proxy_values(task_policy, response, batch)
            return {"response_weight": weight, "task_only_reference": reference, "gate_spec": gate}
        curve = run_updates(active, branch_ids, arguments, folder)
        value = validation_proxy(candidate, response, validation)
        choice = select_task_compatible_checkpoint(
            candidate_proxy=value, task_only_proxy=task_proxy_value,
            tolerance=float(config["training"]["selection_tolerance"]),
            candidate=pointer(folder / "checkpoint.pt"), task_only=task_only_pointer,
        )
        selected[branch] = choice["selected"]
        reports[branch] = {"curve": curve, "validation_task_proxy": value,
                           "selection": choice, "selected_checkpoint": choice["selected"]}
    return {
        "status": "complete", "instance": name, "seed": seed,
        "response_checkpoint": pointer(response_path), "gate_spec": gate.__dict__,
        "sample_stream_sha256": stream, "policies": selected, "branches": reports,
        "device": str(device), "cuda_device_name": torch.cuda.get_device_name(0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text())
    if config.get("version") not in {
        "t2-policy-instances-training-v1",
        "t2-policy-instances-training-v2",
        "t2-policy-instances-training-v3",
    }:
        raise ValueError("frozen T2 training v1/v2 config required")
    expected = config["device"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected["process_visible_devices"]:
        raise RuntimeError("training requires CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != expected["required_visible_device_count"]:
        raise RuntimeError("exactly one logical CUDA device is required")
    device = torch.device(expected["logical_device"])
    for label, value in config["admission"].items():
        document = json.loads(checked(value, label).read_text())
        if label == "behavioral_predeployment" and not document.get("gpu_training_allowed"):
            raise ValueError("behavioral predeployment did not admit GPU training")
    checked(config["actor_manifest"], "actor manifest")
    source_path = checked(config["source_manifest"], "source manifest")
    source = json.loads(source_path.read_text())
    first = next(iter(config["instances"].values()))
    first_response = PublicTaskIssuedResponseEnsemble.from_checkpoint(
        torch.load(checked(first["response_checkpoint"], "first response"), map_location="cpu", weights_only=True), "cpu"
    )
    entries = [dict(row, path=str((ROOT / row["path"]).resolve())) for row in source["entries"]]
    bank = PublicSkillBank(entries, first_response.spec, task="T2")
    raw_fit, raw_validation, data_provenance = prepare_states(config, first_response.spec, bank)
    fit, validation = tensorize(raw_fit, device), tensorize(raw_validation, device)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", config)
    begun = time.monotonic(); instances = {}
    report = {"status": "running", "passed": False, "instances": instances}
    try:
        for name, row in sorted(config["instances"].items()):
            folder = output / name; folder.mkdir()
            instances[name] = train_instance(name, row, config, fit, validation, device, folder)
            write_json(output / "progress.json", instances)
            print(json.dumps({"instance": name, "status": "complete"}), flush=True)
        policy_names = set(config["ablation_objectives"])
        passed = len(instances) == 8 and all(set(row["policies"]) == policy_names for row in instances.values())
        report.update(status="complete" if passed else "failed", passed=passed,
                      data=data_provenance, device={**expected, "logical_name": torch.cuda.get_device_name(0)},
                      training_updates=8 * (int(config["training"]["warmup_updates"]) + 5 * int(config["training"]["branch_updates"])),
                      new_physics_calls=0, confirmation_states_opened=0,
                      old_invalid_outcomes_used=False, wall_seconds=time.monotonic() - begun,
                      artifacts={"config": pointer(config_path), "runner": pointer(Path(__file__).resolve())})
        if not passed:
            raise AssertionError("eight-instance six-policy inventory incomplete")
    except BaseException as error:
        report.update(status="failed", error=repr(error), wall_seconds=time.monotonic() - begun)
        raise
    finally:
        write_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "output": str(output / 'report.json'),
                      "sha256": sha(output / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
