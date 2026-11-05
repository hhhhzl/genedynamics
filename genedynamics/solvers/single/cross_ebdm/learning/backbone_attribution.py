"""Matched T2 MLP/diffusion × task-only/selective training on public logs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .diffusion_transport import PublicTaskDiffusionTransport
from .gated_transport import (PublicAnchoringGateSpec,
                              public_condition_distance, response_disagreement,
                              select_task_compatible_checkpoint)
from .residual_transport import PublicTransportSpec, task_proxy
from .skill_bank import PublicSkillBank
from .task_response import PublicTaskIssuedResponseEnsemble
from .task_transport import (PublicTaskResidualTransport, PublicTaskTransportTrainer,
                             PublicTaskGatedTransportTrainer)
from genedynamics.tasks.transfer.controllers.cross_ebdm.guide_corridor_v2 import PublicCorridorGuideDenseActor
from genedynamics.tasks.transfer.specs import task_spec


VERSION = "T2-backbone-attribution-public-v1"


def _checked(root, pointer, label):
    path = root / pointer["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != pointer["sha256"]:
        raise ValueError(f"{label} changed")
    return path


def _pointer(root, path):
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _atomic_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("."+path.name)
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def _atomic_torch(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("."+path.name)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _concatenate(rows):
    if not rows:
        raise ValueError("empty public training-state group")
    if isinstance(rows[0], dict):
        return {key: _concatenate([row[key] for row in rows]) for key in rows[0]}
    return np.concatenate(rows)


def _tensorize(value, device):
    if isinstance(value, dict):
        return {name: _tensorize(child, device) for name, child in value.items()}
    tensor = torch.as_tensor(value, device=device)
    return tensor.float() if tensor.is_floating_point() else tensor


def _select(value, indices):
    if isinstance(value, dict):
        return {name: _select(child, indices) for name, child in value.items()}
    return value[torch.as_tensor(indices, device=value.device)]


def _policy_values(policy, batch):
    return policy(batch["target_features"], batch["source_features"],
                  batch["source_commands"], batch["base_proto"],
                  batch["phase"], batch["own_context"])


def _proxy_values(policy, response, batch):
    predicted = response(batch["target_features"], _policy_values(policy, batch))
    return task_proxy(predicted, batch["current"], batch["task_reference"]).mean(0)


def _validation_proxy(policy, response, validation):
    with torch.no_grad():
        values = [_proxy_values(policy, response, _select(validation,
                  np.arange(start, min(start+256, len(validation["phase"]))))).cpu()
                  for start in range(0, len(validation["phase"]), 256)]
    return float(torch.cat(values).mean())


def _calibrate_gate(task_only, response, validation, training):
    distance, disagreement = [], []
    with torch.no_grad():
        for start in range(0, len(validation["phase"]), 256):
            batch = _select(validation, np.arange(start, min(start+256, len(validation["phase"]))))
            candidate = _policy_values(task_only, batch)
            predicted = response(batch["target_features"], candidate)
            source = response(batch["source_features"], batch["source_commands"])
            distance.append(public_condition_distance(batch, response.spec).cpu().numpy())
            disagreement.append(response_disagreement(predicted, source).cpu().numpy())
    quantile = training["condition_quantile"]
    method = training["quantile_method"]
    return PublicAnchoringGateSpec(task_tolerance=training["task_tolerance"],
        maximum_condition_distance=float(np.quantile(np.concatenate(distance), quantile, method=method)),
        maximum_response_disagreement=float(np.quantile(np.concatenate(disagreement), quantile, method=method)),
        band_penalty=training["band_penalty"])


def training_plan(contract, root):
    if (contract.get("version") != VERSION or contract.get("status") != "frozen_before_new_training"
            or contract.get("four_cells") != ["mlp/task_only", "mlp/selective",
                                               "diffusion/task_only", "diffusion/selective"]):
        raise ValueError("frozen four-cell T2 attribution contract required")
    recipe_path = _checked(root, contract["source_recipe"], "source T2 recipe")
    _checked(root, contract["source_report"], "source T2 result")
    _checked(root, contract["method_identity_audit"], "actual method identity audit")
    recipe = json.loads(recipe_path.read_text())
    training = recipe["training"]
    shared = contract["shared_training"]
    for name, actual in (("state_stride", training["state_stride"]),
                         ("warmup_updates_per_backbone", training["warmup_updates"]),
                         ("branch_updates_per_cell", training["branch_updates"]),
                         ("batch_size", training["batch_size"]),
                         ("hidden", training["transport_spec"]["hidden"]),
                         ("residual_scale", training["transport_spec"]["residual_scale"])):
        if shared[name] != actual:
            raise ValueError(f"attribution training {name} differs from frozen source")
    if (len(recipe["instances"]) != shared["instances"]
            or contract["diffusion"]["steps"] not in (2, 4, 8)
            or contract["shared_execution"]["post_transport_execution_scale"] != .25
            or contract["shared_execution"]["runtime_selector"] != "none for all four cells"):
        raise ValueError("four-cell instance or execution contract changed")
    device = contract["device"]
    if (device["logical_device"] != "cuda:0" or device["process_visible_devices"] != "1"
            or device["physical_uuid"] != recipe["device"]["physical_uuid"]
            or not 0 < device["maximum_gpu_gib"] <= 8
            or not 1 <= device["maximum_wall_seconds"] <= 43200):
        raise ValueError("bounded physical GPU1 attribution recipe required")
    _checked(root, recipe["frozen_public_inputs"], "frozen public inputs")
    _checked(root, recipe["source_manifest"], "source bank manifest")
    for name, row in recipe["instances"].items():
        _checked(root, row["response_checkpoint"], f"{name} response model")
    if (root / contract["output"]).exists():
        raise FileExistsError("attribution output is immutable")
    return {"status": "ready_not_started", "source_recipe": contract["source_recipe"],
            "instances": shared["instances"], "four_cells": contract["four_cells"],
            "updates": shared["instances"] * (2*shared["warmup_updates_per_backbone"]
                                             + 4*shared["branch_updates_per_cell"]),
            "output": contract["output"], "physical_gpu": device["physical_uuid"]}


def _public_states(root, recipe, bank, response_spec):
    frozen = json.loads(_checked(root, recipe["frozen_public_inputs"], "frozen public inputs").read_text())
    dummy = PublicTaskResidualTransport(response_spec, PublicTransportSpec(residual_scale=.025, hidden=(8,)),
                                        t2_execution_scale=.25)
    groups = {"fit": [], "validation": []}
    episode_sha = {}
    for relative, metadata in frozen["training_validation_episodes"].items():
        if metadata.get("deduplicated_within_split"):
            continue
        path = root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if frozen["sha256"].get(relative) != digest:
            raise ValueError("frozen public episode changed")
        with np.load(path, allow_pickle=False) as archive:
            public = {name[7:]: archive[name].copy() for name in archive.files if name.startswith("public.")}
        actor = PublicCorridorGuideDenseActor(task_spec("T2", robot=metadata["body"]), dummy, bank,
                                             mode=int(metadata["mode"]), device="cpu")
        rows = []
        for tick in range(len(public["tcp_position"])):
            batch, _ = actor.prepare({key: value[tick:tick+1] for key, value in public.items()})
            if tick % recipe["training"]["state_stride"]:
                continue
            batch.pop("source_query_distance")
            rows.append(batch)
        groups[metadata["purpose"]].append(_concatenate(rows))
        episode_sha[relative] = digest
    return _concatenate(groups["fit"]), _concatenate(groups["validation"]), episode_sha


def _make_policy(backbone, response_spec, transport_spec, seed, contract):
    if backbone == "mlp":
        return PublicTaskResidualTransport(response_spec, transport_spec, t2_execution_scale=.25)
    return PublicTaskDiffusionTransport(response_spec, transport_spec,
        denoising_steps=contract["diffusion"]["steps"],
        latent_seed=seed+contract["diffusion"]["latent_seed_offset"], t2_execution_scale=.25)


def _train_instance(root, contract, recipe, fit, validation, name, row, device, folder, started):
    seed = int(row["seed"])
    response_path = _checked(root, row["response_checkpoint"], f"{name} response")
    response = PublicTaskIssuedResponseEnsemble.from_checkpoint(
        torch.load(response_path, map_location="cpu", weights_only=True), str(device)).eval()
    for parameter in response.parameters():
        parameter.requires_grad_(False)
    training = recipe["training"]
    values = dict(training["transport_spec"], hidden=tuple(training["transport_spec"]["hidden"]))
    spec = PublicTransportSpec(**values)
    if response.spec.feature_dim != fit["target_features"].shape[-1]:
        raise ValueError("instance response chart differs from shared public states")
    stream = np.random.default_rng(seed+2)
    branch_ids = [stream.integers(len(fit["phase"]), size=training["batch_size"])
                  for _ in range(training["branch_updates"])]
    stream_sha = hashlib.sha256(b"".join(np.asarray(ids, np.int64).tobytes() for ids in branch_ids)).hexdigest()
    result = {"instance": name, "seed": seed, "response_checkpoint": row["response_checkpoint"],
              "branch_stream_sha256": stream_sha, "backbones": {}}
    for backbone in ("mlp", "diffusion"):
        if time.monotonic()-started > contract["device"]["maximum_wall_seconds"]:
            raise TimeoutError("four-cell attribution wall budget exhausted")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        policy = _make_policy(backbone, response.spec, spec, seed, contract).to(device)
        trainer = PublicTaskTransportTrainer(policy, response, None)
        warm_ids = np.random.default_rng(seed+1)
        backbone_folder = folder / backbone
        backbone_folder.mkdir(parents=True)
        for _ in range(training["warmup_updates"]):
            ids = warm_ids.integers(len(fit["phase"]), size=training["batch_size"])
            trainer.step(_select(fit, ids), response_weight=0.)
        warm_state = deepcopy(trainer.state())
        _atomic_torch(backbone_folder / "warm.pt", warm_state)
        warm_parameters = hashlib.sha256(b"".join(value.numpy().tobytes()
            for value in warm_state["checkpoint"]["state_dict"].values())).hexdigest()
        branches = {}
        for objective in ("task_only", "selective"):
            candidate = type(policy).from_checkpoint(warm_state["checkpoint"], str(device))
            active = (PublicTaskTransportTrainer(candidate, response, None) if objective == "task_only"
                      else PublicTaskGatedTransportTrainer(candidate, response, None))
            active.restore(deepcopy(warm_state))
            gate = None if objective == "task_only" else _calibrate_gate(task_only, response, validation, training)
            curve = []
            for update, ids in enumerate(branch_ids, 1):
                if time.monotonic()-started > contract["device"]["maximum_wall_seconds"]:
                    raise TimeoutError("four-cell attribution wall budget exhausted")
                batch = _select(fit, ids)
                if objective == "task_only":
                    metrics = active.step(batch, response_weight=0.)
                else:
                    with torch.no_grad():
                        reference = _proxy_values(task_only, response, batch)
                    metrics = active.step(batch, task_only_reference=reference, gate_spec=gate)
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("nonfinite four-cell transport objective")
                if update % training["log_interval"] == 0:
                    curve.append({"update": update, "metrics": metrics})
            branch_folder = backbone_folder / objective
            branch_folder.mkdir()
            _atomic_torch(branch_folder / "checkpoint.pt", candidate.checkpoint())
            validation_value = _validation_proxy(candidate, response, validation)
            branch = {"validation_task_proxy": validation_value,
                      "checkpoint": _pointer(root, branch_folder / "checkpoint.pt"),
                      "curve": curve, "warm_parameters_sha256": warm_parameters,
                      "sample_stream_sha256": stream_sha,
                      "gate_spec": None if gate is None else asdict(gate)}
            if objective == "task_only":
                task_only = candidate.eval()
                task_only_value = validation_value
                task_only_pointer = branch["checkpoint"]
            else:
                branch["selection"] = select_task_compatible_checkpoint(
                    candidate_proxy=validation_value, task_only_proxy=task_only_value,
                    tolerance=training["selection_tolerance"],
                    candidate=branch["checkpoint"], task_only=task_only_pointer)
            branches[objective] = branch
            _atomic_json(branch_folder / "report.json", branch)
        result["backbones"][backbone] = {"warm_parameters_sha256": warm_parameters,
                                           "branches": branches}
        _atomic_json(folder / "progress.json", result)
        del policy, trainer, warm_state
        torch.cuda.empty_cache()
    return result


def run_training(contract, root, *, progress=None):
    plan = training_plan(contract, root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("four-cell training requires only physical GPU1 visible")
    properties = torch.cuda.get_device_properties(0)
    if str(properties.uuid).removeprefix("GPU-").lower() != contract["device"]["physical_uuid"].removeprefix("GPU-").lower():
        raise RuntimeError("visible GPU UUID differs from frozen contract")
    torch.cuda.set_per_process_memory_fraction(
        contract["device"]["maximum_gpu_gib"] * 1024**3 / properties.total_memory, 0)
    device = torch.device("cuda:0")
    recipe = json.loads(_checked(root, contract["source_recipe"], "source T2 recipe").read_text())
    source = json.loads(_checked(root, recipe["source_manifest"], "source bank manifest").read_text())
    first = next(iter(recipe["instances"].values()))
    first_response = PublicTaskIssuedResponseEnsemble.from_checkpoint(
        torch.load(_checked(root, first["response_checkpoint"], "first response"), map_location="cpu", weights_only=True))
    entries = [dict(row, path=str((root / row["path"]).resolve())) for row in source["entries"]]
    bank = PublicSkillBank(entries, first_response.spec, task="T2")
    raw_fit, raw_validation, episode_sha = _public_states(root, recipe, bank, first_response.spec)
    fit, validation = _tensorize(raw_fit, device), _tensorize(raw_validation, device)
    output = root / contract["output"]
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "plan.json", contract)
    started = time.monotonic()
    results = {}
    try:
        for name, row in sorted(recipe["instances"].items()):
            folder = output / name
            folder.mkdir()
            results[name] = _train_instance(root, contract, recipe, fit, validation,
                                            name, row, device, folder, started)
            _atomic_json(output / "progress.json", results)
            if progress is not None:
                progress({"instance": name, "status": "complete"})
        report = {"version": VERSION, "status": "complete", "instances": results,
                  "four_cells": contract["four_cells"], "training_updates": plan["updates"],
                  "fit_states": len(fit["phase"]), "validation_states": len(validation["phase"]),
                  "source_bank_sha256": bank.digest, "public_episode_sha256": episode_sha,
                  "gpu_physical_uuid": contract["device"]["physical_uuid"],
                  "new_physics_calls": 0, "confirmation_states_opened": 0,
                  "formal_performance_evidence": False,
                  "wall_seconds": time.monotonic()-started}
    except BaseException as error:
        report = {"version": VERSION, "status": "failed", "error": repr(error),
                  "instances": results, "wall_seconds": time.monotonic()-started}
        _atomic_json(output / "report.json", report)
        raise
    _atomic_json(output / "report.json", report)
    return report
