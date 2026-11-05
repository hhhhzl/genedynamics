"""Task-owned, paired T2 development evaluation of trained transport cells."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from genedynamics.core.artifacts import atomic_json
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.bindings import BINDING_VERSION
from .method_evaluation import GPU1_UUID, _checked, _episode, _gpu1, _request, _root, _sha256


VERSION = "transfer-backbone-attribution-development-v1"
PREFLIGHT_VERSION = "transfer-backbone-attribution-preflight-v1"


def run_numerical_preflight(config: dict) -> dict:
    """Check every checkpoint on real public states before requests are frozen."""
    if config.get("version") != PREFLIGHT_VERSION:
        raise ValueError("explicit four-cell numerical preflight recipe required")
    contract_path = _checked(config["attribution_contract"], "attribution contract")
    training_path = _checked(config["training_report"], "four-cell training report")
    contract = json.loads(contract_path.read_text())
    training = json.loads(training_path.read_text())
    if (training.get("status") != "complete" or training.get("four_cells") != contract.get("four_cells")
            or len(training.get("instances", {})) != 8):
        raise ValueError("complete eight-instance four-cell training required")
    if config.get("execution") != {"device": "cuda", "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID}:
        raise ValueError("preflight requires the bound physical GPU1")
    output = Path(config["output"])
    if output.is_absolute() or not str(output).startswith("results/arm/guided_contact_alignment/validation/"):
        raise ValueError("preflight output must stay task-local")
    if (_root() / output).exists():
        raise FileExistsError("numerical preflight output is immutable")
    _gpu1()
    import torch
    from types import SimpleNamespace
    from genedynamics.solvers.single.cross_ebdm.learning.backbone_attribution import _public_states, _select, _tensorize
    from genedynamics.solvers.single.cross_ebdm.learning.skill_bank import PublicSkillBank
    from genedynamics.solvers.single.cross_ebdm.learning.task_response import PublicTaskIssuedResponseEnsemble
    from genedynamics.tasks.transfer.controllers.bindings import _cross_components
    from genedynamics.tasks.transfer.controllers.cross_ebdm.guide_corridor_v2 import PublicCorridorGuideDenseActor
    from genedynamics.tasks.transfer.specs import task_spec

    recipe = json.loads(_checked(contract["source_recipe"], "source T2 recipe").read_text())
    source = json.loads(_checked(recipe["source_manifest"], "T2 source bank").read_text())
    first = next(iter(recipe["instances"].values()))
    first_response = PublicTaskIssuedResponseEnsemble.from_checkpoint(
        torch.load(_checked(first["response_checkpoint"], "first response"), map_location="cpu", weights_only=True))
    entries = [dict(row, path=str((_root() / row["path"]).resolve())) for row in source["entries"]]
    bank = PublicSkillBank(entries, first_response.spec, task="T2")
    if bank.digest != training["source_bank_sha256"]:
        raise ValueError("preflight source bank differs from trained source bank")
    _, raw_validation, _ = _public_states(_root(), recipe, bank, first_response.spec)
    batch = _select(_tensorize(raw_validation, torch.device("cuda:0")), np.arange(1))
    task = SimpleNamespace(spec=task_spec("T2", robot="xarm7"))
    destination = _root() / output
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / "plan.json", config)
    rows = []
    started = time.monotonic()
    for instance, instance_row in sorted(training["instances"].items()):
        response = recipe["instances"][instance]["response_checkpoint"]
        for cell in contract["four_cells"]:
            backbone, objective = cell.split("/")
            branch = instance_row["backbones"][backbone]["branches"][objective]
            pointer = branch["checkpoint"] if objective == "task_only" else branch["selection"]["selected"]
            _checked(pointer, f"{instance}/{cell} policy")
            binding = {"policy_checkpoint": pointer, "response_checkpoint": response,
                       "source_manifest": recipe["source_manifest"],
                       "source_bank_sha256": training["source_bank_sha256"]}
            policy, _, current_bank = _cross_components(task, binding, "cuda:0")
            PublicCorridorGuideDenseActor(task.spec, policy, current_bank, mode=0, device="cuda:0")
            PublicCorridorGuideDenseActor(task.spec, policy, current_bank, mode=1, device="cuda:0")
            with torch.inference_mode():
                for _ in range(5):
                    value = policy(batch["target_features"], batch["source_features"],
                                   batch["source_commands"], batch["base_proto"],
                                   batch["phase"], batch["own_context"])
                torch.cuda.synchronize()
                elapsed = []
                for _ in range(50):
                    tick = time.perf_counter()
                    value = policy(batch["target_features"], batch["source_features"],
                                   batch["source_commands"], batch["base_proto"],
                                   batch["phase"], batch["own_context"])
                    torch.cuda.synchronize()
                    elapsed.append(time.perf_counter() - tick)
            finite = bool(torch.isfinite(value).all()) and value.shape == batch["base_proto"].shape
            rows.append({"instance": instance, "cell": cell, "checkpoint": pointer,
                         "constructed_both_modes": True, "finite": finite,
                         "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
                         "latency_seconds": {f"p{quantile}": float(np.percentile(elapsed, quantile))
                                             for quantile in (50, 95, 99)}})
            atomic_json(destination / "progress.json", {"completed": len(rows), "requested": 32,
                                                        "latest": rows[-1]})
            del policy, current_bank
            torch.cuda.empty_cache()
    report = {"version": PREFLIGHT_VERSION, "status": "passed" if len(rows) == 32 and all(row["finite"] for row in rows) else "failed",
              "training_report_sha256": config["training_report"]["sha256"],
              "attribution_contract_sha256": config["attribution_contract"]["sha256"],
              "all_four_cells_constructed": len(rows) == 32,
              "all_finite": all(row["finite"] for row in rows),
              "latency_measured": len(rows) == 32 and all(row["latency_seconds"]["p99"] > 0 for row in rows),
              "validation_states_available": len(raw_validation["phase"]),
              "rows": rows, "physics_calls": 0, "confirmation_states_opened": 0,
              "wall_seconds": time.monotonic() - started}
    atomic_json(destination / "report.json", report)
    return report


def evaluation_plan(config: dict) -> dict:
    if config.get("version") != VERSION or config.get("task") != "T2":
        raise ValueError("T2 four-cell development evaluation recipe required")
    contract_path = _checked(config["attribution_contract"], "attribution contract")
    training_path = _checked(config["training_report"], "four-cell training report")
    contract = json.loads(contract_path.read_text())
    training = json.loads(training_path.read_text())
    if (contract.get("status") != "frozen_before_new_training"
            or training.get("status") != "complete"
            or training.get("four_cells") != contract.get("four_cells")
            or len(training.get("instances", {})) != 8
            or training.get("confirmation_states_opened") != 0):
        raise ValueError("complete frozen four-cell training required")
    preflight_path = _checked(config["numerical_preflight"], "four-cell numerical preflight")
    preflight = json.loads(preflight_path.read_text())
    if (preflight.get("status") != "passed"
            or preflight.get("training_report_sha256") != config["training_report"]["sha256"]
            or preflight.get("attribution_contract_sha256") != config["attribution_contract"]["sha256"]
            or preflight.get("all_four_cells_constructed") is not True
            or preflight.get("all_finite") is not True
            or preflight.get("latency_measured") is not True
            or len(preflight.get("rows", [])) != 32
            or preflight.get("physics_calls") != 0
            or preflight.get("confirmation_states_opened") != 0):
        raise ValueError("four-cell construction, numerical and latency preflight required")
    recipe_path = _checked(contract["source_recipe"], "T2 training source recipe")
    recipe = json.loads(recipe_path.read_text())
    source_manifest = recipe["source_manifest"]
    _checked(source_manifest, "T2 source bank")
    task_config = _checked(config["task_config"], "T2 task config")
    experiment = ExperimentConfig.from_yaml(task_config)
    if experiment.method != "cross_ebdm" or experiment.env_name != "guided_contact_alignment":
        raise ValueError("four-cell evaluation requires the shared T2 task and method contract")
    requests = [_request(row, task="T2", index=index) for index, row in enumerate(config["requests"])]
    if (len(requests) != 8 or any(row["robot"] != "xarm7" for row in requests)
            or sorted(row["requested_mode"] for row in requests) != [0] * 4 + [1] * 4
            or len({(row["reset_seed"], row["physical_initial_slot"], row["requested_mode"])
                    for row in requests}) != 8):
        raise ValueError("eight fresh xArm7 T2 requests, two modes by four blocks, required")
    _request_strata(requests)
    execution = {"device": "cuda", "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID}
    if config.get("execution") != execution:
        raise ValueError("four-cell physical evaluation requires physical GPU1")
    budget = config.get("budget", {})
    if (set(budget) != {"maximum_wall_seconds", "maximum_gpu_gib"}
            or not 1 <= budget["maximum_wall_seconds"] <= 14400
            or not 0 < budget["maximum_gpu_gib"] <= 8):
        raise ValueError("bounded four-cell physical evaluation budget required")
    if isinstance(config.get("bootstrap_seed"), bool) or not isinstance(config.get("bootstrap_seed"), int) or config["bootstrap_seed"] < 0:
        raise ValueError("paired instance/request-block bootstrap seed required")
    output = Path(config["output"])
    if output.is_absolute() or not str(output).startswith("results/arm/guided_contact_alignment/validation/"):
        raise ValueError("T2 development output must stay task-local")
    if (_root() / output).exists():
        raise FileExistsError("four-cell development output is immutable")
    jobs = []
    for instance, row in sorted(training["instances"].items()):
        response = recipe["instances"][instance]["response_checkpoint"]
        _checked(response, f"{instance} response model")
        for cell in contract["four_cells"]:
            backbone, objective = cell.split("/")
            branch = row["backbones"][backbone]["branches"][objective]
            policy = branch["checkpoint"] if objective == "task_only" else branch["selection"]["selected"]
            _checked(policy, f"{instance}/{cell} policy")
            task_only = row["backbones"][backbone]["branches"]["task_only"]["checkpoint"]
            binding = {"version": BINDING_VERSION, "task": "T2", "family": "main",
                       "binding_key": "cross_ebdm", "method_identity": cell,
                       "actor": "cross_ebdm", "status": "instances_ready",
                       "instance_id": instance, "policy_checkpoint": deepcopy(policy),
                       "response_checkpoint": deepcopy(response),
                       "source_manifest": deepcopy(source_manifest),
                       "source_bank_sha256": training["source_bank_sha256"],
                       "maximum_query_distance": contract["shared_execution"]["maximum_query_distance"]}
            jobs.append({"instance": instance, "cell": cell, "actor_binding": binding,
                         "selected_alias_of_task_only": objective == "selective" and policy == task_only})
    return {"version": VERSION, "status": "planned_unexecuted", "task": "T2",
            "method": "cross_ebdm", "scope": "independent development four-cell attribution; no main-table claim",
            "task_config": str(task_config.relative_to(_root())),
            "task_config_sha256": _sha256(task_config),
            "attribution_contract": deepcopy(config["attribution_contract"]),
            "training_report": deepcopy(config["training_report"]),
            "numerical_preflight": deepcopy(config["numerical_preflight"]),
            "execution": execution, "requests": requests, "jobs": jobs,
            "budget": deepcopy(budget),
            "bootstrap_seed": config["bootstrap_seed"], "bootstrap_resamples": 5000,
            "output": str(output), "all_failures_retained": True,
            "replacement_requests_allowed": False, "confirmation_states_opened": 0}


def _request_strata(requests):
    """Keep the two reset slots in each frozen mechanism-by-mode cell together."""
    groups = {}
    for index, row in enumerate(requests):
        key = (json.dumps(row["mechanism_parameters"], sort_keys=True),
               int(row["requested_mode"]),
               json.dumps(row["body_support_parameters"], sort_keys=True))
        groups.setdefault(key, []).append(index)
    if (len(groups) != 4 or any(len(indices) != 2 for indices in groups.values())
            or Counter(key[1] for key in groups) != {0: 2, 1: 2}
            or len({key[0] for key in groups}) != 2
            or len({key[2] for key in groups}) != 1):
        raise ValueError("two mechanisms by two modes, two reset slots each, required")
    return tuple(tuple(indices) for indices in groups.values())


def _paired_contrasts(plan, results):
    """Paired descriptive intervals over training instances and request blocks."""
    cells = list(dict.fromkeys(job["cell"] for job in plan["jobs"]))
    instances = sorted({job["instance"] for job in plan["jobs"]})
    requests = [row["request_id"] for row in plan["requests"]]
    lookup = {(row["instance"], row["request_block_id"], row["cell"]): row for row in results}
    success = np.asarray([[[float(lookup[instance, request, cell]["passed"]) for cell in cells]
                           for request in requests] for instance in instances])
    time_score = np.asarray([[[float(lookup[instance, request, cell].get("capped_normalized_completion_time", 1.))
                              for cell in cells] for request in requests] for instance in instances])
    violation = np.asarray([[[float(lookup[instance, request, cell]["constraint_violation"])
                             for cell in cells] for request in requests] for instance in instances])
    rng = np.random.default_rng(plan["bootstrap_seed"])
    draws = plan["bootstrap_resamples"]
    instance_ids = rng.integers(len(instances), size=(draws, len(instances)))
    # Resample the same physical slots across every cell while fixing both
    # mechanism and requested-mode composition in every draw.
    request_ids = np.concatenate([
        rng.choice(indices, size=(draws, len(indices)), replace=True)
        for indices in _request_strata(plan["requests"])
    ], axis=1)
    names = {cell: index for index, cell in enumerate(cells)}
    weights = {
        "mlp_selective_minus_task_only": {"mlp/selective": 1., "mlp/task_only": -1.},
        "diffusion_selective_minus_task_only": {"diffusion/selective": 1., "diffusion/task_only": -1.},
        "diffusion_minus_mlp_task_only": {"diffusion/task_only": 1., "mlp/task_only": -1.},
        "diffusion_minus_mlp_selective": {"diffusion/selective": 1., "mlp/selective": -1.},
        "backbone_by_selective_interaction": {"diffusion/selective": 1., "diffusion/task_only": -1.,
                                               "mlp/selective": -1., "mlp/task_only": 1.},
    }
    paired = {}
    for metric, values in (("constrained_success", success),
                           ("capped_normalized_completion_time", time_score),
                           ("constraint_violation", violation)):
        sampled = values[instance_ids[:, :, None], request_ids[:, None, :]].mean(axis=(1, 2))
        means = values.mean(axis=(0, 1))
        paired[metric] = {}
        for label, row in weights.items():
            coefficient = np.zeros(len(cells))
            for cell, weight in row.items():
                coefficient[names[cell]] = weight
            values_drawn = sampled @ coefficient
            paired[metric][label] = {"point": float(means @ coefficient),
                                     "percentile_95_interval": [float(x) for x in np.quantile(values_drawn, [.025, .975])]}
    return {"unit": "paired training instance by request block; two slots per mechanism-by-mode stratum retained",
            "bootstrap_seed": plan["bootstrap_seed"], "resamples": draws,
            "intervals_are_descriptive_not_familywise_tests": True, "metrics": paired}


def run_evaluation(config: dict) -> dict:
    plan = evaluation_plan(config)
    _gpu1()
    import torch
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(
        plan["budget"]["maximum_gpu_gib"] * 1024**3 / properties.total_memory, 0)
    output = _root() / plan["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", plan)
    results = []
    started = time.monotonic()
    for job in plan["jobs"]:
        episode_plan = {"task": "T2", "method": "cross_ebdm", "task_config": plan["task_config"],
                        "method_identity": job["cell"], "actor_binding": job["actor_binding"]}
        for request in plan["requests"]:
            if time.monotonic() - started > plan["budget"]["maximum_wall_seconds"]:
                atomic_json(output / "report.json", {"version": VERSION, "status": "failed_budget",
                                                    "completed": len(results), "requested": 256,
                                                    "results": results, "wall_seconds": time.monotonic() - started})
                raise TimeoutError("four-cell development wall budget exhausted; partial results retained")
            owned = deepcopy(request)
            owned["request_id"] = (f"T2/attribution/{job['instance']}/{job['cell']}/"
                                   f"{request['request_id']}")
            row = _episode(episode_plan, owned, output)
            row.update(instance=job["instance"], cell=job["cell"],
                       request_block_id=request["request_id"],
                       selected_alias_of_task_only=job["selected_alias_of_task_only"])
            results.append(row)
            atomic_json(output / "progress.json", {"completed": len(results),
                                                     "requested": len(plan["jobs"]) * len(plan["requests"]),
                                                     "latest": row})
    cells = {}
    for cell in dict.fromkeys(job["cell"] for job in plan["jobs"]):
        group = [row for row in results if row["cell"] == cell]
        actor_latency = [row["actor_action_latency_seconds"] for row in group
                         if row.get("actor_action_latency_seconds", {}).get("count", 0)]
        cells[cell] = {"requested": len(group),
                       "constrained_successes": sum(row["passed"] for row in group),
                       "constraint_violations": sum(row["constraint_violation"] for row in group),
                       "execution_errors": sum(row["outcome"] == "reset_or_execution_error" for row in group),
                       "mean_capped_normalized_completion_time": float(np.mean([
                           row.get("capped_normalized_completion_time", 1.) for row in group])),
                       "outcomes": dict(Counter(row["outcome"] for row in group)),
                       "selected_task_only_aliases": sum(row["selected_alias_of_task_only"] for row in group) // len(plan["requests"]),
                       **{f"median_episode_{key}_actor_action_seconds": float(np.median([row[key] for row in actor_latency])) if actor_latency else None
                          for key in ("p50", "p95", "p99")}}
    report = {"version": VERSION, "status": "complete", "task": "T2",
              "scope": plan["scope"], "requests_per_cell_instance": len(plan["requests"]),
              "instances_per_cell": len(plan["jobs"]) // 4, "cells": cells, "results": results,
              "paired_contrasts": _paired_contrasts(plan, results),
              "evidence_audits_passed": all(row.get("evidence", {}).get("audit_passed", False) for row in results),
              "confirmation_states_opened": 0, "formal_performance_evidence": False,
              "wall_seconds": time.monotonic() - started}
    atomic_json(output / "report.json", report)
    return report
