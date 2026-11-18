#!/usr/bin/env python3
"""Run paired critical-contact development cells with clean ICDT ablations."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.solvers.single.cross_ebdm.learning.ablation import treatment_metadata
from genedynamics.solvers.single.cross_ebdm.learning.semantic_phase import PublicSemanticPhaseTransport
from genedynamics.tasks.transfer.compliant_socket import FixtureRelativePolicy
from genedynamics.tasks.transfer.compliant_socket_qualification import _public_contact_probe
from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor, resolve_transfer_binding
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackSpec
from genedynamics.tasks.transfer.controllers.guide_contact import (
    PublicCornerGuidePolicy, adapt_corner_public_policy, bind_corner_public_planner,
)
from genedynamics.tasks.transfer.controllers.mating_feedback import PublicLivenessMatingFeedbackPolicy
from genedynamics.tasks.transfer.episode import audit_episode_evidence, episode_evidence
from genedynamics.tasks.transfer.specs import mechanism_spec
from scripts.paper.cross_ebdm.qualify_method_behavior import _task_from_job

VERSION = "contact-complexity-paired-development-v1"
TREATMENTS = ("full", "no_calibration", "no_response_anchoring", "unconditional_response_anchoring")
COMMON = ("full", "task_only", "adaptsim", "asid", "response_mpc", *TREATMENTS[1:])


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pointer(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def checked(value, label):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} requires exact path/SHA")
    path = ROOT / value["path"]
    if not path.is_file() or sha(path) != value["sha256"]: raise ValueError(f"{label} changed")
    return path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def resolve_binding(path, method, alias, mode):
    experiment = ExperimentConfig.from_yaml(path)
    if experiment.method != method: raise ValueError(f"{method} identity changed")
    binding = deepcopy(experiment.method_params["actor_binding"])
    if binding.get("manifest") is not None:
        binding.update(family="main", key=method, instance="instance_01")
        binding = resolve_transfer_binding({"actor_binding": binding, "mode": mode}, task_alias=alias, instance_id="instance_01")
    return experiment, binding


def checkpoint(task, treatment, scale, phase_count, diagnostic, selection):
    active = list(range(phase_count)) if treatment == "unconditional_response_anchoring" else ([1, 2, 3] if task == "corner_guided_seating" else [2, 3])
    scales = [1.0] * phase_count
    for phase in active:
        scales[phase] = float(scale)
    return {"schema_version": "semantic-phase-transport-v1", "task": task,
            "treatment": treatment_metadata(treatment), "interaction_budget_seconds": 120,
            "phase_scales": scales, "active_phases": active,
            "translation_coordinates": [0, 1, 2], "selection": selection,
            "response_model": diagnostic}


def semantic_base(config, spec, mode, seed):
    if config["task"] == "corner_guided_seating":
        return PublicCornerGuidePolicy(spec, mode=mode)
    feedback = PublicBiasFeedbackSpec(compensation_gain=2., insertion_force_scale=1.,
                                      lift_before_lateral=True, guard_prediction_seconds=0.,
                                      insertion_lead_z_m=.015)
    return PublicLivenessMatingFeedbackPolicy(spec, feedback=feedback, modes=[mode], seed=seed)


def semantic_actor(config, spec, mode, seed, value):
    actor = PublicSemanticPhaseTransport(semantic_base(config, spec, mode, seed), value, task=config["task"])
    return actor if config["task"] == "corner_guided_seating" else FixtureRelativePolicy(actor)


def baseline_actor(config, method_paths, method, mode, seed, device):
    alias = "T2" if config["task"] == "corner_guided_seating" else "T3"
    experiment, binding = resolve_binding(method_paths[method], method, alias, mode)
    request = {"task": alias, "robot": "xarm7", "requested_mode": mode,
               "reset_seed": seed, "mechanism_parameters": {},
               "body_support_parameters": {"robot": "xarm7", "support_controller": "fixed_base"}}
    canonical = _task_from_job(experiment, request)
    actor = create_transfer_actor(canonical, method, {"actor_binding": binding, "mode": mode, "device": device})
    if config["task"] == "corner_guided_seating":
        spec = instantiate_task(config["task"], robot="xarm7",
                                mechanism_spec=mechanism_spec(config["task"], config["cells"][0]["mechanism"]), seed=seed).spec
        return adapt_corner_public_policy(actor, spec, mode=mode) if method in {"adaptsim", "asid"} else bind_corner_public_planner(actor, spec, mode=mode)
    return FixtureRelativePolicy(actor)


def fit_response(samples, device="cuda"):
    if not samples:
        return {"kind": "public-linear-one-step-v1", "train_samples": 0, "validation_samples": 0, "validation_mse": 0.0}, []
    import torch
    data = np.asarray(samples, np.float32); x, y = data[:, 0], data[:, 1]
    mask = np.arange(len(x)) % 5 != 0
    tx, ty = torch.as_tensor(x[mask], device=device), torch.as_tensor(y[mask], device=device)
    vx, vy = torch.as_tensor(x[~mask], device=device), torch.as_tensor(y[~mask], device=device)
    design = torch.cat((tx, torch.ones((len(tx), 1), device=device)), 1)
    weights = torch.linalg.pinv(design) @ ty
    pred = torch.cat((vx, torch.ones((len(vx), 1), device=device)), 1) @ weights
    mse = float(torch.mean((pred - vy) ** 2).cpu()) if len(vx) else 0.0
    return {"kind": "public-linear-one-step-v1", "train_samples": int(len(tx)),
            "validation_samples": int(len(vx)), "validation_mse": mse}, weights.cpu().numpy().tolist()


def execute(config, *, cell, seed, mode, actor, destination, collect=False):
    task = instantiate_task(config["task"], robot=config["robot"],
                            mechanism_spec=mechanism_spec(config["task"], cell["mechanism"]), seed=seed)
    reset_kwargs = {"initial": "jammed"} if config["task"] != "corner_guided_seating" else {}
    public = task.reset((seed,), seed=seed, modes=[mode], **reset_kwargs)
    samples, latencies = [], []
    probe_state = {"trigger_tick": None, "applied_controls": 0}
    import torch
    for tick in range(task.spec.config.timeout_steps):
        if task.state.status[0] != 0: break
        position = np.asarray(public["tcp_position"], float)[0].copy()
        before = time.perf_counter(); action = np.asarray(actor.act(public), np.float32)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        latencies.append(time.perf_counter() - before)
        if config["probe"] is not None:
            action = _public_contact_probe(action, public, tick, probe_state, config["probe"])
        step = task.step(action); following = np.asarray(step.observation["tcp_position"], float)[0].copy()
        if collect: samples.append((action[0, :3].astype(float), (following-position)/task.spec.config.control_dt))
        public = step.observation
    evaluation = task.evaluate(); episode = evaluation["episodes"][-1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.compress(json.dumps(episode_evidence(task.records, task.spec, evaluation), sort_keys=True, allow_nan=False).encode(), mtime=0))
    audit = audit_episode_evidence(destination); values = sorted(latencies)
    key_events = episode.get("event_physics_steps", {k: episode.get(k) for k in ("fixture_motion_physics_step", "loaded_contact_physics_step", "recovery_physics_step", "relative_insertion_physics_step") if episode.get(k) is not None})
    key_event_completed = (
        "endstop" in key_events
        if config["task"] == "corner_guided_seating"
        else bool(episode.get("terminal_joint_stability_passed", False))
    )
    row = {"task": config["task"], "cell": cell["id"], "seed": seed, "mode": mode,
           "success": bool(episode["success"]), "outcome": episode["outcome"],
           "constraint_violation": bool(episode["constraint_violation"]),
           "physical_control_steps": int(episode["physical_control_steps"]),
           "capped_normalized_completion_time": float(episode["capped_normalized_completion_time"]),
           "peak_force_N": float(episode.get("force_peak_N", 0.0)), "key_events": key_events,
           "key_event_completed": key_event_completed,
           "reset_physical_hash": audit["reset_physical_hashes"][0],
           "evidence": pointer(destination), "evidence_audit_passed": bool(audit["passed"]),
           "actor_latency_seconds": {"median": statistics.median(values), "p95": values[min(len(values)-1, int(.95*len(values)))]}}
    for key in ("fixture_max_displacement_m", "fixture_peak_speed_mps", "fixture_peak_acceleration_mps2", "terminal_joint_stability_passed"):
        if key in episode: row[key] = episode[key]
    if config["probe"] is not None: row["public_probe"] = dict(probe_state)
    return row, samples


def summarize(rows):
    return {"episodes": len(rows), "task_success": sum(r["success"] for r in rows)/len(rows),
            "key_event_completion": sum(bool(r["key_event_completed"]) for r in rows)/len(rows),
            "peak_force_N": max(r["peak_force_N"] for r in rows),
            "failure_penalized_normalized_time": statistics.mean(r["capped_normalized_completion_time"] for r in rows),
            "median_actor_p95_seconds": statistics.median(r["actor_latency_seconds"]["p95"] for r in rows)}


def run(config_path):
    config = json.loads(config_path.read_text())
    expected = {"version", "status", "task", "robot", "contract", "source_qualification", "method_configs", "cells", "selection", "evaluation", "treatment_scales", "probe", "execution", "output", "policy"}
    if set(config) != expected or config["version"] != VERSION or config["status"] != "frozen_unexecuted": raise ValueError("contact paired recipe changed")
    contract = checked(config["contract"], "contract"); source = checked(config["source_qualification"], "source")
    if json.loads(source.read_text()).get("passed") is not True: raise ValueError("passed source qualification required")
    method_paths = {name: checked(value, f"{name} config") for name, value in config["method_configs"].items()}
    methods = (*COMMON, "dial_transfer") if config["task"] != "corner_guided_seating" else COMMON
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"): raise RuntimeError("GPU0 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("one visible CUDA device required")
    visible = f"GPU-{torch.cuda.get_device_properties(0).uuid}"
    if visible != config["execution"]["physical_uuid"]: raise RuntimeError("GPU UUID differs")
    output = ROOT/config["output"]; output.mkdir(parents=True, exist_ok=False)
    write_json(output/"plan.json", {"version": VERSION, "config": pointer(config_path), "contract": pointer(contract), "source": pointer(source),
                                    "methods": methods, "dial_corner_exclusion": "T2 DIAL cost requires the legacy corridor base.observer memory; semantic corner planner has no compatible observer" if config["task"] == "corner_guided_seating" else None,
                                    "execution": config["execution"], "policy": config["policy"]})
    started = time.monotonic(); selection_rows=[]; samples=[]; phase_count=5 if config["task"] == "corner_guided_seating" else 4
    for scale_index, scale in enumerate(config["selection"]["candidate_scales"]):
        for cell_index in config["selection"]["cell_indices"]:
            cell = config["cells"][cell_index]
            mode = 0 if config["task"] == "corner_guided_seating" else cell_index
            seed = config["selection"]["seed_anchor"] + 100*scale_index + cell_index
            spec = instantiate_task(config["task"], robot=config["robot"], mechanism_spec=mechanism_spec(config["task"], cell["mechanism"]), seed=seed).spec
            provisional = checkpoint(config["task"], "full", scale, phase_count,
                                     {"kind":"public-linear-one-step-v1","train_samples":0,"validation_samples":0,"validation_mse":0.0}, {"status":"candidate"})
            row, data = execute(config, cell=cell, seed=seed, mode=mode,
                                actor=semantic_actor(config, spec, mode, seed, provisional),
                                destination=output/"selection_evidence"/f"scale_{scale_index}_cell_{cell_index}.json.gz", collect=True)
            row["scale"] = float(scale); selection_rows.append(row); samples.extend(data)
            print(json.dumps({"phase":"selection","task":config["task"],"scale":scale,"cell":cell["id"],"success":row["success"],"steps":row["physical_control_steps"]}), flush=True)
    by_scale={scale:[r for r in selection_rows if r["scale"]==scale] for scale in config["selection"]["candidate_scales"]}
    qualified=[(statistics.mean(r["capped_normalized_completion_time"] for r in rows), scale) for scale,rows in by_scale.items() if len(rows)==len(config["selection"]["cell_indices"]) and all(r["success"] and not r["constraint_violation"] for r in rows)]
    selected = min(qualified)[1] if qualified else 1.0
    diagnostic, weights = fit_response(samples)
    selection={"status":"development","selected_scale":selected,"criterion":"best mean failure-penalized normalized time among candidates passing every development cell", "response_weights":weights}
    checkpoints={}
    for treatment in TREATMENTS:
        scale = selected if treatment in {"full","unconditional_response_anchoring"} else config["treatment_scales"][treatment]
        rationale = {
            "full": "target-calibrated constrained selection",
            "no_calibration": "unit source scale without target calibration",
            "no_response_anchoring": "aggressive task-loss scale without response constraint",
            "unconditional_response_anchoring": "selected response scale applied outside its supported phases",
        }[treatment]
        value=checkpoint(config["task"],treatment,scale,phase_count,diagnostic,{**selection,"clean_treatment":treatment,"applied_scale":scale})
        value["selection"]["treatment_selection"] = rationale
        path=output/"checkpoints"/f"{treatment}.json"; write_json(path,value); checkpoints[treatment]=pointer(path)
    rows=[]; reset_hashes={}
    for cell_index,cell in enumerate(config["cells"]):
        for seed,mode in zip(config["evaluation"]["seeds_by_cell"][cell_index], config["evaluation"]["modes_by_cell"][cell_index]):
            reset_hashes[(cell["id"],seed)] = set()
            for method in methods:
                spec=instantiate_task(config["task"],robot=config["robot"],mechanism_spec=mechanism_spec(config["task"],cell["mechanism"]),seed=seed).spec
                if method in TREATMENTS:
                    value=json.loads((ROOT/checkpoints[method]["path"]).read_text()); actor=semantic_actor(config,spec,mode,seed,value)
                else:
                    actor=baseline_actor(config,method_paths,method,mode,seed,config["execution"]["actor_device"])
                row,_=execute(config,cell=cell,seed=seed,mode=mode,actor=actor,
                              destination=output/"evidence"/method/cell["id"]/f"seed_{seed}.json.gz")
                row["method"]=method;rows.append(row);reset_hashes[(cell["id"],seed)].add(row["reset_physical_hash"])
                write_json(output/"progress.json",{"selection_rows":selection_rows,"completed":len(rows),"rows":rows})
                print(json.dumps({"phase":"paired","task":config["task"],"cell":cell["id"],"method":method,"seed":seed,"success":row["success"],"outcome":row["outcome"],"steps":row["physical_control_steps"]}),flush=True)
                if time.monotonic()-started > config["execution"]["maximum_wall_seconds"]: raise TimeoutError("frozen wall limit exceeded")
    pairing=all(len(v)==1 for v in reset_hashes.values()); audits=all(r["evidence_audit_passed"] for r in selection_rows+rows)
    summary={cell["id"]:{method:summarize([r for r in rows if r["cell"]==cell["id"] and r["method"]==method]) for method in methods} for cell in config["cells"]}
    report={"version":VERSION,"status":"complete_development_only" if pairing and audits else "audit_failed", "task":config["task"],
            "selection":{"rows":selection_rows,"selected_scale":selected,"response_model":diagnostic,"model_mse_role":"diagnostic_only"},
            "checkpoints":checkpoints,"rows":rows,"summary":summary,"pairing_passed":pairing,"all_evidence_reaudited":audits,
            "dial_corner_exclusion":"interface-inapplicable; no physics score assigned" if config["task"]=="corner_guided_seating" else None,
            "formal_performance_evidence":False,"elapsed_seconds":time.monotonic()-started}
    write_json(output/"report.json",report);print(json.dumps({"status":report["status"],"selected_scale":selected,"summary":summary},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);args=parser.parse_args();run(args.config)


if __name__=="__main__":main()
