"""Observed, segmented response prediction for the compliant T3 socket.

This model consumes only the versioned public response bundle.  It predicts
control-rate response summaries; no output is a substep safety certificate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from genedynamics.learning.checkpoints.torch import to_cpu
from .artifact_bundle import load_bundle
from .response import FIXTURE_RESPONSE_SCHEMA, FIXTURE_SEGMENT_SCALE, PublicResponseSpec


VERSION = "compliant-socket-segmented-response-model-v1"
TRAINING_VERSION = "compliant-socket-segmented-response-training-v1"
EVENT_VERSION = "compliant-socket-segmented-event-response-model-v2"
EVENT_TRAINING_VERSION = "compliant-socket-segmented-event-response-training-v2"
CALIBRATED_EVENT_VERSION = "compliant-socket-segmented-calibrated-event-response-model-v3"
CALIBRATED_EVENT_TRAINING_VERSION = "compliant-socket-segmented-calibrated-event-response-training-v3"


@dataclass(frozen=True)
class FixtureResponseModelSpec:
    history_len: int = 4
    horizon: int = 10
    maximum_joints: int = 32
    segment_steps: tuple[int, ...] = (3, 6, 10)
    hidden: tuple[int, ...] = (256, 256)
    members: int = 3
    event_head: bool = False
    event_threshold_m: float = 1e-5

    def __post_init__(self):
        if (min(self.history_len, self.horizon, self.maximum_joints, self.members) < 1
                or len(self.segment_steps) < 1
                or tuple(sorted(set(self.segment_steps))) != self.segment_steps
                or self.segment_steps[-1] != self.horizon
                or not self.hidden or min(self.hidden) < 1
                or not isinstance(self.event_head, bool)
                or not np.isfinite(self.event_threshold_m) or self.event_threshold_m <= 0):
            raise ValueError("explicit positive segmented response shape required")

    @property
    def feature_dim(self):
        base = PublicResponseSpec(history_len=self.history_len, horizon=self.horizon,
                                  maximum_joints=self.maximum_joints)
        return base.feature_dim + 18 * self.history_len

    @property
    def output_dim(self):
        return (len(self.segment_steps) * len(FIXTURE_SEGMENT_SCALE) + 2
                + (len(self.segment_steps) if self.event_head else 0))


class FixtureResponseEnsemble(nn.Module):
    def __init__(self, spec: FixtureResponseModelSpec):
        super().__init__()
        self.spec = spec
        sizes = (spec.feature_dim + 13 * spec.horizon, *spec.hidden, spec.output_dim)
        self.members = nn.ModuleList()
        for _ in range(spec.members):
            layers = []
            for index, (before, after) in enumerate(zip(sizes[:-1], sizes[1:])):
                layers.append(nn.Linear(before, after))
                if index < len(sizes) - 2:
                    layers.append(nn.SiLU())
            self.members.append(nn.Sequential(*layers))

    def forward_with_event(self, features, commands):
        if (features.ndim != 2 or features.shape[1] != self.spec.feature_dim
                or commands.shape != (len(features), self.spec.horizon, 13)):
            raise ValueError("fixture response model requires versioned public history and shared13 chunk")
        inputs = torch.cat((features, commands.flatten(start_dim=1)), dim=1)
        values = torch.stack([member(inputs) for member in self.members])
        segment_width = len(self.spec.segment_steps) * 27
        segments = values[..., :segment_width].reshape(
            self.spec.members, len(features), len(self.spec.segment_steps), 27,
        )
        event_logits = None
        if self.spec.event_head:
            event_logits = values[..., segment_width:-2]
            probability = torch.sigmoid(event_logits)[..., None]
            segments = torch.cat((segments[..., :18], probability * segments[..., 18:]), dim=-1)
        return segments, values[..., -2:], event_logits

    def forward(self, features, commands):
        segments, peaks, _ = self.forward_with_event(features, commands)
        return segments, peaks

    def checkpoint(self, *, data_sha256: str, training_version: str | None = None):
        spec_payload = asdict(self.spec)
        if not self.spec.event_head:
            spec_payload.pop("event_head")
            spec_payload.pop("event_threshold_m")
        if training_version == CALIBRATED_EVENT_TRAINING_VERSION:
            if not self.spec.event_head:
                raise ValueError("calibrated event checkpoint requires event head")
            version = CALIBRATED_EVENT_VERSION
        else:
            version = EVENT_VERSION if self.spec.event_head else VERSION
        return dict(version=version,
                    spec=spec_payload, state_dict=to_cpu(self.state_dict()),
                    data_sha256=data_sha256, observation_schema=FIXTURE_RESPONSE_SCHEMA,
                    input_semantics="causal public fixture and robot history plus issued world requests",
                    output_semantics=(
                        "segmented normalized TCP deltas, event-probability-gated fixture deltas, "
                        "segment event logits and log control-rate force peaks"
                        if self.spec.event_head else
                        "segmented normalized TCP/fixture deltas and log control-rate force peaks"
                    ),
                    private_state=False)

    @classmethod
    def from_checkpoint(cls, payload, *, device="cpu"):
        if (payload.get("version") not in {VERSION, EVENT_VERSION, CALIBRATED_EVENT_VERSION}
                or payload.get("observation_schema") != FIXTURE_RESPONSE_SCHEMA
                or payload.get("private_state") is not False):
            raise ValueError("fixture response checkpoint schema mismatch")
        values = dict(payload["spec"])
        for name in ("segment_steps", "hidden"):
            values[name] = tuple(values[name])
        if bool(values.get("event_head", False)) != (payload["version"] in {
                EVENT_VERSION, CALIBRATED_EVENT_VERSION}):
            raise ValueError("fixture response event head/version mismatch")
        model = cls(FixtureResponseModelSpec(**values)).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


def fixture_member_loss(
    prediction, segment_target, force_target, spec: FixtureResponseModelSpec,
    *, event_importance_weight=None, calibrate_expected_motion: bool = False,
):
    """Fit conditional fixture motion and its public event probability."""
    segment_width = len(spec.segment_steps) * 27
    raw = prediction[:, :segment_width].reshape(-1, len(spec.segment_steps), 27)
    peaks = (prediction[:, -2:] - force_target).square()
    if not spec.event_head:
        physical = (raw - segment_target).square()
        weighted = physical * prediction.new_tensor([1.] * 18 + [8.] * 9)
        return weighted.mean() + .2 * peaks.mean()
    labels = (torch.abs(segment_target[..., 18]) * float(FIXTURE_SEGMENT_SCALE[18])
              > spec.event_threshold_m).to(raw.dtype)
    logits = prediction[:, segment_width:-2]
    motion_error = (raw[..., 18:] - segment_target[..., 18:]).square()
    event_motion = (motion_error * labels[..., None]).sum() / (
        labels.sum().clamp_min(1.) * motion_error.shape[-1]
    )
    event_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if event_importance_weight is not None:
        if event_importance_weight.shape != (len(prediction),):
            raise ValueError("event importance weight must match sampled windows")
        event_loss = event_loss * event_importance_weight[:, None]
    if calibrate_expected_motion:
        probability = torch.sigmoid(logits)[..., None]
        expected_error = (
            probability * raw[..., 18:] - segment_target[..., 18:]
        ).square()
        importance = (
            event_importance_weight[:, None]
            if event_importance_weight is not None
            else torch.ones_like(labels)
        )
        event_weight = importance * labels
        quiet_weight = importance * (1. - labels)
        event_expected = (expected_error * event_weight[..., None]).sum() / (
            event_weight.sum().clamp_min(1.) * expected_error.shape[-1]
        )
        quiet_expected = (expected_error * quiet_weight[..., None]).sum() / (
            quiet_weight.sum().clamp_min(1.) * expected_error.shape[-1]
        )
        return (
            (raw[..., :18] - segment_target[..., :18]).square().mean()
            + 2. * event_motion + 8. * event_expected + 8. * quiet_expected
            + .2 * peaks.mean() + .1 * event_loss.mean()
        )
    return (
        (raw[..., :18] - segment_target[..., :18]).square().mean()
        + 8. * event_motion + .2 * peaks.mean()
        + .1 * event_loss.mean()
    )


def training_plan(config: dict, root: Path):
    """Bind one immutable public bundle and one device before any optimizer step."""
    if config.get("version") not in {
            TRAINING_VERSION, EVENT_TRAINING_VERSION,
            CALIBRATED_EVENT_TRAINING_VERSION}:
        raise ValueError("explicit fixture response training recipe required")
    path = root / config["data"]["path"]
    digest = hashlib.sha256((path / "metadata.json").read_bytes()).hexdigest()
    if digest != config["data"]["metadata_sha256"]:
        raise ValueError("fixture response bundle metadata changed")
    metadata = json.loads((path / "metadata.json").read_text())
    if (metadata.get("artifact_kind") != "fixture_segmented_response_data"
            or metadata.get("response_schema") != FIXTURE_RESPONSE_SCHEMA
            or metadata.get("tensor_sha256") != config["data"]["tensor_sha256"]
            or metadata.get("fit_windows", 0) <= 0 or metadata.get("validation_windows", 0) <= 0):
        raise ValueError("fixture response data identity or split differs")
    spec = FixtureResponseModelSpec(**{
        **config["model"],
        "segment_steps": tuple(config["model"]["segment_steps"]),
        "hidden": tuple(config["model"]["hidden"]),
    })
    if spec.event_head != (config["version"] in {
            EVENT_TRAINING_VERSION, CALIBRATED_EVENT_TRAINING_VERSION}):
        raise ValueError("fixture response event head/training version mismatch")
    if (metadata["segment_steps"] != list(spec.segment_steps)
            or any(metadata["response_spec"][key] != getattr(spec, key)
                   for key in ("history_len", "horizon", "maximum_joints"))):
        raise ValueError("model and response bundle horizons differ")
    execution = config["execution"]
    expected_uuid = execution.get("physical_uuid", "")
    if (execution != {"device": "cuda:0", "cuda_visible_devices": "1", "physical_uuid": expected_uuid}
            or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", expected_uuid) is None):
        raise ValueError("fixture response training requires bound physical GPU1")
    budget = config["budget"]
    if (not 1 <= budget["updates"] <= 4000 or not 1 <= budget["batch_size"] <= 512
            or not 0. < budget["learning_rate"] <= .001
            or not 1 <= budget["maximum_wall_seconds"] <= 3600
            or not 0 < budget["maximum_gpu_gib"] <= 8):
        raise ValueError("bounded fixture response training budget required")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("nonnegative training seed required")
    if config["version"] == CALIBRATED_EVENT_TRAINING_VERSION:
        required = {
            "genedynamics/solvers/single/cross_ebdm/learning/fixture_response.py",
            "genedynamics/solvers/single/cross_ebdm/learning/response.py",
            "genedynamics/solvers/single/cross_ebdm/learning/artifact_bundle.py",
            "scripts/paper/cross_ebdm/train_fixture_response.py",
        }
        bindings = config.get("source_bindings")
        if not isinstance(bindings, list) or {row.get("path") for row in bindings} != required:
            raise ValueError("calibrated event response source roster differs")
        for row in bindings:
            if (set(row) != {"path", "sha256"}
                    or hashlib.sha256((root / row["path"]).read_bytes()).hexdigest()
                    != row["sha256"]):
                raise ValueError(f"calibrated event response source changed: {row.get('path')}")
    if (root / config["output"]).exists():
        raise FileExistsError("fixture response training output is immutable")
    return {"status": "ready_not_started", "data_path": str(path),
            "metadata_sha256": digest, "tensor_sha256": metadata["tensor_sha256"],
            "fit_windows": metadata["fit_windows"], "validation_windows": metadata["validation_windows"],
            "updates": budget["updates"], "gpu": execution}


def _validation_metrics(model, arrays, device):
    index = np.flatnonzero(arrays["split"] == 1)
    features = torch.from_numpy(arrays["features"][index]).to(device)
    commands = torch.from_numpy(arrays["commands"][index]).to(device)
    model.eval()
    with torch.inference_mode():
        predicted, peaks, event_logits = model.forward_with_event(features, commands)
    predicted = predicted.mean(0).cpu().numpy() * FIXTURE_SEGMENT_SCALE
    truth = arrays["segment_targets"][index] * FIXTURE_SEGMENT_SCALE
    force_predicted = np.expm1(peaks.mean(0).cpu().numpy())
    force_truth = np.expm1(arrays["log_control_force_peaks"][index])
    event_probabilities = (
        torch.sigmoid(event_logits).mean(0).cpu().numpy()
        if event_logits is not None else None
    )
    fixture_event = np.max(np.abs(truth[:, :, 18]), axis=1) > 1e-5
    rows = []
    for segment, step in enumerate(model.spec.segment_steps):
        delta = predicted[:, segment] - truth[:, segment]
        event = delta[fixture_event]
        rows.append({"step": step,
                     "tcp_position_mae_m": float(np.abs(delta[:, :3]).mean()),
                     "tcp_twist_mae_mps_radps": float(np.abs(delta[:, 6:12]).mean()),
                     "wrench_mae_N_Nm": float(np.abs(delta[:, 12:18]).mean()),
                     "fixture_position_mae_m": float(np.abs(delta[:, 18:21]).mean()),
                     "fixture_twist_mae_mps_radps": float(np.abs(delta[:, 21:27]).mean()),
                     "fixture_event_position_mae_m": float(np.abs(event[:, 18:21]).mean()) if len(event) else None,
                     "zero_fixture_event_position_mae_m": float(np.abs(truth[fixture_event, segment, 18:21]).mean()) if len(event) else None,
                     "fixture_x_quiet_mae_m": float(np.abs(delta[np.abs(truth[:, segment, 18]) <= model.spec.event_threshold_m, 18]).mean()),
                     "fixture_x_zero_quiet_mae_m": float(np.abs(truth[np.abs(truth[:, segment, 18]) <= model.spec.event_threshold_m, segment, 18]).mean()),
                     "fixture_x_event_brier": (
                         float(np.mean((event_probabilities[:, segment] -
                                        (np.abs(truth[:, segment, 18]) > model.spec.event_threshold_m)) ** 2))
                         if event_probabilities is not None else None),
                     "fixture_x_event_probability_mean": (
                         float(event_probabilities[:, segment].mean())
                         if event_probabilities is not None else None)})
    return {"validation_windows": len(index), "fixture_event_windows": int(fixture_event.sum()),
            "event_threshold_m": 1e-5, "segments": rows,
            "control_force_peak_mae_N": float(np.abs(force_predicted-force_truth).mean())}


def train_fixture_response(config: dict, root: Path):
    plan = training_plan(config, root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("training requires only physical GPU1 visible")
    properties = torch.cuda.get_device_properties(0)
    if str(properties.uuid).removeprefix("GPU-").lower() != config["execution"]["physical_uuid"].removeprefix("GPU-").lower():
        raise RuntimeError("visible GPU UUID differs from GPU1 recipe")
    torch.cuda.set_per_process_memory_fraction(config["budget"]["maximum_gpu_gib"] * 1024**3 / properties.total_memory, 0)
    arrays, metadata = load_bundle(plan["data_path"])
    if metadata["tensor_sha256"] != plan["tensor_sha256"]:
        raise ValueError("fixture response tensor data changed")
    spec = FixtureResponseModelSpec(**{**config["model"], "segment_steps": tuple(config["model"]["segment_steps"]),
                                       "hidden": tuple(config["model"]["hidden"])})
    if (arrays["features"].shape != (metadata["fit_windows"] + metadata["validation_windows"], spec.feature_dim)
            or arrays["commands"].shape[1:] != (spec.horizon, 13)
            or arrays["segment_targets"].shape[1:] != (len(spec.segment_steps), 27)
            or arrays["log_control_force_peaks"].shape[1:] != (2,)):
        raise ValueError("fixture response data shape mismatch")
    fit = np.flatnonzero(arrays["split"] == 0)
    validation = np.flatnonzero(arrays["split"] == 1)
    if len(fit) != metadata["fit_windows"] or len(validation) != metadata["validation_windows"]:
        raise ValueError("response split counts changed")
    if set(arrays["episode_id"][fit]) & set(arrays["episode_id"][validation]):
        raise ValueError("fit and validation share episode IDs")
    device = torch.device("cuda:0")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    model = FixtureResponseEnsemble(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["budget"]["learning_rate"])
    tensor = {key: torch.from_numpy(arrays[key]).to(device) for key in
              ("features", "commands", "segment_targets", "log_control_force_peaks")}
    event = (
        np.max(np.abs(arrays["segment_targets"][:, :, 18]), axis=1)
        * float(FIXTURE_SEGMENT_SCALE[18]) > spec.event_threshold_m
    )
    contact_fit = fit[event[fit]]
    if len(contact_fit) < 10:
        raise ValueError("fit split lacks public fixture-motion windows")
    generator = np.random.default_rng(config["seed"])
    contact_probability = len(contact_fit) / len(fit)
    batch_size = config["budget"]["batch_size"]
    uniform_fraction = (batch_size // 2) / batch_size
    contact_fraction = 1. - uniform_fraction
    noncontact_weight = 1. / uniform_fraction
    contact_weight = 1. / (uniform_fraction + contact_fraction / contact_probability)
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if config["version"] == CALIBRATED_EVENT_TRAINING_VERSION:
        (output / "recipe.json").write_text(json.dumps(config, indent=2) + "\n")
        for row in config["source_bindings"]:
            source = root / row["path"]
            snapshot = output / "source_snapshot" / row["path"]
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, snapshot)
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != row["sha256"]:
                raise RuntimeError("calibrated event response source snapshot differs")
    started = time.monotonic()
    updates = config["budget"]["updates"]
    curve = []
    try:
        for update in range(1, updates + 1):
            if time.monotonic() - started > config["budget"]["maximum_wall_seconds"]:
                raise TimeoutError("fixture response training wall budget exhausted")
            model.train()
            losses = []
            for member in model.members:
                indices = np.concatenate((generator.choice(fit, batch_size // 2),
                                          generator.choice(contact_fit, batch_size - batch_size // 2)))
                sampled = torch.as_tensor(indices, device=device)
                event_importance_weight = torch.as_tensor(
                    np.where(event[indices], contact_weight, noncontact_weight),
                    device=device, dtype=tensor["features"].dtype,
                )
                inputs = torch.cat((tensor["features"][sampled], tensor["commands"][sampled].flatten(start_dim=1)), dim=1)
                prediction = member(inputs)
                target = tensor["segment_targets"][sampled]
                losses.append(fixture_member_loss(
                    prediction, target, tensor["log_control_force_peaks"][sampled], spec,
                    event_importance_weight=event_importance_weight,
                    calibrate_expected_motion=(config["version"] == CALIBRATED_EVENT_TRAINING_VERSION),
                ))
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite fixture response objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if update == 1 or update % max(1, updates // 10) == 0 or update == updates:
                row = {"update": update, "fit_loss": float(loss.detach().cpu()),
                       "wall_seconds": time.monotonic() - started}
                curve.append(row)
                (output / "curve.json").write_text(json.dumps(curve, indent=2) + "\n")
        metrics = _validation_metrics(model, arrays, device)
        def finite_values(value):
            if isinstance(value, dict):
                return all(finite_values(child) for child in value.values())
            if isinstance(value, list):
                return all(finite_values(child) for child in value)
            return value is None or not isinstance(value, (int, float)) or np.isfinite(value)
        if not finite_values(metrics):
            raise FloatingPointError("nonfinite fixture response validation metrics")
        checkpoint = model.checkpoint(data_sha256=plan["tensor_sha256"],
                                      training_version=config["version"])
        checkpoint["training_updates"] = updates
        checkpoint["seed"] = config["seed"]
        temporary = output / ".model.pt"
        torch.save(checkpoint, temporary)
        os.replace(temporary, output / "model.pt")
        restored = FixtureResponseEnsemble.from_checkpoint(
            torch.load(output / "model.pt", map_location="cpu", weights_only=True))
        if any(not torch.equal(a, b) for a, b in zip(model.cpu().parameters(), restored.parameters())):
            raise RuntimeError("fixture response checkpoint changed on roundtrip")
        report = {"version": config["version"], "status": "complete", "gpu_physical_uuid": config["execution"]["physical_uuid"],
                  "updates": updates, "wall_seconds": time.monotonic()-started,
                  "fit_windows": len(fit), "fit_fixture_motion_windows": len(contact_fit),
                  "validation": metrics, "fit_curve": curve, "data": plan,
                  "fixture_motion_loss": (
                      "event_and_quiet_expected_motion_with_conditional_event_amplitude"
                      if config["version"] == CALIBRATED_EVENT_TRAINING_VERSION
                      else "conditional_event_amplitude_and_event_probability"
                  ),
                  "scope": "geometry-controller public trajectories; no learned transport performance or substep safety claim"}
    except BaseException as error:
        (output / "report.json").write_text(json.dumps({"status": "failed", "error": repr(error),
                                                          "curve": curve}, indent=2) + "\n")
        raise
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
