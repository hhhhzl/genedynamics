"""Identification from sealed public cold-reset probes, without target carry.

This path deliberately does not accept the legacy ProbeRecord. The fitted
family is servo gain / integer delay with known nominal payload, not a general
contact world model. Each hypothesis owns its controller and physics state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from genedynamics.envs.adapters.transfer_task import CpuTaskAdapter, FLAT_FIELDS
from genedynamics.tasks.transfer.runtime import MechanismSpec
from .structured_response import MechanismFamily

PROBE_VERSION = "public-cold-reset-probe-v1"
META_FIELDS = {"version", "index", "purpose", "seed", "sample_id", "initial", "kind"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()).encode() + b"\n")
    temporary.replace(path)


@dataclass(frozen=True)
class PublicProbe:
    metadata: dict
    observations: dict
    requests: np.ndarray

    def __post_init__(self):
        meta = dict(self.metadata)
        if set(meta) != META_FIELDS or meta["version"] != PROBE_VERSION:
            raise ValueError("strict public probe metadata required; evaluator/plant fields forbidden")
        if meta["purpose"] not in {"fit", "validation"} or meta["initial"] not in {"clear", "jammed"}:
            raise ValueError("invalid public probe purpose/reset")
        if meta["kind"] not in {"free", "loaded"}:
            raise ValueError("invalid public probe design")
        for key in ("index", "seed", "sample_id"):
            if isinstance(meta[key], bool) or not isinstance(meta[key], int) or meta[key] < 0:
                raise ValueError("nonnegative integer public identities required")
        if set(self.observations) != set(FLAT_FIELDS):
            raise ValueError("public observations only; extra/missing fields forbidden")
        requests = np.array(self.requests, dtype=np.float32, copy=True)
        if requests.ndim != 2 or requests.shape[1] != 13 or not 1 <= len(requests) <= 100 or not np.isfinite(requests).all():
            raise ValueError("probe must contain 1..100 finite13D issued requests")
        observations = {}
        dimensions = dict(joint_position=(7,), joint_velocity=(7,), tcp_position=(3,), tcp_rotation=(3,3),
            tcp_twist=(6,), measured_wrench=(6,), goal_position=(3,), goal_rotation=(3,3),
            previous_action=(13,), previous_request=(13,), reference_position=(3,), reference_rotation=(3,3),
            elapsed_seconds=(1,), body_position=(3,), body_twist=(6,), controller_feedback=(2,))
        for key, value in self.observations.items():
            array = np.array(value, dtype=np.float32, copy=True)
            if array.shape != (len(requests)+1, *dimensions[key]) or not np.isfinite(array).all():
                raise ValueError("one finite public observation before/after each request required")
            array.setflags(write=False)
            observations[key] = array
        ticks = observations["elapsed_seconds"].reshape(len(requests) + 1)
        if not np.allclose(ticks, np.arange(len(ticks)) * .02, atol=1e-6):
            raise ValueError("only complete20ms cold-reset prefixes are supported")
        if np.any(observations["previous_request"][0]) or np.any(observations["previous_action"][0]):
            raise ValueError("probe must start from a fresh public reset")
        if not np.array_equal(observations["previous_request"][1:], requests):
            raise ValueError("public request receipt differs from issued requests")
        requests.setflags(write=False)
        object.__setattr__(self, "metadata", MappingProxyType(meta))
        object.__setattr__(self, "observations", MappingProxyType(observations))
        object.__setattr__(self, "requests", requests)

    @property
    def digest(self):
        value = hashlib.sha256(canonical(dict(self.metadata)))
        for key, array in sorted({**self.observations, "requests": self.requests}.items()):
            value.update(key.encode()); value.update(str(array.shape).encode()); value.update(array.tobytes())
        return value.hexdigest()

    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(path / "public.npz", requests=self.requests,
            **{"observation." + key: value for key, value in self.observations.items()})
        atomic_json(path / "public.json", dict(self.metadata))
        (path / "public.sha256").write_text(self.digest + "\n")

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path / "public.npz", allow_pickle=False) as data:
            expected = {"requests"} | {"observation." + key for key in FLAT_FIELDS}
            if set(data.files) != expected:
                raise ValueError("private/unknown arrays in public probe artifact")
            instance = cls(json.loads((path / "public.json").read_text()),
                {key: data["observation." + key] for key in FLAT_FIELDS}, data["requests"])
        if (path / "public.sha256").read_text().strip() != instance.digest:
            raise ValueError("sealed public probe changed")
        return instance


class PublicQueryLedger:
    """Durable requests before execution; confirmed transitions after execution.

    A crashing in-flight request is retained with an explicit substep upper
    bound, never converted to zero-cost evidence. Existing ledgers cannot be
    silently resumed or overwritten.
    """
    def __init__(self, path, *, limits, wall_deadline=None):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError("charged run exists; choose a new declared attempt")
        self.limits = dict(limits)
        self.wall_deadline = wall_deadline
        self.data = dict(version=1, roles={}, attempts=[], limits=self.limits, status="running")
        self.flush()

    def flush(self):
        for role in self.data["roles"].values():
            role["unconfirmed_command_requests"] = max(0, role.get("command_requests", 0) - role.get("control_steps", 0))
            role["unconfirmed_physics_steps_upper_bound"] = 10 * role["unconfirmed_command_requests"]
        atomic_json(self.path, self.data)

    def charge(self, role, key, value):
        import time
        if key in {"reset_calls", "command_requests"} and self.wall_deadline is not None and time.monotonic() >= self.wall_deadline:
            raise TimeoutError("predeclared wall deadline; incurred accounting retained")
        row = self.data["roles"].setdefault(role, {})
        if key == "command_requests" and row.get(key, 0) + value > self.limits[role]:
            raise RuntimeError("predeclared query ceiling reached")
        row[key] = row.get(key, 0) + int(value)
        self.flush()

    def start(self, role, label):
        self.data["attempts"].append(dict(role=role, label=label, state="requested"))
        self.flush()
        return len(self.data["attempts"]) - 1

    def finish(self, index, state="complete", error=None):
        self.data["attempts"][index].update(state=state, error=error)
        self.flush()


class CountedPublicTask(CpuTaskAdapter):
    def __init__(self, spec, *, ledger, role, **kwargs):
        self.query_ledger, self.query_role = ledger, role
        super().__init__(spec, **kwargs)

    def _charge(self, key, value):
        self.query_ledger.charge(self.query_role, key, value)
        super()._charge(key, value)


class PublicPhysicsTwin:
    """Predict sealed probes by known reset rules; never copy target state.

    Only the probe's public seed/identity/reset and issued action stream enter.
    Initial public encoders are checked against the reproduced cold reset.
    Warm starts / arbitrary controller carry are intentionally unsupported.
    """
    def __init__(self, spec, mechanism, *, ledger=None, role="model_identification"):
        if spec.alias != "T3" or spec.config.robot != "xarm7":
            raise ValueError("public twin qualification currently covers T3 xArm7 only")
        self.spec, self.mechanism, self.ledger, self.role = spec, mechanism, ledger, role

    def predict(self, probe):
        if not isinstance(probe, PublicProbe):
            raise TypeError("sealed PublicProbe required; legacy carry records are forbidden")
        meta = probe.metadata
        attempt = None if self.ledger is None else self.ledger.start(self.role, dict(meta))
        task = (CpuTaskAdapter(self.spec, mechanism_spec=self.mechanism) if self.ledger is None else
            CountedPublicTask(self.spec, mechanism_spec=self.mechanism, ledger=self.ledger, role=self.role))
        try:
            public = task.reset((meta["sample_id"],), seed=meta["seed"], initial=meta["initial"])
            for key in ("joint_position", "joint_velocity", "tcp_position", "tcp_rotation", "reference_position", "reference_rotation"):
                if not np.allclose(public[key][0], probe.observations[key][0], atol=2e-6, rtol=0):
                    raise ValueError("public cold reset cannot be reconstructed: " + key)
            rows = [public]
            ended = False
            for command in probe.requests:
                if ended: break
                result = task.step(command[None])
                public, ended = result.observation, bool(result.done[0])
                rows.append(public)
            result = {key: np.concatenate([row[key] for row in rows]) for key in public}
            if attempt is not None: self.ledger.finish(attempt)
            return result, dict(task.accounting)
        except Exception as exc:
            if attempt is not None: self.ledger.finish(attempt, "failed", repr(exc))
            raise


def public_response(observations):
    return np.concatenate((observations["tcp_position"], observations["tcp_rotation"].reshape(-1, 9),
        observations["measured_wrench"], observations["tcp_twist"]), axis=-1)


def probe_loss(observed, predicted, *, free=False):
    actual = public_response(observed)[1:]
    prediction = public_response(predicted)[1:]
    # Missing hypothesis transitions are explicitly penalized, not masked out.
    scales = np.array(([.0001]*3 + [.001]*9 if free else [.002]*3 + [.03]*9)
        + [1.]*3 + [.1]*3 + [.05]*3 + [.2]*3)
    n = min(len(actual), len(prediction))
    residual = ((actual[:n] - prediction[:n]) / scales)**2
    return float((np.minimum(residual, 1e6).sum() + (len(actual)-n)*len(scales)*1e6) / (len(actual)*len(scales)))


def fit_public_twin(spec, records, *, family=None, grid_points=9, refine_steps=12, ledger=None):
    from scipy.optimize import minimize_scalar
    family = family or MechanismFamily()
    if not records or any(not isinstance(r, PublicProbe) or r.metadata["purpose"] != "fit" for r in records):
        raise ValueError("only sealed fit probes may select mechanism parameters")
    free = [r for r in records if r.metadata["kind"] == "free"]
    if not free: raise ValueError("free excitation required for gain/delay identification")
    calls = []
    def loss(gain, delay, subset, stage):
        twin = PublicPhysicsTwin(spec, MechanismSpec(float(gain), int(delay), family.known_payload_mass), ledger=ledger)
        value = float(np.mean([probe_loss(r.observations, twin.predict(r)[0], free=r.metadata["kind"] == "free") for r in subset]))
        calls.append(dict(gain=float(gain), delay=int(delay), stage=stage, loss=value))
        return value
    grid = np.unique(np.r_[np.linspace(family.gain_min, family.gain_max, grid_points), 1.])
    candidates = []
    for delay in family.delays:
        values = [loss(g, delay, free, "free_grid") for g in grid]
        best = int(np.argmin(values)); gain = float(grid[best]); error = values[best]
        if refine_steps and error > 1e-14:
            result = minimize_scalar(lambda g: loss(g, delay, free, "free_refine"), method="bounded",
                bounds=(grid[max(0,best-1)], grid[min(len(grid)-1,best+1)]), options=dict(maxiter=refine_steps, xatol=1e-5))
            if result.fun < error: gain = float(result.x)
        joint = loss(gain, delay, records, "joint_fit_only")
        candidates.append(dict(gain=gain, delay=delay, loss=joint))
        print(json.dumps(dict(stage="public_identification", **candidates[-1])), flush=True)
    best = min(candidates, key=lambda row: (row["loss"], row["delay"], row["gain"]))
    mechanism = MechanismSpec(best["gain"], best["delay"], family.known_payload_mass)
    return mechanism, dict(fit_loss=best["loss"], profile=candidates, search=calls, family=family.__dict__,
        fit_probe_sha256=[r.digest for r in records], validation_used_for_selection=False,
        scope="gain/integer-delay family, known0.5kg nominal payload, cold-reset probes only; no global identifiability claim")


def validate_public_twin(spec, mechanism, records, *, thresholds, ledger=None):
    if not records or any(r.metadata["purpose"] != "validation" for r in records):
        raise ValueError("independent validation probes required")
    twin = PublicPhysicsTwin(spec, mechanism, ledger=ledger, role="model_validation")
    errors = []; complete = True
    for record in records:
        prediction, _ = twin.predict(record)
        p, y = public_response(prediction)[1:], public_response(record.observations)[1:]
        n = min(len(p), len(y)); complete &= len(p) == len(y) == 100
        errors.append(p[:n]-y[:n])
    error = np.concatenate(errors)
    metrics = dict(position_rmse_m=float(np.sqrt(np.mean(error[:, :3]**2))),
        rotation_matrix_rmse=float(np.sqrt(np.mean(error[:, 3:12]**2))),
        measured_force_rmse_N=float(np.sqrt(np.mean(error[:, 12:15]**2))),
        measured_torque_rmse_Nm=float(np.sqrt(np.mean(error[:, 15:18]**2))))
    return dict(**metrics, complete=complete, thresholds=thresholds,
        supported=bool(complete and all(metrics[key] <= limit for key, limit in thresholds.items())),
        validation_probe_sha256=[r.digest for r in records],
        fitting_fields="public TCP pose/twist and measured TCP wrench; substep force peaks and evaluator labels excluded")
