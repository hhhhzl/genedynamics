"""Episode-grouped development data at an explicit host artifact boundary.

This module schedules complete backend chunks. The numerical collector retains
its own batched implementation; this is not a per-world physics loop. Scripted
routes qualify data plumbing and never imply a learned source prior.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path

from .contracts import (
    ENVIRONMENT_NAME, ExecutionSpec, MechanismSpec, ObstacleTransferConfig,
    RandomRequest, SkillSpec, Status,
)
from .core.artifacts import load_bundle, save_bundle

DATASET_SCHEMA_VERSION = 1
ROUTE_CODES = {"upper": 0, "side": 1}
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
ROLES = frozenset({"source_demonstrations", "response_development", "response_validation"})
# Features available online without actual payload/gain/delay labels. Access
# through policy_features(), rather than handing a training batch to a policy.
POLICY_FEATURES = (
    "chunk.semantic.tcp_position", "chunk.semantic.tcp_rotation",
    "chunk.semantic.tcp_twist", "chunk.semantic.phase",
    "chunk.controller.reference_position", "chunk.controller.reference_rotation",
    "chunk.controller.command_buffer", "chunk.controller.last_command",
    "chunk.controller.integral_error", "task.goal_position", "task.goal_rotation",
)


def _identity(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "little")


@dataclass(frozen=True)
class EpisodeSpec:
    group_id: int
    sample_id: int
    reset_seed: int
    route: str
    split: str
    robot: str
    mechanism: MechanismSpec

    def __post_init__(self):
        RandomRequest(self.reset_seed, (self.sample_id,))
        if not isinstance(self.group_id, int) or not 0 <= self.group_id < 2**32:
            raise ValueError("group_id must be uint32")
        if self.route not in ROUTE_CODES or self.split not in SPLIT_CODES:
            raise ValueError("unknown episode route/split")
        if self.robot not in {"panda", "xarm7"} or not isinstance(self.mechanism, MechanismSpec):
            raise ValueError("invalid episode robot/mechanism")


def make_episode_plan(*, seed=0, episodes_per_route=2, robots=("panda",), mechanisms=(MechanismSpec(),)):
    """Pair route/robot/mechanism variants by group, then split whole groups.

    At least one group per route is held out. Identities are stable under batch
    reordering and microbatch changes. Reset RNG is (reset_seed, sample_id), not
    array position. Different backends use their documented RNG providers.
    """
    RandomRequest(seed, (0,))
    if isinstance(episodes_per_route, bool) or not isinstance(episodes_per_route, int) or episodes_per_route < 2:
        raise ValueError("at least two episodes per route are needed for train/validation")
    if not robots or len(set(robots)) != len(robots) or any(r not in {"panda", "xarm7"} for r in robots):
        raise ValueError("robots must be unique Panda/xArm7 names")
    if not mechanisms or not all(isinstance(m, MechanismSpec) for m in mechanisms):
        raise ValueError("mechanisms must be a nonempty sequence of MechanismSpec")
    # Stable hashed ordering gives reproducibility without an implicit NumPy RNG.
    ordered = sorted(range(episodes_per_route), key=lambda i: _identity(seed, "split", i))
    validation = set(ordered[:max(1, episodes_per_route // 4)])
    episodes = []
    for group in range(episodes_per_route):
        split = "validation" if group in validation else "train"
        for robot in robots:
            for route in ROUTE_CODES:
                for index, mechanism in enumerate(mechanisms):
                    episodes.append(EpisodeSpec(
                        _identity(seed, "group", group),
                        _identity(seed, "episode", group, robot, route, index),
                        seed, route, split, robot, mechanism,
                    ))
    _validate_plan(episodes)
    return tuple(episodes)


def _validate_plan(episodes):
    if not episodes or not all(isinstance(e, EpisodeSpec) for e in episodes):
        raise ValueError("episodes must be nonempty EpisodeSpec records")
    if len({e.sample_id for e in episodes}) != len(episodes):
        raise ValueError("episode sample IDs must be unique")
    group_splits = {}
    for episode in episodes:
        if group_splits.setdefault(episode.group_id, episode.split) != episode.split:
            raise ValueError("one episode group cannot cross dataset splits")
    if len({e.reset_seed for e in episodes}) != 1:
        raise ValueError("one fixed-shape collection uses one reset seed plus unique sample IDs")


def policy_features(tensors):
    """Explicit feature whitelist; mechanism truth and outcomes are excluded."""
    return {name: tensors[name] for name in POLICY_FEATURES}


def code_fingerprint():
    """Hash actual source files, including uncommitted edits; no git assumption."""
    root = Path(__file__).resolve().parents[3]
    roots = (root / "genedynamics/learning/cross_embodiment",
             root / "genedynamics/envs/domains/manipulation/cross_embodiment",
             root / "scripts/cross_embodiment")
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for base in roots for path in sorted(base.rglob("*")) if path.suffix in {".py",".cpp",".h"}}
    aggregate = "\n".join(f"{name}:{digest}" for name, digest in sorted(files.items()))
    return {"algorithm": "sha256", "aggregate": hashlib.sha256(aggregate.encode()).hexdigest(),
            "files": files}


def record_body_state(state):
    """Materialize native body state only at the data I/O boundary."""
    import numpy as np
    physics = state.physics
    if hasattr(physics, "qpos") and hasattr(physics, "qvel"):
        return {"qpos": np.asarray(physics.qpos), "qvel": np.asarray(physics.qvel)}
    if isinstance(physics, tuple) and physics and all(hasattr(x, "qpos") and hasattr(x, "qvel") for x in physics):
        return {"qpos": np.stack([x.qpos for x in physics]), "qvel": np.stack([x.qvel for x in physics])}
    raise TypeError("unknown physics state: provide an explicit state_record_factory")


def observed_route_modes(position, valid):
    """Geometric TCP route evidence, independent of requested script labels.

    -1 means no crossing evidence, 2 means both signatures occurred. This is a
    TCP trajectory classifier, not proof of arm/payload homotopy equivalence.
    """
    import numpy as np
    crossing = valid & (position[...,0] >= .465) & (position[...,0] <= .535)
    upper = (crossing & (position[...,2] >= .64)).any(axis=(1,2))
    side = (crossing & (position[...,1] <= -.15)).any(axis=(1,2))
    return np.where(upper & side, 2, np.where(upper, 0, np.where(side, 1, -1))).astype(np.int32)


def collect_dataset(episodes, *, role, execution=None, task=None, skill=None, chunk_length=10,
                    steps=None, output=None, runner_factory=None, observer_factory=None, command_factory=None,
                    state_record_factory=None):
    """Collect fixed K x L chunks and retain failures in the episode denominator.

    Optional factories allow cheap contract tests. Defaults instantiate actual
    robot execution and the shared semantic observer. All tensor conversion is
    at this host data boundary, outside compiled rollout kernels.
    """
    import numpy as np
    from .cross_embodiment import CrossEmbodiment
    from .reference import reference_commands
    from .source_query import SourceQueryAdapter
    episodes = tuple(episodes)
    _validate_plan(episodes)
    if role not in ROLES:
        raise ValueError("unknown dataset role")
    execution, task, skill = execution or ExecutionSpec(), task or ObstacleTransferConfig(), skill or SkillSpec()
    if execution.execution_mode != "collect":
        raise ValueError("dataset generation requires execution_mode='collect'")
    nominal = MechanismSpec(payload_mass=task.nominal_payload_mass)
    if role == "source_demonstrations" and any(e.robot != "panda" or e.mechanism != nominal for e in episodes):
        raise ValueError("source demonstrations require nominal Panda only")
    steps = task.timeout_steps if steps is None else steps
    if (isinstance(chunk_length, bool) or not isinstance(chunk_length, int) or chunk_length < 1
            or isinstance(steps, bool) or not isinstance(steps, int) or steps < 1
            or steps > task.timeout_steps or steps % chunk_length):
        raise ValueError("steps must be positive, <= task timeout and divisible by fixed chunk_length")
    runner_factory = runner_factory or CrossEmbodiment
    observer_factory = observer_factory or SourceQueryAdapter
    command_factory = command_factory or reference_commands
    state_record_factory = state_record_factory or record_body_state
    observer = observer_factory(execution=execution, task=replace(task, robot="panda"), skill=skill)
    pieces, provenance, sanitization = [], [], {}
    original_order = {episode.sample_id: index for index, episode in enumerate(episodes)}

    def portable(name, value):
        array = np.asarray(value)
        if array.dtype.kind not in "fiub":
            raise ValueError(f"non-numeric dataset field: {name}")
        finite = np.isfinite(array)
        count = int((~finite).sum())
        if count:
            sanitization[name] = sanitization.get(name, 0) + count
            array = np.nan_to_num(array, nan=0., posinf=0., neginf=0.)
        return array

    # Host grouping is by robot topology; mechanisms remain one numeric batch.
    for robot in dict.fromkeys(e.robot for e in episodes):
        group = tuple(e for e in episodes if e.robot == robot)
        robot_task = replace(task, robot=robot)
        runner = runner_factory(execution=execution, task=robot_task, skill=skill,
                                mechanism=tuple(e.mechanism for e in group))
        request = RandomRequest(group[0].reset_seed, tuple(e.sample_id for e in group))
        state = runner.reset_batch(request)
        runner.synchronize(state)
        commands = np.stack([command_factory(robot_task, skill, e.route, steps=steps) for e in group])
        if commands.shape != (len(group), steps, 13):
            raise ValueError("development command factory must return [steps,13]")
        chunks = {}
        initial_body = {name: portable(f"episode.initial_{name}", value)
                        for name,value in state_record_factory(state).items()}
        if set(initial_body) != {"qpos", "qvel"}:
            raise ValueError("body recorder must return qpos and qvel")
        for offset in range(0, steps, chunk_length):
            semantics = observer.observe(runner, state)
            runner.synchronize(semantics)
            features = {}
            for name, value in zip(semantics._fields, semantics):
                features[f"chunk.semantic.{name}"] = value
            for name, value in zip(state.controller._fields, state.controller):
                features[f"chunk.controller.{name}"] = value
            finite_features = np.ones(len(group), dtype=bool)
            for name, value in features.items():
                host = np.asarray(value)
                finite_features &= np.isfinite(host).reshape(len(group), -1).all(axis=1)
                chunks.setdefault(name, []).append(portable(name, host))
            for name, value in state_record_factory(state).items():
                key = f"chunk.body.{name}"
                host = np.asarray(value)
                finite_features &= np.isfinite(host).reshape(len(group), -1).all(axis=1)
                chunks.setdefault(key, []).append(portable(key, host))
            chunks.setdefault("chunk.feature_finite", []).append(finite_features)
            for name in ("steps", "hold_count", "status"):
                key = f"chunk.start_{name}"
                chunks.setdefault(key, []).append(portable(key, getattr(state, name)))
            raw = commands[:, offset:offset+chunk_length]
            result = runner.collect_chunk(state, raw)
            runner.synchronize(result)
            state = result.state
            chunks.setdefault("command.raw", []).append(portable("command.raw", raw))
            chunks.setdefault("command.raw_finite", []).append(np.isfinite(raw).all(axis=-1))
            trace_finite = np.ones((len(group), chunk_length), dtype=bool)
            for name, value in zip(result.trace._fields, result.trace):
                key, host = f"trace.{name}", np.asarray(value)
                trace_finite &= np.isfinite(host).reshape(len(group), chunk_length, -1).all(axis=-1)
                chunks.setdefault(key, []).append(portable(key, host))
            chunks.setdefault("trace.finite", []).append(trace_finite)
        arrays = {name: np.stack(values, axis=1) for name, values in chunks.items()}
        arrays["command.applied"] = arrays["trace.command"].copy()
        arrays["episode.final_status"] = portable("episode.final_status", state.status)
        arrays["episode.final_steps"] = portable("episode.final_steps", state.steps)
        arrays.update({f"episode.initial_{name}": value for name,value in initial_body.items()})
        arrays["episode.order"] = np.array([original_order[e.sample_id] for e in group], dtype=np.int32)
        pieces.append(arrays)
        provenance.append({"robot": robot, "sample_ids": list(request.sample_ids), "scene": runner.describe()})
    tensors = {name: np.concatenate([piece[name] for piece in pieces], axis=0) for name in pieces[0]}
    order = np.argsort(tensors.pop("episode.order"))
    tensors = {name: array[order] for name, array in tensors.items()}
    tensors.update({
        "episode.group_id": np.array([e.group_id for e in episodes], dtype=np.uint32),
        "episode.sample_id": np.array([e.sample_id for e in episodes], dtype=np.uint32),
        "episode.reset_seed": np.array([e.reset_seed for e in episodes], dtype=np.uint32),
        "episode.requested_route": np.array([ROUTE_CODES[e.route] for e in episodes], dtype=np.int32),
        "episode.split": np.array([SPLIT_CODES[e.split] for e in episodes], dtype=np.int32),
        "episode.robot": np.array([0 if e.robot == "panda" else 1 for e in episodes], dtype=np.int32),
        "diagnostic.mechanism_truth": np.array([[e.mechanism.servo_gain, e.mechanism.delay_steps,
                                                e.mechanism.payload_mass] for e in episodes], dtype=np.float32),
        "task.goal_position": np.broadcast_to(np.array(task.goal_position, dtype=np.float32), (len(episodes), 3)).copy(),
    })
    c, s = np.cos(task.goal_yaw), np.sin(task.goal_yaw)
    goal_rotation = np.array([[c,s,0.],[s,-c,0.],[0.,0.,-1.]], dtype=np.float32)
    tensors["task.goal_rotation"] = np.broadcast_to(goal_rotation, (len(episodes),3,3)).copy()
    success = tensors["episode.final_status"] == Status.SUCCESS
    tensors["episode.success"] = success
    tensors["episode.truncated"] = tensors["episode.final_status"] == Status.ACTIVE
    valid = tensors["trace.valid"].astype(bool)
    finite = tensors["trace.finite"] & tensors["command.raw_finite"] & tensors["chunk.feature_finite"][:,:,None]
    observed_mode = observed_route_modes(tensors["trace.tcp_position"], valid & tensors["trace.finite"])
    tensors["episode.observed_mode"] = observed_mode
    mode_matches = observed_mode == tensors["episode.requested_route"]
    eligible = success & mode_matches & (role == "source_demonstrations")
    for name,split in (("training","train"), ("validation","validation")):
        selected = eligible & (tensors["episode.split"] == SPLIT_CODES[split])
        tensors[f"prior.{name}_mask"] = valid & finite & selected[:,None,None]
    counts = {status.name: int((tensors["episode.final_status"] == status).sum()) for status in Status}
    metadata = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION, "environment": ENVIRONMENT_NAME, "role": role,
        "producer": "scripted_public_geometry_routes", "source_prior": "not_trained",
        "code_fingerprint": code_fingerprint(),
        "coverage": "two scripted route families with reset variation; no learned behavioral coverage claim",
        "execution": asdict(execution), "task": asdict(task), "skill": asdict(skill),
        "clocks": {"physics_dt": task.physics_dt, "control_dt": task.control_dt,
                   "H": skill.horizon, "L": chunk_length, "K": steps//chunk_length, "scheduled_steps": steps},
        "split_grouping": "episode.group_id; all route/robot/mechanism variants and all chunks share one split",
        "reset_rng": "backend provider keyed by episode.reset_seed and episode.sample_id; not batch position",
        "route_codes": ROUTE_CODES, "split_codes": SPLIT_CODES, "robot_codes": {"panda":0,"xarm7":1},
        "requested_route_semantics": "script request; actual mode is independently classified from valid TCP trace",
        "observed_mode_codes": {"unknown":-1,"upper":0,"side":1,"ambiguous":2},
        "observed_mode_classifier": {"version":1,"x_band":[.465,.535],"upper_z_min":.64,"side_y_max":-.15,
                                      "scope":"TCP crossing evidence; not full-arm homotopy proof"},
        "initial_state_replay": "episode.initial_qpos/qvel record actual reset; cross-backend RNG equality is not assumed",
        "policy_feature_whitelist": list(POLICY_FEATURES),
        "mechanism_truth_usage": "diagnostic/training label only; excluded from policy_features",
        "mechanism_truth_columns": ["servo_gain", "delay_steps", "payload_mass"],
        "provenance": provenance, "episodes_attempted": len(episodes), "episodes_successful": int(success.sum()),
        "episodes_failed": sum(count for name,count in counts.items() if name not in {"ACTIVE","SUCCESS"}),
        "episodes_truncated": counts["ACTIVE"], "status_counts": counts,
        "terminal_events": int(tensors["trace.terminal"].sum()), "valid_control_steps": int(valid.sum()),
        "nonfinite_values_sanitized": sanitization,
        "gpu_qualification": "not_run",
    }
    validate_dataset(tensors, metadata)
    if output is not None:
        save_bundle(Path(output), tensors, metadata)
    return tensors, metadata


def validate_dataset(tensors, metadata):
    """Reject accidental group leakage and privileged policy feature changes."""
    import numpy as np
    if metadata.get("dataset_schema_version") != DATASET_SCHEMA_VERSION or metadata.get("environment") != ENVIRONMENT_NAME:
        raise ValueError("unsupported dataset schema/environment")
    if metadata.get("role") not in ROLES or metadata.get("policy_feature_whitelist") != list(POLICY_FEATURES):
        raise ValueError("unsupported role or policy feature whitelist")
    policy_features(tensors)
    count = len(tensors["episode.sample_id"])
    if metadata.get("episodes_attempted") != count or any(array.ndim < 1 or array.shape[0] != count for array in tensors.values()):
        raise ValueError("dataset episode dimension/count mismatch")
    if len(set(map(int, tensors["episode.sample_id"]))) != count:
        raise ValueError("duplicate dataset episode identity")
    clocks = metadata["clocks"]
    shape = (count, clocks["K"], clocks["L"])
    for name in ("trace.valid", "trace.terminal", "trace.status", "trace.finite", "command.raw_finite",
                 "prior.training_mask", "prior.validation_mask"):
        if tensors[name].shape != shape:
            raise ValueError(f"invalid fixed chunk shape for {name}")
    for name in ("command.raw", "command.applied", "trace.command"):
        if tensors[name].shape != (*shape,13):
            raise ValueError(f"invalid skill tensor shape for {name}")
    if not np.array_equal(tensors["command.applied"], tensors["trace.command"]):
        raise ValueError("applied commands must match execution trace")
    group_splits = {}
    for group, split in zip(tensors["episode.group_id"], tensors["episode.split"]):
        if group_splits.setdefault(int(group), int(split)) != int(split):
            raise ValueError("episode group leaked across splits")
    if any(array.dtype.kind not in "fiub" or not np.isfinite(array).all() for array in tensors.values()):
        raise ValueError("dataset tensors must be finite and numeric")
    statuses = tensors["episode.final_status"]
    if not np.isin(statuses, [int(s) for s in Status]).all():
        raise ValueError("unknown final episode status")
    valid = tensors["trace.valid"].astype(bool)
    success, truncated = statuses == Status.SUCCESS, statuses == Status.ACTIVE
    if not np.array_equal(tensors["episode.success"], success) or not np.array_equal(tensors["episode.truncated"], truncated):
        raise ValueError("episode success/truncated flags disagree with final status")
    final_steps = tensors["episode.final_steps"]
    timeline = np.arange(clocks["K"] * clocks["L"])[None,:]
    expected_valid = timeline < final_steps[:,None]
    expected_terminal = (~truncated[:,None]) & (timeline == final_steps[:,None]-1)
    expected_status = np.where((~truncated[:,None]) & (timeline >= final_steps[:,None]-1), statuses[:,None], Status.ACTIVE)
    if (not np.array_equal(final_steps, valid.sum(axis=(1,2)))
            or not np.array_equal(valid.reshape(count,-1), expected_valid)
            or not np.array_equal(tensors["trace.terminal"].reshape(count,-1), expected_terminal)
            or not np.array_equal(tensors["trace.status"].reshape(count,-1), expected_status)
            or np.any((~truncated) & (final_steps < 1))):
        raise ValueError("invalid episode valid-prefix/terminal/sticky-status accounting")
    counts = {status.name: int((statuses == status).sum()) for status in Status}
    expected_counts = {
        "episodes_successful": int(success.sum()), "episodes_truncated": int(truncated.sum()),
        "episodes_failed": int((~(success | truncated)).sum()), "status_counts": counts,
        "terminal_events": int(expected_terminal.sum()), "valid_control_steps": int(valid.sum()),
    }
    if any(metadata.get(name) != value for name,value in expected_counts.items()):
        raise ValueError("dataset metadata denominator/counts disagree with recorded episodes")
    observed = observed_route_modes(tensors["trace.tcp_position"], valid & tensors["trace.finite"])
    if not np.array_equal(tensors["episode.observed_mode"], observed):
        raise ValueError("observed mode disagrees with actual trajectory classifier")
    source = metadata["role"] == "source_demonstrations"
    if source:
        nominal = np.array([1.,0.,metadata["task"]["nominal_payload_mass"]])
        if np.any(tensors["episode.robot"] != 0) or not np.allclose(tensors["diagnostic.mechanism_truth"], nominal):
            raise ValueError("source demonstrations require nominal Panda only")
    for name,split in (("training","train"), ("validation","validation")):
        eligible = (source & (statuses == Status.SUCCESS)
                    & (tensors["episode.split"] == SPLIT_CODES[split])
                    & (tensors["episode.observed_mode"] == tensors["episode.requested_route"]))
        allowed = valid & tensors["trace.finite"] & tensors["command.raw_finite"] & tensors["chunk.feature_finite"][:,:,None] & eligible[:,None,None]
        if not np.array_equal(tensors[f"prior.{name}_mask"], allowed):
            raise ValueError("source prior mask disagrees with episode/step eligibility")
    return tensors, metadata


def load_dataset(path):
    return validate_dataset(*load_bundle(path))
