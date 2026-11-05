"""Versioned T1/T3 comparison contract, checked before running an admitted route."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .observation import FEATURE_VERSION, CONTEXT_DIM
from .specs import task_spec

PROTOCOL_VERSIONS = {
    "T1": "cross-embodiment-t1-t3-public-v3",
    "T3": "cross-embodiment-t3-public-v6",
}
PROTOCOL_FILENAMES = {
    "T1": "public_v3.json",
    "T3": "public_v6.json",
}
ROOT = Path(__file__).resolve().parents[3]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def contract_path(task):
    spec = task_spec(task)
    if spec.alias not in PROTOCOL_FILENAMES:
        raise ValueError("only T1/T3 are frozen")
    return ROOT / "configs/arm" / spec.name / "protocol" / PROTOCOL_FILENAMES[spec.alias]


def current_contract(task):
    spec = task_spec(task)
    if spec.alias not in {"T1", "T3"}: raise ValueError("only T1/T3 are frozen")
    if spec.alias == "T3":
        return {
            "version": PROTOCOL_VERSIONS["T3"],
            "extends": [{
                "path": "configs/arm/keyed_insertion_recovery/protocol/public_v5.json",
                "sha256": "f5c26c3e36d8216e26c1db5ff17a244b03b67ee130b511dcdb2cea433c57cf88",
            }],
            "shared_execution": {
                "version": "four-task-shared-execution-v3",
                "insertion_lead_z_m": 0.015,
                "retreat_release_version": "safe_settled_v1",
                "retreat_release_predicate": (
                    "Release after the existing height condition or after phase 3 is safe "
                    "and settled for the existing guard clock."
                ),
            },
            "scope": "Inherited unchanged by every main-table baseline, ablation and Cross-EBDM row.",
            "status": "common_protocol_bound_requires_fresh_predeployment",
            "final_test": False,
        }
    return dict(version=PROTOCOL_VERSIONS[spec.alias], task=spec.name, alias=spec.alias,
        physical_configs={robot: asdict(task_spec(task, robot=robot).config) for robot in ("panda", "xarm7")},
        public_schema=spec.describe()["observations"], excluded_policy_fields=spec.describe()["excluded_policy_fields"],
        shared_memory=dict(version=FEATURE_VERSION, dim=CONTEXT_DIM, clock="each observed20ms tick; T3 phase update every10ticks",
            fields=["issued_reference_relative_pose9", "previous_issued_action13", "phase_onehot4", "requested_mode2",
                    "tracking_integral6", "chunk_cursor1", "chunk_start_tcp_relative_pose9", "chunk_start_issued_reference_relative_pose9"],
            phase_geometry="known upper_clearance/side waypoints for T1; measured contact and known keyway for T3"),
        actions=dict(shared="bounded13D velocity/angular/log_stiffness/normal_force", acp="native19D REF9/VT9/positive scalar stiffness; original anisotropic SPD controller",
            force_off="source gap half-minimum-positive threshold is available to every method; continuous positive forces retained",
            chunks="native method horizon; causal memory observes every executed control tick"),
        evaluation=dict(cost="0 iff independently successful;1 for failure/rejection/incomplete", denominator="all requested episodes",
            terminal="unchanged scene deadline, geometric tolerances, force constraints and hold duration in physical_configs",
            dense_shaping="training only; excluded from paper cost", replay="every qualification episode; physics calls charged separately",
            mode="independently classified physical trace must match requested mode"),
        cells=dict(source=dict(robot="panda",servo_gain=1.,delay_steps=0,payload_mass=.5),
            development_targets=[dict(robot="xarm7",servo_gain=1.,delay_steps=0,payload_mass=.5),
                                 dict(robot="xarm7",servo_gain=.65,delay_steps=3,payload_mass=.8)],
            deferred=["UR5e until separate scene/source-to-target cell qualification", "T2", "T4"]),
        permissions={"A_fixed":"same charged short probe buffer; no complete target task training, target reward or successful target demonstrations; updates query only calibrated twin",
            "A_active":"method chooses short probes under the same count/duration/validation budget; report separately from shared-buffer ablations",
            "B_log":"same predeclared target logs and total executed transitions for every method; no target reward/model selection",
            "B_online":"same charged target task duration with reward; no extra expert demonstrations",
            "C_supervised":"same predeclared charged target calibration and validation episodes per method; no final-test selection"},
        development_budgets=dict(A_fixed=dict(probe_counts=[0,5,10,30],max_probe_seconds=2.,max_control_steps_per_probe=100,
                fit_validation_split={"0":[0,0],"5":[3,2],"10":[6,4],"30":[18,12]},
                held_out_physical_validation="included in the stated probe count; no free replacements or target full-task tuning"),
            B_log=dict(action_seconds=[30,60,120],reward=False,shared_buffer_control=True),
            B_online=dict(action_seconds=[30,60,120],reward=True,expert_demonstrations=False),
            C_supervised=dict(total_requested_target_demonstrations=[2,4,8],balanced_modes=True,
                validation="one complete demonstration per mode reserved within each total;2-demo point is validation-only, no supervised fit"),
            accounting="reset/preparation, failures, interrupted transitions, all real validation and exact replay separately reported and never rolled back",
            test="future sealed final-test cohort; these P3 development scenes/seeds never become independent final test",
            source_training="all pretraining, checkpoints, source physics and tuning attempts disclosed; source development is not target-data budget"),
        admission=dict(scope="per task/method/permission; development only, not final paper ranking",
            source_gate="4 fresh complete episodes, both modes twice;4/4 independent success and exact replay; no status mismatch",
            scene_gate="public expert reaches both modes of the named development cell; exact charged replay",
            method_gate="frozen checkpoint SHA, bound chart, batch1/4, continuation, evaluator/accounting checks",
            paper_gate="separate preregistered equal target budgets, independent final seeds, uncertainty and complete baseline training"))


def load_frozen_contract(task):
    path = contract_path(task)
    value, digest = load_contract_artifact(task, path.name)
    if canonical(value) != canonical(current_contract(task)):
        raise ValueError("physical/public contract drift: create a reviewed new version before running")
    return value, digest


def load_contract_artifact(task, filename):
    """Load one immutable historical contract without treating it as current."""
    spec = task_spec(task)
    path = ROOT / "configs/arm" / spec.name / "protocol" / filename
    value = json.loads(path.read_text())
    digest = hashlib.sha256(canonical(value)).hexdigest()
    if path.with_suffix(".sha256").read_text().strip() != digest:
        raise ValueError("frozen contract digest changed")
    parents = value.get("extends", [])
    if isinstance(parents, dict):
        parents = [parents]
    for parent in parents:
        parent_path = ROOT / parent["path"]
        parent_digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
        if parent_digest != parent["sha256"]:
            raise ValueError("frozen parent contract digest changed")
    return value, digest


def freeze_new_contract(task):
    path = contract_path(task)
    if path.exists() or path.with_suffix(".sha256").exists():
        raise FileExistsError("frozen contracts cannot be overwritten")
    value = current_contract(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    path.with_suffix(".sha256").write_text(hashlib.sha256(canonical(value)).hexdigest() + "\n")
    return load_frozen_contract(task)
