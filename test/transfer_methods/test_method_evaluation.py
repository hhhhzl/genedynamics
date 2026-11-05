from copy import deepcopy
import json
from pathlib import Path

import pytest

from genedynamics.tasks.transfer.method_evaluation import (
    GPU1_UUID,
    INDEPENDENT_ACTOR_VERSION,
    SOURCE_POSITIVE_CONTROL_VERSION,
    VERSION,
    evaluation_plan,
    manifest_roster_config,
    run_evaluation,
    source_positive_control_plan,
)
from genedynamics.tasks.transfer.attribution_evaluation import _paired_contrasts, _request_strata


ROOT = Path(__file__).resolve().parents[2]


def _config():
    repair = json.loads((
        ROOT / "configs/arm/obstacle_transfer/training/fada_planner_repair_idm_pilot.json"
    ).read_text())
    return {
        "version": VERSION,
        "task": "T1",
        "method": "fada",
        "method_identity": "FADA_repair_development_pilot",
        "task_config": "configs/arm/obstacle_transfer/baseline/fada.yaml",
        "checkpoint": repair["source_checkpoint"],
        "execution": {
            "device": "cuda", "cuda_visible_devices": "1",
            "physical_uuid": GPU1_UUID,
        },
        "requests": [{
            "robot": "xarm7", "requested_mode": 0, "reset_seed": 991001,
            "physical_initial_slot": 0,
            "mechanism_parameters": {
                "delay_steps": 0, "payload_mass": 0.5, "servo_gain": 1.0,
            },
            "body_support_parameters": {},
        }],
        "output": "results/arm/obstacle_transfer/baseline/fada/validation/test_only",
        "plan_output": "results/_runtime/test_only/fada_plan.json",
    }


def test_development_plan_locks_checkpoint_task_and_gpu1_without_physics():
    plan = evaluation_plan(_config())
    assert plan["status"] == "planned_unexecuted"
    assert plan["task"] == "T1" and plan["method"] == "fada"
    assert plan["checkpoint"]["sha256"] == _config()["checkpoint"]["sha256"]
    assert plan["execution"]["physical_uuid"] == GPU1_UUID
    assert plan["requests"][0]["performance_evidence"] is False


def test_development_plan_rejects_duplicate_request_identity():
    config = _config()
    config["requests"].append(deepcopy(config["requests"][0]))
    with pytest.raises(ValueError, match="unique reset/slot/mode"):
        evaluation_plan(config)


def test_t2_public_geometry_uses_the_same_frozen_roster_without_a_checkpoint():
    config = json.loads((
        ROOT / "configs/arm/guided_contact_alignment/validation/"
        "public_geometry_fresh_actor_pilot.json"
    ).read_text())
    plan = evaluation_plan(config)
    assert plan["method"] == "public_geometry"
    assert "checkpoint" not in plan
    assert plan["actor_binding"]["key"] == "public_geometry"
    assert plan["request_roster"] == config["request_roster"]
    assert len(plan["requests"]) == 16
    assert plan["requests"][0]["request_id"] == "T2/ADAPTSIM-FRESH/source/s0/m0"
    assert plan["requests"][-1]["scene_cell"] == "payload_shift"

    altered = deepcopy(config)
    altered["request_roster"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="request roster SHA changed"):
        evaluation_plan(altered)

    altered = deepcopy(config)
    altered["checkpoint"] = {"path": "unused", "sha256": "0" * 64}
    with pytest.raises(ValueError, match="checkpoint-free"):
        evaluation_plan(altered)


def test_manifest_bound_development_requires_a_real_actor_instance():
    config = _config()
    config.update({
        "method": "task_only", "method_identity": "task_only",
        "task_config": "configs/arm/obstacle_transfer/baseline/task_only.yaml",
        "actor_binding_source": "task_config_manifest",
        "instance_id": "instance_01",
    })
    del config["checkpoint"]
    plan = evaluation_plan(config)
    assert plan["instance_id"] == "instance_01"
    assert "checkpoint" not in plan
    assert plan["actor_binding"]["key"] == "task_only"
    assert plan["actor_binding"]["instance"] == "instance_01"

    invalid = deepcopy(config)
    invalid["instance_id"] = "logical_copy_01"
    with pytest.raises(ValueError, match="instance_01 through instance_08"):
        evaluation_plan(invalid)

    rejected = deepcopy(config)
    rejected.update({
        "method": "acp", "method_identity": "ACP_task_adapted",
        "task_config": "configs/arm/obstacle_transfer/baseline/acp.yaml",
    })
    with pytest.raises(ValueError, match="manifest-bound actor"):
        evaluation_plan(rejected)


def test_independent_actor_plan_binds_one_of_eight_audited_checkpoints():
    config = json.loads((
        ROOT / "configs/arm/obstacle_transfer/validation/"
        "asid_independent_source_nominal_8801002.json"
    ).read_text())
    plan = evaluation_plan(config)
    assert plan["version"] == INDEPENDENT_ACTOR_VERSION
    assert plan["instance_id"] == "independent_8801002"
    assert plan["actor_binding"]["checkpoint"] == config["checkpoint"]
    assert len(plan["requests"]) == 8
    assert all(row["performance_evidence"] is False for row in plan["requests"])

    changed = deepcopy(config)
    changed["instance_id"] = "independent_8801003"
    with pytest.raises(ValueError, match="package identity differs"):
        evaluation_plan(changed)

    changed = deepcopy(config)
    changed["actor_package"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="package SHA changed"):
        evaluation_plan(changed)


@pytest.mark.parametrize("task_dir,task,roster_file", [
    ("arm/guided_contact_alignment", "T2", "method_source_nominal_actor_roster.json"),
    ("arm/keyed_insertion_recovery", "T3", "method_source_nominal_actor_roster.json"),
    ("arm/keyed_insertion_recovery", "T3", "method_source_nominal_actor_roster_native.json"),
    ("quadruped/mobile_keyed_insertion_recovery", "T4", "method_source_nominal_actor_roster_native.json"),
])
def test_compact_actor_roster_resolves_seven_real_task_actors_without_physics(task_dir, task, roster_file):
    roster_path = ROOT / "configs" / task_dir / "validation" / roster_file
    roster = json.loads(roster_path.read_text())
    plans = [evaluation_plan(manifest_roster_config(roster_path, method))
             for method in roster["methods"]]
    assert len(plans) == 7
    assert {plan["task"] for plan in plans} == {task}
    assert all(len(plan["requests"]) == 8 for plan in plans)
    assert all(plan["request_roster"]["path"] == str(roster_path.relative_to(ROOT))
               for plan in plans)
    assert all(row["performance_evidence"] is False
               for plan in plans for row in plan["requests"])
    assert {plan["instance_id"] for plan in plans} == {"instance_01", "fixed"}
    assert len({(row["reset_seed"], row["physical_initial_slot"], row["requested_mode"])
                for row in plans[0]["requests"]}) == 8


def test_t3_native_roster_rejects_wrong_library_before_any_physics(monkeypatch):
    roster_path = ROOT / "configs/arm/keyed_insertion_recovery/validation/method_source_nominal_actor_roster_native.json"
    config = manifest_roster_config(roster_path, "cross_ebdm")
    report_path = ROOT / config["output"] / "report.json"
    before = report_path.read_bytes() if report_path.exists() else None

    def forbidden(*_args, **_kwargs):
        pytest.fail("wrong native ABI path reached GPU or physics execution")

    monkeypatch.setattr("genedynamics.tasks.transfer.method_evaluation._gpu1", forbidden)
    monkeypatch.setattr("genedynamics.tasks.transfer.method_evaluation._run_plan", forbidden)
    monkeypatch.setenv("CROSS_EBDM_NATIVE_LIBRARY", "/tmp/wrong-native-library.so")
    with pytest.raises(ValueError, match="differs from the task ABI binding"):
        run_evaluation(config)
    assert (report_path.read_bytes() if report_path.exists() else None) == before


def test_fada_repaired_source_plan_locks_exact_cpu_checkpoint_without_physics():
    checkpoint = ROOT / (
        "results/arm/obstacle_transfer/baseline/fada/training/"
        "planner_repair_idm_target_budget_30s/instance_01/source_checkpoint.pt"
    )
    import hashlib

    requests = []
    for mode in (0, 1):
        for slot in range(4):
            requests.append({
                "robot": "panda", "requested_mode": mode,
                "reset_seed": 995100 + 4 * mode + slot,
                "physical_initial_slot": slot,
                "mechanism_parameters": {
                    "delay_steps": 0, "payload_mass": 0.5, "servo_gain": 1.0,
                },
                "body_support_parameters": {},
            })
    config = {
        "version": SOURCE_POSITIVE_CONTROL_VERSION,
        "task": "T1", "method": "fada",
        "method_identity": "FADA_repaired_source_positive_control",
        "task_config": "configs/arm/obstacle_transfer/baseline/fada.yaml",
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "execution": {"device": "cpu"}, "requests": requests,
        "output": "results/arm/obstacle_transfer/baseline/fada/validation/test_source",
    }
    plan = source_positive_control_plan(config)
    assert plan["actor_role"] == "repaired_source_policy"
    assert plan["checkpoint_schema"]["cmd_dim"] == 67
    assert plan["source_only"] and plan["target_physical_execution_count"] == 0
    assert {row["robot"] for row in plan["requests"]} == {"panda"}

    broken = deepcopy(config)
    broken["requests"][0]["robot"] = "xarm7"
    with pytest.raises(ValueError, match="nominal Panda"):
        source_positive_control_plan(broken)


def test_fada_qualified_source_reference_requires_bound_source_only_proof():
    config = json.loads((
        ROOT / "configs/arm/obstacle_transfer/validation/"
        "fada_qualified_source_reference_positive_control.json"
    ).read_text())
    plan = source_positive_control_plan(config)
    assert plan["actor_role"] == "qualified_source_reference"
    assert plan["checkpoint_schema"]["repair_phase"] is None
    assert plan["source_qualification"]["source_final_successes"] == 4
    assert plan["source_qualification"]["consecutive_successful_monitor_stages"] == 2
    assert plan["source_only"] and plan["target_physical_execution_count"] == 0

    broken = deepcopy(config)
    broken["qualification_report"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="qualification report SHA changed"):
        source_positive_control_plan(broken)


def test_four_cell_paired_contrast_preserves_instance_and_request_blocks():
    cells = ["mlp/task_only", "mlp/selective", "diffusion/task_only", "diffusion/selective"]
    requests = [{"request_id": f"block_{i}", "requested_mode": (i % 4) // 2,
                 "mechanism_parameters": {"delay_steps": 3 if i < 4 else 1},
                 "body_support_parameters": {"robot": "xarm7"}} for i in range(8)]
    plan = {"jobs": [{"instance": f"instance_{instance:02d}", "cell": cell}
                     for instance in range(1, 9) for cell in cells], "requests": requests,
            "bootstrap_seed": 12, "bootstrap_resamples": 200}
    rows = []
    for instance in range(1, 9):
        for request in requests:
            for cell in cells:
                selective = cell.endswith("selective")
                rows.append({"instance": f"instance_{instance:02d}",
                             "request_block_id": request["request_id"], "cell": cell,
                             "passed": selective,
                             "constraint_violation": False,
                             "capped_normalized_completion_time": .5 if selective else 1.})
    result = _paired_contrasts(plan, rows)
    success = result["metrics"]["constrained_success"]
    assert success["mlp_selective_minus_task_only"]["point"] == 1.
    assert success["mlp_selective_minus_task_only"]["percentile_95_interval"] == [1., 1.]
    assert success["backbone_by_selective_interaction"]["percentile_95_interval"] == [0., 0.]
    assert result["metrics"]["capped_normalized_completion_time"]["diffusion_selective_minus_task_only"]["point"] == -.5


def test_four_cell_bootstrap_fixes_mechanism_and_mode_composition():
    cells = ["mlp/task_only", "mlp/selective", "diffusion/task_only", "diffusion/selective"]
    requests = [{"request_id": f"block_{i}", "requested_mode": (i % 4) // 2,
                 "mechanism_parameters": {"delay_steps": 3 if i < 4 else 1},
                 "body_support_parameters": {"robot": "xarm7"}} for i in range(8)]
    plan = {"jobs": [{"instance": f"instance_{instance:02d}", "cell": cell}
                     for instance in range(1, 9) for cell in cells], "requests": requests,
            "bootstrap_seed": 23, "bootstrap_resamples": 1000}
    rows = [{"instance": f"instance_{instance:02d}", "cell": cell,
             "request_block_id": request["request_id"],
             "passed": cell.endswith("selective") and request["requested_mode"] == 1,
             "constraint_violation": False,
             "capped_normalized_completion_time": 1.}
            for instance in range(1, 9) for request in requests for cell in cells]
    contrast = _paired_contrasts(plan, rows)["metrics"]["constrained_success"]
    for name in ("mlp_selective_minus_task_only", "diffusion_selective_minus_task_only"):
        assert contrast[name]["point"] == .5
        assert contrast[name]["percentile_95_interval"] == [.5, .5]
    broken = deepcopy(requests)
    broken[3]["requested_mode"] = 0
    with pytest.raises(ValueError, match="two mechanisms by two modes"):
        _request_strata(broken)
