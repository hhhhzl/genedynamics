"""Regression gates for the four-task paper execution contract."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.experiments.plugins.methods.transfer.registry import (
    transfer_method_plugins,
)
from genedynamics.experiments.runner import _formal_readiness_errors
from genedynamics.tasks.transfer.evaluation import aggregate_episode_rows
from genedynamics.tasks.transfer.controllers.bindings import (
    create_transfer_actor,
    resolve_transfer_binding,
)
from genedynamics.tasks.transfer.confirmation import (
    CONFIRMATION_PACKAGE_VERSION,
    CONFIRMATION_REVISION,
    CONFIRMATION_VERSION,
    _audit_input_passes,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOTS = (
    ROOT / "configs/arm/obstacle_transfer",
    ROOT / "configs/arm/guided_contact_alignment",
    ROOT / "configs/arm/keyed_insertion_recovery",
    ROOT / "configs/quadruped/mobile_keyed_insertion_recovery",
)
MAIN_METHODS = {
    "cross_ebdm", "task_only", "fada", "acp", "human2sim2robot",
    "dial_transfer", "response_mpc", "public_geometry", "public_source_prior",
}
ABLATION_CELLS = {
    "frozen_warm", "task_only", "matching_no_band", "band_only",
    "ungated_full", "gated_full",
}


def test_confirmation_invalid_predecessor_requires_zero_reuse_invalid_state():
    valid = {
        "status": "invalid_incomplete_run",
        "performance_evidence_admitted": False,
        "reusable_records": 0,
        "failure_replacement": False,
    }
    assert _audit_input_passes("invalid_predecessor", valid)
    assert not _audit_input_passes(
        "invalid_predecessor", {**valid, "reusable_records": 1},
    )
    assert not _audit_input_passes(
        "invalid_predecessor", {**valid, "performance_evidence_admitted": True},
    )
    assert _audit_input_passes(
        "invalid_predecessor",
        {**valid, "status": "invalid_incomplete_qualification"},
    )
    assert _audit_input_passes("gpu_package_calibration", {"passed": True})
    assert not _audit_input_passes("gpu_package_calibration", valid)


def test_confirmation_seed_overlap_search_is_exact_and_targeted(tmp_path):
    script = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/prepare_confirmation.py"),
        run_name="cross_ebdm_prepare_confirmation_seed_test",
    )
    results = tmp_path / "results"
    results.mkdir()
    (results / "history.jsonl").write_text(
        '{"reset_seed": 9471000}\n'
        '{"confirmation_seed": 94710010}\n'
    )
    (results / "seed_9471001").mkdir()
    script["prior_seed_conflicts"].__globals__["ROOT"] = tmp_path

    assert script["prior_seed_conflicts"]({9471000, 9471001, 9471002}) == {
        9471000, 9471001,
    }


def test_confirmation_entrypoints_share_v14_protocol_identity():
    script = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/prepare_confirmation.py"),
        run_name="cross_ebdm_prepare_confirmation_identity_test",
    )

    assert CONFIRMATION_REVISION == 14
    assert CONFIRMATION_VERSION == "four-task-transfer-confirmation-v14"
    assert CONFIRMATION_PACKAGE_VERSION == (
        "four-task-transfer-confirmation-package-v14"
    )
    assert script["CONFIRMATION_REVISION"] == CONFIRMATION_REVISION
    assert script["PACKAGE_VERSION"] == CONFIRMATION_PACKAGE_VERSION
    assert script["CONFIRMATION_LABEL"] == "CONFIRMATION-v14"
    assert script["SEED_ANCHORS"] == {
        "T1": 9_681_000, "T2": 9_682_000,
        "T3": 9_683_000, "T4": 9_684_000,
    }


def test_mode_conditioned_t2_manifest_has_strict_parent_diff_audit():
    script = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/verify_shared_protocol.py"),
        run_name="cross_ebdm_mode_conditioned_manifest_audit_test",
    )
    manifest = json.loads((
        ROOT
        / "configs/arm/guided_contact_alignment/protocol/"
        "actor_bindings_mode_conditioned_contact_establishment_v2.json"
    ).read_text())

    result = script["verify_actor_revision"](manifest)

    check = result["mode_conditioned_contact_establishment_check"]
    assert check["passed"] is True
    assert check["affected_rows"] == [
        "main/cross_ebdm",
        "ablation/band_only",
        "ablation/frozen_warm",
        "ablation/gated_full",
        "ablation/matching_no_band",
        "ablation/task_only",
        "ablation/ungated_full",
    ]


@pytest.mark.parametrize(("task", "version"), [("T2", 16), ("T3", 10)])
def test_gpu0_predeployment_binding_reaches_every_physical_job(task, version):
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/qualify_method_behavior.py"),
        run_name=f"cross_ebdm_{task.lower()}_gpu0_plan_binding_test",
    )

    plan = runner["build_plan"]((task,), version)

    expected = {
        "device": "cuda", "cuda_visible_devices": "0", "physical_index": 0,
        "physical_uuid": "GPU-253f0813-4432-2d51-a149-3e73b38bf77a",
    }
    assert plan["execution"] == expected
    assert len(plan["jobs"]) == 720
    assert all(all(job[field] == value for field, value in expected.items())
               for job in plan["jobs"])


@pytest.mark.parametrize(("task", "version"), [("T2", 17), ("T3", 11)])
def test_gpu1_predeployment_binding_reaches_every_physical_job(task, version):
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/qualify_method_behavior.py"),
        run_name=f"cross_ebdm_{task.lower()}_gpu1_plan_binding_test",
    )

    plan = runner["build_plan"]((task,), version)

    expected = {
        "device": "cuda", "cuda_visible_devices": "1", "physical_index": 1,
        "physical_uuid": "GPU-58b54684-3577-11d3-f039-0e8469190739",
    }
    assert plan["execution"] == expected
    assert len(plan["jobs"]) == 720
    assert all(all(job[field] == value for field, value in expected.items())
               for job in plan["jobs"])


def test_execution_admission_retains_task_failures_for_final_comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/paper/cross_ebdm"))
    auditor = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/audit_behavioral_predeployment.py"),
        run_name="cross_ebdm_execution_admission_test",
    )
    report = {"rows": [
        {
            "row_id": "T2/main/public_geometry", "family": "main",
            "key": "public_geometry", "task": "T2", "physical_jobs": 1,
            "admission_status": "behavior_rejected",
            "final_for_current_protocol": True,
        },
        {
            "row_id": "T2/main/fada", "family": "main", "key": "fada",
            "task": "T2", "physical_jobs": 2,
            "admission_status": "behavior_rejected",
            "final_for_current_protocol": True,
        },
        {
            "row_id": "T2/main/acp", "family": "main", "key": "acp",
            "task": "T2", "physical_jobs": 0,
            "admission_status": "retained_algorithm_predeployment_rejection",
            "final_for_current_protocol": True,
        },
    ]}
    results = [
        {"record_id": "public", "row_id": "T2/main/public_geometry",
         "passed": False, "outcome": "overforce"},
        {"record_id": "fada-timeout", "row_id": "T2/main/fada",
         "passed": False, "outcome": "timeout"},
        {"record_id": "fada-force", "row_id": "T2/main/fada",
         "passed": False, "outcome": "overforce"},
    ]
    evidence = [{"record_id": row["record_id"], "passed": True} for row in results]

    admitted = auditor["execution_admission"](
        {}, report, results, evidence, integrity_passed=True,
    )

    rows = {row["key"]: row for row in admitted["rows"]}
    assert admitted["formal_evaluation_allowed"] is True
    assert rows["fada"]["admission_status"] == "behavior_qualified"
    assert rows["fada"]["task_performance_failures_retained"] == 2
    assert rows["fada"]["retained_outcomes"] == {"timeout": 1, "overforce": 1}
    assert rows["acp"]["admission_status"] == (
        "retained_algorithm_predeployment_rejection"
    )


def test_causal_reference_unloading_parameters_round_trip_through_paper_runner():
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/qualify_method_behavior.py"),
        run_name="cross_ebdm_qualify_method_behavior_contract_test",
    )
    interface = "public-T2-completion-budget-causal-reference-unloading-v17"
    parameters = {
        "completion_reserve_controls": 150,
        "reserve_taper_controls": 100,
        "reference_lag_full_authority_m": .008,
        "reference_lag_zero_authority_m": .012,
        "maximum_authority_decay_per_control": .05,
        "completion_progress_floor_mps": .020,
        "post_traverse_reserve_controls": 100,
        "completion_budget_taper_controls": 60,
        "force_speed_full_authority_mps": .010,
        "force_speed_zero_authority_mps": .025,
    }
    parse = runner["_task_actor_interface_parameters"]

    serialized = parse(interface, parameters)

    assert serialized == parameters
    assert parse(interface, serialized) == parameters
    accelerated = "public-T2-completion-budget-causal-traverse-v18"
    assert parse(accelerated, parameters) == parameters
    assert accelerated in runner["T2_DIAGNOSTIC_ACTOR_INTERFACES"]


def test_transfer_code_ownership_has_one_plugin_and_one_solver_tree():
    plugin_root = ROOT / "genedynamics/experiments/plugins/methods/transfer"
    assert {
        path.relative_to(plugin_root).as_posix()
        for path in plugin_root.rglob("*.py")
    } == {"__init__.py", "registry.py"}
    assert not (ROOT / "solvers").exists()
    forbidden_learning = {
        "cross_ebdm", "cross_embodiment", "baselines", "acp", "fada",
    }
    learning = ROOT / "genedynamics/learning"
    assert not {path.name for path in learning.iterdir()} & forbidden_learning


def test_frozen_method_and_ablation_matrix_matches_every_task():
    registered = {plugin.name for plugin in transfer_method_plugins()}
    assert registered == MAIN_METHODS | {"task_reference"}
    for root in TASK_ROOTS:
        configs = [
            ExperimentConfig.from_yaml(path)
            for category in ("main", "baseline")
            for path in (root / category).glob("*.yaml")
        ]
        assert {config.method for config in configs} == MAIN_METHODS
        cells = {
            ExperimentConfig.from_yaml(path).method_params["ablation_cell"]
            for path in (root / "ablation").glob("*.yaml")
        }
        cells.add(
            ExperimentConfig.from_yaml(
                root / "main/cross_ebdm.yaml"
            ).method_params["ablation_cell"]
        )
        cells.add(
            ExperimentConfig.from_yaml(
                root / "baseline/task_only.yaml"
            ).method_params["ablation_cell"]
        )
        assert cells == ABLATION_CELLS
        readiness = _formal_readiness_errors(
            ExperimentConfig.from_yaml(root / "main/cross_ebdm.yaml")
        )
        assert not any("frozen_design artifact" in error for error in readiness)


def test_every_method_inherits_one_task_owned_shared_execution_contract():
    expected = {
        "T2": {
            "version": "four-task-shared-execution-v7",
            "strict_evaluator_version": "strict-guide-substep-v4",
            "physics_dt_seconds": .001,
            "control_dt_seconds": .02,
            "contact_time_constant_seconds": .04,
            "contact_damping_ratio": 2.,
            "execution_profile": "substep_wrench_governor",
            "guide_mode0_transit_speed_mps": .00675,
            "guide_mode0_post_outlet_route": "concurrent_monotone_lift_v3",
            "guide_reference_admission": "response_normalized_completion",
            "guide_controller_version": "public-T2-response-normalized-hybrid-force-v50",
            "guide_route_version": "public-T2-response-normalized-guide-route-v4",
            "guide_clearance_route_version": "public-T2-response-normalized-clearance-traverse-route-v3",
            "terminal_actor_interface": "public-T2-completion-budget-causal-traverse-v18",
            "terminal_actor_interface_parameters": {
                "completion_reserve_controls": 150,
                "reserve_taper_controls": 100,
                "reference_lag_full_authority_m": .008,
                "reference_lag_zero_authority_m": .012,
                "maximum_authority_decay_per_control": .05,
                "completion_progress_floor_mps": .02,
                "post_traverse_reserve_controls": 100,
                "completion_budget_taper_controls": 60,
                "force_speed_full_authority_mps": .01,
                "force_speed_zero_authority_mps": .025,
            },
            "surface_force_target_N": 5.,
            "terminal_surface_force_target_N": 6.,
            "mode0_preoutlet_stiffness_ramp_m": .002,
            "mode0_transition_spring_budget_N": 8.8,
            "log_stiffness_limit": 1.5,
            "tangential_wrench_limit_N": 16.,
            "timeout_seconds": 20.,
            "terminal_position_tolerance_m": .002,
            "terminal_orientation_tolerance_rad": math.radians(2.),
            "terminal_dwell_controls": 15,
            "force_limit_N": 25.,
            "lateral_force_limit_N": 10.,
        },
        "T3": {
            "version": "four-task-shared-execution-v3",
            "insertion_lead_z_m": .015,
            "retreat_release_version": "safe_settled_v1",
        },
        "T4": {
            "version": "four-task-shared-execution-v3",
            "execution_profile": "joint_margin_public_reference",
            "reset_profile": "folded_stance",
            "reset_joint_noise_rad": .002,
            "body_intent_profile": "public_planar_v1",
            "body_velocity_world": [.005, .005, 0.],
            "body_intent_start_seconds": 0.,
            "body_intent_duration_seconds": 6.,
        },
    }
    for alias, root in zip(("T1", "T2", "T3", "T4"), TASK_ROOTS):
        configs = [
            ExperimentConfig.from_yaml(path)
            for category in ("main", "baseline", "ablation")
            for path in (root / category).glob("*.yaml")
        ]
        if alias == "T1":
            assert all("shared_execution" not in config.env_params for config in configs)
            continue
        assert configs
        assert all(
            config.env_params["shared_execution"] == expected[alias]
            for config in configs
        )
        pointers = {config.metadata["shared_task_contract"]["path"] for config in configs}
        hashes = {config.metadata["shared_task_contract"]["sha256"] for config in configs}
        assert len(pointers) == len(hashes) == 1
        path = ROOT / pointers.pop()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes.pop()


def test_shared_execution_rejects_drift_before_physics():
    from genedynamics.tasks.transfer.specs import (
        mechanism_spec, task_spec, validate_shared_execution,
    )

    t2 = task_spec("T2", config_overrides={
        "physics_dt": .001, "execution_profile": "substep_wrench_governor",
        "guide_mode0_transit_speed_mps": .00675,
        "guide_mode0_post_outlet_route": "concurrent_monotone_lift_v3",
        "guide_reference_admission": "response_normalized_completion",
        "log_stiffness_limit": 1.5,
        "tangential_wrench_limit": 16.,
    })
    values = {
        "version": "four-task-shared-execution-v7",
        "strict_evaluator_version": "strict-guide-substep-v4",
        "physics_dt_seconds": .001, "control_dt_seconds": .02,
        "contact_time_constant_seconds": .04, "contact_damping_ratio": 2.,
        "execution_profile": "substep_wrench_governor",
        "guide_mode0_transit_speed_mps": .00675,
        "guide_mode0_post_outlet_route": "concurrent_monotone_lift_v3",
        "guide_reference_admission": "response_normalized_completion",
        "guide_controller_version": "public-T2-response-normalized-hybrid-force-v50",
        "guide_route_version": "public-T2-response-normalized-guide-route-v4",
        "guide_clearance_route_version": "public-T2-response-normalized-clearance-traverse-route-v3",
        "terminal_actor_interface": "public-T2-completion-budget-causal-traverse-v18",
        "terminal_actor_interface_parameters": {
            "completion_reserve_controls": 150,
            "reserve_taper_controls": 100,
            "reference_lag_full_authority_m": .008,
            "reference_lag_zero_authority_m": .012,
            "maximum_authority_decay_per_control": .05,
            "completion_progress_floor_mps": .02,
            "post_traverse_reserve_controls": 100,
            "completion_budget_taper_controls": 60,
            "force_speed_full_authority_mps": .01,
            "force_speed_zero_authority_mps": .025,
        },
        "surface_force_target_N": 5.,
        "terminal_surface_force_target_N": 6.,
        "mode0_preoutlet_stiffness_ramp_m": .002,
        "mode0_transition_spring_budget_N": 8.8,
        "log_stiffness_limit": 1.5,
        "tangential_wrench_limit_N": 16.,
        "timeout_seconds": 20.,
        "terminal_position_tolerance_m": .002,
        "terminal_orientation_tolerance_rad": math.radians(2.),
        "terminal_dwell_controls": 15,
        "force_limit_N": 25.,
        "lateral_force_limit_N": 10.,
    }
    validate_shared_execution(
        t2, mechanism_spec("T2", {"contact_time_constant": .04}), values,
    )
    values["physics_dt_seconds"] = .002
    with pytest.raises(ValueError, match="constructed physics"):
        validate_shared_execution(
            t2, mechanism_spec("T2", {"contact_time_constant": .04}), values,
        )


def test_learned_actor_cannot_run_without_a_versioned_binding():
    with pytest.raises(ValueError, match="actor_binding.version"):
        create_transfer_actor(object(), "cross_ebdm", {})


def test_diagnostic_terminal_interface_replaces_shared_wrapper(monkeypatch):
    import genedynamics.tasks.transfer.controllers.bindings as bindings
    from genedynamics.tasks.transfer.controllers.guide_contact import (
        PublicTerminalOwnedGuideActor,
    )
    from genedynamics.tasks.transfer.specs import task_spec

    candidate = SimpleNamespace()
    monkeypatch.setattr(
        bindings, "_create_transfer_actor",
        lambda task, method, config: candidate,
    )
    task = SimpleNamespace(
        spec=task_spec("T2"),
        shared_execution={
            "terminal_actor_interface": (
                PublicTerminalOwnedGuideActor.CLEARANCE_PROGRESS_VERSION
            ),
        },
        _experiment_mode=1,
    )
    parameters = {
        "completion_reserve_controls": 150,
        "reserve_taper_controls": 100,
        "reference_lag_full_authority_m": .008,
        "reference_lag_zero_authority_m": .012,
        "maximum_authority_decay_per_control": .05,
    }
    actor = create_transfer_actor(task, "cross_ebdm", {
        "mode": 1,
        "task_actor_interface": (
            PublicTerminalOwnedGuideActor.HANDOFF_VIABILITY_VERSION
        ),
        "task_actor_interface_parameters": parameters,
    })
    assert isinstance(actor, PublicTerminalOwnedGuideActor)
    assert actor.candidate is candidate
    assert actor.version == PublicTerminalOwnedGuideActor.HANDOFF_VIABILITY_VERSION
    assert actor.handoff_parameters == parameters


def test_public_geometry_bypasses_shared_terminal_parameters(monkeypatch):
    import genedynamics.tasks.transfer.controllers.bindings as bindings
    from genedynamics.tasks.transfer.controllers.guide_contact import (
        PublicTerminalOwnedGuideActor,
    )
    from genedynamics.tasks.transfer.specs import task_spec

    sentinel = SimpleNamespace()
    monkeypatch.setattr(
        bindings, "_create_transfer_actor",
        lambda task, method, config: sentinel,
    )
    task = SimpleNamespace(
        spec=task_spec("T2"),
        shared_execution={
            "terminal_actor_interface": (
                PublicTerminalOwnedGuideActor.HANDOFF_VIABILITY_VERSION
            ),
            "terminal_actor_interface_parameters": {
                "completion_reserve_controls": 150,
                "reserve_taper_controls": 100,
                "reference_lag_full_authority_m": .008,
                "reference_lag_zero_authority_m": .012,
                "maximum_authority_decay_per_control": .05,
            },
        },
        _experiment_mode=1,
    )

    actor = create_transfer_actor(task, "public_geometry", {"mode": 1})

    assert actor is sentinel


def test_canonical_factory_accepts_stage_debt_viability_parameters(monkeypatch):
    import genedynamics.tasks.transfer.controllers.bindings as bindings
    from genedynamics.tasks.transfer.controllers.guide_contact import (
        PublicTerminalOwnedGuideActor,
    )
    from genedynamics.tasks.transfer.specs import task_spec

    candidate = SimpleNamespace()
    monkeypatch.setattr(
        bindings, "_create_transfer_actor",
        lambda task, method, config: candidate,
    )
    task = SimpleNamespace(
        spec=task_spec("T2"), shared_execution={}, _experiment_mode=1,
    )
    parameters = {
        "completion_reserve_controls": 150,
        "reserve_taper_controls": 100,
        "reference_lag_full_authority_m": .006,
        "reference_lag_zero_authority_m": .010,
        "maximum_authority_decay_per_control": .05,
        "clearance_debt_full_controls": 343,
        "clearance_debt_zero_controls": 371,
    }
    actor = create_transfer_actor(task, "cross_ebdm", {
        "mode": 1,
        "task_actor_interface": (
            PublicTerminalOwnedGuideActor.STAGE_DEBT_VIABILITY_VERSION
        ),
        "task_actor_interface_parameters": parameters,
    })
    assert actor.version == PublicTerminalOwnedGuideActor.STAGE_DEBT_VIABILITY_VERSION
    assert actor.handoff_parameters == parameters


def test_canonical_factory_accepts_completion_budget_parameters(monkeypatch):
    import genedynamics.tasks.transfer.controllers.bindings as bindings
    from genedynamics.tasks.transfer.controllers.guide_contact import (
        PublicTerminalOwnedGuideActor,
    )
    from genedynamics.tasks.transfer.specs import task_spec

    candidate = SimpleNamespace()
    monkeypatch.setattr(
        bindings, "_create_transfer_actor",
        lambda task, method, config: candidate,
    )
    task = SimpleNamespace(
        spec=task_spec("T2"), shared_execution={}, _experiment_mode=1,
    )
    parameters = {
        "completion_reserve_controls": 150,
        "reserve_taper_controls": 100,
        "reference_lag_full_authority_m": .008,
        "reference_lag_zero_authority_m": .012,
        "maximum_authority_decay_per_control": .05,
        "completion_progress_floor_mps": .020,
        "post_traverse_reserve_controls": 100,
        "completion_budget_taper_controls": 60,
        "force_speed_full_authority_mps": .010,
        "force_speed_zero_authority_mps": .025,
    }
    actor = create_transfer_actor(task, "cross_ebdm", {
        "mode": 1,
        "task_actor_interface": (
            PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_VIABILITY_VERSION
        ),
        "task_actor_interface_parameters": parameters,
    })
    assert actor.version == (
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_VIABILITY_VERSION
    )
    assert actor.handoff_parameters == parameters

    liveness_spec = task_spec("T2")
    liveness_task = SimpleNamespace(
        spec=replace(
            liveness_spec,
            config=replace(
                liveness_spec.config,
                guide_reference_admission="response_normalized_completion",
                log_stiffness_limit=1.5,
                tangential_wrench_limit=16.0,
            ),
        ),
        shared_execution={},
        _experiment_mode=1,
    )
    liveness = create_transfer_actor(liveness_task, "cross_ebdm", {
        "mode": 1,
        "task_actor_interface": (
            PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_LIVENESS_VERSION
        ),
        "task_actor_interface_parameters": parameters,
    })
    from genedynamics.tasks.transfer.controllers.guide_contact import (
        PublicCompletionBudgetCaptureBandGuideCompletionPolicy,
        PublicCompletionBudgetCaptureGuideCompletionPolicy,
        PublicCompletionBudgetGuideCompletionPolicy,
        PublicCompletionBudgetPlanarBrakingGuideCompletionPolicy,
        PublicCompletionBudgetPreDescentSettlingGuideCompletionPolicy,
        PublicCompletionBudgetReferenceUnloadingGuideCompletionPolicy,
        PublicCompletionBudgetReferenceDebtUnloadingGuideCompletionPolicy,
        PublicCompletionBudgetCausalReferenceUnloadingGuideCompletionPolicy,
        PublicCompletionBudgetCausalTraverseGuideCompletionPolicy,
    )
    assert liveness.version == (
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_LIVENESS_VERSION
    )
    assert isinstance(liveness.common, PublicCompletionBudgetGuideCompletionPolicy)
    assert liveness.common.CLEARANCE_TRAVERSE_REQUIRED_SPEED_MPS == .045
    capture = create_transfer_actor(liveness_task, "cross_ebdm", {
        "mode": 1,
        "task_actor_interface": (
            PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_CAPTURE_VERSION
        ),
        "task_actor_interface_parameters": parameters,
    })
    assert capture.version == (
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_CAPTURE_VERSION
    )
    assert isinstance(
        capture.common, PublicCompletionBudgetCaptureGuideCompletionPolicy,
    )
    assert capture.common.CAPTURE_REQUIRED_SPEED_MPS == .030
    capture_band = create_transfer_actor(liveness_task, "cross_ebdm", {
        "mode": 1,
        "task_actor_interface": (
            PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_CAPTURE_BAND_VERSION
        ),
        "task_actor_interface_parameters": parameters,
    })
    assert capture_band.version == (
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_CAPTURE_BAND_VERSION
    )
    assert isinstance(
        capture_band.common,
        PublicCompletionBudgetCaptureBandGuideCompletionPolicy,
    )
    assert capture_band.common.CAPTURE_BAND_M == .010
    planar_braking = create_transfer_actor(liveness_task, "cross_ebdm", {
        "mode": 1,
        "task_actor_interface": (
            PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_PLANAR_BRAKING_VERSION
        ),
        "task_actor_interface_parameters": parameters,
    })
    assert planar_braking.version == (
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_PLANAR_BRAKING_VERSION
    )
    assert isinstance(
        planar_braking.common,
        PublicCompletionBudgetPlanarBrakingGuideCompletionPolicy,
    )
    assert planar_braking.common.TERMINAL_PLANAR_SPEED_LIMIT_MPS == .010
    pre_descent_settling = create_transfer_actor(liveness_task, "cross_ebdm", {
        "mode": 1,
        "task_actor_interface": (
            PublicTerminalOwnedGuideActor.
            COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION
        ),
        "task_actor_interface_parameters": parameters,
    })
    assert pre_descent_settling.version == (
        PublicTerminalOwnedGuideActor.
        COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION
    )
    assert isinstance(
        pre_descent_settling.common,
        PublicCompletionBudgetPreDescentSettlingGuideCompletionPolicy,
    )
    assert pre_descent_settling.common.PRE_DESCENT_SETTLING_CONTROLS == 4
    reference_unloading = create_transfer_actor(liveness_task, "cross_ebdm", {
        "mode": 1,
        "task_actor_interface": (
            PublicTerminalOwnedGuideActor.
            COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION
        ),
        "task_actor_interface_parameters": parameters,
    })
    assert reference_unloading.version == (
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION
    )
    assert isinstance(
        reference_unloading.common,
        PublicCompletionBudgetReferenceUnloadingGuideCompletionPolicy,
    )
    assert (
        reference_unloading.common.
        PRE_DESCENT_REFERENCE_UNLOADING_LIMIT_MPS
        == .010
    )
    debt_unloading = create_transfer_actor(liveness_task, "cross_ebdm", {
        "mode": 1,
        "task_actor_interface": (
            PublicTerminalOwnedGuideActor.
            COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION
        ),
        "task_actor_interface_parameters": parameters,
    })
    assert debt_unloading.version == (
        PublicTerminalOwnedGuideActor.
        COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION
    )
    assert isinstance(
        debt_unloading.common,
        PublicCompletionBudgetReferenceDebtUnloadingGuideCompletionPolicy,
    )
    assert (
        debt_unloading.common.PRE_DESCENT_REFERENCE_UNLOADING_LIMIT_MPS
        == debt_unloading.common.MAXIMUM_CONTACT_SPEED_MPS
    )
    causal_unloading = create_transfer_actor(liveness_task, "cross_ebdm", {
        "mode": 1,
        "task_actor_interface": (
            PublicTerminalOwnedGuideActor.
            COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION
        ),
        "task_actor_interface_parameters": parameters,
    })
    assert causal_unloading.version == (
        PublicTerminalOwnedGuideActor.
        COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION
    )
    assert isinstance(
        causal_unloading.common,
        PublicCompletionBudgetCausalReferenceUnloadingGuideCompletionPolicy,
    )
    assert causal_unloading.common.PUBLIC_REFERENCE_DELAY_CANDIDATES == (
        1, 2, 3,
    )
    assert causal_unloading.common.PUBLIC_REFERENCE_DELAY_MINIMUM_SAMPLES == 8
    causal_traverse = create_transfer_actor(liveness_task, "cross_ebdm", {
        "mode": 1,
        "task_actor_interface": (
            PublicTerminalOwnedGuideActor.
            COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION
        ),
        "task_actor_interface_parameters": parameters,
    })
    assert causal_traverse.version == (
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION
    )
    assert isinstance(
        causal_traverse.common,
        PublicCompletionBudgetCausalTraverseGuideCompletionPolicy,
    )
    assert causal_traverse.common.CLEARANCE_TRAVERSE_REQUIRED_SPEED_MPS == .060
    assert causal_traverse.common.PUBLIC_REFERENCE_DELAY_CANDIDATES == (1, 2, 3)
    viability = create_transfer_actor(liveness_task, "cross_ebdm", {
        "mode": 1,
        "task_actor_interface": (
            PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_VIABILITY_VERSION
        ),
        "task_actor_interface_parameters": parameters,
    })
    assert viability.common.CLEARANCE_TRAVERSE_REQUIRED_SPEED_MPS == .030


def test_t2_public_geometry_manifest_binds_the_qualified_reference():
    config = ExperimentConfig.from_yaml(
        TASK_ROOTS[1] / "baseline/public_geometry.yaml"
    )
    binding = resolve_transfer_binding(
        config.method_params, task_alias="T2"
    )
    assert binding["reference_version"] == "public_finite_state_guide_contact_v2"


def test_t3_public_manifest_binds_the_confirmed_v21_transport_pair():
    transport = ExperimentConfig.from_yaml(
        TASK_ROOTS[2] / "main/cross_ebdm.yaml"
    )
    task_only = ExperimentConfig.from_yaml(
        TASK_ROOTS[2] / "baseline/task_only.yaml"
    )
    report_path = (
        ROOT / "results/arm/keyed_insertion_recovery/validation/"
        "finite_time_mode_specific_dual_physical_selected_transport_confirmation/"
        "gpu1/report.json"
    )
    report = json.loads(report_path.read_text())
    assert report["passed"] is True
    assert report["contribution_contrast"][
        "transport_claim_supported_confirmation"
    ] is True
    plan_pointer = report["artifacts"]["plan"]
    plan_path = ROOT / plan_pointer["path"]
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == plan_pointer["sha256"]
    plan = json.loads(plan_path.read_text())
    expected = {
        (job["instance"], int(job["request"]["requested_mode"])): job[
            "actor_binding"
        ]
        for job in plan["jobs"]
        if job["key"] == "mode_specific_physical_selected_transport"
    }
    assert len(expected) == 16

    for instance in (f"instance_{index:02d}" for index in range(1, 9)):
        paired = resolve_transfer_binding(
            {
                **task_only.method_params,
                "mode": 0,
                "actor_binding": {
                    **task_only.method_params["actor_binding"],
                    "family": "main",
                    "key": "task_only",
                    "instance": instance,
                },
            },
            task_alias="T3",
        )
        assert paired["method_identity"] == "task_only_shared_start_mode_specific_v9"
        for mode in (0, 1):
            resolved = resolve_transfer_binding(
                {
                    **transport.method_params,
                    "mode": mode,
                    "actor_binding": {
                        **transport.method_params["actor_binding"],
                        "family": "main",
                        "key": "cross_ebdm",
                        "instance": instance,
                    },
                },
                task_alias="T3",
            )
            source = expected[(instance, mode)]
            assert resolved["method_identity"] == (
                "ours_task_first_mode_specific_dual_physical_selected_v21"
            )
            assert resolved["task_dominance_version"] == (
                "public_mode_specific_recorded_completion_dominance_v15"
            )
            assert resolved["policy_checkpoint"] == source["policy_checkpoint"]
            assert resolved["admission_task_only_checkpoint"] == (
                source["admission_task_only_checkpoint"]
            )
            assert resolved["admission_task_only_checkpoint"] == paired[
                "policy_checkpoint"
            ]


def test_accessed_confirmation_cannot_be_relaunched_as_current_formal_protocol(
    tmp_path,
):
    config = ExperimentConfig.from_yaml(
        TASK_ROOTS[0] / "main/cross_ebdm.yaml"
    )
    errors = _formal_readiness_errors(config)
    assert not any("confirmation_accessed" in error for error in errors)
    assert not any("protocol_status" in error for error in errors)
    assert any("requires actor instance_01 through instance_08" in error for error in errors)
    assert not any("frozen_design artifact" in error for error in errors)

    archived = deepcopy(config)
    archived.metadata["frozen_design"]["confirmation_accessed"] = True
    archived.metadata["protocol_status"] = "development_training_pending"
    archived_errors = _formal_readiness_errors(archived)
    assert any("confirmation_accessed" in error for error in archived_errors)
    assert any("development_training_pending" in error for error in archived_errors)

    config.use_development_output_root(tmp_path)
    assert any(
        "requires actor instance_01 through instance_08" in error
        for error in _formal_readiness_errors(config)
    )
    public = ExperimentConfig.from_yaml(
        TASK_ROOTS[0] / "baseline/public_geometry.yaml"
    )
    public.use_development_output_root(tmp_path)
    assert not any(
        "confirmation_accessed" in error
        for error in _formal_readiness_errors(public)
    )
    public.metadata["frozen_design"]["confirmation_accessed"] = True
    assert any(
        "confirmation_accessed" in error
        for error in _formal_readiness_errors(public)
    )
    # Explicit development configs remain usable; isolation alone cannot
    # relabel the accessed confirmation cohort as fresh formal evidence.
    development = ExperimentConfig.from_yaml(
        TASK_ROOTS[2] / "validation/mechanism_factorial.yaml"
    )
    assert _formal_readiness_errors(development) == []


def test_frozen_evidence_sha_is_enforced():
    config = ExperimentConfig.from_yaml(
        TASK_ROOTS[0] / "main/cross_ebdm.yaml"
    )
    config.metadata = deepcopy(config.metadata)
    config.metadata["frozen_design"]["artifacts"]["protocol_validity"]["sha256"] = "0" * 64
    assert any(
        "protocol_validity" in error for error in _formal_readiness_errors(config)
    )


def test_common_metric_aggregation_keeps_failures_and_groups():
    rows = [
        {
            "success": True, "constraint_violation": False,
            "predeployment_rejection": False, "group": "mode_0",
            "capped_normalized_completion_time": 0.25, "task_cost": 0.0,
            "status_mismatches": 0, "outcome": "success",
        },
        {
            "success": False, "constraint_violation": True,
            "predeployment_rejection": False, "group": "mode_1",
            "capped_normalized_completion_time": 1.0, "task_cost": 1.0,
            "status_mismatches": 0, "outcome": "collision",
        },
    ]
    report = aggregate_episode_rows(rows, scope="fixture")
    assert report["requested_episodes"] == 2
    assert report["constrained_success_rate"] == 0.5
    assert report["constraint_violation_rate"] == 0.5
    assert report["capped_normalized_completion_time"] == 0.625
    assert report["worst_group_constrained_success"] == 0.0


def test_t2_handoff_attribution_decision_requires_exact_common_failure_inside_envelope():
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/run_t2_handoff_attribution.py"),
        run_name="cross_ebdm_t2_handoff_attribution_contract_test",
    )
    classify = runner["classify"]

    def branch(outcome, *, audited=True):
        return {
            "outcome": outcome,
            "success": outcome == "success",
            "evidence": {"audit_passed": audited},
        }

    assert classify("success", branch("success"), branch("success"), True) == (
        "successful_control_preserved"
    )
    assert classify("timeout", branch("timeout"), branch("success"), True) == (
        "method_or_interface_failure"
    )
    assert classify("timeout", branch("timeout"), branch("timeout"), True) == (
        "shared_completion_defect_supported"
    )
    assert classify("timeout", branch("timeout"), branch("timeout"), False) == (
        "inadmissible_method_handoff"
    )
    assert classify("timeout", branch("success"), branch("timeout"), True) == (
        "inconclusive"
    )

    digest = runner["restorable_task_digest"]
    checkpoint = {
        "schema_version": 1,
        "integration": [1., 2.],
        "episodes": [{"state": "active"}],
        "lifetime_accounting_at_export": {"control_steps": 10},
    }
    charged_branch = deepcopy(checkpoint)
    charged_branch["lifetime_accounting_at_export"]["control_steps"] = 20
    assert digest(charged_branch) == digest(checkpoint)
    charged_branch["integration"][0] = 3.
    assert digest(charged_branch) != digest(checkpoint)
    stable_digest = runner["stable_digest"]
    assert stable_digest({"utilization": 0.25}) == stable_digest({
        "utilization": __import__("numpy").float64(0.25),
    })


def test_t3_contribution_gate_uses_only_exact_paired_matched_control_effect():
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/qualify_method_behavior.py"),
        run_name="cross_ebdm_t3_contribution_contract_test",
    )
    contrast = runner["_paired_contribution_contrast"]
    acceptance = {
        "transport": "cross_ebdm_matched",
        "task_only": "task_only_matched",
        "required_pairs": 8,
        "minimum_aggregate_success_rate_gain": .05,
        "minimum_nonnegative_instances": 6,
        "require_zero_extra_transport_constraint_violations": True,
    }
    results = []
    for index in range(1, 9):
        instance = f"instance_{index:02d}"
        for key in ("cross_ebdm_matched", "task_only_matched"):
            passed = key == "cross_ebdm_matched" or index != 1
            results.append({
                "key": key,
                "instance": instance,
                "request": {"request_id": "paired_00"},
                "passed": passed,
                "outcome": "success" if passed else "timeout",
                "constraint_violation": False,
                "physical_control_steps": 100,
                "evidence": {"audit_passed": True},
            })

    report = contrast({"acceptance": acceptance}, results)

    assert report["execution_complete"] is True
    assert report["integrity_passed"] is True
    assert report["transport_minus_task_only_success_rate"] == pytest.approx(.125)
    assert report["transport_claim_supported_development"] is True
    assert contrast({"acceptance": acceptance}, results[:-1]) == {
        "execution_complete": False,
        "transport_claim_supported_development": False,
        "reason": "paired denominator incomplete",
    }


def test_t3_same_start_variants_only_change_response_anchoring_weight():
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/train_t4_completion.py"),
        run_name="cross_ebdm_t3_same_start_training_contract_test",
    )
    variant = runner["training_variant"]
    initialization = {
        "kind": "recentered_current_checkpoint_fresh_optimizer",
    }
    anchored = variant({
        "training": {
            "variant": "same_start_anchored",
            "initialization": initialization,
        },
    })
    task_only = variant({
        "training": {
            "variant": "same_start_task_only",
            "initialization": initialization,
        },
    })

    assert anchored == {
        "name": "same_start_anchored",
        "initialization_policy": "task_only",
        "response_anchoring_weight": 1.0,
    }
    assert task_only == {
        "name": "same_start_task_only",
        "initialization_policy": "task_only",
        "response_anchoring_weight": 0.0,
    }

    relative_anchored = variant({
        "training": {
            "variant": "same_start_relative_anchored",
            "initialization": initialization,
        },
    })
    relative_task_only = variant({
        "training": {
            "variant": "same_start_relative_task_only",
            "initialization": initialization,
        },
    })
    assert relative_anchored == {
        "name": "same_start_relative_anchored",
        "initialization_policy": "task_only",
        "response_anchoring_weight": 1.0,
    }
    assert relative_task_only == {
        "name": "same_start_relative_task_only",
        "initialization_policy": "task_only",
        "response_anchoring_weight": 0.0,
    }

    directional_anchored = variant({
        "training": {
            "variant": "task_first_directional_anchored",
            "initialization": initialization,
        },
    })
    directional_task_only = variant({
        "training": {
            "variant": "task_first_directional_task_only",
            "initialization": initialization,
        },
    })
    assert directional_anchored == {
        "name": "task_first_directional_anchored",
        "initialization_policy": "task_only",
        "response_anchoring_weight": 1.0,
    }
    assert directional_task_only == {
        "name": "task_first_directional_task_only",
        "initialization_policy": "task_only",
        "response_anchoring_weight": 0.0,
    }
    margin_anchored = variant({
        "training": {
            "variant": "task_first_margin_anchored",
            "initialization": initialization,
        },
    })
    margin_task_only = variant({
        "training": {
            "variant": "task_first_margin_task_only",
            "initialization": initialization,
        },
    })
    assert margin_anchored["response_anchoring_weight"] == 1.0
    assert margin_task_only["response_anchoring_weight"] == 0.0
    assert margin_anchored["initialization_policy"] == "task_only"
    assert margin_task_only["initialization_policy"] == "task_only"
    stable_margin = variant({
        "training": {
            "variant": "task_first_stable_margin_anchored",
            "initialization": initialization,
        },
    })
    assert stable_margin == {
        "name": "task_first_stable_margin_anchored",
        "initialization_policy": "task_only",
        "response_anchoring_weight": 1.0,
    }


def test_task_first_selector_requires_each_frozen_transfer_mode():
    from genedynamics.solvers.single.cross_ebdm.learning.gated_transport import (
        PublicFiniteTimeProgressSpec,
    )

    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/train_t4_completion.py"),
        run_name="cross_ebdm_mode_stratified_selector_contract_test",
    )
    select = runner["progress_selection_decision"]
    config = {
        "selection": {
            "required_modes": [0, 1],
            "task_tolerance": .05,
            "progress_tolerance": 1e-7,
            "minimum_useful_fraction": .25,
        },
    }
    good = {
        "task_proxy": 1.0,
        "linear_deficit_p95": 0.0,
        "angular_deficit_p95": 0.0,
        "useful_fraction": 1.0,
    }
    reference = {**good, "task_proxy": 1.0}
    regressed = {**good, "task_proxy": 1.051}

    decision = select(
        config, PublicFiniteTimeProgressSpec(), good, reference,
        {"0": good, "1": regressed},
        {"0": reference, "1": reference},
        candidate="transport", task_only="task-only",
    )

    assert decision["per_mode"]["0"]["accepted"] is True
    assert decision["per_mode"]["1"]["accepted"] is False
    assert decision["accepted"] is False
    assert decision["selected"] == "task-only"


def test_all_paired_training_variants_share_the_task_only_start():
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/train_t4_completion.py"),
        run_name="cross_ebdm_shared_start_dispatch_contract_test",
    )
    uses_task_only_start = runner["uses_task_only_start"]

    assert all(uses_task_only_start(name) for name in (
        "task_first_directional_anchored",
        "task_first_directional_task_only",
        "task_first_margin_anchored",
        "task_first_margin_task_only",
        "task_first_stable_margin_anchored",
    ))
    assert not uses_task_only_start("transport")


def test_paired_checkpoint_score_uses_the_worst_held_out_mode_tail():
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/train_t4_completion.py"),
        run_name="cross_ebdm_paired_checkpoint_score_contract_test",
    )
    score = runner["checkpoint_paired_progress_score"]

    assert score(
        {"normalized_gain_p05": .4},
        {
            "0": {"normalized_gain_p05": .3},
            "1": {"normalized_gain_p05": -.2},
        },
    ) == pytest.approx(.2)


def test_relative_response_matching_is_dimensionless_and_detaches_control_reference():
    import torch
    from genedynamics.solvers.single.cross_ebdm.learning.gated_transport import (
        PublicRelativeResponseSpec,
        relative_response_matching,
    )

    matching = torch.tensor([.004, .00005], requires_grad=True)
    reference = torch.tensor([.002, .00001], requires_grad=True)
    actual = relative_response_matching(
        matching, reference, PublicRelativeResponseSpec(reference_floor=.0001),
    )

    torch.testing.assert_close(actual, torch.tensor([2., .5]))
    actual.sum().backward()
    torch.testing.assert_close(matching.grad, torch.tensor([500., 10000.]))
    assert reference.grad is None


def test_directional_response_spec_rejects_negative_source_task_advantage():
    from genedynamics.solvers.single.cross_ebdm.learning.gated_transport import (
        PublicDirectionalResponseSpec,
    )

    assert PublicDirectionalResponseSpec(
        reference_floor=.0001, minimum_source_task_advantage=0.0,
    ).minimum_source_task_advantage == 0.0
    with pytest.raises(ValueError):
        PublicDirectionalResponseSpec(minimum_source_task_advantage=-1e-6)
    with pytest.raises(ValueError):
        PublicDirectionalResponseSpec(source_task_state="unregistered")


def test_directional_source_task_gate_uses_explicit_target_state_counterfactual():
    import torch
    from genedynamics.solvers.single.cross_ebdm.learning.gated_transport import (
        PublicDirectionalResponseSpec, directional_source_task_advantage,
    )

    class Response:
        def __call__(self, features, commands):
            prediction = torch.zeros((1, len(features), 20))
            prediction[0, :, 0] = features[:, 0]
            return prediction

    batch = {
        "source_features": torch.zeros((1, 1)),
        "target_features": torch.ones((1, 1)),
        "source_commands": torch.zeros((1, 13)),
        "current": {
            "position": torch.zeros((1, 3)),
            "rotation": torch.eye(3).reshape(1, 3, 3),
            "wrench": torch.zeros((1, 6)),
        },
        "task_reference": torch.cat((
            torch.tensor([[.01, 0., 0.]]),
            torch.eye(3).reshape(1, 9), torch.zeros((1, 1)),
        ), dim=1),
    }
    common = {"response": Response(), "batch": batch,
              "task_only_reference": torch.tensor([.1])}
    historical = directional_source_task_advantage(
        **common, spec=PublicDirectionalResponseSpec(),
    )
    matched = directional_source_task_advantage(
        **common, spec=PublicDirectionalResponseSpec(source_task_state="target"),
    )
    assert historical.item() < 0 < matched.item()


def test_ensemble_robust_task_optimizes_the_worst_same_member_gap():
    import torch
    from genedynamics.solvers.single.cross_ebdm.learning.gated_transport import (
        PublicEnsembleRobustTaskSpec,
        ensemble_robust_paired_task,
    )

    candidate = torch.tensor([
        [0.8, 1.1],
        [1.2, 0.7],
        [0.9, 0.9],
    ], requires_grad=True)
    task_only = torch.ones_like(candidate, requires_grad=True)
    gap, worst = ensemble_robust_paired_task(
        candidate, task_only,
        PublicEnsembleRobustTaskSpec(reduction="worst_paired_member"),
    )

    torch.testing.assert_close(gap, candidate.detach() - 1.)
    torch.testing.assert_close(worst, torch.tensor([0.2, 0.1]))
    worst.sum().backward()
    torch.testing.assert_close(candidate.grad, torch.tensor([
        [0., 1.],
        [1., 0.],
        [0., 0.],
    ]))
    assert task_only.grad is None


def test_t3_relative_response_plan_uses_frozen_selection_and_fallback_outputs():
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/qualify_method_behavior.py"),
        run_name="cross_ebdm_relative_response_plan_contract_test",
    )
    plan = runner["build_diagnostic_plan"](
        ROOT / "configs/arm/keyed_insertion_recovery/validation/finite_time_relative_response_contrast.json"
    )

    assert plan["counts"]["physical_jobs"] == 256
    rows = {row["key"]: row for row in plan["rows"]}
    transport = rows["relative_response_transport"]["candidate_policy_set"]
    task_only = rows["relative_response_task_only"]["candidate_policy_set"]
    assert [
        name for name, row in transport["selection_proof"].items()
        if row["fallback"]
    ] == ["instance_03"]
    assert all(not row["fallback"] for row in task_only["selection_proof"].values())
    assert transport["pairing_proof"] == task_only["pairing_proof"]
    assert plan["policy"]["same_start_relative_response_is_only_training_factor"]


def test_t3_task_first_directional_plan_is_exactly_paired_and_mode_selected():
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/qualify_method_behavior.py"),
        run_name="cross_ebdm_task_first_directional_plan_contract_test",
    )
    plan = runner["build_diagnostic_plan"](
        ROOT / "configs/arm/keyed_insertion_recovery/validation/finite_time_task_first_directional_contrast.json"
    )

    assert plan["counts"]["physical_jobs"] == 256
    rows = {row["key"]: row for row in plan["rows"]}
    transport = rows["directional_transport"]["candidate_policy_set"]
    task_only = rows["directional_task_only"]["candidate_policy_set"]
    assert transport["pairing_proof"] == task_only["pairing_proof"]
    assert transport["resolved_shared_start_report"] == task_only[
        "resolved_shared_start_report"
    ]
    assert all(row["accepted"] for row in transport["selection_proof"].values())
    assert all(row["accepted"] for row in task_only["selection_proof"].values())
    assert plan["policy"][
        "task_first_directional_response_is_only_training_factor"
    ]


def test_t3_same_state_directional_plan_rejects_unmatched_task_only(tmp_path):
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/qualify_method_behavior.py"),
        run_name="cross_ebdm_same_state_directional_plan_contract_test",
    )
    path = ROOT / "configs/arm/keyed_insertion_recovery/validation/finite_time_task_first_directional_same_state_contrast.json"
    plan = runner["build_diagnostic_plan"](path)
    assert plan["counts"]["physical_jobs"] == 256
    transport, task_only = (row["candidate_policy_set"] for row in plan["rows"])
    assert transport["pairing_proof"] == task_only["pairing_proof"]
    assert transport["data_sha256"] == task_only["data_sha256"]
    assert all(row["accepted"] for row in transport["selection_proof"].values())
    assert all(row["accepted"] for row in task_only["selection_proof"].values())
    assert plan["policy"]["directional_source_task_uses_target_state"]

    changed = json.loads(path.read_text())
    changed["methods"][1]["candidate_policy_set"] = {
        "method_identity": "task_first_directional_task_only_v1",
        "training_report": {
            "path": "results/arm/keyed_insertion_recovery/training/finite_time_task_first_directional/task_only/checkpoints/report.json",
            "sha256": "20ac36a124b46479e6cb35344b0e618780c7b555ca55fd803b5c0e44e969e6aa",
        },
    }
    old_control = tmp_path / "unmatched_directional_contrast.json"
    old_control.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="directional response gate"):
        runner["build_diagnostic_plan"](old_control)


def test_t3_runtime_dominance_plan_binds_each_paired_task_only_checkpoint():
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/qualify_method_behavior.py"),
        run_name="cross_ebdm_runtime_dominance_plan_contract_test",
    )
    plan = runner["build_diagnostic_plan"](
        ROOT / "configs/arm/keyed_insertion_recovery/validation/finite_time_task_first_runtime_dominance_contrast.json"
    )
    rows = {row["key"]: row for row in plan["rows"]}
    expected = rows["runtime_task_only"]["candidate_policy_set"]["checkpoints"]
    transport_jobs = [
        job for job in plan["jobs"] if job["key"] == "runtime_transport"
    ]

    assert len(transport_jobs) == 128
    assert plan["policy"]["runtime_task_dominance_is_only_execution_factor"]
    assert all(
        job["actor_binding"]["task_dominance_version"]
        == "public_transport_over_task_only_dominance_v1"
        and job["actor_binding"]["admission_task_only_checkpoint"]
        == expected[job["instance"]]
        for job in transport_jobs
    )

    full = runner["build_diagnostic_plan"](
        ROOT / "configs/arm/keyed_insertion_recovery/validation/finite_time_task_first_full_runtime_dominance_contrast.json"
    )
    full_jobs = [
        job for job in full["jobs"] if job["key"] == "full_runtime_transport"
    ]
    assert len(full_jobs) == 128
    assert full["policy"][
        "full_runtime_task_dominance_is_only_execution_factor"
    ]
    assert all(
        job["actor_binding"]["task_dominance_version"]
        == "public_full_transport_over_task_only_dominance_v2"
        for job in full_jobs
    )

    margin = runner["build_diagnostic_plan"](
        ROOT / "configs/arm/keyed_insertion_recovery/validation/finite_time_task_first_margin_runtime_contrast.json"
    )
    margin_rows = {row["key"]: row for row in margin["rows"]}
    assert margin_rows["margin_runtime_transport"]["candidate_policy_set"][
        "paired_progress"
    ] == {"linear_margin": 1e-5, "angular_margin": 1e-4, "weight": 1.0}
    assert margin_rows["margin_runtime_task_only"]["candidate_policy_set"][
        "paired_progress"
    ] is None
    assert margin["policy"][
        "margin_training_and_runtime_dominance_define_transport"
    ]

    adaptive = runner["build_diagnostic_plan"](
        ROOT / "configs/arm/keyed_insertion_recovery/validation/finite_time_task_first_adaptive_step_contrast.json"
    )
    adaptive_jobs = [
        job for job in adaptive["jobs"]
        if job["key"] == "adaptive_step_transport"
    ]
    assert len(adaptive_jobs) == 128
    assert adaptive["policy"][
        "adaptive_step_transport_is_only_execution_change"
    ]
    assert all(
        job["actor_binding"]["task_dominance_version"]
        == "public_adaptive_step_transport_dominance_v3"
        for job in adaptive_jobs
    )

    stable = runner["build_diagnostic_plan"](
        ROOT / "configs/arm/keyed_insertion_recovery/validation/finite_time_task_first_stable_adaptive_step_contrast.json"
    )
    stable_rows = {row["key"]: row for row in stable["rows"]}
    assert stable["policy"][
        "stable_paired_selector_and_adaptive_step_define_transport"
    ]
    assert stable_rows["stable_adaptive_step_transport"][
        "candidate_policy_set"
    ]["checkpoint_selection"]["rule"] == "best_validation_paired_progress"

    expanded = runner["build_diagnostic_plan"](
        ROOT / "configs/arm/keyed_insertion_recovery/validation/finite_time_task_first_expanded_adaptive_step_hard_cell_contrast.json"
    )
    expanded_jobs = [
        job for job in expanded["jobs"]
        if job["key"] == "expanded_adaptive_step_transport"
    ]
    assert len(expanded_jobs) == 64
    assert expanded["policy"][
        "expanded_adaptive_step_is_only_execution_change"
    ]
    assert all(
        job["actor_binding"]["task_dominance_version"]
        == "public_expanded_adaptive_step_transport_dominance_v4"
        for job in expanded_jobs
    )


def test_t3_relative_response_gate_requires_effect_size_consistency_and_exact_sign_test():
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/qualify_method_behavior.py"),
        run_name="cross_ebdm_relative_response_effect_contract_test",
    )
    contrast = runner["_paired_contribution_contrast"]
    acceptance = {
        "transport": "relative_response_transport",
        "task_only": "relative_response_task_only",
        "required_pairs": 128,
        "minimum_aggregate_success_rate_gain": 0.0,
        "minimum_nonnegative_instances": 8,
        "minimum_mean_control_step_reduction_fraction": 0.02,
        "minimum_nonnegative_time_instances": 6,
        "maximum_one_sided_exact_sign_p_value": 0.05,
        "require_transport_to_preserve_every_task_only_success": True,
        "require_zero_extra_transport_constraint_violations": True,
    }
    results = []
    for index in range(1, 9):
        instance = f"instance_{index:02d}"
        for request_index in range(16):
            request = {"request_id": f"paired_{request_index:02d}"}
            for key, steps in (
                ("relative_response_transport", 96 + request_index % 2),
                ("relative_response_task_only", 100 + request_index % 2),
            ):
                results.append({
                    "key": key, "instance": instance, "request": request,
                    "passed": True, "outcome": "success",
                    "constraint_violation": False,
                    "physical_control_steps": steps,
                    "evidence": {"audit_passed": True},
                })

    report = contrast({"acceptance": acceptance}, results)

    assert report["mean_control_step_reduction_fraction"] > .02
    assert report["nonnegative_time_instances"] == 8
    assert report["one_sided_exact_sign_p_value"] < .05
    assert report["transport_claim_supported_development"] is True
    weakened = deepcopy(results)
    for row in weakened:
        if row["key"] == "relative_response_transport":
            row["physical_control_steps"] = 100
    rejected = contrast({"acceptance": acceptance}, weakened)
    assert rejected["transport_claim_supported_development"] is False


def test_t4_factorial_separates_transport_projection_and_interaction_effects():
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/validate_t4_completion.py"),
        run_name="cross_ebdm_t4_factorial_contract_test",
    )
    effects = runner["_factorial_effects"]
    results = []
    for index in range(1, 9):
        instance = f"instance_{index:02d}"
        cells = {
            "task_only_projection_off": index > 2,
            "task_only_projection_on": index > 1,
            "transport_projection_off": index > 1,
            "transport_projection_on": True,
        }
        for key, passed in cells.items():
            results.append({
                "key": key,
                "instance": instance,
                "request": {"request_id": "paired_00"},
                "passed": passed,
                "constraint_violation": False,
            })
    decision = {
        "minimum_primary_success_rate_gain": .05,
        "minimum_nonnegative_instances": 6,
        "maximum_two_sided_exact_mcnemar_p_value": .05,
    }

    report = effects(results, decision)

    assert report["primary_transport_on_minus_task_only_on"] == pytest.approx(.125)
    assert report["task_only_projection_main_effect"] == pytest.approx(.125)
    assert report["transport_projection_main_effect"] == pytest.approx(.125)
    assert report["transport_by_projection_interaction"] == pytest.approx(0.)
    assert report["primary_two_sided_exact_mcnemar_p_value"] == pytest.approx(1.)
    assert report["transport_policy_increment_supported_development"] is False
    assert report["transport_vs_strong_task_only_direction_supported"] is False
    unmatched = effects(
        results, decision, matched_control_admitted=False,
    )
    assert unmatched["transport_vs_strong_task_only_direction_supported"] is False
    assert unmatched["matched_task_only_training_admitted"] is False
    assert unmatched["response_anchoring_causal_claim_supported"] is False
    assert unmatched["transport_claim_supported_development"] is False


def test_t4_factorial_requires_qualified_composite_and_statistical_direction():
    runner = runpy.run_path(
        str(ROOT / "scripts/paper/cross_ebdm/validate_t4_completion.py"),
        run_name="cross_ebdm_t4_factorial_statistical_contract_test",
    )
    effects = runner["_factorial_effects"]
    results = []
    for index in range(1, 9):
        instance = f"instance_{index:02d}"
        for request_index in range(8):
            task_only = request_index >= 2
            cells = {
                "task_only_projection_off": task_only,
                "task_only_projection_on": task_only,
                "transport_projection_off": True,
                "transport_projection_on": True,
            }
            for key, passed in cells.items():
                results.append({
                    "key": key,
                    "instance": instance,
                    "request": {"request_id": f"paired_{request_index:02d}"},
                    "passed": passed,
                    "constraint_violation": False,
                })
    decision = {
        "minimum_primary_success_rate_gain": .05,
        "minimum_nonnegative_instances": 6,
        "maximum_two_sided_exact_mcnemar_p_value": .05,
    }

    admitted = effects(results, decision, matched_control_admitted=False)

    assert admitted["primary_transport_on_minus_task_only_on"] == pytest.approx(.25)
    assert admitted["primary_two_sided_exact_mcnemar_p_value"] < .05
    assert admitted["transport_policy_increment_supported_development"] is True
    assert admitted["response_anchoring_causal_claim_supported"] is False
    rejected = effects(
        results, decision, matched_control_admitted=False,
        transport_composite_admitted=False,
    )
    assert rejected["transport_policy_increment_supported_development"] is False
