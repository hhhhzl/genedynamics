"""Formal task evaluation cannot open sealed requests without cell admission."""

import hashlib
import json

import pytest

import genedynamics.tasks.transfer.main_table_evaluation as formal
import genedynamics.tasks.transfer.method_evaluation as method_evaluation
from genedynamics.experiments.framework import ExperimentConfig
from scripts.paper.cross_ebdm.qualify_legacy_main_table_actors import (
    _frozen as frozen_legacy_cohort,
    _instance_config as legacy_instance_config,
    audit_fada_source_lineage,
)
import scripts.paper.cross_ebdm.assemble_independent_main_table as table_assembly
from scripts.paper.cross_ebdm.assemble_independent_main_table import (
    assemble as assemble_table,
    inventory as table_inventory,
)
from genedynamics.tasks.transfer.controllers.bindings import resolve_transfer_binding
from genedynamics.tasks.transfer.main_table_evaluation import (
    GPU0_UUID, GPU1_UUID, ROOT, TASK_ROOTS, _binding, _roster, _task_config, formal_plan,
    VERSION,
)


def test_frozen_rosters_and_legacy_actor_instances_are_resolvable_without_physics():
    for task in TASK_ROOTS:
        pointer, roster = _roster(task)
        assert len(roster["requests"]) == (80 if task == "T4" else 40)
        assert pointer["sha256"] == hashlib.sha256(
            (ROOT / pointer["path"]).read_bytes()
        ).hexdigest()
        for method, instance in (("cross_ebdm", "instance_01"),
                                 ("public_geometry", "fixed")):
            experiment = ExperimentConfig.from_yaml(_task_config(task, method))
            manifest = json.loads((ROOT / experiment.method_params[
                "actor_binding"]["manifest"]["path"]).read_text())
            identity = manifest["main"][method]["method_identity"]
            binding = _binding(task, method, instance,
                               {"method_identity": identity}, experiment)
            resolved = resolve_transfer_binding(
                {"actor_binding": binding, "mode": 0}, task_alias=task,
            )
            assert resolved["task"] == task
            assert resolved["instance_id"] == instance


def _fixed_config():
    task, method = "T1", "public_geometry"
    task_config = _task_config(task, method)
    experiment = ExperimentConfig.from_yaml(task_config)
    manifest = json.loads((ROOT / experiment.method_params[
        "actor_binding"]["manifest"]["path"]).read_text())
    roster_pointer, _ = _roster(task)
    return {
        "version": VERSION, "task": task, "method": method,
        "method_identity": manifest["main"][method]["method_identity"],
        "instance_id": "fixed",
        "task_config": {"path": str(task_config.relative_to(ROOT)),
                        "sha256": hashlib.sha256(task_config.read_bytes()).hexdigest()},
        "request_roster": roster_pointer,
        "cell_admission": {
            "path": str(formal._admission_path(task, method).relative_to(ROOT)),
            "sha256": "0" * 64,
        },
        "output": ("results/arm/obstacle_transfer/baseline/public_geometry/"
                   "confirmation/independent_main_v1/fixed"),
        "execution": {"device": "cuda", "cuda_visible_devices": "1",
                      "physical_uuid": GPU1_UUID},
    }


def test_formal_plan_refuses_an_unadmitted_cell_before_physics():
    with pytest.raises(ValueError, match="cell admission missing or changed"):
        formal_plan(_fixed_config())


def test_formal_plan_builds_only_from_checked_cell_provenance(tmp_path, monkeypatch):
    config = _fixed_config()
    task, method = config["task"], config["method"]
    admission_path = tmp_path / "admission.json"
    source_path = tmp_path / "source_nominal.json"
    fidelity_path = tmp_path / "fidelity.json"

    def pointer(path):
        return {"path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    source_path.write_text(json.dumps({
        "task": task, "method": method, "status": "passed",
        "physical_evidence_verified": 8, "formal_performance_evidence": False,
    }))
    fidelity_path.write_text(json.dumps({
        "task": task, "method": method, "status": "passed",
        "method_identity": config["method_identity"],
        "task_config": config["task_config"],
        "manifest": ExperimentConfig.from_yaml(
            _task_config(task, method)
        ).method_params["actor_binding"]["manifest"],
        "source_nominal_qualification": pointer(source_path),
        "formal_request_seed_access": 0,
        "physical_calls": 0,
        "controller_source": pointer(ROOT / "genedynamics/tasks/transfer/controllers/bindings.py"),
        "actor_binding_sha256": hashlib.sha256(
            json.dumps(_binding(task, method, "fixed", config,
                                ExperimentConfig.from_yaml(_task_config(task, method))),
                       sort_keys=True).encode()
        ).hexdigest(),
    }))
    experiment = ExperimentConfig.from_yaml(_task_config(task, method))
    binding = _binding(task, method, "fixed", config, experiment)
    admission_path.write_text(json.dumps({
        "version": formal.ADMISSION_VERSION, "status": "passed",
        "independent_table_eligible": True,
        "task": task, "method": method,
        "method_identity": config["method_identity"],
        "request_roster": config["request_roster"],
        "formal_request_seed_access": 0,
        "instance_ids": ["fixed"],
        "source_nominal_qualification": pointer(source_path),
        "method_fidelity_review": pointer(fidelity_path),
        "actor_binding_sha256": {
            "fixed": hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        },
    }))
    monkeypatch.setattr(formal, "_admission_path", lambda *_: admission_path)
    config["cell_admission"] = pointer(admission_path)
    plan = formal_plan(config)
    assert plan["scope"] == "independent formal main table"
    assert len(plan["requests"]) == 40
    assert all(row["confirmation_state"] is True for row in plan["requests"])
    assert all(row["performance_evidence"] is True for row in plan["requests"])


def test_legacy_cohort_resolves_a_second_actor_without_formal_access():
    config, base, _, output = frozen_legacy_cohort("T1", "task_only")
    second = legacy_instance_config(base, output, 2)
    assert config["maximum_new_physical_episodes"] == 56
    assert config["formal_request_seed_access"] == 0
    assert second["instance_id"] == "instance_02"
    assert second["output"].endswith(
        "/validation/independent_source_nominal_cohort_v1/instance_02"
    )


def test_fada_target_instances_share_qualified_source_but_distinct_adaptation_seeds():
    report = audit_fada_source_lineage("T1")
    assert report["status"] == "qualified_source_lineage_only"
    assert report["source_successes"] == report["source_requests"] == 4
    assert report["distinct_target_adaptation_seeds"] == 8
    assert report["target_method_fidelity_qualified"] is False


def test_main_table_assembly_refuses_unexecuted_formal_cells(tmp_path, monkeypatch):
    state = table_inventory()
    assert state["cell_count"] == 36
    assert 0 <= state["fully_audited_cells"] <= 36
    monkeypatch.setattr(table_assembly, "inventory", lambda: {"status": "pending"})
    with pytest.raises(ValueError, match="independent main table is incomplete"):
        assemble_table(tmp_path / "unscored")


def test_assembly_resolves_fixed_legacy_and_direct_actor_config_rosters():
    for method, expected in (("public_geometry", 1), ("cross_ebdm", 8),
                             ("asid", 8)):
        entries = table_assembly._configs("T1", method)
        assert len(entries) == expected
        assert len({config["instance_id"] for _, config in entries}) == expected
        assert all(formal_plan(config)["method"] == method
                   for _, config in entries)


def test_formal_continuation_refuses_ambiguous_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(method_evaluation, "_root", lambda: tmp_path)
    request = {"request_id": "INDEPENDENT-MAIN-v1/T1/source/fixed/i00/m0"}
    plan = {"scope": "independent formal main table", "output": "cell",
            "requests": [request]}
    output = tmp_path / "cell"
    episode = output / "episodes" / hashlib.sha256(
        request["request_id"].encode()
    ).hexdigest()
    episode.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan))
    (episode / "reset_receipt.json").write_text("{}")
    with pytest.raises(ValueError, match="ambiguous physical attempt"):
        method_evaluation._run_plan(plan, continue_claimed=True)


def test_gpu0_revision_is_seed_disjoint_and_preserves_paired_actor_requests():
    from scripts.paper.cross_ebdm.freeze_independent_main_table_requests import verify

    assert verify("v2")["execution"] == {
        "device": "cuda", "cuda_visible_devices": "0", "physical_uuid": GPU0_UUID,
    }
    for task in TASK_ROOTS:
        _, old = _roster(task)
        _, fresh = _roster(task, "v2")
        assert {row["reset_seed"] for row in fresh["requests"]}.isdisjoint(
            row["reset_seed"] for row in old["requests"]
        )
        ids = [row["request_id"] for row in fresh["requests"]]
        for method in ("cross_ebdm", "public_geometry", "asid", "adaptsim"):
            for _, config in table_assembly._configs(task, method, "v2"):
                plan = formal_plan(config)
                assert plan["execution"]["physical_uuid"] == GPU0_UUID
                assert [row["request_id"] for row in plan["requests"]] == ids
                assert "/independent_main_v2/" in plan["output"]


def test_gpu0_formal_plan_rejects_a_gpu1_execution_binding():
    config = table_assembly._configs("T1", "public_geometry", "v2")[0][1]
    config["execution"] = {
        "device": "cuda", "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
    }
    with pytest.raises(ValueError, match="physical GPU0"):
        formal_plan(config)
