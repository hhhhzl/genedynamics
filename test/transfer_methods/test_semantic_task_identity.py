"""Semantic runtime identities preserve frozen T1--T4 artifact reads."""

from pathlib import Path

import pytest

from genedynamics.tasks.transfer.main_table_configs import config_paths, configs
from genedynamics.tasks.transfer.main_table_evaluation import GPU0_UUID
from genedynamics.tasks.transfer.main_table_execution import (
    ExecutionSettings,
    validate_settings,
)
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.task_identity import (
    CANONICAL_TASKS,
    canonical_task_id,
    canonicalize_task_mapping,
    legacy_task_id,
)


EXPECTED = {
    "T1": "obstacle_transfer",
    "T2": "guided_contact_alignment",
    "T3": "keyed_insertion_recovery",
    "T4": "mobile_keyed_insertion_recovery",
}


def test_runtime_identity_is_semantic_and_legacy_alias_is_reader_only():
    assert tuple(EXPECTED.values()) == CANONICAL_TASKS
    for legacy, semantic in EXPECTED.items():
        assert canonical_task_id(legacy) == semantic
        assert canonical_task_id(semantic) == semantic
        assert legacy_task_id(semantic) == legacy
        spec = task_spec(semantic)
        assert spec.name == semantic
        assert spec.alias == legacy
    with pytest.raises(ValueError, match="unknown transfer task identity"):
        canonical_task_id("T5")


def test_task_keyed_reader_rejects_alias_semantic_collision():
    assert canonicalize_task_mapping({"T1": 1, "guided_contact_alignment": 2}) == {
        "obstacle_transfer": 1,
        "guided_contact_alignment": 2,
    }
    with pytest.raises(ValueError, match="both a semantic name and legacy alias"):
        canonicalize_task_mapping({"T1": 1, "obstacle_transfer": 2})


def test_historical_actor_configs_resolve_identically_from_semantic_names():
    for legacy, semantic in EXPECTED.items():
        assert config_paths(legacy, "public_geometry", "v2") == config_paths(
            semantic, "public_geometry", "v2"
        )
        old = configs(legacy, "public_geometry", "v2")
        new = configs(semantic, "public_geometry", "v2")
        assert old == new
        assert old[0][1]["task"] == legacy


def test_execution_machine_and_output_locations_are_explicit_settings(tmp_path):
    settings = ExecutionSettings(
        output=Path("results/_analysis/four_task_transfer/test_execution"),
        table_output=Path("results/_analysis/four_task_transfer/test_table"),
        site_packages=None,
        temporary_directory=tmp_path,
        cuda_visible_devices="0",
        physical_gpu_uuid=GPU0_UUID,
    )
    validate_settings(settings)
    with pytest.raises(ValueError, match="analysis path"):
        validate_settings(ExecutionSettings(
            output=Path("/tmp/not-a-result"),
            table_output=settings.table_output,
            site_packages=None,
            temporary_directory=tmp_path,
            cuda_visible_devices="0",
            physical_gpu_uuid=GPU0_UUID,
        ))
    with pytest.raises(ValueError, match="analysis path"):
        validate_settings(ExecutionSettings(
            output=Path("results/_analysis/four_task_transfer/../../../outside"),
            table_output=settings.table_output,
            site_packages=None,
            temporary_directory=tmp_path,
            cuda_visible_devices="0",
            physical_gpu_uuid=GPU0_UUID,
        ))


def test_package_runner_does_not_import_paper_entrypoints():
    source = Path(
        "genedynamics/tasks/transfer/main_table_execution.py"
    ).read_text()
    assert "scripts.paper" not in source
    assert "scripts/paper" not in source
