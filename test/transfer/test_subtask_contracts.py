"""Schema and admission gates for the eight semantic transfer subtasks."""
from copy import deepcopy
from pathlib import Path

import pytest

from genedynamics.tasks.transfer.subtask_contracts import (
    CONTRACT_PATHS,
    FAMILY_SUITES,
    PENDING_STATUS,
    RUNNABLE_STATUS,
    load_all_subtask_contracts,
    load_family_suite,
    load_subtask_contract,
    validate_subtask_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def test_eight_semantic_contracts_are_unique_versioned_and_path_bound():
    contracts = load_all_subtask_contracts()
    ids = [contract.semantic_id for contract in contracts]

    assert len(contracts) == len(ids) == 8
    assert len(set(ids)) == 8
    assert set(ids) == set(CONTRACT_PATHS)
    assert not ({"T1", "T2", "T3", "T4"} & set(ids))
    assert all(contract.path.is_file() for contract in contracts)
    assert all(contract.payload["contract_version"].endswith("-v1") for contract in contracts)


def test_four_families_each_bind_two_frozen_semantic_suites():
    assert len(FAMILY_SUITES) == 4
    for family in FAMILY_SUITES:
        suite = load_family_suite(family)
        assert len(suite) == 2
        assert {contract.family for contract in suite} == {family}
        assert {contract.implementation_status for contract in suite} <= {
            RUNNABLE_STATUS, PENDING_STATUS,
        }
    assert {contract.semantic_id for contract in load_all_subtask_contracts()
            if contract.runnable} == {
        "dual_route_carry", "dual_channel_contact", "corner_guided_seating",
        "jam_recovery", "compliant_fixture_insertion", "stance_insertion",
        "aperture_brake",
    }


def test_pending_suites_fail_before_physics_when_execution_is_requested():
    for semantic_id in ("reposition_then_dock",):
        with pytest.raises(RuntimeError, match="implementation_pending"):
            load_subtask_contract(semantic_id, require_runnable=True)

    for semantic_id in (
        "dual_route_carry", "dual_channel_contact", "corner_guided_seating",
        "jam_recovery", "compliant_fixture_insertion", "stance_insertion",
        "aperture_brake",
    ):
        assert load_subtask_contract(semantic_id, require_runnable=True).runnable


def test_contracts_declare_required_fields_splits_and_resolving_paths():
    for contract in load_all_subtask_contracts():
        payload = contract.payload
        assert payload["event_order"]
        assert payload["interface"]["observations"]["public"]
        assert payload["interface"]["action_authority"]["method"]
        assert payload["evaluation"]["success"]
        assert payload["evaluation"]["violations"]
        assert payload["evaluation"]["anti_bypass"]
        assert payload["source_qualification"]["requirements"]
        assert payload["evaluator_evidence"]["required_signals"]
        for axis in payload["mechanism_axes"]:
            assert axis["train"] and axis["development"] and axis["heldout"]
        implementation = ROOT / payload["evaluator_evidence"]["implementation_path"]
        assert implementation.is_file()


def test_target_budget_overrides_are_rejected_by_shared_schema():
    contract = load_subtask_contract("dual_route_carry")
    mutated = deepcopy(contract.payload)
    mutated["interface"]["target_budget_seconds"] = 120

    with pytest.raises(ValueError, match="target budgets"):
        validate_subtask_contract(mutated, path=contract.path)


def test_numbered_current_identity_is_rejected():
    contract = load_subtask_contract("dual_route_carry")
    mutated = deepcopy(contract.payload)
    mutated["semantic_id"] = "T1"

    with pytest.raises(ValueError, match="unknown semantic task id|historical aliases"):
        validate_subtask_contract(mutated, path=contract.path)
