"""CPU contracts for the paper's matched ICDT ablations."""

from dataclasses import asdict
from pathlib import Path

import pytest

from genedynamics.solvers.single.cross_ebdm.learning.ablation import (
    EXPECTED_TREATMENT_DIFFERENCES,
    TREATMENTS,
    load_contract,
    treatment,
    treatment_metadata,
    treatment_objective_kwargs,
    treatment_trainer_class,
)
from genedynamics.solvers.single.cross_ebdm.learning.gated_transport import (
    PublicRecoveryGatedTransportTrainer,
)
from scripts.paper.cross_ebdm.train_t4_completion import load_training_config


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT / "configs/arm/keyed_insertion_recovery/ablation/"
    "clean_treatment_contract.json"
)


def test_clean_ablation_contract_changes_exactly_one_theoretical_field():
    document = load_contract(CONTRACT)
    full = asdict(TREATMENTS["full"])

    for name, expected_fields in EXPECTED_TREATMENT_DIFFERENCES.items():
        actual = asdict(TREATMENTS[name])
        changed = tuple(key for key in full if actual[key] != full[key])
        assert changed == expected_fields
        assert document["treatments"][name]["only_difference_from_full"] == list(
            expected_fields
        )


def test_clean_ablation_contract_locks_training_and_execution_components():
    document = load_contract(CONTRACT)

    assert document["shared"]["warm_start"]["policy_set"] == (
        "task_only_shared_start_v9"
    )
    assert document["shared"]["data"]["batch_index_stream"] == (
        "exact replay from one persisted seed stream"
    )
    assert document["shared"]["action_authority"]["decoder"] == (
        "recovery_policy_commands"
    )
    assert document["shared"]["checkpoint_selection"]["selected_once"] is True
    assert all(
        row["evidence_reusable"] is False
        for row in document["legacy_compatibility"].values()
    )


def test_treatments_select_only_response_weight_calibration_or_gate():
    assert treatment_objective_kwargs("no_response_anchoring") == {
        "response_weight": 0.0, "apply_anchoring_gate": True,
    }
    assert treatment("no_calibration").calibration == "nominal_uncalibrated"
    assert treatment_trainer_class("full") is PublicRecoveryGatedTransportTrainer
    assert (
        treatment_trainer_class("unconditional_response_anchoring")
        is PublicRecoveryGatedTransportTrainer
    )
    metadata = treatment_metadata("unconditional_response_anchoring")
    assert metadata["treatment_only"] is True
    assert metadata["anchoring_gate"] == "unconditional"
    with pytest.raises(ValueError, match="unknown clean ablation treatment"):
        treatment("band_only")


def test_clean_pilot_trains_full_and_each_ablation_for_both_modes():
    paths = sorted((
        ROOT / "configs/arm/keyed_insertion_recovery/training"
    ).glob("clean_*_mode[01].json"))
    configs = [load_training_config(path) for path in paths]

    assert len(configs) == 8
    assert {
        (config["training"]["clean_treatment"], mode)
        for config in configs
        for mode in config["selection"]["required_modes"]
    } == {
        (name, mode) for name in TREATMENTS for mode in (0, 1)
    }
    assert all(config["training"]["updates"] == 1500 for config in configs)
    assert all(config["limits"]["wall_seconds"] == 7200 for config in configs)
    assert all(set(config["instances"]) == {"instance_01"} for config in configs)
