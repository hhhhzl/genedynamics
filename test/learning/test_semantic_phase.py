import numpy as np
import pytest

from genedynamics.solvers.single.cross_ebdm.learning.ablation import treatment_metadata
from genedynamics.solvers.single.cross_ebdm.learning.semantic_phase import (
    PublicSemanticPhaseTransport, select_learned_scale, validate_checkpoint,
)


def checkpoint():
    selection = select_learned_scale([
        {"scale": 1.0, "task_loss": 1.0,
         "calibrated_response_loss": .2,
         "uncalibrated_response_loss": 0.},
        {"scale": 1.15, "task_loss": .9,
         "calibrated_response_loss": .01,
         "uncalibrated_response_loss": .2},
    ], treatment_name="full")
    fit = {"weights": np.zeros((4, 3)).tolist(), "train_samples": 8,
           "validation_samples": 2, "validation_mse": 1e-4}
    return {
        "schema_version": "semantic-response-transport-v2",
        "task": "aperture_brake",
        "treatment": treatment_metadata("full"),
        "interaction_budget_seconds": 60,
        "phase_scales": [1.0, 1.0, 1.15, 1.15],
        "active_phases": [2, 3],
        "translation_coordinates": [0, 1, 2],
        "selection": selection,
        "response_model": {
            "kind": "public-linear-step-response-v2",
            "source": fit, "target": fit,
            "input_semantics": "xyz command plus bias",
            "output_semantics": "observed xyz velocity",
        },
    }


def test_semantic_phase_transport_applies_only_declared_phase():
    class Actor:
        def __init__(self):
            self.phase = np.asarray([2])

        def act(self, public):
            return np.asarray([[.01, -.01, .002, *([0.] * 10)]], np.float32)

    actor = Actor()
    execution = PublicSemanticPhaseTransport(
        actor, checkpoint(), task="aperture_brake",
    )
    np.testing.assert_allclose(
        execution.act({})[0, :3], [.0115, -.0115, .0023], atol=1e-7,
    )
    actor.phase[:] = 1
    np.testing.assert_allclose(
        execution.act({})[0, :3], [.01, -.01, .002], atol=1e-7,
    )


def test_semantic_phase_checkpoint_rejects_treatment_drift():
    value = checkpoint()
    value["treatment"]["response_weight"] = 0.0
    with pytest.raises(ValueError, match="treatment metadata"):
        validate_checkpoint(value, task="aperture_brake")


def test_learned_selection_changes_only_declared_ablation_term():
    candidates = [
        {"scale": 1.0, "task_loss": 1.0,
         "calibrated_response_loss": .4,
         "uncalibrated_response_loss": 0.},
        {"scale": 1.2, "task_loss": .95,
         "calibrated_response_loss": 0.,
         "uncalibrated_response_loss": .4},
    ]
    assert select_learned_scale(
        candidates, treatment_name="full",
    )["selected_scale"] == 1.2
    assert select_learned_scale(
        candidates, treatment_name="no_calibration",
    )["selected_scale"] == 1.0
