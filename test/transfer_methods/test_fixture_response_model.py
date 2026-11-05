"""The moving-fixture response model binds public data and stays differentiable."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from genedynamics.solvers.single.cross_ebdm.learning.fixture_response import (
    CALIBRATED_EVENT_TRAINING_VERSION, CALIBRATED_EVENT_VERSION,
    EVENT_VERSION, FixtureResponseEnsemble, FixtureResponseModelSpec,
    fixture_member_loss, training_plan,
)


def test_fixture_response_model_has_finite_input_gradients_and_strict_checkpoint():
    spec = FixtureResponseModelSpec(hidden=(16,), members=2)
    model = FixtureResponseEnsemble(spec)
    features = torch.randn(3, spec.feature_dim, requires_grad=True)
    commands = torch.randn(3, spec.horizon, 13, requires_grad=True)
    segments, peaks = model(features, commands)
    assert segments.shape == (2, 3, 3, 27)
    assert peaks.shape == (2, 3, 2)
    (segments.square().mean() + peaks.square().mean()).backward()
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(commands.grad).all()
    payload = model.checkpoint(data_sha256="a" * 64)
    assert "event_head" not in payload["spec"]
    assert "event_threshold_m" not in payload["spec"]
    restored = FixtureResponseEnsemble.from_checkpoint(payload)
    with torch.no_grad():
        actual = restored(features.detach(), commands.detach())
        expected = model(features.detach(), commands.detach())
    for one, other in zip(actual, expected):
        torch.testing.assert_close(one, other, rtol=0, atol=0)
    with pytest.raises(ValueError, match="schema mismatch"):
        FixtureResponseEnsemble.from_checkpoint(dict(payload, observation_schema="static"))
    with pytest.raises(ValueError, match="versioned public history"):
        model(features[:, :-1], commands)


def test_fixture_response_plan_binds_immutable_public_bundle(tmp_path):
    root = Path(__file__).resolve().parents[2]
    path = root / "configs/arm/compliant_keyed_insertion_recovery/training/segmented_response_gpu1.json"
    config = json.loads(path.read_text())
    config["output"] = str(tmp_path / "unused_output")
    assert training_plan(config, root)["fit_windows"] == 910
    altered = deepcopy(config)
    altered["data"]["tensor_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="identity or split"):
        training_plan(altered, root)
    altered = deepcopy(config)
    altered["execution"]["cuda_visible_devices"] = "0"
    with pytest.raises(ValueError, match="physical GPU1"):
        training_plan(altered, root)


def test_fixture_response_event_head_suppresses_quiet_motion_and_roundtrips():
    spec = FixtureResponseModelSpec(hidden=(8,), members=1, event_head=True)
    model = FixtureResponseEnsemble(spec)
    with torch.no_grad():
        for layer in model.members[0]:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        output = model.members[0][-1]
        for segment in range(3):
            output.bias[segment * 27 + 18] = 1.
            output.bias[3 * 27 + segment] = -20.
    features = torch.zeros((2, spec.feature_dim))
    commands = torch.zeros((2, spec.horizon, 13))
    segments, peaks, logits = model.forward_with_event(features, commands)
    assert logits.shape == (1, 2, 3)
    assert peaks.shape == (1, 2, 2)
    assert torch.max(torch.abs(segments[..., 18])).item() < 1e-7
    payload = model.checkpoint(data_sha256="b" * 64)
    assert payload["version"] == EVENT_VERSION
    restored = FixtureResponseEnsemble.from_checkpoint(payload)
    torch.testing.assert_close(restored(features, commands)[0], model(features, commands)[0])
    with pytest.raises(ValueError, match="event head/version mismatch"):
        FixtureResponseEnsemble.from_checkpoint(dict(payload, version="compliant-socket-segmented-response-model-v1"))


def test_fixture_response_event_training_updates_event_logits_and_motion():
    spec = FixtureResponseModelSpec(hidden=(8,), members=1, event_head=True)
    model = FixtureResponseEnsemble(spec)
    inputs = torch.randn((4, spec.feature_dim + 13 * spec.horizon))
    target = torch.zeros((4, 3, 27))
    target[:2, :, 18] = .002
    prediction = model.members[0](inputs)
    loss = fixture_member_loss(prediction, target, torch.zeros((4, 2)), spec)
    assert torch.isfinite(loss)
    loss.backward()
    gradient = model.members[0][-1].bias.grad
    assert torch.isfinite(gradient).all()
    assert torch.any(gradient[3 * 27:3 * 27 + 3] != 0)
    assert torch.any(gradient[[18, 45, 72]] != 0)


def test_event_importance_weights_restore_public_event_probability():
    spec = FixtureResponseModelSpec(hidden=(8,), members=1, event_head=True)
    width = len(spec.segment_steps) * 27
    target = torch.zeros((2, len(spec.segment_steps), 27))
    target[0, :, 18] = .002  # 20 micrometers in physical units
    logit = torch.tensor(-2.1972246, requires_grad=True)  # public event rate 0.1
    prediction = torch.zeros((2, width + len(spec.segment_steps) + 2))
    prediction[:, width:width + len(spec.segment_steps)] = logit
    loss = fixture_member_loss(
        prediction, target, torch.zeros((2, 2)), spec,
        event_importance_weight=torch.tensor([.2, 1.8]),
    )
    loss.backward()
    assert abs(logit.grad.item()) < 1e-7

    unweighted_logit = torch.tensor(-2.1972246, requires_grad=True)
    unweighted_prediction = torch.zeros_like(prediction)
    unweighted_prediction[:, width:width + len(spec.segment_steps)] = unweighted_logit
    fixture_member_loss(
        unweighted_prediction, target, torch.zeros((2, 2)), spec,
    ).backward()
    assert unweighted_logit.grad.item() < -0.03


def test_calibrated_event_loss_penalizes_deployed_quiet_motion():
    spec = FixtureResponseModelSpec(hidden=(8,), members=1, event_head=True)
    width = len(spec.segment_steps) * 27
    target = torch.zeros((2, len(spec.segment_steps), 27))
    target[0, :, 18] = .002
    values = torch.zeros((2, width + len(spec.segment_steps) + 2))
    values[1, 18] = .02
    old = values.clone().requires_grad_()
    fixture_member_loss(old, target, torch.zeros((2, 2)), spec).backward()
    assert old.grad[1, 18] == 0

    new = values.clone().requires_grad_()
    fixture_member_loss(
        new, target, torch.zeros((2, 2)), spec,
        calibrate_expected_motion=True,
    ).backward()
    assert torch.isfinite(new.grad).all()
    assert new.grad[1, 18] > 0

    model = FixtureResponseEnsemble(spec)
    payload = model.checkpoint(
        data_sha256="c" * 64,
        training_version=CALIBRATED_EVENT_TRAINING_VERSION,
    )
    assert payload["version"] == CALIBRATED_EVENT_VERSION
    restored = FixtureResponseEnsemble.from_checkpoint(payload)
    torch.testing.assert_close(
        restored(torch.zeros((1, spec.feature_dim)), torch.zeros((1, spec.horizon, 13)))[0],
        model(torch.zeros((1, spec.feature_dim)), torch.zeros((1, spec.horizon, 13)))[0],
    )


def test_calibrated_event_training_requires_source_bindings(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = json.loads((
        root / "configs/arm/compliant_keyed_insertion_recovery/training/"
        "segmented_response_event_gpu1.json"
    ).read_text())
    config["version"] = CALIBRATED_EVENT_TRAINING_VERSION
    config["output"] = str(tmp_path / "calibrated")
    paths = (
        "genedynamics/solvers/single/cross_ebdm/learning/fixture_response.py",
        "genedynamics/solvers/single/cross_ebdm/learning/response.py",
        "genedynamics/solvers/single/cross_ebdm/learning/artifact_bundle.py",
        "scripts/paper/cross_ebdm/train_fixture_response.py",
    )
    config["source_bindings"] = [
        {"path": path, "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest()}
        for path in paths
    ]
    assert training_plan(config, root)["validation_windows"] == 5480
    config["source_bindings"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source changed"):
        training_plan(config, root)
