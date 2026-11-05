import numpy as np
import torch

from genedynamics.solvers.single.cross_ebdm.learning.response import (
    PUBLIC_FIELDS,
    PublicResponseSpec,
)
from genedynamics.solvers.single.cross_ebdm.learning.substep_risk import (
    PublicJointAdmissionCalibration,
    PublicJointAdmissionEnsemble,
    PublicJointAdmissionTrainer,
    PublicSubstepRiskCalibration,
    PublicSubstepRiskEnsemble,
    PublicSubstepRiskTrainer,
    calibrated_force_upper_bound,
    calibrated_joint_upper_bound,
    decode_joint_admission_risks,
    emergency_reference_brake_horizon,
    fit_block_conformal_calibration,
    fit_joint_block_conformal_calibration,
    fit_split_conformal_calibration,
    interpolation_candidates,
    joint_admission_targets,
    risk_admission_candidates,
    select_admissible_candidate,
    select_admissible_fraction,
    select_joint_admissible_candidate,
    substep_risk_windows,
)


def public_episode(count):
    values = {
        "joint_position": np.zeros((count, 7), np.float32),
        "joint_velocity": np.zeros((count, 7), np.float32),
        "tcp_position": np.zeros((count, 3), np.float32),
        "tcp_rotation": np.tile(np.eye(3, dtype=np.float32), (count, 1, 1)),
        "tcp_twist": np.zeros((count, 6), np.float32),
        "measured_wrench": np.zeros((count, 6), np.float32),
        "goal_position": np.zeros((count, 3), np.float32),
        "goal_rotation": np.tile(np.eye(3, dtype=np.float32), (count, 1, 1)),
        "previous_action": np.zeros((count, 13), np.float32),
        "previous_request": np.zeros((count, 13), np.float32),
        "reference_position": np.zeros((count, 3), np.float32),
        "reference_rotation": np.tile(np.eye(3, dtype=np.float32), (count, 1, 1)),
        "elapsed_seconds": np.arange(count, dtype=np.float32)[:, None] * .02,
        "body_position": np.zeros((count, 3), np.float32),
        "body_twist": np.zeros((count, 6), np.float32),
        "controller_feedback": np.zeros((count, 2), np.float32),
    }
    assert set(values) == PUBLIC_FIELDS
    return values


def test_substep_windows_use_exact_maximum_inside_each_horizon():
    spec = PublicResponseSpec(history_len=2, horizon=3, maximum_joints=8, hidden=(8,), members=2)
    total = np.asarray([1., 2., 30., 4., 5.], np.float32)
    lateral = np.asarray([.5, 11., 1., 2., 3.], np.float32)
    rows = substep_risk_windows(
        public_episode(6), np.zeros((5, 13), np.float32),
        total_force_peaks_N=total, lateral_force_peaks_N=lateral,
        body="xarm7", calibration_context=(0., 0., 0.), spec=spec,
    )
    np.testing.assert_allclose(
        np.expm1(rows["targets"]),
        [[30., 11.], [30., 11.], [30., 3.]], rtol=1e-6,
    )
    assert rows["commands"].shape == (3, 3, 13)


def test_checkpoint_round_trip_preserves_explicit_substep_semantics():
    torch.manual_seed(3)
    spec = PublicResponseSpec(history_len=2, horizon=3, maximum_joints=8, hidden=(8,), members=2)
    model = PublicSubstepRiskEnsemble(spec)
    restored = PublicSubstepRiskEnsemble.from_checkpoint(model.checkpoint())
    features = torch.randn(4, spec.feature_dim)
    commands = torch.randn(4, spec.horizon, 13)
    torch.testing.assert_close(model(features, commands), restored(features, commands))


def test_conformal_upper_bound_and_largest_safe_fraction():
    predicted = np.zeros((2, 4, 2), dtype=np.float32)
    truth = np.log1p(np.asarray([[2., 1.], [3., 2.], [4., 3.], [5., 4.]]))
    calibration = fit_split_conformal_calibration(
        predicted, truth, miscoverage=.25,
        soft_force_limits_N=(20., 8.),
    )
    raw = torch.zeros(2, 1, 4, 2)
    # Four interpolation candidates: the first two are unsafe after calibration,
    # the third is explicitly below the limits and zero remains the fallback.
    correction = torch.tensor(calibration.additive_log_correction)
    desired = torch.tensor([[[25., 9.], [21., 7.], [19., 7.], [100., 100.]]])
    raw[:, 0] = torch.log1p(desired) - correction
    upper = calibrated_force_upper_bound(raw.reshape(2, 4, 2), calibration).reshape(1, 4, 2)
    selected, index = select_admissible_fraction(upper, calibration)
    assert index.item() == 2
    assert selected.item() == .25


def test_no_interpolation_is_forced_safe_when_the_safe_set_is_empty():
    calibration = PublicSubstepRiskCalibration()
    upper = torch.full((2, 4, 2), 100.0)
    selected, index = select_admissible_fraction(upper, calibration)
    assert torch.isnan(selected).all()
    torch.testing.assert_close(index, torch.full((2,), -1, dtype=torch.int64))


def test_block_conformal_treats_each_episode_as_one_exchangeable_unit():
    predicted = np.zeros((2, 6, 2), dtype=np.float32)
    truth = np.asarray([
        [0.1, 0.2], [0.2, 0.3],
        [0.4, 0.1], [0.8, 0.2], [0.3, 0.7],
        [0.5, 0.4],
    ], dtype=np.float32)
    calibration = fit_block_conformal_calibration(
        predicted, truth, np.asarray([10, 10, 20, 20, 20, 30]),
        miscoverage=.25,
    )
    # Episode maxima are [0.2, 0.3], [0.8, 0.7], [0.5, 0.4].  With three
    # exchangeable blocks the finite-sample upper quantile is the maximum.
    np.testing.assert_allclose(
        calibration.additive_log_correction, [0.8, 0.7], rtol=1e-6,
    )


def test_risk_trainer_updates_every_ensemble_member():
    torch.manual_seed(7)
    spec = PublicResponseSpec(
        history_len=2, horizon=3, maximum_joints=8, hidden=(8,), members=2,
    )
    model = PublicSubstepRiskEnsemble(spec)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    trainer = PublicSubstepRiskTrainer(model, underprediction_weight=3.0)
    metrics = trainer.step(
        torch.randn(5, spec.feature_dim),
        torch.randn(5, spec.horizon, 13),
        torch.rand(5, 2),
    )
    assert trainer.steps == 1
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_interpolation_candidates_end_in_bitwise_base_fallback():
    torch.manual_seed(11)
    candidate = torch.randn(2, 10, 13)
    base = torch.randn(2, 10, 13)
    values = interpolation_candidates(
        candidate, base, PublicSubstepRiskCalibration(),
    )
    torch.testing.assert_close(values[:, 0], candidate)
    assert torch.equal(values[:, -1], base)
    torch.testing.assert_close(values[:, 1], base + .5 * (candidate - base))


def test_emergency_reference_brake_preserves_every_task_owned_coordinate():
    torch.manual_seed(19)
    base = torch.randn(2, 10, 13)
    emergency = emergency_reference_brake_horizon(base)
    assert torch.equal(emergency[..., [2, 6, 7, 8, 9, 10, 11, 12]],
                       base[..., [2, 6, 7, 8, 9, 10, 11, 12]])
    assert torch.equal(emergency[..., [0, 1, 3, 4, 5]],
                       torch.zeros_like(emergency[..., [0, 1, 3, 4, 5]]))


def test_risk_admission_selects_emergency_only_when_earlier_candidates_fail():
    torch.manual_seed(23)
    calibration = PublicSubstepRiskCalibration()
    candidate = torch.randn(2, 10, 13)
    base = torch.randn(2, 10, 13)
    choices = risk_admission_candidates(candidate, base, calibration)
    assert torch.equal(choices[:, 0], candidate)
    assert torch.equal(choices[:, -2], base)
    assert torch.equal(choices[:, -1], emergency_reference_brake_horizon(base))
    upper = torch.full((2, 5, 2), 100.0)
    upper[:, -1] = torch.tensor([10.0, 4.0])
    index, admitted = select_admissible_candidate(upper, calibration)
    assert admitted.all()
    torch.testing.assert_close(index, torch.full((2,), 4, dtype=torch.int64))


def test_risk_admission_exposes_an_empty_set_without_executing_a_fallback():
    calibration = PublicSubstepRiskCalibration()
    upper = torch.full((3, 5, 2), 100.0)
    index, admitted = select_admissible_candidate(upper, calibration)
    assert not admitted.any()
    torch.testing.assert_close(index, torch.full((3,), -1, dtype=torch.int64))


def test_joint_targets_encode_missing_outlet_contact_as_unsafe():
    transformed = joint_admission_targets(
        np.asarray([[5.0, 2.0], [6.0, 3.0]], np.float32),
        np.asarray([0.02, 0.0], np.float32),
        np.asarray([0.8, 1.0], np.float32),
        np.asarray([False, True]),
    )
    decoded = decode_joint_admission_risks(transformed)
    np.testing.assert_allclose(decoded[0], [5.0, 2.0, 0.02, 0.2], rtol=1e-5)
    assert decoded[1, 2] > 0.15
    assert decoded[1, 3] > 0.99


def test_joint_checkpoint_trainer_and_block_calibration():
    torch.manual_seed(29)
    spec = PublicResponseSpec(
        history_len=2, horizon=3, maximum_joints=8, hidden=(8,), members=2,
    )
    model = PublicJointAdmissionEnsemble(spec)
    before = [value.detach().clone() for value in model.parameters()]
    trainer = PublicJointAdmissionTrainer(model, underprediction_weight=3.0)
    features = torch.randn(6, spec.feature_dim)
    commands = torch.randn(6, spec.horizon, 13)
    targets = torch.rand(6, 4)
    metrics = trainer.step(features, commands, targets)
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    restored = PublicJointAdmissionEnsemble.from_checkpoint(model.checkpoint())
    torch.testing.assert_close(model(features, commands), restored(features, commands))
    predicted = np.zeros((2, 6, 4), np.float32)
    calibration = fit_joint_block_conformal_calibration(
        predicted, targets.numpy(), np.asarray([0, 0, 1, 1, 2, 2]),
        miscoverage=0.25,
    )
    assert len(calibration.additive_transformed_correction) == 4


def test_joint_selector_requires_force_and_contact_viability():
    calibration = PublicJointAdmissionCalibration()
    upper = torch.tensor([[
        [10.0, 4.0, 0.20, 0.1],
        [10.0, 4.0, 0.10, 0.7],
        [10.0, 4.0, 0.10, 0.2],
        [25.0, 4.0, 0.10, 0.2],
        [10.0, 9.0, 0.10, 0.2],
    ]])
    index, admitted = select_joint_admissible_candidate(upper, calibration)
    assert admitted.item()
    assert index.item() == 2
    transformed = torch.zeros(2, 5, 4)
    transformed[:, 2] = torch.tensor([np.log1p(10.0), np.log1p(4.0),
                                      np.log1p(100.0), -np.log(0.8)])
    zero = PublicJointAdmissionCalibration()
    decoded = calibrated_joint_upper_bound(transformed, zero)
    assert decoded.shape == (5, 4)
