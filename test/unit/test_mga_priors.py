"""Fedguide gates: shared priors package + the additive prior= seam.

Covers the brax-free parts (protocols, registry, elite buffer, jax diffusion
prior, and the prior= warm-start seam's additive safety on a mock rollout). The
brax RL prior (jax backend = brax.training) is exercised in the docker test."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

from genedynamics.learning.priors import (
    DiffusionPrior, EliteBuffer, Prior, ProposalBatch, StructuredPrior,
    list_priors, make_prior,
)
from genedynamics.learning.reliability import (
    FEATURE_NAMES, LinearReliabilityModel, PEG_INSERT_FEATURE_NAMES,
    PEG_INSERT_RISK_NAMES, RISK_NAMES, samples_from_records,
)
from genedynamics.solvers.single.mga.backends.mga_jax import MgaBackendJax
from genedynamics.solvers.common.rl_policy_controller import RLPolicyController
from genedynamics.experiments.framework.config import ExperimentConfig
from genedynamics.experiments.utils.metrics import (
    aggregate_receding_diagnostics,
)
from scripts.tasks.robot.arm.train_mga_reliability import (
    _assert_disjoint_protocol,
    _peg_insert_record_from_result,
)
from scripts.tasks.robot.humanoid.train_box_push_reliability import (
    FEATURE_NAMES as H1_FEATURE_NAMES,
    _fit_audit as h1_reliability_fit_audit,
    _model_diagnostics as h1_reliability_diagnostics,
    _sequence_rows as h1_reliability_sequence_rows,
    _validate_splits as h1_reliability_validate_splits,
)


# --- registry / elite buffer ---


def test_reliability_protocol_rejects_split_overlap_and_formal_seed_leakage():
    train = [{"suite": "soft", "seed": 0}]
    calibration = [{"suite": "soft", "seed": 1}]
    _assert_disjoint_protocol(train, calibration, set(range(10, 20)))
    with pytest.raises(ValueError, match="overlap"):
        _assert_disjoint_protocol(train, train, set(range(10, 20)))
    with pytest.raises(ValueError, match="formal evaluation seeds"):
        _assert_disjoint_protocol(
            [{"suite": "soft", "seed": 10}], calibration, set(range(10, 20))
        )


def _h1_reliability_payload(n=10, *, width=12, is_walk=False):
    actions = np.random.default_rng(113).uniform(-1, 1, (n, width))
    post_features = np.zeros((n, len(H1_FEATURE_NAMES)))
    post_features[:, :12] = np.arange(n)[:, None]
    post_features[:, -1] = float(is_walk)
    return {
        "actions": actions,
        "states": [{"info": {"prev_action": np.zeros(width)}}] + [
            {"info": {"prev_action": action}} for action in actions
        ],
        "task_signals": {
            "reliability_features": post_features,
            "reliability_risk": np.zeros((n, 4)),
            "task_success": np.zeros(n),
        },
    }


def test_h1_reliability_uses_prestate_and_same_future_risk_window():
    payload = _h1_reliability_payload()
    risks = payload["task_signals"]["reliability_risk"]
    risks[0] = [1, 1, 100, 100]  # Past risk must not leak into the target.
    risks[6] = [1, 1, 0.6, 6]    # Last transition of decision-1's six-step window.
    risks[5, 2:] = [0.2, 2]
    risks[7] = [1, 1, 100, 100]  # Outside the window.
    x, y, meta = h1_reliability_sequence_rows(payload, {}, horizon_steps=6)
    assert meta["decision_steps"] == [1, 2, 3, 4]
    np.testing.assert_array_equal(x[0, :12], np.zeros(12))
    np.testing.assert_allclose(y[0], [1, 1, 0.4, 8 / 6])
    assert y[1, 2] > y[0, 2]
    # Unknown future measured states are labels, never part of s_t features.
    payload["task_signals"]["reliability_features"][1:, :12] = 999
    x_changed, _, _ = h1_reliability_sequence_rows(payload, {}, horizon_steps=6)
    np.testing.assert_array_equal(x_changed[0], x[0])


@pytest.mark.parametrize("use_base,is_walk", [(False, False), (True, False), (False, True)])
def test_h1_reliability_features_match_deployment_without_physics(use_base, is_walk):
    from genedynamics.core.control.stiffness import PrimitiveSpec
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    spec = PrimitiveSpec(pos_dim=8 if use_base else 5, stiff_dim=3, feed_dim=1)
    width = spec.total_width + (11 if is_walk else 0)
    payload = _h1_reliability_payload(width=width, is_walk=is_walk)
    # Persisted controller memory, rather than the last submitted action, is
    # authoritative.  These differ on an absorbing terminal in real traces.
    previous = np.full(width, 0.37)
    payload["states"][1]["info"]["prev_action"] = previous
    state = SimpleNamespace(
        pipeline_state=SimpleNamespace(
            x=SimpleNamespace(
                pos=jnp.asarray([[0.9, 0.1, 0.5], [0.03, 0.01, 1.0]]),
                rot=jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
            ),
            site_xpos=jnp.asarray([[0.0, -0.1, 0.0], [0.0, 0.1, 0.0]]),
        ),
        info={"box_x0": 0.8, "box_goal_x": 1.0, "contact_acquired": 1.0,
              "unjam_released": 0.0, "force_int": 3.0,
              "prev_action": jnp.asarray(previous)},
    )
    env = SimpleNamespace(
        spec=spec, _is_walk=is_walk, _half=0.5, _box_idx=1, _pelvis_idx=2,
        _feet_site_id=jnp.asarray([0, 1]),
        _bcfg=SimpleNamespace(push_dist=0.2, f_max=60.0, support_radius=0.25,
                              force_int_max=30.0, unjam_release_clearance=0.005),
        _box_contact_forces=lambda ps: {"hand": 15.0, "wall": 2.0, "nonhand": 0.0},
        _corridor_clearance=lambda ps: 0.01,
    )
    expected = HumanoidBoxPushEnv.reliability_features_sequence(
        env, state, jnp.asarray(payload["actions"][1:5]),
    )
    payload["task_signals"]["reliability_features"][0, :12] = np.asarray(expected[:12])
    x, _, _ = h1_reliability_sequence_rows(
        payload, {"env_params": {"use_base": use_base,
                  "level": "push_walk" if is_walk else "push_to_line"}},
        horizon_steps=4,
    )
    np.testing.assert_allclose(x[0], np.asarray(expected), rtol=1e-6, atol=1e-7)


def test_h1_reliability_excludes_terminal_starts_and_incomplete_windows():
    payload = _h1_reliability_payload(n=8)
    _, _, meta = h1_reliability_sequence_rows(payload, {}, horizon_steps=3)
    assert meta["decision_steps"] == [1, 2, 3, 4, 5]
    assert meta["unobserved_reset_starts"] == 1
    assert meta["right_censored_starts_excluded"] == 2
    payload["task_signals"]["task_success"][2:] = 1
    # Absorbing success can retain an unsafe contact; reaching the goal must
    # not erase that event from windows that started before success.
    payload["task_signals"]["reliability_risk"][2:, 0] = 1
    _, y, meta = h1_reliability_sequence_rows(payload, {}, horizon_steps=3)
    assert meta["decision_steps"] == [1, 2]
    assert meta["absorbing_starts_excluded"] == 5
    np.testing.assert_array_equal(y[:, 0], [1, 1])
    x, y, _ = h1_reliability_sequence_rows(payload, {}, horizon_steps=20)
    assert x.shape == (0, len(H1_FEATURE_NAMES)) and y.shape == (0, 4)


def test_h1_reliability_preserves_unsafe_examples_and_single_step_boundary():
    payload = _h1_reliability_payload(n=4)
    payload["task_signals"]["reliability_risk"][1] = [1, 0, 0, 0]
    payload["task_signals"]["reliability_risk"][2] = [0, 1, 0.5, 2]
    x, y, meta = h1_reliability_sequence_rows(payload, {}, horizon_steps=1)
    np.testing.assert_array_equal(y, [[1, 0, 0, 0], [0, 1, 0.5, 2], [0, 0, 0, 0]])
    np.testing.assert_array_equal(x[:, H1_FEATURE_NAMES.index("action_roughness")], 0)
    assert meta["decision_steps"] == [1, 2, 3]
    with pytest.raises(ValueError, match="positive"):
        h1_reliability_sequence_rows(payload, {}, horizon_steps=0)
    payload["task_signals"]["reliability_risk"][1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        h1_reliability_sequence_rows(payload, {}, horizon_steps=1)


def test_h1_reliability_diagnostics_expose_degenerate_labels_and_no_support():
    model = SimpleNamespace(
        predict=lambda x: np.zeros((len(x), 4)),
        predict_upper=lambda x: np.zeros((len(x), 4)),
        support_score_state=lambda x: np.full(len(x), 2.0),
        support_score=lambda x: np.full(len(x), 3.0),
    )
    report = h1_reliability_diagnostics(
        model, np.zeros((2, 24)), np.asarray([[1, 0, 0, 0], [0, 0, 0, 0]]),
    )
    assert report["labels"]["invalid_contact_or_fall"]["positive"] == 0
    assert report["labels"]["force_violation"]["positive_fraction"] == 0.5
    assert report["binary_brier"]["force_violation"] == 0.5
    assert report["binary_threshold_errors"]["force_violation"]["false_negative_rate"] == 1.0
    assert report["binary_threshold_errors"]["force_violation"]["false_positive_rate"] == 0.0
    assert report["binary_threshold_errors"]["invalid_contact_or_fall"]["false_negative_rate"] is None
    assert set(report["continuous_upper_coverage"]) == {"balance", "force_mae"}
    assert report["state_support_fraction"] == 0.0
    assert report["in_state_support_continuous_upper_coverage"]["balance"] is None


def test_h1_reliability_reserves_canonical_and_development_evaluation_seeds():
    excluded = h1_reliability_validate_splits({101}, {102}, {110, 111}, set(range(10)))
    assert excluded == set(range(10)) | {110, 111}
    for train, calibration in [({0}, {102}), ({101}, {110}), ({101}, {101})]:
        with pytest.raises(ValueError, match="must be disjoint"):
            h1_reliability_validate_splits(train, calibration, {110, 111}, set(range(10)))


def test_h1_reliability_fit_audit_exposes_duplicates_missing_classes_and_brier():
    train_y = np.asarray([[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]])
    cal_y = np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]])
    prediction = np.asarray([[0.5, 0.25, 0, 0], [0.5, 0.25, 0, 0]])
    train_sources = [
        {"result": "train_a", "feature_risk_sha256": "same"},
        {"result": "train_b", "feature_risk_sha256": "same"},
    ]
    cal_sources = [{"result": "cal", "feature_risk_sha256": "same"}]
    audit = h1_reliability_fit_audit(train_y, cal_y, prediction, train_sources, cal_sources)
    assert audit["source_counts"]["training"] == {
        "window_rows": 4, "source_trajectory_count": 2,
        "unique_feature_risk_trajectory_count": 1,
        "unique_feature_risk_sha256": ["same"],
    }
    assert len(audit["identical_training_calibration_sources"]) == 2
    reasons = audit["promotion_blocking_reasons"]
    assert "missing_negative_class:training:invalid_contact_or_fall" in reasons
    assert "missing_negative_class:calibration:force_violation" in reasons
    assert "missing_positive_class:calibration:invalid_contact_or_fall" in reasons
    assert "identical_training_calibration_feature_risk_trajectories" in reasons
    brier = audit["calibration_brier_vs_training_prevalence"]["force_violation"]
    assert brier["training_prevalence"] == 0.25
    assert brier["constant_calibration_brier"] == 0.5625
    assert brier["model_calibration_brier"] == 0.25
    assert brier["model_minus_constant"] == -0.3125


def test_h1_reliability_fit_without_data_warnings_still_cannot_claim_promotion():
    y = np.asarray([[0, 1, 0, 0], [1, 0, 0, 0]])
    audit = h1_reliability_fit_audit(
        y, y, y,
        [{"result": "train", "feature_risk_sha256": "a"}],
        [{"result": "cal", "feature_risk_sha256": "b"}],
    )
    assert audit["fit_scope"] == "development_fit_only"
    assert audit["performance_validated"] is False
    assert audit["promotion_eligible"] is False
    assert audit["trajectory_independence_validated"] is False
    assert audit["identical_training_calibration_sources"] == []
    assert audit["promotion_blocking_reasons"] == [
        "independent_candidate_and_performance_validation_not_performed",
    ]


def test_h1_reliability_rejects_existing_output_before_collecting_or_fitting(tmp_path, monkeypatch):
    from scripts.tasks.robot.humanoid import train_box_push_reliability as trainer

    output = tmp_path / "frozen.json"
    output.write_text("existing artifact")
    monkeypatch.setattr("sys.argv", ["train", str(tmp_path), "--output", str(output)])
    monkeypatch.setattr(trainer, "_rows", lambda *a, **k: pytest.fail("data read before output guard"))
    with pytest.raises(FileExistsError, match="already exists"):
        trainer.main()
    assert output.read_text() == "existing artifact"


def test_h1_policy_interface_checks_semantics_and_preserves_legacy_loads():
    from genedynamics.experiments.plugins.methods.contact_receding import _validate_policy_interface

    interface = {"action_layout": {"primitive_width": 12, "mapping": "joint_target"}}
    env = SimpleNamespace(policy_interface=interface, _config=SimpleNamespace(dt=0.02))
    _validate_policy_interface(env, {"policy_interface": interface})
    residual = {
        "action_bias": [0.0, -0.4], "action_scale": [1.0, 0.1],
        "mapping": "residual",
    }
    residual_env = SimpleNamespace(
        policy_interface=interface, policy_action_transform=residual,
        _config=SimpleNamespace(dt=0.02),
    )
    residual_checkpoint = {
        "policy_interface": interface, "policy_action_transform": residual,
        "action_bias": residual["action_bias"], "action_scale": residual["action_scale"],
    }
    _validate_policy_interface(residual_env, residual_checkpoint)
    with pytest.raises(ValueError, match="policy action transform mismatch"):
        _validate_policy_interface(residual_env, {"policy_interface": interface})
    with pytest.raises(ValueError, match="action_bias"):
        _validate_policy_interface(
            residual_env, {**residual_checkpoint, "action_bias": [0.0, 0.0]},
        )
    for checkpoint in ({}, {"policy_interface": {"mapping": "residual"}}):
        with pytest.raises(ValueError, match="policy interface mismatch"):
            _validate_policy_interface(env, checkpoint)
    _validate_policy_interface(SimpleNamespace(), {})
    _validate_policy_interface(SimpleNamespace(policy_interface=None), {"policy_interface": interface})
    with pytest.raises(ValueError, match="model/execution"):
        _validate_policy_interface(env, {"policy_interface": interface}, execution_env=SimpleNamespace())
    transform = {"Kc": 1.0, "action_limit": 1.0}
    checkpoint = {"policy_interface": interface, "atacom_transform": {**transform, "time_step": 0.02}}
    _validate_policy_interface(env, checkpoint, atacom=transform)
    with pytest.raises(ValueError, match="transform mismatch"):
        _validate_policy_interface(env, checkpoint, atacom={**transform, "Kc": 2.0})
    peg_transform = {**transform, "alpha_limit": 0.58}
    peg_env = SimpleNamespace(
        policy_interface=None,
        _config=SimpleNamespace(dt=0.02),
        atacom_constraint_residual=lambda state, action: (state, action),
    )
    peg_checkpoint = {
        "atacom_transform": {**peg_transform, "time_step": 0.02},
    }
    _validate_policy_interface(peg_env, peg_checkpoint, atacom=peg_transform)
    with pytest.raises(ValueError, match="transform mismatch"):
        _validate_policy_interface(
            peg_env, peg_checkpoint,
            atacom={**peg_transform, "alpha_limit": 0.6},
        )


def _h1_reliability_contract_env(*, walk=False):
    from genedynamics.envs.domains.humanoid.box_push_brax import (
        HumanoidBoxPushConfig, HumanoidBoxPushEnv,
    )

    env = SimpleNamespace(
        _bcfg=HumanoidBoxPushConfig(level="push_walk" if walk else "push_to_line"),
        dt=0.02, _is_walk=walk, _robot_profile=SimpleNamespace(model_id="h1"),
        action_size=23 if walk else 12, spec=SimpleNamespace(total_width=12),
        policy_interface={"observation_layout": {"observation_size": 91,
                          "version": "current_history"}} if walk else None,
        reliability_features=lambda *args: None, reliability_feature_size=24,
    )
    env.reliability_contract = lambda horizon_steps=None: HumanoidBoxPushEnv.reliability_contract(env, horizon_steps)
    env.validate_reliability_checkpoint = lambda payload, **kw: HumanoidBoxPushEnv.validate_reliability_checkpoint(env, payload, **kw)
    return env


def test_h1_reliability_contract_rejects_same_width_old_risk_horizon_and_policy():
    from copy import deepcopy
    from genedynamics.envs.domains.humanoid.box_push_brax import H1_RELIABILITY_SCHEMA

    env = _h1_reliability_contract_env(walk=True)
    contract = env.reliability_contract(25)
    payload = {name: deepcopy(H1_RELIABILITY_SCHEMA[name]) for name in (
        "feature_names", "risk_names", "state_feature_count", "support_state_feature_count",
        "probability_risk_count", "classification_probabilities",
    )}
    with pytest.raises(ValueError, match="contract mismatch"):
        env.validate_reliability_checkpoint(payload, horizon_steps=25)
    payload["metadata"] = {"reliability_contracts": [contract]}
    env.validate_reliability_checkpoint(payload, horizon_steps=25)
    with pytest.raises(ValueError, match="contract mismatch"):
        env.validate_reliability_checkpoint(payload, horizon_steps=17)
    for key in ("risk_names", "feature_names"):
        altered = deepcopy(payload)
        altered[key][0] = "old_same_width_semantics"
        with pytest.raises(ValueError, match="mismatch"):
            env.validate_reliability_checkpoint(altered, horizon_steps=25)
    altered = deepcopy(payload)
    altered["metadata"]["reliability_contracts"][0]["policy_interface"]["observation_layout"]["observation_size"] = 78
    with pytest.raises(ValueError, match="contract mismatch"):
        env.validate_reliability_checkpoint(altered, horizon_steps=25)
    altered = deepcopy(payload)
    altered["metadata"]["reliability_contracts"][0]["schema"]["risk_definitions"][3] = "old_constant_target_error"
    with pytest.raises(ValueError, match="contract mismatch"):
        env.validate_reliability_checkpoint(altered, horizon_steps=25)
    env._bcfg.f_max = 61.0
    with pytest.raises(ValueError, match="contract mismatch"):
        env.validate_reliability_checkpoint(payload, horizon_steps=25)

    # A development checkpoint may be inspected under model-based abstention
    # even when its collected domain is not the active OOD contract.  The
    # promotion flags remain false, so this path cannot silently become a
    # learned veto.
    env._bcfg.f_max = 60.0
    diagnostic = deepcopy(payload)
    diagnostic["metadata"] = {
        "reliability_contracts": [],
        "performance_validated": False,
        "promotion_eligible": False,
    }
    env.validate_reliability_checkpoint(
        diagnostic, horizon_steps=25, allow_abstaining_ood=True
    )
    diagnostic["metadata"]["promotion_eligible"] = True
    with pytest.raises(ValueError, match="contract mismatch"):
        env.validate_reliability_checkpoint(
            diagnostic, horizon_steps=25, allow_abstaining_ood=True
        )


def test_h1_reliability_collection_requires_saved_contract_and_matching_sources():
    from copy import deepcopy
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv
    from scripts.tasks.robot.humanoid.train_box_push_reliability import _collection_contract, _fitted_contracts

    with pytest.raises(ValueError, match="audit-only"):
        _collection_contract({}, horizon_steps=17)
    env = _h1_reliability_contract_env()
    signals = {"task_metadata": {
        "reliability_contract": env.reliability_contract(),
        "reliability_source_sha256": HumanoidBoxPushEnv.reliability_source_sha256(),
    }}
    contract, _ = _collection_contract(signals, horizon_steps=17)
    assert contract["horizon_steps"] == 17
    assert signals["task_metadata"]["reliability_contract"]["horizon_steps"] is None
    altered = deepcopy(signals)
    altered["task_metadata"]["reliability_source_sha256"] = {"old_env": "old_sha"}
    with pytest.raises(ValueError, match="source mismatch"):
        _collection_contract(altered, horizon_steps=17)
    rows = [{"reliability_contract": contract}]
    assert _fitted_contracts(rows, rows) == [contract]
    with pytest.raises(ValueError, match="contracts differ"):
        _fitted_contracts(rows, [{"reliability_contract": env.reliability_contract(25)}])


def test_h1_reliability_loader_checks_task_contract_before_shared_model(tmp_path, monkeypatch):
    from genedynamics.experiments.plugins.methods.contact_receding import make_mga

    path = tmp_path / "legacy_q95.json"
    path.write_text("{}")
    monkeypatch.setattr(LinearReliabilityModel, "load", lambda *args: pytest.fail("shared load before validation"))
    with pytest.raises(ValueError, match="H1 reliability checkpoint"):
        make_mga("humanoid_box_push", model_env=_h1_reliability_contract_env(),
                 reliability_ckpt=str(path))
    class ReachedSharedLoader(Exception):
        pass
    def unchanged_loader(*args):
        raise ReachedSharedLoader()
    monkeypatch.setattr(LinearReliabilityModel, "load", unchanged_loader)
    for task in ("manipulator_surface_scan", "manipulator_peg_insert"):
        with pytest.raises(ReachedSharedLoader):
            make_mga(task, model_env=SimpleNamespace(reliability_features=lambda *args: None),
                     reliability_ckpt=str(path))


def test_h1_reliability_static_metadata_keeps_metric_arrays_and_json_serializable():
    import json
    from genedynamics.experiments.framework.experiment import convert_to_json_serializable
    from genedynamics.experiments.plugins.metrics.general import GeneralMetricsPlugin

    metadata = {"reliability_contract": _h1_reliability_contract_env().reliability_contract()}
    signals = {"force": np.array([1., 3., 2.]), "task_metadata": metadata}
    plugin = GeneralMetricsPlugin(["force_peak"], extractor=lambda *a, **k: signals,
                                  persist_signals=True)
    assert plugin.compute(None, None, None, None)["force_peak"] == 3.0
    saved = json.loads(json.dumps(convert_to_json_serializable(plugin.pop_artifacts()), allow_nan=False))
    assert saved["task_signals"]["task_metadata"] == metadata


@pytest.mark.parametrize("consumer", ["mga", "atacom_prior", "rl", "issa", "atacom"])
def test_h1_checkpoint_loaders_reject_old_matching_dimension_policy(monkeypatch, consumer):
    from genedynamics.experiments.plugins.methods.contact_receding import make_controller, make_mga
    from genedynamics.learning import train_rl_policy as learning

    env = SimpleNamespace(action_size=23, observation_size=76,
                          manifold_constraint_size=1, inequality_constraint_size=3,
                          policy_interface={"mapping": "joint_target"})
    checkpoint = {"action_size": 22 if "atacom" in consumer else 23,
                  "observation_size": 76, "protocol": "old_atacom" if "atacom" in consumer else "old_walk"}
    monkeypatch.setattr(learning, "load_policy", lambda path: (None, checkpoint))
    with pytest.raises(ValueError, match="policy interface mismatch"):
        if consumer in {"mga", "atacom_prior"}:
            key = "policy_ckpt" if consumer == "mga" else "atacom_policy_ckpt"
            make_mga("humanoid_box_push", model_env=env, **{key: "old.pkl"})
        else:
            make_controller("humanoid_box_push", consumer, model_env=env, policy_ckpt="old.pkl")


@pytest.mark.parametrize("consumer", ["atacom", "atacom_prior"])
def test_h1_p3_loaders_reject_old_one_dimensional_tangent_policy(monkeypatch, consumer):
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv
    from genedynamics.experiments.plugins.methods.contact_receding import make_controller, make_mga
    from genedynamics.learning import train_rl_policy as learning

    task = SimpleNamespace(_is_walk=False, _face_select=True)
    env = SimpleNamespace(action_size=12, observation_size=76, policy_interface=None,
                          manifold_constraint_size=HumanoidBoxPushEnv.manifold_constraint_size.fget(task),
                          inequality_constraint_size=3)
    checkpoint = {"action_size": 1, "observation_size": 76,
                  "protocol": "humanoid_box_push_atacom_p3_ppo_v1"}
    monkeypatch.setattr(learning, "load_policy", lambda path: (None, checkpoint))
    with pytest.raises(ValueError, match="action mismatch:.*expected 2"):
        if consumer == "atacom_prior":
            make_mga("humanoid_box_push", model_env=env, atacom_policy_ckpt="old.pkl")
        else:
            make_controller("humanoid_box_push", "atacom", model_env=env, policy_ckpt="old.pkl")


def test_h1_p3_training_exports_new_tangent_width_without_changing_raw_policy(monkeypatch):
    from brax.training.agents.ppo import train as ppo_train
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv
    from genedynamics.learning.train_rl_policy import train_rl_policy
    from genedynamics.solvers.single.atacom.wrapper import AtacomEnvWrapper

    task = SimpleNamespace(_is_walk=False, _face_select=True)
    raw_env = SimpleNamespace(action_size=12, observation_size=76, policy_interface=None,
                              manifold_constraint_size=HumanoidBoxPushEnv.manifold_constraint_size.fget(task),
                              inequality_constraint_size=3)
    wrapped = AtacomEnvWrapper(raw_env)
    calls = []

    def fake_train(**kwargs):
        calls.append(kwargs["environment"].action_size)
        return None, None, {}

    monkeypatch.setattr(ppo_train, "train", fake_train)
    for env, expected in ((wrapped, 2), (raw_env, 12)):
        _, config = train_rl_policy(env, num_timesteps=0, episode_length=100, num_envs=1)
        assert config["action_size"] == expected
        assert config["observation_size"] == 76
    assert calls == [2, 12]


def test_h1_training_p4_overrides_are_explicit_scoped_and_do_not_mutate_canonical():
    from scripts.tasks.robot.humanoid.train_box_push_rl import (
        _development_env_specs, _stage_episode_lengths, _walk_curriculum_groups,
    )

    specs = [{"level": "push_walk", "push_dist": 0.30}]
    unchanged, overrides = _development_env_specs("walk", specs, None)
    assert unchanged == specs and overrides == {}
    for schema in ("walk", "atacom_p4"):
        resolved, overrides = _development_env_specs(schema, specs,
            '{"walk_leg_control":"joint_target","walk_success_mode":"locomotion","gait_cadence":0.7}')
        assert resolved[0]["walk_leg_control"] == "joint_target"
        assert resolved[0]["walk_success_mode"] == "locomotion"
        assert resolved[0]["gait_cadence"] == overrides["gait_cadence"] == 0.7
    assert specs == [{"level": "push_walk", "push_dist": 0.30}]
    suites = [SimpleNamespace(n_steps=100)]
    assert _stage_episode_lengths("walk", suites) == [100]
    assert _stage_episode_lengths("walk", suites, 300) == [300]
    assert _stage_episode_lengths("walk", suites, 300, stage_count=2) == [300, 300]
    groups = _walk_curriculum_groups("walk", [{
        "level": "push_walk", "f_target": 30.0, "w_contact": 5.0,
        "w_force": 2.0, "w_force_limit": 10.0, "w_nonhand": 10.0,
        "w_bal": 0.5, "w_stiffness_nominal": 2.0,
    }], True)
    assert len(groups) == 2
    assert groups[0][0]["f_target"] == 0.0
    assert groups[0][0]["f_max"] == 1.0
    assert groups[0][0]["emergency_retract_force"] == 0.0
    assert groups[0][0]["fixed_force_target"] is True
    assert groups[0][0]["s_ref_diag"] == groups[0][0]["s_scale"] == 0.0
    assert groups[0][0]["w_contact"] == groups[0][0]["w_nonhand"] == 0.0
    assert groups[1][0]["f_target"] == 30.0
    with pytest.raises(ValueError, match="single-domain walk"):
        _walk_curriculum_groups("atacom_p4", [{"level": "push_walk"}], True)
    for schema, raw in (
        ("fixed", '{}'), ("atacom_p3", '{}'), ("walk", '[]'),
        ("walk", '{"unknown_field":1}'), ("walk", '{}'),
        ("walk", '{"walk_leg_control":"joint_target","walk_success_mode":"legacy"}'),
        ("walk", '{"walk_leg_control":"joint_target","walk_success_mode":"locomotion","level":"unjam"}'),
        ("walk", '{"walk_leg_control":"joint_target","walk_success_mode":"locomotion","use_base":true}'),
        ("walk", '{"walk_leg_control":"joint_target","walk_success_mode":"locomotion","dt":NaN}'),
    ):
        with pytest.raises(ValueError):
            _development_env_specs(schema, specs, raw)


def test_h1_walk_expert_loader_keeps_only_verified_representable_leg_teacher(tmp_path):
    import json
    from scripts.tasks.robot.humanoid.train_box_push_rl import _load_walk_expert

    path = tmp_path / "trajectory.json"
    actions = np.zeros((3, 23), dtype=float)
    actions[:, 12:] = np.linspace(-1.0, 1.0, 11)
    path.write_text(json.dumps({
        "actions": actions.tolist(),
        "states": [{"obs": np.full(91, i, dtype=float).tolist()} for i in range(4)],
        "task_signals": {
            "task_fallen": [0, 0, 0],
            "walk_left_steps": [0, 1, 1],
            "walk_right_steps": [0, 0, 1],
        },
    }))
    bias = np.r_[np.zeros(12), np.full(11, -0.2)]
    scale = np.r_[np.full(12, 0.25), np.full(11, 1.2)]
    obs, target, report = _load_walk_expert(
        path, observation_size=91, action_size=23,
        action_transform={"action_bias": bias.tolist(), "action_scale": scale.tolist()},
    )
    assert obs.shape == (3, 91) and target.shape == (3, 23)
    np.testing.assert_array_equal(target[:, :12], 0.0)
    assert report["verified_left_steps"] == report["verified_right_steps"] == 1
    assert report["absolute_leg_action_rmse"] < 5e-4
    broken = json.loads(path.read_text())
    broken["task_signals"]["task_fallen"][-1] = 1
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="fall-free bilateral"):
        _load_walk_expert(path, observation_size=91, action_size=23,
                          action_transform={"action_bias": bias.tolist(),
                                            "action_scale": scale.tolist()})


def test_h1_walk_expert_loader_does_not_double_add_shared_dial_reference(tmp_path):
    import hashlib
    import json
    from scripts.tasks.robot.humanoid.train_box_push_rl import _load_walk_expert

    path = tmp_path / "trajectory.json"
    actions = np.zeros((3, 23), dtype=float)
    actions[:, :12] = 0.75  # Unloaded contact coordinates are not a teacher.
    actions[:, 12:] = np.linspace(-1.0, 1.0, 11)
    path.write_text(json.dumps({
        "actions": actions.tolist(),
        "states": [{"obs": np.full(91, i, dtype=float).tolist()} for i in range(4)],
        "task_signals": {
            "task_fallen": [0, 0, 0],
            "walk_left_steps": [0, 1, 1],
            "walk_right_steps": [0, 0, 1],
        },
    }))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    transform = {
        "center": "time_indexed_dial_reference",
        "action_bias": np.zeros(23).tolist(),
        "action_scale": np.ones(23).tolist(),
    }
    interface = {"action_layout": {"planner_reference_sha256": digest}}
    _, target, report = _load_walk_expert(
        path, observation_size=91, action_size=23,
        action_transform=transform, policy_interface=interface,
    )
    np.testing.assert_array_equal(target, 0.0)
    assert report["absolute_leg_action_rmse"] == 0.0
    assert report["target_semantics"] == (
        "zero_residual_around_verified_DIAL_planner_reference"
    )

    interface["action_layout"]["planner_reference_sha256"] = "wrong"
    with pytest.raises(ValueError, match="exactly match"):
        _load_walk_expert(
            path, observation_size=91, action_size=23,
            action_transform=transform, policy_interface=interface,
        )


def test_h1_walk_residual_expert_requires_current_safe_executed_prefix(tmp_path):
    import json
    from scripts.tasks.robot.humanoid.train_box_push_rl import _load_walk_expert

    path = tmp_path / "trajectory.json"
    interface = {"action_layout": {"action_size": 23}, "task": {"mode": "current"}}
    actions = np.zeros((3, 23), dtype=float)
    actions[:, 12:] = 0.025
    payload = {
        "actions": actions.tolist(),
        "states": [{"obs": np.full(91, i, dtype=float).tolist()} for i in range(4)],
        "execution_status": {
            "state": "aborted_unrecoverable",
            "executed_steps": 3,
            "metrics_scope": "actual_execution_prefix_only",
        },
        "task_signals": {
            "task_fallen": [0, 0, 0],
            "walk_left_steps": [0, 1, 1],
            "walk_right_steps": [0, 0, 1],
            "physics_samples_valid": [1, 1, 1],
            "physics_safety_margins": np.full((3, 5, 4), -0.1).tolist(),
            "task_metadata": {
                "reliability_contract": {"policy_interface": interface},
            },
        },
    }
    path.write_text(json.dumps(payload))
    transform = {
        "center": "time_indexed_dial_reference",
        "action_bias": np.zeros(23).tolist(),
        "action_scale": np.r_[np.full(12, 0.01), np.full(11, 0.05)].tolist(),
    }
    obs, target, report = _load_walk_expert(
        path, observation_size=91, action_size=23,
        action_transform=transform, policy_interface=interface,
        expert_target="residual",
    )
    assert obs.shape == (3, 91)
    np.testing.assert_array_equal(target[:, :12], 0.0)
    np.testing.assert_allclose(target[:, 12:], 0.5)
    assert report["target_semantics"] == (
        "zero_contact_primitive_plus_model_based_leg_residual"
    )

    payload["task_signals"]["physics_safety_margins"][1][2][0] = 0.01
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="violation-free fast-loop"):
        _load_walk_expert(
            path, observation_size=91, action_size=23,
            action_transform=transform, policy_interface=interface,
            expert_target="residual",
        )


def test_h1_behavior_anchor_is_training_only_and_skips_success_padding():
    from brax.envs.base import State
    from scripts.tasks.robot.humanoid.train_box_push_rl import (
        _BehaviorAnchoredH1Env,
    )

    class Env:
        domains = (object(),)
        action_size = observation_size = 2
        backend = "generalized"
        dt = 0.02

        def reset(self, rng):
            del rng
            return None

        def step(self, state, action):
            del action
            return state.replace(reward=jnp.float32(2.0))

    wrapper = _BehaviorAnchoredH1Env(
        Env(), lambda obs: jnp.ones_like(obs) * 0.5, weight=4.0,
    )
    state = State(
        pipeline_state=None, obs=jnp.zeros(2), reward=jnp.float32(0.0),
        done=jnp.float32(0.0), metrics={}, info={"success_padding": False},
    )
    assert float(wrapper.step(state, jnp.zeros(2)).reward) == pytest.approx(1.0)
    padding = state.replace(info={"success_padding": True})
    assert float(wrapper.step(padding, jnp.zeros(2)).reward) == pytest.approx(2.0)


def test_h1_walk_expert_set_preserves_per_trajectory_provenance(monkeypatch):
    import hashlib
    import json
    from scripts.tasks.robot.humanoid import train_box_push_rl as training

    reports = {
        "a.json": (2, "a" * 64, 0.1, 1, 2),
        "b.json": (3, "b" * 64, 0.2, 2, 1),
    }

    def load(path, **kwargs):
        del kwargs
        count, digest, rmse, left, right = reports[str(path)]
        return (
            np.full((count, 4), count, np.float32),
            np.full((count, 2), -count, np.float32),
            {
                "path": str(path), "sha256": digest,
                "transitions": count, "verified_left_steps": left,
                "verified_right_steps": right,
                "absolute_leg_action_rmse": rmse,
                "target_semantics": "model_based_residual",
            },
        )

    monkeypatch.setattr(training, "_load_walk_expert", load)
    obs, targets, report = training._load_walk_expert_set(
        ["a.json", "b.json"], observation_size=4, action_size=2,
        action_transform={}, expert_target="residual",
    )
    assert obs.shape == (5, 4)
    assert targets.shape == (5, 2)
    assert report["paths"] == ["a.json", "b.json"]
    assert report["trajectory_sha256"] == ["a" * 64, "b" * 64]
    assert report["trajectory_count"] == 2
    assert report["transitions"] == 5
    assert report["verified_left_steps"] == 1
    assert report["verified_right_steps"] == 1
    assert report["path"] is None
    expected = hashlib.sha256(json.dumps(
        ["a" * 64, "b" * 64], separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    assert report["sha256"] == expected
    assert report["absolute_leg_action_rmse"] == pytest.approx(
        np.sqrt((2 * 0.1 ** 2 + 3 * 0.2 ** 2) / 5)
    )


def test_h1_walk_expert_initialization_is_globally_zero_about_shared_reference(
        tmp_path):
    import hashlib
    import json
    import jax
    import jax.numpy as jnp
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from scripts.tasks.robot.humanoid.train_box_push_rl import (
        _pretrain_walk_policy_from_expert,
    )

    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({
        "actions": np.zeros((3, 23), dtype=float).tolist(),
        "states": [{"obs": np.full(91, i, dtype=float).tolist()}
                   for i in range(4)],
        "task_signals": {
            "task_fallen": [0, 0, 0],
            "walk_left_steps": [0, 1, 1],
            "walk_right_steps": [0, 0, 1],
        },
    }))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    transform = {
        "center": "time_indexed_dial_reference",
        "action_bias": np.zeros(23).tolist(),
        "action_scale": np.ones(23).tolist(),
    }
    interface = {"action_layout": {"planner_reference_sha256": digest}}
    env = SimpleNamespace(observation_size=91, action_size=23)
    params, report = _pretrain_walk_policy_from_expert(
        env, transform, interface, path, seed=101, hidden_sizes=(16, 16),
        normalize_observations=False, normalizer_std_eps=0.05,
        init_noise_std=0.25, steps=10,
    )
    networks = ppo_networks.make_ppo_networks(
        observation_size=91, action_size=23,
        preprocess_observations_fn=lambda x, y: x,
        policy_hidden_layer_sizes=(16, 16), init_noise_std=0.25,
    )
    normalizer, policy_params, _ = params
    observations = jax.random.normal(jax.random.PRNGKey(7), (8, 91)) * 100.0
    logits = networks.policy_network.apply(normalizer, policy_params, observations)
    actions = networks.parametric_action_distribution.mode(logits)
    np.testing.assert_allclose(np.asarray(actions), 0.0, atol=1e-7)
    distribution = networks.parametric_action_distribution.create_dist(logits)
    np.testing.assert_allclose(np.asarray(distribution.scale), 0.25, atol=1e-6)
    assert report["initialization"] == "analytic_global_zero_policy_mean"
    assert report["initial_exploration_std"] == pytest.approx(0.25)
    assert report["exploration_initialization"] == (
        "constant_tanh_normal_scale_head"
    )
    assert report["pretrain_steps"] == 0
    assert report["requested_pretrain_steps"] == 10


def test_policy_prior_current_observation_mode_avoids_second_dynamics_graph(monkeypatch):
    from genedynamics.learning import train_rl_policy as learning
    from genedynamics.learning.priors import rl
    from genedynamics.experiments.plugins.methods.contact_receding import make_mga

    captured = {}

    def fake_prior(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(rl, "RLPrior", fake_prior)
    env = SimpleNamespace(step=lambda state, action: (state, action))
    config = {
        "algo": "ppo", "observation_size": 4, "action_size": 2,
        "normalize_observations": False,
        "policy_hidden_layer_sizes": (8, 8),
    }
    learning.build_policy_prior(
        object(), config, env=env, Hsample=6, Hnode=2,
        rollout_mode="current_observation",
    )
    assert captured["rollout_step"] is None
    with pytest.raises(ValueError, match="rollout_mode"):
        learning.build_policy_prior(
            object(), config, env=env, rollout_mode="unknown",
        )
    with pytest.raises(ValueError, match="does not support stochastic"):
        make_mga(
            "humanoid_box_push", prior_rollout_mode="current_observation",
            prior_stochastic_samples=1,
        )


def test_h1_training_exports_actual_underlying_interface_and_atacom_options():
    from dataclasses import dataclass
    from scripts.tasks.robot.humanoid.train_box_push_rl import (
        _atacom_training_options, _environment_policy_action_transform,
        _environment_training_contract,
    )

    @dataclass
    class Config:
        dt: float = 0.02
        walk_leg_control: str = "joint_target"

    interface = {"action_layout": {"action_size": 23}}
    action_transform = {"action_bias": [0.0], "action_scale": [0.2]}
    domain = SimpleNamespace(
        _bcfg=Config(), _config=Config(), policy_interface=interface,
        policy_action_transform=action_transform,
    )
    saved, params, transform = _environment_training_contract([domain], {"Kc": 2.0, "action_limit": 0.8})
    assert saved == interface
    assert params == [{"dt": 0.02, "walk_leg_control": "joint_target"}]
    assert transform == {"Kc": 2.0, "action_limit": 0.8, "time_step": 0.02}
    assert _environment_policy_action_transform([domain]) == action_transform
    assert _atacom_training_options("walk") is None
    config = ExperimentConfig(name="atacom", env_name="humanoid_box_push", method="atacom",
                              output_dir="results/_development/test",
                              method_params={"Kc": 2.0, "action_limit": 0.8},
                              suites=[{"name": "p4_walk_push", "level": "push_walk"}])
    assert _atacom_training_options("atacom_p4", config) == {"Kc": 2.0, "action_limit": 0.8}
    legacy = SimpleNamespace(_bcfg=Config(), _config=Config(), policy_interface=None)
    with pytest.raises(ValueError, match="share policy interface"):
        _environment_training_contract([domain, legacy])
    with pytest.raises(ValueError, match="share policy action transform"):
        _environment_policy_action_transform([domain, legacy])


def test_registry_names():
    assert set(list_priors()) == {"rl", "diffusion"}


def test_elite_buffer_topk_and_capacity():
    eb = EliteBuffer(capacity=3, seed=0)
    for r in [0.1, 0.9, 0.5, 0.2]:
        eb.add(np.zeros((2, 3)), np.full((2, 3), r), r)
    assert len(eb) == 3                                  # lowest (0.1) evicted
    _, _, rw = eb.topk(2)
    assert np.allclose(rw, [0.9, 0.5])
    o, a, r = eb.sample(4, elite_frac=0.5)
    assert o.shape[0] == 4 and a.shape[0] == 4


# --- jax diffusion prior (pure jax, no brax) ---

def test_diffusion_prior_protocol_and_ops():
    dp = make_prior("diffusion", dim=12, nu=2, n_warm_nodes=6, n_steps=10)
    assert isinstance(dp, DiffusionPrior) and isinstance(dp, Prior)
    s = dp.score(jnp.zeros(12), 3)
    assert s.shape == (12,) and jnp.all(jnp.isfinite(s))
    dp.train(np.random.default_rng(0).standard_normal((24, 12)).astype("float32"),
             steps=5, batch=8)
    assert dp.warm_start(None).shape == (6, 2)


# --- prior= seam additive safety (mock rollout, no brax) ---

class _FakePrior:
    """Minimal Prior: constant warm-start, for additive-safety testing."""
    output_dim = 2

    def __init__(self, hn1, nu, val=0.3):
        self._u = jnp.full((hn1, nu), val, jnp.float32)

    def act(self, obs, *, key=None, deterministic=True):
        return self._u[0]

    def logp_of_sequence(self, obs_seq, act_seq):
        return jnp.array(0.0)

    def warm_start(self, state):
        return self._u


class _FakeStructuredPrior(_FakePrior):
    def sample_horizons(self, state, *, key, n_samples):
        return ProposalBatch(
            trajectories=jnp.broadcast_to(
                self._u, (int(n_samples),) + self._u.shape
            ),
            log_prob=jnp.zeros((int(n_samples),), jnp.float32),
            expert_id=jnp.ones((int(n_samples),), jnp.int32),
        )


COMMON = dict(nu=2, Hnode=4, Hsample=16, Nsample=32, Ndiffuse_init=4, Ndiffuse=2,
              temp_sample=0.06, action_limit=1.0, ctrl_dt=0.02, seed=0)
TARGET = jnp.array([0.4, -0.2])


def _mock_rollout(state, us, t0):
    return -jnp.sum((us - TARGET) ** 2, axis=-1)


def _replan(backend, rng):
    return backend.replan(jnp.zeros((1,)), backend.init_plan_var(),
                          backend.make_schedule(backend.Ndiffuse_init), rng)


def test_fake_prior_conforms():
    assert isinstance(_FakePrior(5, 2), Prior)


def test_additive_prior_preserves_gaussian_search_and_rejects_bad_expert():
    def build(value=None):
        backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
        backend.prior_mode = "additive"
        backend.prior_fallback_mode = "receding_incumbent"
        backend.score_risk_fn = lambda state, us, lam, rho: (
            -jnp.mean((us - TARGET) ** 2), jnp.zeros(4)
        )
        backend.risk_safe_fn = lambda risk: jnp.all(risk[:3] == 0)
        if value is not None:
            backend.prior = _FakeStructuredPrior(5, 2, val=value)
            backend.use_rl_prior = True
            backend.prior_stochastic_samples = 3
        return backend

    rng = jax.random.PRNGKey(31)
    reference = _replan(build(), rng)
    bad = build(-1.0)
    selected, info = bad.replan_with_info(
        jnp.zeros(1), bad.init_plan_var(),
        bad.make_schedule(bad.Ndiffuse_init), rng,
    )
    np.testing.assert_array_equal(selected, reference)
    assert float(info["additive_prior_selected"]) == 0.0
    good = build(TARGET)
    selected, info = good.replan_with_info(
        jnp.zeros(1), good.init_plan_var(),
        good.make_schedule(good.Ndiffuse_init), rng,
    )
    np.testing.assert_allclose(selected, jnp.broadcast_to(TARGET, (5, 2)))
    assert float(info["additive_prior_selected"]) == 1.0


@pytest.mark.parametrize("rejection", ["physical", "learned", "unsupported"])
def test_additive_prior_rejects_high_scoring_unsafe_expert(rejection):
    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior_mode = "additive"
    backend.prior_fallback_mode = "receding_incumbent"
    backend.prior = _FakeStructuredPrior(5, 2, val=TARGET)
    backend.use_rl_prior = True
    backend.prior_stochastic_samples = 3
    is_expert = lambda us: (jnp.mean((us - TARGET) ** 2) < 1.0e-8).astype(jnp.float32)
    backend.score_risk_fn = lambda state, us, lam, rho: (
        -jnp.mean((us - TARGET) ** 2),
        jnp.zeros(4).at[0].set(is_expert(us) if rejection == "physical" else 0.0),
    )
    backend.risk_safe_fn = lambda risk: jnp.all(risk[:3] == 0)
    if rejection != "physical":
        backend.reliability_sequence_feature_fn = lambda state, us: is_expert(us)[None]
        backend.reliability_support_mode = "joint"
        backend.reliability_ood_policy = "veto"
        backend.reliability_hard_limits = jnp.ones(4) * 0.5
        backend.reliability_model = SimpleNamespace(
            predict_upper=lambda features: jnp.broadcast_to(
                features if rejection == "learned" else jnp.zeros_like(features),
                (*features.shape[:-1], 4),
            ),
            support_score=lambda features: (
                2.0 * features[..., 0] if rejection == "unsupported"
                else jnp.zeros(features.shape[:-1])
            ),
        )
    selected, info = backend.replan_with_info(
        jnp.zeros(1), backend.init_plan_var(),
        backend.make_schedule(backend.Ndiffuse_init), jax.random.PRNGKey(31),
    )
    assert float(info["additive_prior_selected"]) == 0.0
    assert float(info["additive_prior_any_safe"]) == 0.0
    assert not np.allclose(selected, TARGET)


def test_logistic_reliability_handles_separable_contact_data():
    rng = np.random.default_rng(72)
    x = rng.normal(size=(300, len(PEG_INSERT_FEATURE_NAMES)))
    y = np.column_stack([
        x[:, 0] > 0.5, x[:, 1] > 1, x[:, 2] > 0,
        np.maximum(x[:, 3], 0),
    ]).astype(float)
    model = LinearReliabilityModel.fit(
        x[:200], y[:200], x[200:], y[200:],
        feature_names=PEG_INSERT_FEATURE_NAMES, risk_names=PEG_INSERT_RISK_NAMES,
        state_feature_count=32, support_state_feature_count=32,
        probability_risk_count=3, classification_probabilities=True,
    )
    probabilities = np.asarray(model.predict(x[200:]))[:, :3]
    assert np.all(np.isfinite(model.weights))
    assert np.mean((probabilities - y[200:, :3]) ** 2) < 0.12


def test_structured_prior_batch_is_jax_pytree_and_backward_compatible():
    prior = _FakeStructuredPrior(5, 2)
    assert isinstance(prior, Prior)
    assert isinstance(prior, StructuredPrior)
    batch = prior.sample_horizons(
        None, key=jax.random.PRNGKey(0), n_samples=3
    )
    assert batch.trajectories.shape == (3, 5, 2)
    assert batch.log_prob.shape == (3,)
    assert batch.expert_id.shape == (3,)
    assert len(jax.tree_util.tree_leaves(batch)) == 3


def test_prior_seam_additive_safety():
    rng = jax.random.PRNGKey(5)
    base = _replan(MgaBackendJax(rollout_fn=_mock_rollout, **COMMON), rng)

    # lambda_shift=1.0 => U_init = 1*U_shift + 0*U_rl = U_shift => byte-identical
    b1 = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    b1.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    b1.use_rl_prior, b1.prior_lambda_shift = True, 1.0
    # Isolate the legacy warm-start seam. The full Phase-4 path intentionally
    # keeps U_rl as a separate candidate even when lambda_shift=1.
    b1.prior_include_incumbent = False
    b1.prior_trust_radius = 0.0
    b1.prior_acceptance = False
    y1 = _replan(b1, rng)
    assert float(jnp.max(jnp.abs(y1 - base))) < 1e-6      # prior mix is identity at lam=1

    # lambda_shift=0.0 => U_init = U_rl (nonzero) => changes the plan
    b0 = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    b0.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    b0.use_rl_prior, b0.prior_lambda_shift = True, 0.0
    y0 = _replan(b0, rng)
    assert float(jnp.max(jnp.abs(y0 - base))) > 1e-4      # warm-start replaced => differs
    assert jnp.all(jnp.isfinite(y0))


def test_structured_proposals_use_fixed_gaussian_budget_and_are_opt_in():
    rng = jax.random.PRNGKey(23)
    legacy = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    legacy.prior = _FakeStructuredPrior(
        COMMON["Hnode"] + 1, COMMON["nu"], val=0.3
    )
    legacy.use_rl_prior = True
    legacy.prior_stochastic_samples = 0
    y_legacy = _replan(legacy, rng)

    structured = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    structured.prior = legacy.prior
    structured.use_rl_prior = True
    structured.prior_stochastic_samples = 4
    y_structured, info = structured.replan_with_info(
        jnp.zeros((1,)), structured.init_plan_var(),
        structured.make_schedule(structured.Ndiffuse_init), rng,
    )
    assert y_structured.shape == y_legacy.shape
    assert jnp.all(jnp.isfinite(y_structured))
    assert float(info["proposal_stochastic_count"]) == 4.0
    # Four structured candidates replace four Gaussian candidates; the reverse
    # kernel's declared stochastic budget itself never changes.
    assert structured.Nsample == legacy.Nsample == COMMON["Nsample"]


def test_prior_none_is_unchanged():
    rng = jax.random.PRNGKey(9)
    base = _replan(MgaBackendJax(rollout_fn=_mock_rollout, **COMMON), rng)
    b = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)   # prior stays None
    assert b.prior is None
    assert float(jnp.max(jnp.abs(_replan(b, rng) - base))) == 0.0


def test_prior_acceptance_and_force_veto():
    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    U_rl = jnp.full((COMMON["Hnode"] + 1, COMMON["nu"]), 0.3)
    refined = jnp.broadcast_to(TARGET, U_rl.shape)

    backend.risk_fn = lambda state, us: jnp.zeros((4,), jnp.float32)
    selected, info = backend._accept_refinement(
        X0 := jnp.zeros((1,)), U_rl, refined, jnp.float32(0.0)
    )
    np.testing.assert_allclose(selected, refined)
    assert float(info["prior_accepted"]) == 1.0
    assert float(info["prior_predicted_improvement"]) > 0.0

    def force_risk(state, us):
        violation = jnp.mean((us[:, 0] > 0.35).astype(jnp.float32))
        return jnp.asarray([violation, 0.0, 0.0, 0.0])

    backend.risk_fn = force_risk
    selected, info = backend._accept_refinement(
        X0, U_rl, refined, jnp.float32(0.0)
    )
    np.testing.assert_allclose(selected, U_rl)
    assert float(info["prior_accepted"]) == 0.0
    assert float(info["prior_force_veto"]) == 1.0


def test_task_owned_risk_comparison_allows_safe_quality_tradeoff():
    """A task may score soft quality risks instead of vetoing them twice."""
    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    incumbent = jnp.full((COMMON["Hnode"] + 1, COMMON["nu"]), 0.3)
    refined = jnp.broadcast_to(TARGET, incumbent.shape)

    def quality_risk(state, us):
        del state
        return jnp.asarray([
            0.0,
            0.0,
            jnp.mean(jnp.abs(us)),
            jnp.mean(us * us),
        ])

    backend.risk_fn = quality_risk
    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), incumbent, refined, jnp.float32(1.0)
    )
    np.testing.assert_allclose(selected, incumbent)
    assert float(info["prior_risk_ok"]) == 0.0

    backend.risk_compare_fn = (
        lambda candidate, fallback, tolerance:
        candidate[0] <= fallback[0] + tolerance[0]
    )
    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), incumbent, refined, jnp.float32(1.0)
    )
    np.testing.assert_allclose(selected, refined)
    assert float(info["prior_risk_ok"]) == 1.0


def test_learned_reliability_is_additional_acceptance_veto():
    class Reliability:
        def predict_upper(self, features):
            return jnp.asarray([features[0], 0.0, 0.0, 0.0])

        def support_score(self, features):
            return jnp.asarray(0.0)

    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    backend.risk_fn = lambda state, us: jnp.zeros((4,), jnp.float32)
    backend.reliability_model = Reliability()
    backend.reliability_feature_fn = lambda state, action: action[:1]
    backend.reliability_force_limit = 0.35
    backend.reliability_deformation_limit = float("inf")
    backend.reliability_risk_tolerance = jnp.zeros((4,), jnp.float32)
    incumbent = jnp.full((COMMON["Hnode"] + 1, COMMON["nu"]), 0.3)
    refined = jnp.broadcast_to(TARGET, incumbent.shape)

    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), incumbent, refined, jnp.float32(1.0)
    )
    np.testing.assert_allclose(selected, incumbent)
    assert float(info["prior_accepted"]) == 0.0
    assert float(info["prior_force_veto"]) == 1.0
    assert "reliability_risk_refined" in info


def test_ood_reliability_can_abstain_to_model_based_certificate():
    """OOD confidence is unknown, not a permanent unsafe classification."""

    class Reliability:
        def predict_upper(self, features):
            del features
            return jnp.asarray([1.0, 1.0, 10.0, 1.0])

        def support_score(self, features):
            del features
            return jnp.asarray(2.0)

    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    backend.risk_fn = lambda state, us: jnp.zeros((4,), jnp.float32)
    backend.reliability_model = Reliability()
    backend.reliability_feature_fn = lambda state, action: action[:1]
    backend.reliability_deformation_limit = 0.5
    incumbent = jnp.full((COMMON["Hnode"] + 1, COMMON["nu"]), 0.3)
    refined = jnp.broadcast_to(TARGET, incumbent.shape)

    # The conservative default retains the historical OOD veto.
    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), incumbent, refined, jnp.float32(1.0)
    )
    np.testing.assert_allclose(selected, incumbent)
    assert float(info["prior_accepted"]) == 0.0

    # Surface scanning opts into a complete task-owned model certificate.  The
    # unsupported learned model abstains and that certificate accepts the safe
    # improvement instead of freezing the incumbent forever.
    backend.reliability_ood_policy = "model_based"
    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), incumbent, refined, jnp.float32(1.0)
    )
    np.testing.assert_allclose(selected, refined)
    assert float(info["prior_accepted"]) == 1.0
    assert float(info["reliability_abstained"]) == 1.0


def test_sequence_reliability_uses_candidate_horizon_when_available():
    class Reliability:
        def predict_upper(self, features):
            return jnp.asarray([features[0], 0.0, 0.0, 0.0])

        def support_score(self, features):
            return jnp.asarray(0.0)

    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    backend.risk_fn = lambda state, us: jnp.zeros((4,), jnp.float32)
    backend.reliability_model = Reliability()
    backend.reliability_feature_fn = lambda state, action: action[:1] * 0.0
    backend.reliability_sequence_feature_fn = (
        lambda state, actions: jnp.max(actions[:, :1], axis=0)
    )
    backend.reliability_force_limit = 0.35
    backend.reliability_deformation_limit = float("inf")
    backend.reliability_risk_tolerance = jnp.zeros((4,), jnp.float32)
    incumbent = jnp.full((COMMON["Hnode"] + 1, COMMON["nu"]), 0.3)
    refined = incumbent.at[-1, 0].set(0.8)

    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), incumbent, refined, jnp.float32(1.0)
    )
    np.testing.assert_allclose(selected, incumbent)
    assert float(info["prior_force_veto"]) == 1.0


def test_task_reliability_hard_limits_cover_every_risk_channel():
    class Reliability:
        def predict_upper(self, features):
            return jnp.asarray([0.0, features[0], 0.0, 0.0])

        def support_score(self, features):
            return jnp.asarray(0.0)

    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    backend.risk_fn = lambda state, us: jnp.zeros((4,), jnp.float32)
    backend.reliability_model = Reliability()
    backend.reliability_feature_fn = lambda state, action: action[:1]
    backend.reliability_sequence_feature_fn = None
    backend.reliability_hard_limits = jnp.asarray(
        [1.0, 0.35, 1.0, jnp.inf], jnp.float32
    )
    backend.reliability_risk_tolerance = jnp.ones((4,), jnp.float32)
    incumbent = jnp.full((COMMON["Hnode"] + 1, COMMON["nu"]), 0.3)
    refined = jnp.broadcast_to(TARGET, incumbent.shape)

    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), incumbent, refined, jnp.float32(1.0)
    )
    np.testing.assert_allclose(selected, incumbent)
    assert float(info["prior_force_veto"]) == 1.0


def test_learned_reliability_rejects_unsafe_incumbent_for_emergency():
    class Reliability:
        def predict_upper(self, features):
            return jnp.asarray([features[0], 0.0, 0.0, 0.0])

        def support_score(self, features):
            return jnp.asarray(0.0)

    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    backend.risk_fn = lambda state, us: jnp.zeros((4,), jnp.float32)
    backend.reliability_model = Reliability()
    backend.reliability_feature_fn = lambda state, action: action[:1]
    backend.reliability_sequence_feature_fn = None
    backend.reliability_hard_limits = jnp.asarray(
        [0.2, 1.0, 1.0, jnp.inf], jnp.float32
    )
    backend.reliability_risk_tolerance = jnp.ones((4,), jnp.float32)
    incumbent = jnp.full((COMMON["Hnode"] + 1, COMMON["nu"]), 0.3)
    refined = jnp.full_like(incumbent, 0.4)
    emergency = jnp.zeros_like(incumbent)

    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), incumbent, refined, jnp.float32(1.0),
        emergency=emergency,
    )
    np.testing.assert_allclose(selected, emergency)
    assert float(info["incumbent_revalidated_safe"]) == 0.0
    assert float(info["refined_revalidated_safe"]) == 0.0
    assert float(info["emergency_selected"]) == 1.0


def test_linear_reliability_fit_calibration_and_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(160, len(FEATURE_NAMES))).astype(np.float32)
    weights = rng.normal(scale=0.1, size=(len(FEATURE_NAMES), len(RISK_NAMES)))
    train_y = np.maximum(train_x @ weights + 0.2, 0.0).astype(np.float32)
    cal_x = rng.normal(size=(80, len(FEATURE_NAMES))).astype(np.float32)
    cal_y = np.maximum(cal_x @ weights + 0.25, 0.0).astype(np.float32)
    model = LinearReliabilityModel.fit(
        train_x, train_y, cal_x, cal_y, quantile=0.9
    )
    upper = np.asarray(model.predict_upper(cal_x))
    assert upper.shape == cal_y.shape
    assert np.all(np.mean(cal_y <= upper + 1e-6, axis=0) >= 0.9)
    assert np.mean(np.asarray(model.in_support(cal_x))) >= 0.9
    assert not bool(model.in_support(np.full(len(FEATURE_NAMES), 1.0e6)))
    action_only_ood = np.zeros(len(FEATURE_NAMES), np.float32)
    action_only_ood[model.state_feature_count:] = 1.0e6
    assert bool(model.support_score_state(action_only_ood) <= 1.0)
    assert bool(model.support_score_action(action_only_ood) > 1.0)

    path = tmp_path / "reliability.json"
    model.save(path, metadata={"split": "unit"})
    loaded = LinearReliabilityModel.load(path)
    np.testing.assert_allclose(loaded.predict_upper(cal_x), upper, atol=1e-6)


def test_peg_insert_reliability_schema_roundtrip(tmp_path):
    rng = np.random.default_rng(17)
    nx, ny = len(PEG_INSERT_FEATURE_NAMES), len(PEG_INSERT_RISK_NAMES)
    train_x = rng.normal(size=(192, nx)).astype(np.float32)
    train_y = np.maximum(rng.normal(0.1, 0.05, size=(192, ny)), 0.0).astype(np.float32)
    cal_x = rng.normal(size=(96, nx)).astype(np.float32)
    cal_y = np.maximum(rng.normal(0.12, 0.05, size=(96, ny)), 0.0).astype(np.float32)
    model = LinearReliabilityModel.fit(
        train_x,
        train_y,
        cal_x,
        cal_y,
        feature_names=PEG_INSERT_FEATURE_NAMES,
        risk_names=PEG_INSERT_RISK_NAMES,
        state_feature_count=32,
        support_state_feature_count=12,
        probability_risk_count=3,
        classification_probabilities=True,
        basis_mode="contact_phase",
    )
    assert model.predict_upper(cal_x).shape == (len(cal_x), ny)
    assert model.state_feature_count == 32
    assert model.support_state_feature_count == 12
    assert model.classification_probabilities
    assert model.basis_mode == "contact_phase"
    path = tmp_path / "peg_reliability.json"
    model.save(path)
    loaded = LinearReliabilityModel.load(path)
    assert loaded.feature_names == PEG_INSERT_FEATURE_NAMES
    assert loaded.risk_names == PEG_INSERT_RISK_NAMES
    assert loaded.probability_risk_count == 3
    assert loaded.basis_mode == "contact_phase"
    np.testing.assert_allclose(
        loaded.predict_upper(cal_x), model.predict_upper(cal_x), atol=1e-6
    )


def test_peg_insert_reliability_labels_the_committed_horizon():
    n = 5
    actions = np.zeros((n, 13), np.float32)
    actions[1:4, 2] = [0.1, 0.2, 0.3]
    series = {
        "actions": actions.tolist(),
        # State t=0 is at the socket plane, so row t=1 owns three committed
        # actions and must see the violation at post-state t=3.
        "pose": np.zeros((n, 3), np.float32).tolist(),
        "angle_vec": np.zeros((n, 3), np.float32).tolist(),
        "measured_lateral_force": [0.0] * n,
        "measured_axial_force": [0.0] * n,
        "measured_bending_torque": [0.0] * n,
        "measured_wrench_delta": [0.0] * n,
        "contact_count": [0.0] * n,
        "stall_steps": [0.0] * n,
        "force_violation": [0.0, 0.0, 0.0, 1.0, 0.0],
        "torque_violation": [0.0] * n,
        "jammed": [0.0] * n,
        "axial_force": [0.0] * n,
        "socket_depth": 0.04,
        "lateral_force_limit": 20.0,
        "f_max": 30.0,
        "bending_torque_limit": 1.5,
        "jam_dwell_steps": 4,
        "f_target": 10.0,
    }
    x, y, provenance = samples_from_records([{
        "task": "manipulator_peg_insert",
        "suite": "unit",
        "seed": 7,
        "series": series,
    }])
    assert x.shape == (n - 1, len(PEG_INSERT_FEATURE_NAMES))
    assert y.shape == (n - 1, len(PEG_INSERT_RISK_NAMES))
    assert y[0, 0] == 1.0
    axial_idx = PEG_INSERT_FEATURE_NAMES.index("cumulative_axial_action")
    assert np.isclose(x[0, axial_idx], 0.6)
    assert provenance[0]["step"] == 1


def test_peg_insert_unified_result_preserves_hidden_execution_signals(tmp_path):
    import json

    n = 5
    run = tmp_path / "level_ood_sensing" / "seed_106"
    trajectory_dir = run / "trajectory"
    trajectory_dir.mkdir(parents=True)
    actions = np.zeros((n, 13), np.float32)
    actions[:, 2] = 0.25
    scalar = [0.0] * n
    signals = {
        "controls": actions.tolist(),
        "pose": np.zeros((n, 3), np.float32).tolist(),
        "angle_vec": np.zeros((n, 3), np.float32).tolist(),
        "measured_lateral_force": [0.70710677] * n,
        "measured_axial_force": scalar,
        "measured_bending_torque": scalar,
        "measured_wrench_delta": scalar,
        "contact_count": scalar,
        "stall_steps": scalar,
        "force_violation": [0.0, 0.0, 0.0, 1.0, 0.0],
        "torque_violation": scalar,
        "jammed": scalar,
        "axial_force": scalar,
    }
    trajectory = {"actions": actions.tolist(), "task_signals": signals}
    trajectory_path = trajectory_dir / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    result = {
        "suite": "ood_sensing",
        "level": "wide",
        "seed": 106,
        "config_snapshot": {
            "env_name": "manipulator_peg_insert",
            "method": "mga",
            "name": "reliability_collection",
            "env_params": {"f_max": 30.0, "jam_dwell_steps": 4},
            "execution_env_params": {
                "level": "wide", "action_delay_steps": 1,
                "sensor_delay_steps": 1,
            },
            "metadata": {"policy_training_seed": 0},
        },
        "provenance": {"checkpoints": {"policy_ckpt": {"sha256": "a" * 64}}},
    }
    result_path = run / "results.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    record = _peg_insert_record_from_result(result_path)
    assert record["sample_source"] == "executed_commit_horizon"
    assert record["series"]["action_delay_steps"] == 1
    assert record["series"]["sensor_delay_steps"] == 1
    assert record["series"]["measured_lateral_force"] == [0.70710677] * n
    assert record["policy_checkpoint_sha256"] == "a" * 64
    assert record["behavior_method"] == "mga"
    x, y, provenance = samples_from_records([record])
    assert x.shape == (n - 1, len(PEG_INSERT_FEATURE_NAMES))
    assert y[0, 0] == 1.0
    assert provenance[0]["sample_source"] == "executed_commit_horizon"
    assert provenance[0]["source_result"] == str(result_path)
    assert provenance[0]["behavior_method"] == "mga"


def test_peg_insert_reliability_extends_labels_for_execution_delay():
    n = 6
    actions = np.zeros((n, 13), np.float32)
    series = {
        "actions": actions.tolist(),
        "pose": np.zeros((n, 3), np.float32).tolist(),
        "angle_vec": np.zeros((n, 3), np.float32).tolist(),
        "measured_lateral_force": [0.0] * n,
        "measured_axial_force": [0.0] * n,
        "measured_bending_torque": [0.0] * n,
        "measured_wrench_delta": [0.0] * n,
        "contact_count": [0.0] * n,
        "stall_steps": [0.0] * n,
        # For row t=1, a delayed candidate owns post-states t=1..4.
        "force_violation": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "torque_violation": [0.0] * n,
        "jammed": [0.0] * n,
        "axial_force": [0.0] * n,
        "socket_depth": 0.04,
        "lateral_force_limit": 20.0,
        "f_max": 30.0,
        "bending_torque_limit": 1.5,
        "jam_dwell_steps": 4,
        "f_target": 10.0,
        "action_delay_steps": 1,
    }
    _, y, _ = samples_from_records([{
        "task": "manipulator_peg_insert",
        "suite": "unit",
        "seed": 7,
        "series": series,
    }])
    assert y[0, 0] == 1.0


def test_peg_insert_reliability_jam_labels_require_applied_insertion_intent():
    n = 6
    actions = np.zeros((n, 13), np.float32)
    # Canonical f_min=0 maps to raw=-0.6 for [-10, 40] N.  The old trace says
    # the residual contact was stalled/jammed, but the delayed applied command
    # is already a zero-force hold, so the current contract must not label jam.
    actions[:, 12] = -0.6
    series = {
        "actions": actions.tolist(),
        "pose": np.zeros((n, 3), np.float32).tolist(),
        "angle_vec": np.zeros((n, 3), np.float32).tolist(),
        "measured_lateral_force": [0.0] * n,
        "measured_axial_force": [10.0] * n,
        "measured_bending_torque": [0.0] * n,
        "measured_wrench_delta": [0.0] * n,
        "contact_count": [4.0] * n,
        "stall_steps": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "force_violation": [0.0] * n,
        "torque_violation": [0.0] * n,
        "jammed": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "axial_force": [10.0] * n,
        "socket_depth": 0.04,
        "lateral_force_limit": 20.0,
        "f_min": 0.0,
        "f_max": 30.0,
        "f_cmd_pad": 10.0,
        "bending_torque_limit": 1.5,
        "jam_dwell_steps": 4,
        "f_target": 10.0,
        "action_delay_steps": 1,
    }
    _, safe_y, _ = samples_from_records([{
        "task": "manipulator_peg_insert",
        "suite": "unit",
        "seed": 7,
        "series": series,
    }])
    assert np.max(safe_y[:, 2]) == 0.0

    # Positive axial commands are true insertion intent and must preserve the
    # same historical jam event under the relabeling pass.
    insertion_series = dict(series)
    insertion_actions = actions.copy()
    insertion_actions[:, 2] = 0.5
    insertion_series["actions"] = insertion_actions.tolist()
    _, jam_y, _ = samples_from_records([{
        "task": "manipulator_peg_insert",
        "suite": "unit",
        "seed": 7,
        "series": insertion_series,
    }])
    assert np.max(jam_y[:, 2]) == 1.0


def test_peg_insert_reliability_promotion_requires_paired_non_regression(
    tmp_path,
):
    import hashlib
    import json

    from scripts.tasks.robot.arm.promote_mga_reliability import audit

    policy_hash = "a" * 64
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"metadata": {
        "task": "manipulator_peg_insert",
        "fit_scope": "development_fit_only",
        "performance_validated": False,
        "promotion_eligible": False,
        "training_seeds": [106],
        "calibration_seeds": [107],
        "training_policy_checkpoint_sha256": [policy_hash],
        "calibration_policy_checkpoint_sha256": [policy_hash],
    }}))
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    learned_root = tmp_path / "learned"
    control_root = tmp_path / "control"

    def write_result(root, seed, *, learned, safe):
        run = root / f"seed_{seed}"
        trajectory = run / "trajectory"
        trajectory.mkdir(parents=True, exist_ok=True)
        (trajectory / "trajectory.json").write_text(json.dumps({
            "actions": [[float(learned), 0.0] for _ in range(2)],
        }))
        method_params = {
            "controller_method": "mga_controllable_gate",
            "learned_reliability": learned,
            "reliability_ckpt": "candidate.json" if learned else None,
        }
        if learned:
            method_params["reliability_validation_authoritative"] = True
        payload = {
            "suite": "id_wide",
            "seed": seed,
            "config_snapshot": {
                "env_name": "manipulator_peg_insert",
                "method_params": method_params,
                "env_params": {"level": "wide"},
                "execution_env_params": {},
            },
            "component_contract": {
                "learned_reliability": learned,
                "reliability_gate_authoritative": learned,
                "reliability_validation_authoritative": learned,
                "reliability_promotion_eligible": False,
            },
            "provenance": {"checkpoints": {
                "policy_ckpt": {"sha256": policy_hash},
                **({"reliability_ckpt": {"sha256": candidate_hash}}
                   if learned else {}),
            }},
            "metrics": {"peg_insert_metrics": {
                "insertion_success": 1.0,
                "safe_insertion_success": float(safe),
                "max_insertion_depth": 0.033,
            }},
            "diagnostics": (
                {"reliability_abstained": 0.1} if learned else {}
            ),
        }
        (run / "results.json").write_text(json.dumps(payload))

    for seed in (110, 111):
        write_result(learned_root, seed, learned=True, safe=True)
        write_result(control_root, seed, learned=False, safe=True)
    report = audit(
        candidate,
        [str(learned_root)],
        [str(control_root)],
        required_suites=("id_wide",),
        minimum_seeds=2,
        formal_seeds=set(range(10)),
        expected_policy_sha256=policy_hash,
    )
    assert report["passed"] is True
    assert report["changed_action_pairs"] == 2

    write_result(learned_root, 111, learned=True, safe=False)
    blocked = audit(
        candidate,
        [str(learned_root)],
        [str(control_root)],
        required_suites=("id_wide",),
        minimum_seeds=2,
        formal_seeds=set(range(10)),
        expected_policy_sha256=policy_hash,
    )
    assert (
        "candidate_safe_success_below_control"
        in blocked["blocking_reasons"]
    )


def test_no_acceptance_skips_counterfactual_rollouts():
    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior_acceptance = False
    backend.risk_fn = lambda state, us: (_ for _ in ()).throw(
        AssertionError("unchecked ablation must not evaluate risk")
    )
    U_rl = jnp.full((COMMON["Hnode"] + 1, COMMON["nu"]), 0.3)
    refined = jnp.broadcast_to(TARGET, U_rl.shape)
    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), U_rl, refined, jnp.float32(0.0)
    )
    np.testing.assert_allclose(selected, refined)
    assert float(info["prior_accepted"]) == 1.0
    assert bool(jnp.isnan(info["prior_predicted_improvement"]))


def test_receding_incumbent_fallback_never_returns_raw_policy():
    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    backend.prior_fallback_mode = "receding_incumbent"
    incumbent = jnp.zeros((COMMON["Hnode"] + 1, COMMON["nu"]), jnp.float32)
    worse = jnp.full_like(incumbent, -0.8)
    backend.risk_fn = lambda state, us: jnp.zeros((4,), jnp.float32)

    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), incumbent, worse, jnp.float32(1.0)
    )
    np.testing.assert_allclose(selected, incumbent)
    assert float(info["prior_accepted"]) == 0.0
    assert "prior_risk_incumbent" in info

    # Cold start is no longer an unconditional refined-plan bypass.  Both
    # candidates receive the same model-based certificate and the safe,
    # higher-scoring incumbent remains selected.
    cold, cold_info = backend._accept_refinement(
        jnp.zeros((1,)), incumbent, worse, jnp.float32(0.0)
    )
    np.testing.assert_allclose(cold, incumbent)
    assert float(cold_info["prior_accepted"]) == 0.0


def test_atacom_pareto_incumbent_can_dominate_receding_and_refinement():
    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    backend.prior_fallback_mode = "receding_incumbent"
    backend.risk_fn = lambda state, us: jnp.zeros((4,), jnp.float32)
    receding = jnp.zeros((5, 2), jnp.float32)
    atacom = jnp.broadcast_to(TARGET, receding.shape)
    worse_refinement = jnp.full_like(receding, -0.8)
    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), receding, worse_refinement, jnp.float32(1.0),
        atacom,
    )
    np.testing.assert_allclose(selected, atacom)
    assert float(info["atacom_incumbent_selected"]) == 1.0
    assert float(info["prior_accepted"]) == 0.0
    assert float(info["prior_score_fallback"]) == float(
        info["prior_score_atacom"]
    )


def test_atacom_default_is_lower_bound_when_refinement_is_worse():
    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    backend.prior_fallback_mode = "receding_incumbent"
    backend.prior_atacom_default = True
    backend.prior_atacom_strict_risk = True
    backend.risk_fn = lambda state, us: jnp.asarray([
        0.0, 0.0, jnp.mean(jnp.abs(us[:, 0])), 0.0
    ])
    receding = jnp.broadcast_to(TARGET, (5, 2))
    atacom = jnp.zeros_like(receding)
    risky = jnp.full_like(receding, 0.8)
    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), receding, risky, jnp.float32(2.0), atacom
    )
    np.testing.assert_allclose(selected, atacom)
    assert float(info["atacom_incumbent_selected"]) == 1.0
    assert float(info["prior_accepted"]) == 0.0


def test_receding_incumbent_uses_terminal_hold_shift():
    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    nodes = jnp.linspace(
        -0.4, 0.8, (COMMON["Hnode"] + 1) * COMMON["nu"]
    ).reshape(COMMON["Hnode"] + 1, COMMON["nu"])

    backend.prior_fallback_mode = "rl"
    np.testing.assert_allclose(
        backend.shift(nodes), backend.spline.shift_nodes(nodes)
    )
    backend.prior_fallback_mode = "receding_incumbent"
    np.testing.assert_allclose(
        backend.shift(nodes), backend.spline.shift_nodes_terminal_hold(nodes)
    )


def test_prior_trust_region_bounds_refinement():
    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    U_rl = jnp.full((COMMON["Hnode"] + 1, COMMON["nu"]), 0.2)
    backend.prior = _FakePrior(
        COMMON["Hnode"] + 1, COMMON["nu"], val=0.2
    )
    backend.use_rl_prior = True
    backend.prior_trust_radius = 0.1
    projected = backend._project_to_prior(
        jnp.full_like(U_rl, 1.0), U_rl
    )
    rms = jnp.sqrt(jnp.mean((projected - U_rl) ** 2))
    assert float(rms) <= 0.100001


def test_union_trust_region_preserves_nearest_expert_mode():
    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    backend.prior_trust_radius = 0.1
    centers = jnp.stack([
        jnp.full((5, 2), -0.6, jnp.float32),
        jnp.full((5, 2), 0.6, jnp.float32),
    ])
    plans = jnp.stack([
        jnp.full((5, 2), -0.9, jnp.float32),
        jnp.full((5, 2), 0.9, jnp.float32),
    ])
    projected = backend._project_to_prior_union(plans, centers)
    np.testing.assert_allclose(projected[0], -0.7, atol=2e-5)
    np.testing.assert_allclose(projected[1], 0.7, atol=2e-5)


def test_raw_rl_controller_matches_receding_horizon_contract():
    class State(NamedTuple):
        obs: jax.Array

    class Env:
        def step(self, state, action):
            return State(obs=state.obs + action)

    controller = RLPolicyController(
        Env(), lambda obs, key: jnp.ones_like(obs) * 0.25
    )
    x0 = State(obs=jnp.zeros((2,), jnp.float32))
    result = controller.run_receding(
        x0,
        3,
        jax.random.PRNGKey(0),
        collect_states=False,
    )
    assert result.horizon == 3
    assert len(result.states) == 2
    np.testing.assert_allclose(result.states[-1].obs, [0.75, 0.75])


def test_formal_surface_algorithm_matrix_and_policy_routing():
    root = Path("configs/arm/surface_scan")
    paths = [root / "main/mga.yaml", *sorted((root / "baseline").glob("*.yaml"))]
    configs = [ExperimentConfig.from_yaml(path) for path in paths]
    assert len(configs) == 8
    assert {c.name for c in configs} == {
        "dial", "mppi", "pegasusflow", "issa", "atacom",
        "standalone_rl", "model_based_only", "mga",
    }
    assert {suite["name"] for suite in configs[0].suites} == {
        "rigid_plane", "rigid_cylinder", "rigid_convex", "rigid_bumpy",
        "rigid_unseen", "soft_plane", "soft_cylinder", "soft_convex",
        "soft_bumpy", "soft_unseen", "hybrid_stripes",
        "hybrid_center_hard", "hybrid_center_soft",
    }
    assert all(c.seeds == list(range(10)) for c in configs)
    for cfg in configs:
        uses_policy = cfg.name in {
            "standalone_rl", "issa", "atacom", "mga",
        }
        assert ("policy_ckpt" in cfg.method_params) == uses_policy
    full = next(c for c in configs if c.name == "mga")
    params = full.method_params
    assert params["prior_acceptance"] is True
    assert params["prior_fallback_mode"] == "receding_incumbent"
    assert params["reliability_deformation_limit"] == 0.5
    assert "atacom_policy_ckpt" in params
    assert params["prior_stochastic_samples"] == 8
    assert params["prior_atacom_samples"] == 8
    assert params["prior_union_trust"] is True
    assert params["prior_risk_tolerance"] == [0.0, 0.0, 0.0, 0.0]
    assert params["reliability_risk_tolerance"] == [0.0, 0.0, 0.0, 0.0]
    soft_unseen = next(s for s in full.suites if s["name"] == "soft_unseen")
    resolved = full.for_suite(soft_unseen)
    assert resolved.env_params["f_target"] == 15.0
    assert resolved.env_params["grav_comp"] == 0.45
    training = full.metadata["training"]["rl"]
    assert training["num_timesteps"] == 2_000_000
    assert training["seed"] == 0
    assert training["checkpoint"].endswith("shared_ppo_seed{seed}.pkl")


def test_prior_diagnostics_aggregate_acceptance_and_risk():
    result = SimpleNamespace(infos=[
        {
            "prior_accepted": jnp.asarray(1.0),
            "prior_predicted_improvement": jnp.asarray(0.2),
            "prior_risk_ok": jnp.asarray(1.0),
            "prior_force_veto": jnp.asarray(0.0),
            "prior_risk_rl": jnp.asarray([0.0, 0.1, 0.2, 0.3]),
            "prior_risk_refined": jnp.asarray([0.0, 0.0, 0.1, 0.2]),
        },
        {
            "prior_accepted": jnp.asarray(0.0),
            "prior_predicted_improvement": jnp.asarray(-0.1),
            "prior_risk_ok": jnp.asarray(0.0),
            "prior_force_veto": jnp.asarray(1.0),
            "prior_risk_rl": jnp.asarray([0.0, 0.2, 0.4, 0.6]),
            "prior_risk_refined": jnp.asarray([0.1, 0.3, 0.5, 0.7]),
        },
    ])
    diagnostics = aggregate_receding_diagnostics(result.infos)
    assert diagnostics["prior_acceptance_rate"] == pytest.approx(0.5)
    assert diagnostics["prior_fallback_rate"] == pytest.approx(0.5)
    assert diagnostics["prior_force_veto_rate"] == pytest.approx(0.5)
    assert diagnostics["prior_risk_rl_contact_loss"] == pytest.approx(0.15)
    insert_diagnostics = aggregate_receding_diagnostics(
        result.infos, task="manipulator_peg_insert"
    )
    assert insert_diagnostics["prior_risk_rl_torque_violation"] == pytest.approx(
        0.15
    )
    assert insert_diagnostics["prior_risk_rl_jam"] == pytest.approx(0.3)


def test_best_eval_selector_pairs_params_with_later_metrics():
    from scripts.tasks.robot.arm.train_rl_baseline import _BestEvalSelector

    selector = _BestEvalSelector("eval/episode_realized_progress")
    selector.observe(0, {"eval/episode_realized_progress": 0.0})
    selector.capture(0, None, jnp.asarray([0.0]))
    selector.capture(100, None, jnp.asarray([1.0]))
    selector.observe(100, {
        "eval/episode_reward": -2.0,
        "eval/episode_realized_progress": 0.8,
    })
    selector.capture(200, None, jnp.asarray([2.0]))
    selector.observe(200, {
        "eval/episode_reward": -1.0,
        "eval/episode_realized_progress": 0.5,
    })

    assert selector.step == 100
    assert selector.score == pytest.approx(0.8)
    assert selector.metrics["eval/episode_reward"] == pytest.approx(-2.0)
    np.testing.assert_allclose(selector.params, [1.0])


def test_arm_cpu_low_memory_batch_matches_vector_width():
    """The canonical four-env PegInsert PPO must pass Brax's batch invariant."""
    from scripts.tasks.robot.arm.train_rl_baseline import (
        _cpu_low_memory_ppo_kwargs,
    )

    kwargs = _cpu_low_memory_ppo_kwargs(4)
    assert kwargs["policy_hidden_layer_sizes"] == (16, 16)
    assert kwargs["batch_size"] * kwargs["num_minibatches"] % 4 == 0
    with pytest.raises(ValueError, match="num_envs must be positive"):
        _cpu_low_memory_ppo_kwargs(0)


def test_arm_atacom_training_options_match_deployment_contract():
    from scripts.tasks.robot.arm.train_rl_baseline import (
        _atacom_training_options,
    )

    assert _atacom_training_options({}) == {
        "Kc": 1.0,
        "action_limit": 1.0,
        "alpha_limit": 1.0,
    }
    assert _atacom_training_options({
        "method_params": {
            "Kc": 2.0, "action_limit": 1.0, "alpha_limit": 0.6,
        },
    }) == {"Kc": 2.0, "action_limit": 1.0, "alpha_limit": 0.6}
    with pytest.raises(ValueError, match="Kc"):
        _atacom_training_options({"method_params": {"Kc": 0.0}})
    with pytest.raises(ValueError, match="action_limit"):
        _atacom_training_options({"method_params": {"action_limit": 1.1}})
    with pytest.raises(ValueError, match="alpha_limit"):
        _atacom_training_options({"method_params": {"alpha_limit": 0.0}})


def test_arm_policy_validation_health_gate_and_checkpoint_provenance(tmp_path):
    import json

    from scripts.tasks.robot.arm.train_rl_baseline import (
        _policy_validation_summary,
    )

    sha256 = "a" * 64
    for suite in ("id_wide", "ood_pose"):
        for seed in (104, 105):
            path = tmp_path / f"level_{suite}" / f"seed_{seed}" / "results.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "level": suite,
                "seed": seed,
                "provenance": {"checkpoints": {
                    "policy_ckpt": {"sha256": sha256},
                }},
                "metrics": {"peg_insert_metrics": {
                    "insertion_success": 1.0,
                    "safe_insertion_success": 0.0,
                    "max_insertion_depth": 0.04,
                    "force_torque_violation_rate": 0.1,
                    "jam_rate": 1.0,
                    "rho_cvar95": 1.2,
                }},
            }), encoding="utf-8")

    summary = _policy_validation_summary(
        tmp_path,
        expected_suites=("id_wide", "ood_pose"),
        expected_seeds=(104, 105),
        evaluated_sha256=sha256,
    )
    assert summary["run_count"] == 4
    assert summary["insertion_success_rate"] == 1.0
    assert summary["safe_insertion_success_rate"] == 0.0
    with pytest.raises(ValueError, match="nonzero whole-window safe"):
        _policy_validation_summary(
            tmp_path,
            expected_suites=("id_wide", "ood_pose"),
            expected_seeds=(104, 105),
            evaluated_sha256=sha256,
            require_nonzero_safe_success=True,
        )

    changed = tmp_path / "level_id_wide/seed_104/results.json"
    payload = json.loads(changed.read_text())
    payload["provenance"]["checkpoints"]["policy_ckpt"]["sha256"] = "b" * 64
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="used policy sha256"):
        _policy_validation_summary(
            tmp_path,
            expected_suites=("id_wide", "ood_pose"),
            expected_seeds=(104, 105),
            evaluated_sha256=sha256,
        )


def test_panda_residual_action_wrapper_adds_and_clips_bias():
    from genedynamics.envs.domains.manipulation.panda_brax import (
        PandaResidualActionEnv,
    )

    class State(NamedTuple):
        obs: jax.Array

    class Env:
        action_size = 2
        observation_size = 1
        backend = "mjx"
        dt = 0.02

        def step(self, state, action):
            return State(obs=action)

    wrapped = PandaResidualActionEnv(Env(), [0.25, -0.5], [0.5, 0.25])
    result = wrapped.step(
        State(obs=jnp.zeros((1,), jnp.float32)),
        jnp.asarray([0.9, 0.25]),
    )
    np.testing.assert_allclose(result.obs, [0.7, -0.4375])


def _h1_normal_only_reliability_payload(n=10):
    payload = _h1_reliability_payload(n)
    signals = payload["task_signals"]
    signals.update({
        "mga_execution_mode": np.zeros(n, np.int32),
        "reliability_execution_applicable": np.ones(n, bool),
        "reliability_risk_valid": np.ones(n, bool),
        "safety_margins": np.full((n, 4), -1.),
        "task_metadata": {"reliability_contract": {"schema": {"version": 3}}},
    })
    return payload


def test_h1_reliability_excludes_unload_windows_not_real_physical_observations():
    payload = _h1_normal_only_reliability_payload()
    signals = payload["task_signals"]
    signals["mga_execution_mode"][4] = 1
    signals["reliability_execution_applicable"][4] = False
    signals["reliability_risk_valid"][4] = False
    signals["reliability_risk"][4] = np.nan
    # The UNLOAD endpoint is an observed physical state. It may seed a later
    # NORMAL candidate, while a candidate crossing its UNLOAD action is N/A.
    signals["safety_margins"][4] = [.1, -.1, -.1, -.1]
    _, y, meta = h1_reliability_sequence_rows(payload, {}, horizon_steps=3)
    assert meta["decision_steps"] == [1, 5, 6, 7]
    assert meta["unload_crossing_starts_excluded"] == 3
    assert meta["unload_transitions_excluded_from_normal_labels"] == 1
    assert y[1, 0] == 1.  # initial s_5 force, not the previous interval envelope
    assert np.isfinite(y).all()


def test_h1_reliability_v3_initial_endpoint_not_past_peak_sets_candidate_risk():
    payload = _h1_normal_only_reliability_payload()
    signals = payload["task_signals"]
    signals["reliability_risk"][0] = [1., 1., 99., 99.]
    _, y, _ = h1_reliability_sequence_rows(payload, {}, horizon_steps=3)
    np.testing.assert_array_equal(y[0], 0.)
    signals["safety_margins"][0] = [.1, .2, .3, -.1]
    _, y, _ = h1_reliability_sequence_rows(payload, {}, horizon_steps=3)
    np.testing.assert_allclose(y[0], [1., 1., .3, 0.])


@pytest.mark.parametrize("bad", ["normal_nan", "normal_unknown", "mode_mismatch", "unload_finite",
                                 "missing_mode", "instantaneous_nan"])
def test_h1_reliability_normal_only_mask_does_not_hide_unknown_or_corrupt_labels(bad):
    payload = _h1_normal_only_reliability_payload()
    signals = payload["task_signals"]
    signals["mga_execution_mode"][4] = 1
    signals["reliability_execution_applicable"][4] = False
    signals["reliability_risk_valid"][4] = False
    signals["reliability_risk"][4] = np.nan
    if bad == "normal_nan":
        signals["reliability_risk"][0, 0] = np.nan
    elif bad == "normal_unknown":
        signals["reliability_risk_valid"][0] = False
    elif bad == "mode_mismatch":
        signals["reliability_execution_applicable"][4] = True
    elif bad == "unload_finite":
        signals["reliability_risk"][4] = 0.
    elif bad == "missing_mode":
        del signals["mga_execution_mode"]
    else:
        signals["safety_margins"][0, 0] = np.nan
    with pytest.raises(ValueError):
        h1_reliability_sequence_rows(payload, {}, horizon_steps=3)


def test_h1_reliability_rejects_aborted_collection_before_reading_prefix(tmp_path):
    import json
    from scripts.tasks.robot.humanoid.train_box_push_reliability import _rows

    path = tmp_path / "results.json"
    path.write_text(json.dumps({
        "seed": 101, "config_snapshot": {"env_name": "humanoid_box_push", "name": "mga"},
        "execution_status": {"state": "aborted_unrecoverable",
                             "reason": "invalid_unload_entry"},
    }))
    # A trajectory is deliberately absent: rejection must precede loading or
    # length checks, even if an aborted run would have had 17+ normal actions.
    with pytest.raises(ValueError, match="aborted H1 collection.*results.json.*invalid_unload_entry"):
        _rows([path], {101}, {"mga"})
