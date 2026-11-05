"""Focused CPU contracts for the H2S2R robot-source learning-chain audit."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from genedynamics.solvers.single.human2sim2robot.learning.diagnostics import (
    VERSION,
    native_tracking_reward_profile,
    normalize_physical_actions,
    reference_clock_trace,
)
from genedynamics.solvers.single.human2sim2robot.learning import pilot as pilot_module


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = (
    "configs/arm/obstacle_transfer/training/human2sim2robot_learning_chain.json",
    "configs/arm/guided_contact_alignment/training/human2sim2robot_learning_chain.json",
    "configs/arm/keyed_insertion_recovery/training/human2sim2robot_learning_chain.json",
    "configs/quadruped/mobile_keyed_insertion_recovery/training/human2sim2robot_learning_chain.json",
)


def test_native_reward_stop_boundary_is_audited() -> None:
    rows = native_tracking_reward_profile([0.0, 0.2, 0.21], 0.2)
    assert rows[0]["tracking_reward"] == 1.0
    assert np.isclose(rows[1]["tracking_reward"], np.exp(-2.0))
    assert np.isclose(rows[2]["tracking_reward"], 0.1 * np.exp(-2.1))
    assert rows[2]["tracking_reward"] < rows[1]["tracking_reward"] / 10


def test_action_scaling_round_trip_handles_inactive_axis() -> None:
    low = np.array([-0.1, -2.0, 0.0], dtype=np.float32)
    high = np.array([0.1, 2.0, 0.0], dtype=np.float32)
    physical = np.stack((low, (low + high) / 2, high))
    normalized, active = normalize_physical_actions(physical, low, high)
    np.testing.assert_array_equal(active, [True, True, False])
    np.testing.assert_allclose(normalized, [[-1, -1, 0], [0, 0, 0], [1, 1, 0]])


def test_reference_clock_stops_outside_native_tracking_radius() -> None:
    demonstration = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    positions = np.array([[0, 0, 0], [0.3, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
    rotations = np.tile(np.eye(3, dtype=np.float32), (4, 1, 1))
    trace = reference_clock_trace(
        positions, rotations, demonstration,
        control_dt=0.02, demo_dt=0.02, stop_error_m=0.2,
    )
    np.testing.assert_allclose(trace["reference_indices"], [0, 0, 1, 2])
    assert trace["advanced_steps"] == 2
    assert trace["stopped_steps"] == 1


def test_four_task_configs_keep_gpu1_plan_task_local() -> None:
    seen = set()
    for relative in CONFIGS:
        path = ROOT / relative
        config = json.loads(path.read_text())
        assert config["version"] == VERSION
        assert config["task"] not in seen
        seen.add(config["task"])
        assert config["pilot"]["status"] == "planned_not_started"
        assert config["pilot"]["physical_gpu"] == "GPU1"
        assert config["pilot"]["expected_physical_uuid"] == (
            "GPU-58b54684-3577-11d3-f039-0e8469190739"
        )
        assert config["pilot"]["cuda_visible_devices"] == "1"
        assert config["pilot"]["process_device"] == "cuda:0"
        assert config["pilot"]["target_calls_during_training"] == 0
        assert config["pilot"]["training_transitions"] == (
            config["pilot"]["num_actors"]
            * config["pilot"]["horizon_steps"]
            * config["pilot"]["epochs"]
        )
        assert config["output"].endswith(
            "/baseline/human2sim2robot/diagnostics/learning_chain_cpu"
        )
        expected_output = (
            "/baseline/human2sim2robot/training/robot_source_pilot/seed_"
            + str(config["seed"] + 100000)
        )
        assert (
            config["pilot_output"].endswith(expected_output)
            or config["pilot_output"].endswith(
                expected_output + "_checkpoint_writer_retry_01"
            )
        )
    assert seen == {"T1", "T2", "T3", "T4"}


def test_gpu_binding_checks_visible_uuid_without_allocating(monkeypatch) -> None:
    expected = "GPU-58b54684-3577-11d3-f039-0e8469190739"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(pilot_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pilot_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        pilot_module.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(uuid=expected, total_memory=24 * 1024**3),
    )
    properties = pilot_module._verify_gpu_binding({
        "cuda_visible_devices": "1", "expected_physical_uuid": expected,
    })
    assert properties.uuid == expected


def test_gpu_binding_accepts_torch_uuid_without_gpu_prefix(monkeypatch) -> None:
    expected = "GPU-58b54684-3577-11d3-f039-0e8469190739"
    actual = expected.removeprefix("GPU-")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(pilot_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pilot_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        pilot_module.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(uuid=actual, total_memory=24 * 1024**3),
    )
    properties = pilot_module._verify_gpu_binding({
        "cuda_visible_devices": "1", "expected_physical_uuid": expected,
    })
    assert properties.uuid == actual


def test_gpu_binding_rejects_wrong_uuid(monkeypatch) -> None:
    expected = "GPU-58b54684-3577-11d3-f039-0e8469190739"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(pilot_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pilot_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        pilot_module.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(uuid="GPU-wrong", total_memory=24 * 1024**3),
    )
    with np.testing.assert_raises_regex(RuntimeError, "UUID differs"):
        pilot_module._verify_gpu_binding({
            "cuda_visible_devices": "1", "expected_physical_uuid": expected,
        })


def test_atomic_torch_checkpoint_uses_supported_temporary_name(tmp_path) -> None:
    path = tmp_path / "checkpoints" / "epoch_64.pt"
    value = {"weight": torch.tensor([1.0, 2.0])}
    pilot_module._atomic_torch(path, value)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.equal(restored["weight"], value["weight"])
