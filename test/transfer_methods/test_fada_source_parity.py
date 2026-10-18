"""Numerical parity with pinned official FADA definitions, not a surrogate policy.

The reference clone is optional for ordinary package users. Reproduction qualification
must run this suite with the pinned source populated in ``baselines/FADA``.
"""

from __future__ import annotations

import ast
from bisect import bisect_right
import copy
import dataclasses
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn, optim
from torch.nn import functional as F

from genedynamics.solvers.single.fada.learning import adaptation
from genedynamics.solvers.single.fada.learning import FADAIDMAdapter, IDMAdaptationConfig
from genedynamics.solvers.single.fada.learning.data import Trajectory, TrajectoryWindowDataset
from genedynamics.solvers.single.fada import FADASolver
from genedynamics.solvers.single.fada.backends import lora
from genedynamics.solvers.single.fada.backends.fada_torch import lora_inference_context
from genedynamics.solvers.single.fada.backends.model import PlannerIDMPolicy
from genedynamics.solvers.single.fada.backends.norm_stats import validate_norm_stats

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "baselines/FADA/src/holosoma/holosoma/fada"
DEVICE = torch.device(os.environ.get("GENEDYNAMICS_BASELINE_TEST_DEVICE", "cpu"))


def _reference_definitions(relative, names, **dependencies):
    """Execute original AST definitions without importing its Isaac simulator stack."""
    path = SOURCE / relative
    module = ModuleType("_fada_reference_" + relative.replace("/", "_").replace(".", "_"))
    sys.modules[module.__name__] = module
    module.__dict__.update(
        torch=torch, nn=nn, optim=optim, F=F, np=np, math=math,
        copy=copy, dataclasses=dataclasses, Any=object,
        bisect_right=bisect_right, **dependencies,
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in tree.body} == set(names)
    exec(compile(tree, str(path), "exec", flags=__import__("__future__").annotations.compiler_flag), module.__dict__)
    return module


@pytest.fixture(scope="module")
def reference():
    if not SOURCE.is_dir():
        pytest.skip("Pinned FADA clone is required for source parity")
    manifest = json.loads((ROOT / "genedynamics/solvers/single/fada/source_manifest.json").read_text())
    for entry in manifest["files"]:
        source_path = ROOT / "baselines/FADA" / entry["source"]
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == entry["source_sha256"]
    pos = _reference_definitions("common/backbone.py", ["SinusoidalPositionalEncoding"])
    model = _reference_definitions(
        "planner_idm/model.py", ["_PositionalEncodingMixin", "Planner", "IDM", "PlannerIDMPolicy"],
        SinusoidalPositionalEncoding=pos.SinusoidalPositionalEncoding,
    )
    lora_names = ["LoRALinear", "LoRAMultiheadAttention", "_resolve_linear_weight_bias", "_inject_lora_modules",
                  "_inject_lora_multihead_attention_modules", "build_merged_state_dict"]
    ref_lora = _reference_definitions("common/lora_utils.py", lora_names)
    loss = _reference_definitions("planner_idm/loss_utils.py", ["weighted_horizon_mse"])
    train_names = ["_normalize_idm_if_enabled", "_normalize_lora_target_scope", "_freeze_all_parameters",
                   "_mark_lora_trainables", "_is_idm_backbone_linear", "_is_idm_backbone_attention",
                   "inject_lora_into_idm_backbone", "_loss_metric_key", "_run_epoch", "TrajectoryWindowDataset"]
    train = _reference_definitions(
        "planner_idm/finetune_idm_lora.py", train_names,
        **{name: getattr(ref_lora, name) for name in lora_names if name != "build_merged_state_dict"},
        weighted_horizon_mse=loss.weighted_horizon_mse,
    )
    return model, ref_lora, train


def _config(**changes):
    cfg = dict(
        obs_dim=5, act_dim=3, cmd_dim=2, history_len=3, pred_horizon=2,
        planner_d_model=16, planner_nhead=2, planner_num_layers=1,
        planner_dim_feedforward=24, planner_dropout=0.0,
        planner_use_learned_positional_encoding=False, planner_use_action_history=True,
        idm_d_model=16, idm_nhead=2, idm_encoder_num_layers=1, idm_decoder_num_layers=1,
        idm_dim_feedforward=24, idm_dropout=0.0,
        idm_use_learned_positional_encoding=False,
        idm_use_current_command_for_history=True, planner_predict_delta=False,
    )
    cfg.update(changes)
    return cfg


def _batch():
    generator = torch.Generator().manual_seed(77)
    batch = dict(
        history_obs=torch.randn(2, 3, 5, generator=generator),
        history_act=torch.randn(2, 3, 3, generator=generator),
        current_command=torch.randn(2, 2, generator=generator),
        history_valid_mask=torch.tensor([[False, True, True], [True, True, True]]),
        future_observations=torch.randn(2, 2, 5, generator=generator),
        target_actions=torch.randn(2, 2, 3, generator=generator),
    )
    return {name: value.to(DEVICE) for name, value in batch.items()}


@pytest.mark.parametrize("delta,learned,history", [(False, False, False), (True, False, True), (False, True, True)])
def test_native_planner_idm_outputs_and_input_gradients(reference, delta, learned, history):
    ref_model, _, _ = reference
    cfg = _config(planner_predict_delta=delta, planner_use_action_history=history,
                  idm_use_current_command_for_history=history,
                  planner_use_learned_positional_encoding=learned,
                  idm_use_learned_positional_encoding=learned)
    torch.manual_seed(31)
    source = ref_model.PlannerIDMPolicy(**cfg).to(DEVICE).train()
    port = PlannerIDMPolicy(**cfg).to(DEVICE).train()
    port.load_state_dict(source.state_dict(), strict=True)
    batch = _batch()
    observations = [batch["history_obs"].clone().requires_grad_() for _ in range(2)]
    outputs = []
    for model, obs in zip((source, port), observations):
        actions, future = model(obs, batch["history_act"], batch["current_command"],
                                history_valid_mask=batch["history_valid_mask"], return_obs=True)
        (actions.square().mean() + future.square().mean()).backward()
        outputs.append((actions, future))
    for left, right in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(observations[0].grad, observations[1].grad, rtol=0, atol=0)


def test_native_target_record_windows_are_right_aligned(reference):
    _, _, ref_train = reference
    trajectory = Trajectory(
        obs=np.arange(35, dtype=np.float32).reshape(7, 5),
        actions=np.arange(21, dtype=np.float32).reshape(7, 3),
        current_command=np.arange(14, dtype=np.float32).reshape(7, 2), source="public_target_record",
    )
    expected = ref_train.TrajectoryWindowDataset([trajectory], history_len=3, pred_horizon_k=2)
    actual = TrajectoryWindowDataset([trajectory], history_len=3, pred_horizon_k=2)
    assert len(actual) == len(expected) == 5
    for index in (0, 1, 4):
        for key in expected[index]:
            torch.testing.assert_close(actual[index][key], expected[index][key], rtol=0, atol=0)
    assert not actual[0]["history_act"].any()
    torch.testing.assert_close(actual[1]["history_act"][-1], torch.as_tensor(trajectory.actions[0]))


def test_qkv_lora_adaptation_update_and_merged_checkpoint(reference):
    ref_model, ref_lora, ref_train = reference
    torch.manual_seed(12)
    source = ref_model.PlannerIDMPolicy(**_config()).to(DEVICE)
    port = PlannerIDMPolicy(**_config()).to(DEVICE)
    port.load_state_dict(source.state_dict())
    baseline_planner = {key: value.clone() for key, value in port.planner.state_dict().items()}
    kwargs = dict(r=2, alpha=4.0, dropout=0.0, target_scope="encoder_decoder_qkv")
    torch.manual_seed(91)
    expected_replaced = ref_train.inject_lora_into_idm_backbone(source, **kwargs)
    torch.manual_seed(91)
    actual_replaced = adaptation.inject_lora_into_idm_backbone(port, **kwargs)
    assert expected_replaced == actual_replaced
    assert all(not param.requires_grad for param in port.planner.parameters())
    before = {key: value.clone() for key, value in port.named_parameters() if value.requires_grad}
    metrics = []
    for model, run_epoch in ((source, ref_train._run_epoch), (port, adaptation._run_epoch)):
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-3)
        metrics.append(run_epoch(
            model=model, loader=[_batch()], optimizer=optimizer, device=DEVICE,
            io_norm_enabled=False, obs_stats=None, action_stats=None, command_stats=None,
            pred_horizon_k=2, grad_clip=0.5, show_progress=False, epoch_desc="parity",
            loss_target="idm", horizon_gamma=0.0,
        ))
    assert metrics[0] == metrics[1]
    assert any(not torch.equal(before[name], value) for name, value in port.named_parameters() if name in before)
    for (name_a, param_a), (name_b, param_b) in zip(source.named_parameters(), port.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(param_a, param_b, rtol=0, atol=0)
        if param_a.grad is not None:
            torch.testing.assert_close(param_a.grad, param_b.grad, rtol=0, atol=0)
    for name, value in port.planner.state_dict().items():
        assert torch.equal(value, baseline_planner[name])
    expected_merged = ref_lora.build_merged_state_dict(source)
    merged = lora.build_merged_state_dict(port)
    assert expected_merged.keys() == merged.keys()
    for key in merged:
        torch.testing.assert_close(expected_merged[key], merged[key], rtol=0, atol=0)
    deployed = PlannerIDMPolicy(**_config()).to(DEVICE).eval()
    deployed.load_state_dict(merged, strict=True)
    batch = _batch()
    port.eval()
    with torch.inference_mode(), lora_inference_context(port):
        adapted = port(batch["history_obs"], batch["history_act"], batch["current_command"],
                       history_valid_mask=batch["history_valid_mask"])
        merged_output = deployed(batch["history_obs"], batch["history_act"], batch["current_command"],
                                 history_valid_mask=batch["history_valid_mask"])
    torch.testing.assert_close(adapted, merged_output, rtol=2e-5, atol=2e-6)


def test_solver_native_normalization_split_checkpoint_and_cache(reference):
    ref_model, _, _ = reference
    torch.manual_seed(22)
    source = ref_model.PlannerIDMPolicy(**_config()).to(DEVICE).eval()
    cfg = _config()
    # Native releases read normalization.io_enabled, not a cfg-only flag.
    cfg["io_normalization"] = False
    payload = dict(
        cfg=cfg, obs_dim=5, act_dim=3, cmd_dim=2,
        normalization={"io_enabled": True},
        planner_state_dict=source.planner_state_dict(), idm_state_dict=source.idm_state_dict(),
        obs_norm_stats={"obs_mean": torch.arange(5) / 5, "obs_std": torch.ones(5) * 1.5},
        action_norm_stats={"action_mean": torch.arange(3) / 4, "action_std": torch.ones(3) * 0.7},
        command_norm_stats={"command_mean": torch.arange(2) / 3, "command_std": torch.ones(2) * 1.2},
    )
    solver = FADASolver.from_checkpoint(payload, device=str(DEVICE), open_loop_deploy_steps=2)
    batch = _batch()
    with torch.inference_mode():
        expected_actions, expected_future = source(
            (batch["history_obs"] - payload["obs_norm_stats"]["obs_mean"].to(DEVICE)) / 1.5,
            (batch["history_act"] - payload["action_norm_stats"]["action_mean"].to(DEVICE)) / 0.7,
            (batch["current_command"] - payload["command_norm_stats"]["command_mean"].to(DEVICE)) / 1.2,
            history_valid_mask=batch["history_valid_mask"], return_obs=True,
        )
    actual_actions, actual_future = solver.predict(
        batch["history_obs"], batch["history_act"], batch["current_command"],
        history_valid_mask=batch["history_valid_mask"], return_obs=True,
    )
    torch.testing.assert_close(actual_actions, expected_actions * 0.7 + payload["action_norm_stats"]["action_mean"].to(DEVICE))
    torch.testing.assert_close(actual_future, expected_future * 1.5 + payload["obs_norm_stats"]["obs_mean"].to(DEVICE))
    torch.testing.assert_close(solver.act(batch), actual_actions[:, 0])
    torch.testing.assert_close(solver.act(batch), actual_actions[:, 1])
    solver.reset()
    torch.testing.assert_close(solver.act(batch), actual_actions[:, 0])
    with pytest.raises(ValueError, match="right aligned"):
        solver.predict(batch["history_obs"], batch["history_act"], batch["current_command"],
                       history_valid_mask=torch.zeros(2, 3, dtype=torch.bool))


def test_adaptation_api_preserves_source_update_and_roundtrips_native_checkpoint(reference):
    ref_model, ref_lora, ref_train = reference
    torch.manual_seed(20)
    source = ref_model.PlannerIDMPolicy(**_config()).to(DEVICE)
    payload = dict(
        cfg=_config(), obs_dim=5, act_dim=3, cmd_dim=2,
        normalization={"io_enabled": False}, model_state_dict=source.state_dict(),
    )
    torch.manual_seed(101)
    ref_train.inject_lora_into_idm_backbone(
        source, r=2, alpha=4.0, dropout=0.0, target_scope="encoder_decoder_qkv",
    )
    torch.manual_seed(101)
    adapter = FADAIDMAdapter(
        payload, device=str(DEVICE), config=IDMAdaptationConfig(rank=2, alpha=4.0, dropout=0.0),
    )
    optimizer = optim.AdamW([p for p in source.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-3)
    expected = ref_train._run_epoch(
        model=source, loader=[_batch()], optimizer=optimizer, device=DEVICE,
        io_norm_enabled=False, obs_stats=None, action_stats=None, command_stats=None,
        pred_horizon_k=2, grad_clip=0.5, show_progress=False, epoch_desc="source", horizon_gamma=0.0,
    )
    assert adapter.step(_batch()) == expected
    for expected_param, actual_param in zip(source.parameters(), adapter.model.parameters()):
        torch.testing.assert_close(expected_param, actual_param, rtol=0, atol=0)
    checkpoint = adapter.export_checkpoint()
    assert checkpoint["genedynamics_adaptation"]["optimizer_steps"] == 1
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    stream.seek(0)
    deployed = FADASolver.from_checkpoint(torch.load(stream, weights_only=True), device=str(DEVICE))
    batch = _batch()
    source.eval()
    native_merged = ref_model.PlannerIDMPolicy(**_config()).to(DEVICE).eval()
    native_merged.load_state_dict(ref_lora.build_merged_state_dict(source), strict=True)
    # Native merged checkpoint evaluation uses the same optimized Transformer path.
    # Compare this path exactly; CUDA fused-vs-unfused roundoff is a separate issue.
    with torch.inference_mode():
        native_merged_actions = native_merged(
            batch["history_obs"], batch["history_act"], batch["current_command"],
            history_valid_mask=batch["history_valid_mask"],
        )
    torch.testing.assert_close(deployed.act(batch), native_merged_actions[:, 0], rtol=0, atol=0)
    # Independently check that merging preserves LoRA arithmetic under the same
    # unfused execution path, retaining the original strict numerical tolerance.
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.inference_mode():
            source_actions = source(batch["history_obs"], batch["history_act"], batch["current_command"],
                                    history_valid_mask=batch["history_valid_mask"])
        torch.testing.assert_close(deployed.act(batch), source_actions[:, 0], rtol=2e-5, atol=2e-6)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)


def test_resume_optimizer_normalization_and_rng_replays_next_real_update():
    torch.manual_seed(32)
    payload = dict(
        cfg=_config(), obs_dim=5, act_dim=3, cmd_dim=2, normalization={"io_enabled": True},
        model_state_dict=PlannerIDMPolicy(**_config()).state_dict(),
        obs_norm_stats={"obs_mean": torch.arange(5) / 5, "obs_std": torch.ones(5) * 1.5},
        action_norm_stats={"action_mean": torch.arange(3) / 4, "action_std": torch.ones(3) * 0.7},
        command_norm_stats={"command_mean": torch.arange(2) / 3, "command_std": torch.ones(2) * 1.2},
    )
    adapter = FADAIDMAdapter(payload, device=str(DEVICE), config=IDMAdaptationConfig(rank=2, alpha=4.0, dropout=0.15))
    adapter.step(_batch())
    stream = io.BytesIO()
    torch.save(adapter.training_state_dict(), stream)
    expected = adapter.step(_batch())
    stream.seek(0)
    resumed = FADAIDMAdapter.from_training_state(torch.load(stream, weights_only=True), device=str(DEVICE))
    assert resumed.backend.io_normalization
    assert resumed.step(_batch()) == expected
    assert adapter.steps == resumed.steps == 2
    for expected_param, actual_param in zip(adapter.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(expected_param, actual_param, rtol=0, atol=0)


@pytest.mark.parametrize("std", [torch.tensor([float("nan")]), torch.tensor([-1.0])])
def test_normalization_rejects_corrupted_stats(std):
    with pytest.raises(RuntimeError):
        validate_norm_stats(torch.zeros(1), std, 1e-6, key="obs_norm_stats")
