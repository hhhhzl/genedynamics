"""Numerical parity against the pinned author's code, rather than a second local formula.

Core policy tests use a small observation-encoder fixture in BOTH implementations.
This verifies the real sparse diffusion/U-Net/normalizer/loss/checkpoint path; it
does not qualify the native pretrained vision/force encoder or physical controller.
Set GENEDYNAMICS_BASELINE_TEST_DEVICE=cuda:0 to run the same checks on the GPU.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from genedynamics.solvers.single.acp.acp import ACPSolver
from genedynamics.solvers.single.acp.compliance import (
    VirtualTargetEstimator, decode_action19, decode_stiffness_tool,
)
from genedynamics.solvers.single.acp.policy import ACPDiffusionPolicy

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "baselines" / "ACP"
COMMIT = "74d68eca3bfb5cb59da131c6351d9e1c9b129ff6"


def _source(path):
    if not (SOURCE / ".git").exists():
        pytest.skip("source parity requires the pinned official baselines/ACP checkout")
    return subprocess.check_output(
        ["git", "-C", str(SOURCE), "show", f"{COMMIT}:{path}"], text=True,
    )


def _definition(path, name, namespace):
    """Compile an unchanged source definition, bypassing unrelated optional imports."""
    node = next(
        n for n in ast.parse(_source(path)).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(SOURCE / path), "exec"), namespace)
    return namespace[name]


@pytest.fixture(scope="module")
def native_spatial():
    path = SOURCE / "PyriteUtility/spatial_math/spatial_utilities.py"
    if not path.exists():
        pytest.skip("official ACP spatial source must be checked out")
    spec = importlib.util.spec_from_file_location("acp_author_spatial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_fingerprints_match_pinned_commit():
    manifest = json.loads((ROOT / "genedynamics/solvers/single/acp/SOURCE.json").read_text())
    assert manifest["commit"] == COMMIT
    for entry in manifest["source_files"]:
        actual = hashlib.sha256(_source(entry["source"]).encode()).hexdigest()
        assert actual == entry["sha256"], entry["source"]
        checked_out = SOURCE / entry["source"]
        if checked_out.is_file():
            assert hashlib.sha256(checked_out.read_bytes()).hexdigest() == actual


@pytest.mark.parametrize("dim", [3, 6])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_virtual_target_labels_match_author_across_force_and_motion_regimes(native_spatial, dim, dtype):
    native_class = _definition(
        "PyriteUtility/planning_control/compliance_helpers.py", "VirtualTargetEstimator",
        {"np": np, "su": native_spatial},
    )
    kwargs = dict(k_max=5000, k_min=200, f_low=0.5, f_high=5, dim=dim,
                  characteristic_length=0.02, vel_tol=0.001)
    native, local = native_class(**kwargs), VirtualTargetEstimator(**kwargs)
    for force_scale, moving in [(0.01, False), (1.0, False), (8.0, False), (3.0, True)]:
        wrench = np.array([1.0, -0.4, 0.2, 0.02, -0.01, 0.03], dtype=dtype) * force_scale
        velocity = (np.array([0.02, 0.01, -0.03, 0.4, -0.2, 0.3], dtype=dtype)
                    if moving else np.zeros(6, dtype=dtype))
        a = native.update(wrench.copy(), velocity.copy())
        b = local.update(wrench.copy(), velocity.copy())
        assert a[0] == b[0]
        np.testing.assert_allclose(a[1], b[1], rtol=1e-12, atol=1e-12)
        assert a[2] == b[2]
        np.testing.assert_array_equal(native.last_wrench_T, local.last_wrench_T)
        # Unlike the upstream in-place rotational regularization, callers retain SI velocity.
        untouched = velocity.copy()
        local.update(wrench, velocity)
        np.testing.assert_array_equal(velocity, untouched)


@pytest.mark.parametrize("arms", [1, 2])
def test_action19_world_frames_and_rotation_row_convention_match_author(native_spatial, arms):
    native_fn = _definition(
        "PyriteConfig/tasks/common/common_type_conversions.py", "action19_postprocess",
        {"np": np, "su": native_spatial},
    )
    rng = np.random.default_rng(812)
    action = rng.normal(size=(3, 8, arms * 19))
    current = np.tile(np.eye(4), (3, arms, 1, 1))
    current[..., :3, 3] = rng.normal(size=(3, arms, 3))
    for b in range(3):
        for arm in range(arms):
            current[b, arm, :3, :3] = native_spatial.so3_to_SO3(rng.normal(size=3))
    reference, virtual, stiffness = decode_action19(action, current)
    for b in range(3):
        a_ref, a_vt, a_k = native_fn(action[b], list(current[b]), list(range(arms)))
        np.testing.assert_allclose(reference[b], np.stack(a_ref, axis=1), atol=1e-12)
        np.testing.assert_allclose(virtual[b], np.stack(a_vt, axis=1), atol=1e-12)
        np.testing.assert_array_equal(stiffness[b], np.stack(a_k, axis=1))


def test_anisotropic_stiffness_matches_author_deployment_block(native_spatial):
    source = _source("PyriteEnvSuites/env_runners/virtual_target_real_env_runner.py")
    tree = ast.parse(source)
    # Execute the exact official stiffness reconstruction loop without hardware/UI imports.
    loop = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
        and n.target.id == "i" and "target_stiffness" in ast.unparse(n)
    )
    rng = np.random.default_rng(52)
    current = np.eye(4)
    current[:3, :3] = native_spatial.so3_to_SO3(np.array([0.4, 0.2, -0.3]))
    current[:3, 3] = [0.3, -0.1, 0.5]
    reference = np.tile(np.eye(4), (8, 1, 1))
    reference[:, :3, 3] = rng.normal(size=(8, 3))
    virtual = reference.copy()
    virtual[:, :3, 3] += rng.normal(size=(8, 3)) * 0.03
    virtual[0] = reference[0]  # preserve author's <1mm x-axis fallback
    scalar = np.linspace(200, 5000, 8)
    namespace = dict(np=np, id=0, SE3_TW=native_spatial.SE3_inv(current),
                     ts_targets_virtual=np.empty((8, 7)),
                     action_sparse_target_mats=[reference], action_sparse_vt_mats=[virtual],
                     action_stiffnesses=[scalar], ts_stiffnesses=np.zeros((6, 48)))
    module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
    exec(compile(module, "author_deployment_stiffness_loop", "exec"), namespace)
    expected = np.stack(np.split(namespace["ts_stiffnesses"], 8, axis=1))
    actual = decode_stiffness_tool(reference, virtual, scalar, current)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_parallel_basis_direction_is_finite_and_retains_soft_axis():
    reference = np.eye(4)
    virtual = reference.copy()
    virtual[2, 3] = 0.02  # native cross(X, z) basis is singular in this case
    stiffness = decode_stiffness_tool(reference, virtual, 200.0, reference)
    np.testing.assert_allclose(np.diag(stiffness), [5000, 5000, 200, 100, 100, 100])
    assert np.isfinite(stiffness).all()


class _FixtureEncoder(nn.Module):
    """Small identical encoder to isolate the real native diffusion core in parity tests."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6, 8)

    def output_shape(self):
        return (8,)

    def forward(self, obs):
        return self.proj(obs["robot0_eef_wrench"].mean(dim=1))


@pytest.fixture
def policy_pair():
    pytest.importorskip("einops")
    pytest.importorskip("zarr")
    diffusers = pytest.importorskip("diffusers")
    if not (SOURCE / "PyriteML/diffusion_policy").exists():
        pytest.skip("official ACP policy sources must be checked out")
    torch.set_num_threads(1)
    device = torch.device(os.environ.get("GENEDYNAMICS_BASELINE_TEST_DEVICE", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.fail("CUDA source parity was explicitly requested but CUDA is unavailable")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    sys.path.insert(0, str(SOURCE / "PyriteML"))
    try:
        original_normalizer = importlib.import_module("diffusion_policy.model.common.normalizer")
        original_unet = importlib.import_module("diffusion_policy.model.diffusion.conditional_unet1d")
        original_base = importlib.import_module("diffusion_policy.policy.base_image_policy")
        for module in (original_normalizer, original_unet, original_base):
            assert Path(module.__file__).resolve().is_relative_to(SOURCE.resolve())
        native_class = _definition(
            "PyriteML/diffusion_policy/policy/diffusion_unet_timm_mod1_policy.py",
            "DiffusionUnetTimmMod1Policy", dict(
                Dict=Dict, torch=torch, nn=nn, F=torch.nn.functional, np=np,
                DDPMScheduler=diffusers.DDPMScheduler,
                TimmObsEncoderWithForce=nn.Module,
                BaseImagePolicy=original_base.BaseImagePolicy,
                LinearNormalizer=original_normalizer.LinearNormalizer,
                ConditionalUnet1D=original_unet.ConditionalUnet1D,
            ),
        )
    finally:
        sys.path.pop(0)
    shape_meta = {
        "action": {"shape": [19]},
        "obs": {"robot0_eef_wrench": {"shape": [6], "type": "low_dim"}},
        "sample": {
            "obs": {"sparse": {"robot0_eef_wrench": {"horizon": 4, "down_sample_steps": 1}}},
            "action": {"sparse": {"horizon": 8, "down_sample_steps": 50}},
        },
    }
    scheduler_kwargs = dict(num_train_timesteps=8, beta_start=0.0001, beta_end=0.02,
                            beta_schedule="squaredcos_cap_v2", clip_sample=True,
                            set_alpha_to_one=True, steps_offset=0, prediction_type="epsilon")
    kwargs = dict(shape_meta=shape_meta, num_inference_steps=4,
                  diffusion_step_embed_dim=8, down_dims=(8, 16), kernel_size=3, n_groups=8)
    torch.manual_seed(58)
    native = native_class(noise_scheduler=diffusers.DDIMScheduler(**scheduler_kwargs),
                          obs_encoder=_FixtureEncoder(), **kwargs).to(device)
    normalizer = original_normalizer.LinearNormalizer()
    normalizer.fit({"robot0_eef_wrench": torch.randn(32, 4, 6), "action": torch.randn(32, 8, 19)})
    native.set_normalizer(normalizer)
    native.to(device)
    local = ACPDiffusionPolicy(noise_scheduler=diffusers.DDIMScheduler(**scheduler_kwargs),
                               obs_encoder=_FixtureEncoder(), **kwargs).to(device)
    local.load_state_dict(copy.deepcopy(native.state_dict()), strict=True)
    local.to(device)
    batch = {"obs": {"sparse": {"robot0_eef_wrench": torch.randn(2, 4, 6, device=device)}},
             "action": {"sparse": torch.randn(2, 8, 19, device=device)}}
    return native, local, batch, shape_meta, device


def test_real_sparse_diffusion_loss_gradients_adamw_update_and_sampling_match_author(policy_pair):
    native, local, batch, _, device = policy_pair
    assert list(native.state_dict()) == list(local.state_dict())
    native.train()
    local.train()
    torch.manual_seed(1234)
    native_loss = native.compute_loss(batch, None)
    torch.manual_seed(1234)
    local_loss = local.compute_loss(batch, None)
    torch.testing.assert_close(local_loss, native_loss, rtol=1e-6, atol=1e-7)
    native_loss.backward()
    local_loss.backward()
    for (name_a, a), (name_b, b) in zip(native.named_parameters(), local.named_parameters()):
        assert name_a == name_b
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(b.grad, a.grad, rtol=1e-5, atol=1e-7)
    optim_kwargs = dict(lr=3e-4, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    torch.optim.AdamW(native.parameters(), **optim_kwargs).step()
    torch.optim.AdamW(local.parameters(), **optim_kwargs).step()
    for key, value in native.state_dict().items():
        torch.testing.assert_close(local.state_dict()[key], value, rtol=1e-5, atol=1e-7)
    native.eval()
    local.eval()
    with torch.inference_mode():
        torch.manual_seed(789)
        expected = native.predict_action(batch["obs"])["sparse"]
        torch.manual_seed(789)
        actual = local.predict_action(batch["obs"])["sparse"]
    assert actual.device == device
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(local.sparse_nobs_encode, native.sparse_nobs_encode)


def test_native_checkpoint_schema_and_structured_solver_path(policy_pair, tmp_path):
    native, local, batch, shape_meta, device = policy_pair
    solver = ACPSolver(local, shape_meta, device=str(device), execution_horizon=4)
    checkpoint = tmp_path / "native_tensor_state.ckpt"
    torch.save({"state_dicts": {"ema_model": native.state_dict()}}, checkpoint)
    solver.load_checkpoint(checkpoint, use_ema=True)
    with pytest.raises(KeyError, match="model"):
        solver.load_checkpoint(checkpoint, use_ema=False)
    current = np.tile(np.eye(4), (2, 1, 1, 1))
    torch.manual_seed(401)
    native.eval()
    with torch.inference_mode():
        raw_expected = native.predict_action(batch["obs"])["sparse"]
    torch.manual_seed(401)
    result = solver.act({**batch["obs"], "current_tool_pose_world": current})
    torch.testing.assert_close(result["raw_action"], raw_expected)
    assert result["reference_pose_world"].shape == (2, 8, 1, 4, 4)
    assert result["virtual_target_pose_world"].shape == (2, 8, 1, 4, 4)
    assert result["stiffness_tool"].shape == (2, 8, 1, 6, 6)
    np.testing.assert_allclose(result["waypoint_times_s"], np.arange(8) * 0.15)
    assert result["execution_horizon"] == 4
    solver.reset()
    assert solver.policy.sparse_nobs_encode is None
    assert solver.policy.sparse_naction_pred is None


def test_solver_requires_complete_native_sensor_history_schema(policy_pair):
    _, local, _, shape_meta, device = policy_pair
    solver = ACPSolver(local, shape_meta, device=str(device), execution_horizon=4)
    current = np.tile(np.eye(4), (1, 1, 1, 1))
    with pytest.raises(KeyError, match="robot0_eef_wrench"):
        solver.act({"sparse": {}, "current_tool_pose_world": current})
    with pytest.raises(ValueError, match="history length"):
        solver.act({"sparse": {"robot0_eef_wrench": np.zeros((1, 2, 6))},
                    "current_tool_pose_world": current})
