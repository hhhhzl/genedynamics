"""Build actual ACP vision/force policies and load the author's released checkpoints."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import importlib
from pathlib import Path
import subprocess
import sys
import types

COMMIT = "74d68eca3bfb5cb59da131c6351d9e1c9b129ff6"
SUBMODULE_COMMIT = "e4bd73e3e6ff0d4334d12aabbe0e9575669b63e7"


@contextmanager
def native_source_imports(source_root, submodule_root):
    source_root, submodule_root = Path(source_root).resolve(), Path(submodule_root).resolve()
    for path, expected in [(source_root, COMMIT), (submodule_root, SUBMODULE_COMMIT)]:
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise ValueError(f"native ACP source revision mismatch: {path}")
    module = sys.modules.get("multimodal_representation")
    if module is None:
        module = types.ModuleType("multimodal_representation")
        module.__path__ = [str(submodule_root)]
        sys.modules["multimodal_representation"] = module
    elif list(module.__path__) != [str(submodule_root)]:
        raise RuntimeError("another multimodal_representation source is already loaded")
    paths = [str(source_root / "PyriteML"), str(source_root)]
    previous = list(sys.path)
    sys.path[:0] = paths
    try:
        yield
    finally:
        sys.path[:] = previous


@contextmanager
def checkpoint_initialization():
    """Skip redundant pretrained network downloads; the exact checkpoint replaces all weights."""
    import timm
    original = timm.create_model

    def create_model(*args, **kwargs):
        kwargs["pretrained"] = False
        return original(*args, **kwargs)

    timm.create_model = create_model
    try:
        yield
    finally:
        timm.create_model = original


def load_author_payload(checkpoint):
    """Load a SHA-verified, author-origin dill/Hydra workspace checkpoint on CPU."""
    import dill
    import torch
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    if not isinstance(payload, dict) or "cfg" not in payload or "state_dicts" not in payload:
        raise ValueError("expected an original ACP workspace checkpoint")
    return payload


def build_policy(cfg, *, upstream=False, checkpoint_weights=False, source_root=None, submodule_root=None):
    """Construct native topology using the author's config, never a fixture encoder."""
    import hydra
    from omegaconf import OmegaConf
    policy_cfg = copy.deepcopy(cfg.policy)
    OmegaConf.set_struct(policy_cfg, False)
    removed_flags = {}
    for flag in ("hack_no_obs_encoder_for_dense",):
        if flag in policy_cfg:
            if policy_cfg[flag] is not False:
                raise ValueError(f"released checkpoint requires unavailable legacy behavior: {flag}")
            removed_flags[flag] = policy_cfg.pop(flag)
    encoder_target = str(policy_cfg.obs_encoder._target_)
    if encoder_target.endswith(".TimmObsEncoderWithForceSpec"):
        local_encoder = "genedynamics.solvers.single.acp.learning.models.vision_force_spec.TimmObsEncoderWithForceSpec"
    elif encoder_target.endswith(".TimmObsEncoderWithForce"):
        local_encoder = "genedynamics.solvers.single.acp.learning.models.vision_force.TimmObsEncoderWithForce"
    else:
        raise ValueError(f"unsupported native ACP observation encoder: {encoder_target}")
    if not str(policy_cfg._target_).endswith(".DiffusionUnetTimmMod1Policy"):
        raise ValueError("native ACP checkpoint must use the original Mod1 sparse policy")
    if not upstream:
        policy_cfg._target_ = "genedynamics.solvers.single.acp.learning.models.diffusion_policy.DiffusionUnetTimmMod1Policy"
        policy_cfg.obs_encoder._target_ = local_encoder

    def construct():
        if checkpoint_weights:
            with checkpoint_initialization():
                return hydra.utils.instantiate(policy_cfg)
        return hydra.utils.instantiate(policy_cfg)

    if upstream:
        if source_root is None or submodule_root is None:
            raise ValueError("upstream policy construction requires both pinned source checkouts")
        with native_source_imports(source_root, submodule_root):
            model = construct()
    else:
        model = construct()
    model._native_checkpoint_compatibility = {"removed_inactive_legacy_flags": removed_flags}
    shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    return model, shape_meta


def load_native_policy(checkpoint, *, device="cuda:0", use_ema=True, upstream=False,
                       source_root=None, submodule_root=None):
    payload = load_author_payload(checkpoint)
    model, shape_meta = build_policy(payload["cfg"], upstream=upstream, checkpoint_weights=True,
                                     source_root=source_root, submodule_root=submodule_root)
    key = "ema_model" if use_ema else "model"
    if key not in payload["state_dicts"]:
        raise KeyError(f"author ACP checkpoint has no {key}")
    model.load_state_dict(payload["state_dicts"][key], strict=True)
    model.to(device).eval()
    model.reset()
    return model, shape_meta, payload["cfg"]
