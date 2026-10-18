"""Actual author ACP demonstration loading and source-derived label qualification."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import zarr

from genedynamics.solvers.single.acp.backends.compliance import VirtualTargetEstimator
from genedynamics.solvers.single.acp.backends.native_source import native_source_imports


def subset_store(downloaded_store, inventory_path, destination):
    """Create an explicitly labelled episode subset without mutating downloaded evidence."""
    downloaded_store, destination = Path(downloaded_store), Path(destination)
    inventory = json.loads(Path(inventory_path).read_text())
    originals = sorted({item["path"].split("/")[2] for item in inventory["members"]
                        if "/data/episode_" in item["path"]})
    actual = sorted(path.name for path in (downloaded_store / "data").iterdir()
                    if path.is_dir() and path.name.startswith("episode_"))
    indices = [originals.index(name) for name in actual]
    source = zarr.open(str(downloaded_store), mode="r")
    lengths = [("episode_rgb0_len", "rgb_0"), ("episode_robot0_len", "ts_pose_fb_0"),
               ("episode_wrench0_len", "wrench_0")]
    for key, sensor in lengths:
        for name, index in zip(actual, indices):
            # Read the zarray metadata so the image codec is not required for this check.
            shape = json.loads((downloaded_store/"data"/name/sensor/".zarray").read_text())["shape"]
            if int(source["meta"][key][index]) != shape[0]:
                raise ValueError("author metadata episode ordering differs from the selected source inventory")
    evidence = {"kind": "released_demonstration_subset", "original_episode_count": len(originals),
                "selected_episodes": actual, "original_episode_indices": indices,
                "complete_original_training_dataset": len(actual) == len(originals)}
    marker = destination / ".subset.json"
    if destination.exists():
        if not marker.exists() or json.loads(marker.read_text()) != evidence:
            raise ValueError("refusing to replace an unrelated dataset subset")
        return evidence
    shutil.copytree(downloaded_store, destination)
    output = zarr.open(str(destination), mode="a")
    for key, array in source["meta"].arrays():
        values = array[:]
        if values.shape[0] != len(originals):
            raise ValueError("unrecognized native episode metadata shape")
        output["meta"].array(key, values[indices], overwrite=True)
    marker.write_text(json.dumps(evidence, indent=2)+"\n")
    return evidence


def build_native_dataset(cfg, dataset_path, *, source_root, submodule_root):
    import copy
    import hydra
    from omegaconf import OmegaConf
    data_cfg = copy.deepcopy(cfg.task.dataset)
    OmegaConf.set_struct(data_cfg, False)
    removed_flags = {}
    for flag in ("hack_linear_interpolated_dense_action",):
        if flag in data_cfg:
            if data_cfg[flag] is not False:
                raise ValueError(f"released dataset requires unavailable legacy behavior: {flag}")
            removed_flags[flag] = data_cfg.pop(flag)
    with native_source_imports(source_root, submodule_root):
        # The original class registers the author's JPEG-XL codec and runs its actual
        # raw_to_obs/raw_to_action19 -> timestamp sampler -> relative-pose chain.
        dataset = hydra.utils.instantiate(data_cfg, dataset_path=str(Path(dataset_path).resolve()),
                                          val_ratio=0.0)
    dataset._native_checkpoint_compatibility = {"removed_inactive_legacy_flags": removed_flags}
    return dataset


def audit_native_labels(dataset_path, *, source_root):
    """Regenerate every selected real episode label with the pinned native estimator.

    Reports agreement with released labels independently from port/source agreement;
    disagreement with an author's released artifact is retained as evidence.
    """
    source_root = Path(source_root)
    from spatialmath import SE3
    from spatialmath.base import q2r, r2q
    path = source_root / "PyriteUtility/spatial_math/spatial_utilities.py"
    spec = importlib.util.spec_from_file_location("acp_native_spatial_labels", path)
    spatial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spatial)
    source = (source_root / "PyriteUtility/planning_control/compliance_helpers.py").read_text()
    node = next(item for item in ast.parse(source).body
                if isinstance(item, ast.ClassDef) and item.name == "VirtualTargetEstimator")
    scope = {"np": np, "su": spatial}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), scope)
    native_class = scope["VirtualTargetEstimator"]
    root = zarr.open(str(dataset_path), mode="r")
    reports = []
    for name in root["data"].keys():
        raw = root["data"][name]
        wrench, wrench_times = raw["wrench_0"][:], raw["wrench_time_stamps_0"][:]
        robot_times, poses = raw["robot_time_stamps_0"][:], raw["ts_pose_fb_0"][:]
        filtered = np.stack([np.convolve(wrench[:, i], np.ones(7000)/7000, mode="same")
                             for i in range(6)], axis=-1)
        options = dict(k_max=5000, k_min=200, f_low=0.5, f_high=5, dim=3,
                       characteristic_length=0.02, vel_tol=999.002)
        native, local = native_class(**options), VirtualTargetEstimator(**options)
        generated = np.empty((len(poses), 4, 4))
        generated_pose7 = np.empty((len(poses), 7))
        stiffness = np.empty(len(poses))
        port_error = 0.0
        native_pose = spatial.pose7_to_SE3(poses)
        # The author's label script uses spatialmath for world-from-tool poses,
        # while using its own spatial utilities for the velocity difference.
        world_pose = np.array([SE3.Rt(q2r(pose[3:7]), pose[:3], check=False).A for pose in poses])
        for t, timestamp in enumerate(robot_times):
            nearest = int(np.argmin(np.abs(wrench_times-timestamp)))
            start, end = max(0, t-10), min(len(poses)-1, t+10)
            velocity = spatial.SE3_to_spt(spatial.SE3_inv(native_pose[start]) @ native_pose[end])
            k, offset, adjusted = native.update(filtered[nearest].copy(), velocity.copy())
            own_k, own_offset, own_adjusted = local.update(filtered[nearest].copy(), velocity.copy())
            port_error = max(port_error, abs(k-own_k), float(np.max(np.abs(offset-own_offset))))
            if adjusted != own_adjusted:
                raise AssertionError("ported/native ACP label adjustment differs")
            transform = np.eye(4)
            transform[:3, 3] = offset
            generated[t], stiffness[t] = world_pose[t] @ transform, k
            generated_pose7[t] = np.concatenate([generated[t, :3, 3], r2q(generated[t, :3, :3])])
        # Compare the actual emitted pose7 labels after the native r2q conversion;
        # nonunit rounded input quaternions make pre-conversion matrix comparison misleading.
        generated = spatial.pose7_to_SE3(generated_pose7)
        released = spatial.pose7_to_SE3(raw["ts_pose_virtual_target_0"][:])
        reports.append({"episode": name, "labels": len(poses), "wrench_samples": len(wrench),
                        "source_port_max_error": port_error,
                        "released_stiffness_max_error": float(np.max(np.abs(stiffness-raw["stiffness_0"][:]))),
                        "released_virtual_pose_max_error": float(np.max(np.abs(generated-released))),
                        "source_parameters": options, "force_filter": "native 7000-sample moving average",
                        "real_tool_frame_wrench": True})
    return reports
