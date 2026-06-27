"""Stage 6 — unified robotization: point cloud + Gaussian splat → SoftBodySpec.

CPU, no GPU / no 3D-gen: synthetic points and hand-made Gaussians robotize
through the SAME validity tail as the mesh path, producing a SoftBodySpec the
simulator already consumes.
"""

from __future__ import annotations

import numpy as np
import pytest

from genedynamics.learning.priors.morphology import (
    MeshRobotizeConfig,
    robotize_point_cloud,
    robotize_gaussians,
    SoftBodySpec,
)


def _check_valid_spec(spec, cfg):
    assert isinstance(spec, SoftBodySpec)
    assert spec.particles_x0.ndim == 2 and spec.particles_x0.shape[1] == 3
    N = spec.particles_x0.shape[0]
    assert spec.actuator_id.shape == (N,)
    assert spec.voxel_id.shape == (N,)
    assert int(spec.actuator_id.min()) >= -1
    assert int(spec.actuator_id.max()) < cfg.n_actuators
    assert int(spec.voxel_id.min()) >= 0
    assert int(spec.voxel_id.max()) < int(np.prod(cfg.voxel_dims))
    assert spec.fiber_dirs.shape == (cfg.n_actuators, 3)
    assert spec.n_voxels == int(np.prod(cfg.voxel_dims))


def test_point_cloud_robotizes_with_default_gates():
    rng = np.random.default_rng(0)
    # Dense elongated block (worm-like): many filled cells, rests on floor.
    pts = rng.uniform([-1.0, -0.5, -0.5], [1.0, 0.5, 0.5], size=(3000, 3))
    cfg = MeshRobotizeConfig()
    spec, report = robotize_point_cloud(pts, cfg)
    assert spec is not None, report.failure_reasons
    assert report.success
    assert report.n_filled_cells >= cfg.min_filled_cells
    _check_valid_spec(spec, cfg)


def test_point_cloud_too_few_points_fails_softly():
    pts = np.random.default_rng(1).standard_normal((10, 3))
    spec, report = robotize_point_cloud(pts)  # default min_filled_cells=64
    assert spec is None
    assert not report.success and report.failure_reasons


def test_gaussians_robotize():
    # Lattice of Gaussians spanning an elongated body → solid when thresholded.
    xs = np.linspace(-1.0, 1.0, 8)
    ys = np.linspace(-0.4, 0.4, 3)
    zs = np.linspace(-0.4, 0.4, 3)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    scales = np.full((centers.shape[0],), 0.3)
    cfg = MeshRobotizeConfig(min_filled_cells=8)
    spec, report = robotize_gaussians(centers, scales, None, cfg, density_threshold=0.3)
    assert spec is not None, report.failure_reasons
    assert report.success
    _check_valid_spec(spec, cfg)


def test_pc_and_gs_specs_have_same_structure():
    rng = np.random.default_rng(2)
    pts = rng.uniform([-1.0, -0.5, -0.5], [1.0, 0.5, 0.5], size=(2000, 3))
    cfg = MeshRobotizeConfig(min_filled_cells=8)
    spec_pc, _ = robotize_point_cloud(pts, cfg)
    spec_gs, _ = robotize_gaussians(pts, np.full(pts.shape[0], 0.15), None, cfg,
                                    density_threshold=0.3)
    assert spec_pc is not None and spec_gs is not None
    # Same simulator-facing structure (dims, ranges) regardless of front end.
    assert spec_pc.n_voxels == spec_gs.n_voxels == int(np.prod(cfg.voxel_dims))
    for spec in (spec_pc, spec_gs):
        _check_valid_spec(spec, cfg)
