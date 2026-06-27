"""Unit tests for co-design Table-3 shape / connectivity / diversity metrics."""

import numpy as np
import pytest

from genedynamics.learning.priors.morphology.shape_metrics import (
    compute_shape_metrics, connectivity_metrics, diversity_metrics,
    occupancy_grid_from_voxel_id, occupancy_grid_from_occ,
)


def test_solid_cube_is_ideal():
    m = compute_shape_metrics(np.ones((3, 3, 3), bool))
    assert m["n_components"] == 1
    assert m["single_cc"] == 1.0
    assert m["largest_cc_fraction"] == 1.0
    assert m["bilateral_symmetry"] == pytest.approx(1.0)
    assert m["cavity_count"] == 0.0


def test_scattered_specks_flagged_ugly():
    specks = np.zeros((4, 4, 4), bool)
    specks[::2, ::2, ::2] = True
    cube = np.ones((3, 3, 3), bool)
    ms, mc = compute_shape_metrics(specks), compute_shape_metrics(cube)
    assert ms["single_cc"] == 0.0 and ms["n_components"] > 1
    assert ms["sa_to_volume"] > mc["sa_to_volume"]          # jagged
    assert ms["bilateral_symmetry"] < mc["bilateral_symmetry"]


def test_two_blobs_not_single_cc():
    g = np.zeros((5, 3, 3), bool)
    g[0:2] = True
    g[3:5] = True
    m = connectivity_metrics(g)
    assert m["n_components"] == 2
    assert m["single_cc"] == 0.0
    assert m["largest_cc_fraction"] == pytest.approx(0.5)


def test_hollow_box_has_one_cavity():
    g = np.ones((5, 5, 5), bool)
    g[2, 2, 2] = False
    assert compute_shape_metrics(g)["cavity_count"] == 1.0


def test_empty_grid_is_safe():
    m = compute_shape_metrics(np.zeros((3, 3, 3), bool))
    assert m["n_components"] == 0 and m["single_cc"] == 0.0


def test_grid_constructors():
    vid = np.array([0, 0, 1, 26], dtype=np.int64)
    g = occupancy_grid_from_voxel_id(vid, (3, 3, 3))
    assert g.shape == (3, 3, 3) and g.sum() == 3   # voxels 0,1,26 occupied
    occ = np.zeros(27, np.float32); occ[0] = 0.9; occ[1] = 0.1
    g2 = occupancy_grid_from_occ(occ, (3, 3, 3), threshold=0.5)
    assert g2.sum() == 1                            # only voxel 0 > 0.5


def test_diversity_detects_collapse():
    same = np.ones((4, 27), np.float32)
    varied = np.random.RandomState(0).rand(4, 27).astype(np.float32)
    assert diversity_metrics(same)["mean_pairwise_l2"] == pytest.approx(0.0)
    assert diversity_metrics(varied)["mean_pairwise_l2"] > 0.5
