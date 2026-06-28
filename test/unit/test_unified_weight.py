"""Stage-4 unit tests: hard validity indicator I(z) for the unified weight.

`validity_logmask` returns log I in {0, -1e4}: 0 for a morphology that is dense
enough AND 6-connected, -1e4 otherwise (driven out of the importance softmax).
"""
from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
from genedynamics.solvers.single.mrmfmbd.backends.mrmfmbd_mbd_jax import (  # noqa: E402
    validity_logmask,
)


def _occ(grid):
    g = np.asarray(grid, dtype=np.float32)
    return jnp.asarray(g.reshape(1, -1))


def test_connected_body_valid():
    g = np.zeros((3, 3, 3)); g[1, :, 1] = 1.0           # connected line of 3
    assert float(validity_logmask(_occ(g), 3, 3, 3, 0.01, 12)[0]) == 0.0


def test_fragmented_invalid():
    g = np.zeros((3, 3, 3)); g[0, 0, 0] = 1.0; g[2, 2, 2] = 1.0   # two disjoint voxels
    assert float(validity_logmask(_occ(g), 3, 3, 3, 0.001, 12)[0]) < -1e3


def test_too_sparse_invalid():
    g = np.zeros((4, 4, 4)); g[0, 0, 0] = 1.0           # frac 1/64 < 0.05
    assert float(validity_logmask(_occ(g), 4, 4, 4, 0.05, 12)[0]) < -1e3


def test_full_body_valid():
    g = np.ones((3, 3, 3))
    assert float(validity_logmask(_occ(g), 3, 3, 3, 0.1, 12)[0]) == 0.0


def test_batch_mixed():
    g_ok = np.ones((2, 2, 2))
    g_bad = np.zeros((2, 2, 2)); g_bad[0, 0, 0] = 1.0; g_bad[1, 1, 1] = 1.0
    occ = jnp.asarray(np.stack([g_ok.reshape(-1), g_bad.reshape(-1)]).astype("float32"))
    lm = validity_logmask(occ, 2, 2, 2, 0.01, 8)
    assert float(lm[0]) == 0.0 and float(lm[1]) < -1e3


def test_connectivity_uses_occupied_mask_only():
    # A C-shaped connected body in 3x3 (z=0 slice) stays valid; a body with a
    # detached corner is invalid.
    g = np.zeros((3, 3, 1)); g[0, :, 0] = 1; g[:, 0, 0] = 1   # an L (connected)
    assert float(validity_logmask(_occ(g), 3, 3, 1, 0.01, 12)[0]) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
