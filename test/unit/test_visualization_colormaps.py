"""Compatibility checks for Matplotlib's public colormap registry."""

import numpy as np

from genedynamics.experiments.plugins.visualizations.backends.colormaps import (
    get_colormap,
)


def test_corridor_ghost_colormap_uses_public_registry():
    colors = np.asarray([get_colormap("viridis")(value) for value in (0.0, 1.0)])
    assert colors.shape == (2, 4)
    assert np.isfinite(colors).all()


def test_named_discrete_colormap_uses_public_registry():
    colors = np.asarray([get_colormap("tab10")(index) for index in range(4)])
    assert colors.shape == (4, 4)
    assert np.isfinite(colors).all()
