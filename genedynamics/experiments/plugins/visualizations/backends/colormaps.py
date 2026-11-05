"""Matplotlib colormap compatibility through its supported public registry."""

from matplotlib import colormaps


def get_colormap(name: str):
    """Return a named colormap (``cm.get_cmap`` was removed in 3.11)."""
    return colormaps.get_cmap(name)
