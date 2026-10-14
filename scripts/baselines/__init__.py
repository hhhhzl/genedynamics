"""Pinned upstream sources and disposable native baseline environments."""
from pathlib import Path


def require_non_paper_output(path):
    """Keep baseline experiment artifacts out of the manuscript tree."""
    path = Path(path).resolve()
    paper = Path(__file__).resolve().parents[2] / "latex"
    if path == paper or paper in path.parents:
        raise ValueError("baseline experiment artifacts belong outside latex/")
    return path
