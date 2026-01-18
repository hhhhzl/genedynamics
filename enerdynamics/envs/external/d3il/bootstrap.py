from __future__ import annotations

import sys
from pathlib import Path


def ensure_d3il_on_path(project_root: str | Path | None = None) -> Path:
    """
    Ensure vendored D3IL is importable as `environments.d3il.*`.

    D3IL uses absolute imports like `environments.d3il.d3il_sim...`, so the
    directory that contains the `environments/` package must be on sys.path.

    We vendor D3IL at:
      <project_root>/third_party/environments/d3il/

    This helper adds:
      <project_root>/third_party/

    Returns:
        The resolved path to `<project_root>/third_party`.
    """
    if project_root is None:
        # enerdynamics/envs/external/d3il/bootstrap.py -> enerdynamics -> repo root
        project_root = Path(__file__).resolve().parents[4]
    else:
        project_root = Path(project_root).resolve()

    third_party = (project_root / "third_party").resolve()

    # Support two common layouts:
    #
    # 1) Vendored subtree (current):
    #    third_party/environments/d3il/  -> import environments.d3il.*
    #
    # 2) Full upstream repo as submodule:
    #    third_party/d3il/ (ALRhub/d3il)
    #      environments/d3il/            -> import environments.d3il.*
    #
    candidates = [
        third_party,                # layout (1): third_party/environments/...
        (third_party / "d3il"),     # layout (2): third_party/d3il/environments/...
    ]

    for base in candidates:
        environments_pkg = base / "environments"
        d3il_pkg = environments_pkg / "d3il"
        if d3il_pkg.exists():
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            return base

    raise FileNotFoundError(
        "D3IL not found in supported locations. Expected one of:\n"
        f"- {third_party / 'environments' / 'd3il'} (vendored subtree)\n"
        f"- {third_party / 'd3il' / 'environments' / 'd3il'} (full upstream repo)\n"
        "Run the Milestone A setup (copy/submodule) first."
    )


