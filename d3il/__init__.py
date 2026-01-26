# Lightweight shim to support imports like `d3il.environments.*` (vendored
# under `third_party/environments`) and, when available, external DPCC `d3il`
# sources (e.g., `../dpcc/d3il`). We add any discovered roots to `sys.path` so
# submodules such as `d3il.agents.*` resolve during DPCC checkpoint loading.
from pathlib import Path
import sys

# Prepare search paths for d3il.* subpackages (agents, etc.)
repo_root = Path(__file__).resolve().parents[1]
external_candidates = [
    repo_root.parent / "dpcc" / "d3il" / "d3il",  # dpcc/d3il/d3il (contains agents)
    repo_root.parent / "dpcc" / "d3il",          # dpcc/d3il (parent)
    repo_root.parent / "dpcc" / "src" / "d3il",  # dpcc/src/d3il
]

paths = []
for ext in external_candidates:
    ext = ext.resolve()
    if ext.exists() and ext.is_dir():
        if str(ext) not in sys.path:
            sys.path.insert(0, str(ext))
        paths.append(str(ext))

# Make this a namespace-style package that also searches external paths
try:
    __path__ = list(__path__) + paths  # type: ignore[name-defined]
except Exception:
    pass
