"""
Compatibility shim: expose vendored D3IL as `d3il.environments.*`.

The real package lives at `third_party/environments`. We add that directory to
`sys.path`, import the real `environments` package, and mirror its `__path__`
so submodules (e.g., `d3il.environments.d3il.envs...`) resolve normally.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


# repo_root/d3il/environments/__init__.py -> repo_root
_repo_root = Path(__file__).resolve().parents[2]
_third_party = (_repo_root / "third_party").resolve()
if str(_third_party) not in sys.path:
    sys.path.insert(0, str(_third_party))

# Import the actual package and mirror its search path.
_env_pkg = importlib.import_module("environments")
__path__ = getattr(_env_pkg, "__path__", [])  # type: ignore[assignment]
__all__ = getattr(_env_pkg, "__all__", [])
