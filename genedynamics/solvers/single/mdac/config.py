"""YAML experiment-config loader for the MDAC harness (pure Python, no brax).

The MDAC comparison is configured by a tree of method yamls subdivided BY
ENVIRONMENT, e.g. ``configs/arm/impedence/rigid/<environment>/<role>/<method>.yaml``
(environment = surface family plane/cylinder/convex/bumpy/unseen; role =
baseline/main/ablation). Each method yaml inherits a shared ``_base.yaml`` (task /
seeds / horizon / sampling budget) via a ``base: <relative-path>`` key and sets
only ``level`` / ``method`` / ``name`` / ``output_dir`` — so every method in every
environment runs at an IDENTICAL budget (``method_registry.assert_fair``),
differing only by the method's component flags.

``discover_configs`` walks the tree and picks up any method yaml whose parent dir
is a role dir, so the older flat layout (``rigid/<role>/<method>.yaml``) also works.

Kept separate from ``experiment.py``/``run_experiment.py`` so config loading pulls
in NO brax (importable + testable on fedguide).
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List

import yaml

# role dir name -> canonical group (accept singular/plural "baseline(s)").
_ROLE_CANON = {"baseline": "baseline", "baselines": "baseline",
               "main": "main", "ablation": "ablation"}
_ROLE_RANK = {"main": 0, "baseline": 1, "ablation": 2}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins). Nested dicts
    merge key-wise; scalars / lists replace. Neither input is mutated."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_experiment_config(path: str) -> Dict[str, Any]:
    """Load one method yaml, resolving its ``base:`` chain (relative to the yaml's
    own directory) and deep-merging base <- this. Returns the merged dict with the
    ``base`` key stripped and ``config_path`` recorded."""
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("base", None)
    if base_ref is not None:
        base_path = os.path.normpath(os.path.join(os.path.dirname(path), base_ref))
        merged = deep_merge(load_experiment_config(base_path), cfg)
    else:
        merged = cfg
    merged["config_path"] = path
    return merged


def discover_configs(config_dir: str) -> List[Dict[str, Any]]:
    """Walk ``config_dir`` and load every method yaml whose parent dir is a role dir
    (baseline / main / ablation), skipping leading-underscore shared bases. Works for
    the per-environment tree (``<env>/<role>/<method>.yaml``) and the flat layout
    (``<role>/<method>.yaml``). Each result is tagged with its canonical ``group``;
    ordered by (level, role-rank main<baseline<ablation, method)."""
    out: List[Dict[str, Any]] = []
    for root, _dirs, files in os.walk(config_dir):
        group = _ROLE_CANON.get(os.path.basename(root))
        if group is None:
            continue
        for fname in sorted(files):
            if not fname.endswith((".yaml", ".yml")) or fname.startswith("_"):
                continue
            cfg = load_experiment_config(os.path.join(root, fname))
            cfg["group"] = group
            out.append(cfg)
    out.sort(key=lambda c: (str(c.get("level", "")),
                            _ROLE_RANK.get(c.get("group"), 9),
                            str(c.get("method", ""))))
    return out


__all__ = ["deep_merge", "load_experiment_config", "discover_configs"]
