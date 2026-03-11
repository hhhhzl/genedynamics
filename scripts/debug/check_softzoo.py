#!/usr/bin/env python3
"""
SoftZoo environment health check.

Validates:
- Code root exists
- Config root exists
- Assets (optional)
- One rollout (optional, requires Taichi/PyTorch)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _get_paths():
    """Inline path resolution to avoid importing genedynamics (which pulls jax)."""
    code_root = ROOT / "third_party" / "environments" / "softzoo"
    assets_root = ROOT / "data" / "softzoo" / "assets"
    pkg_assets = code_root / "softzoo" / "assets"
    if not assets_root.exists() and pkg_assets.exists():
        assets_root = pkg_assets
    return code_root, assets_root


def _validate():
    ok, msgs = True, []
    code_root, assets_root = _get_paths()
    if not code_root.exists():
        ok, msgs = False, [f"Code root missing: {code_root}"]
    else:
        msgs.append(f"Code root OK: {code_root}")
    if not (code_root / "softzoo").exists():
        ok, msgs = False, msgs + [f"Package missing: {code_root / 'softzoo'}"]
    env_configs = code_root / "softzoo" / "configs" / "env_configs"
    if not env_configs.exists():
        ok, msgs = False, msgs + [f"Configs missing: {env_configs}"]
    else:
        msgs.append(f"Config root OK: {env_configs}")
    msgs.append(f"Assets root: {assets_root}")
    return ok, msgs


def _ensure_on_path():
    code_root, _ = _get_paths()
    if not code_root.exists():
        raise FileNotFoundError(f"SoftZoo not found: {code_root}")
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    return code_root


def main():
    ap = argparse.ArgumentParser(description="SoftZoo health check")
    ap.add_argument("--require-assets", action="store_true", help="Require assets dir")
    ap.add_argument("--require-pcd", type=str, default=None, help="Require PCD file (e.g. Caterpillar)")
    ap.add_argument("--run-rollout", action="store_true", help="Run one rollout (requires Taichi)")
    args = ap.parse_args()

    print("=== SoftZoo Health Check ===\n")

    # Path resolution
    ok, messages = _validate()
    if args.require_pcd:
        pcd_path = ROOT / "data" / "softzoo" / "assets" / "meshes" / "pcd" / f"{args.require_pcd}.pcd"
        if not pcd_path.exists():
            pcd_path = ROOT / "third_party" / "environments" / "softzoo" / "softzoo" / "assets" / "meshes" / "pcd" / f"{args.require_pcd}.pcd"
        if not pcd_path.exists():
            ok, messages = False, messages + [f"PCD missing: {args.require_pcd}.pcd"]
        else:
            messages.append(f"PCD OK: {pcd_path}")
    for m in messages:
        print(m)
    print()

    if not ok:
        print("FAILED: Path validation failed.")
        return 1

    # Ensure on path
    try:
        code_root = _ensure_on_path()
        print(f"Code root on path: {code_root}\n")
    except FileNotFoundError as e:
        print(f"FAILED: {e}")
        return 1

    # Optional: one rollout
    if args.run_rollout:
        print("Running one rollout (dummy env)...")
        try:
            from genedynamics.envs.external.softzoo.config import SoftZooRuntimeConfig
            from genedynamics.envs.external.softzoo.adapters import make_softzoo_env
            from genedynamics.envs.external.softzoo.task_registry import get_task_spec
            task_spec = get_task_spec("crawling_ground")
            runtime_config = SoftZooRuntimeConfig(
                use_renderer=False,
                device="torch_cpu",
                out_dir="/tmp/softzoo_check",
                suppress_init_print=True,
                project_root=ROOT,
            )
            env = make_softzoo_env(
                task_spec=task_spec,
                runtime_config=runtime_config,
            )
            design = {}
            env.reset(design)
            total_reward = 0.0
            for _ in range(10):
                obs, reward, done, info = env.step(env.action_space.sample())
                total_reward += reward
                if done:
                    break
            env.close()
            print(f"  Rollout OK: return={total_reward:.4f}\n")
        except Exception as e:
            print(f"  Rollout FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            return 1

    print("PASSED: SoftZoo environment OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
