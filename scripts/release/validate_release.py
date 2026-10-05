"""Validate the configuration and source boundary for the V1 release."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import yaml

from genedynamics.experiments.framework.config import ExperimentConfig


RELEASE_CONFIG_ROOTS = (
    "configs/single_2d",
    "configs/d3il_avoiding",
    "configs/quadruped/stepping_stones_2d",
    "configs/humanoid/corridor_2d",
    "configs/arm/peg_insert",
    "configs/arm/surface_scan",
)

FORBIDDEN_SOURCE_ROOTS = (
    "configs/3dgs",
    "configs/soft_robot",
    "genedynamics/morphology",
    "genedynamics/solvers/single/mbd3d",
    "genedynamics/solvers/single/mrmfmbd",
    "scripts/tasks/3dgs",
    "scripts/tasks/soft_robot",
)

RELEASE_METHODS = {
    "2go",
    "atacom",
    "cfsmbd_full",
    "d3il_unified",
    "dial",
    "dpcc",
    "ebmbd",
    "issa",
    "mbd",
    "mdoc",
    "mga",
    "model_based_only",
    "mppi",
    "pegasusflow",
    "safediffuser",
    "standalone_rl",
}

RELEASE_ENVIRONMENTS = {
    "d3il_avoiding",
    "d3il_avoiding_9d",
    "humanoid_corridor_2d",
    "manipulator_peg_insert",
    "manipulator_surface_scan",
    "quadruped_stepping_stones_2d",
    "single_integrator_box_2d",
}


def _contains_files(path: Path) -> bool:
    if not path.exists():
        return False
    return any(
        candidate.is_file()
        and candidate.name != ".DS_Store"
        and "__pycache__" not in candidate.parts
        and candidate.suffix not in {".pyc", ".pyo"}
        for candidate in path.rglob("*")
    )


def validate(root: Path) -> tuple[int, int, Counter[str], Counter[str]]:
    errors: list[str] = []
    method_counts: Counter[str] = Counter()
    environment_counts: Counter[str] = Counter()
    config_count = 0
    deploy_config_count = 0

    manifest_path = root / "release" / "v1_compatibility.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        errors.append(f"{manifest_path.relative_to(root)}: {type(exc).__name__}: {exc}")
        manifest = {}
    if set(manifest.get("methods", [])) != RELEASE_METHODS:
        errors.append("release manifest method catalogue differs from validator")
    if set(manifest.get("environments", [])) != RELEASE_ENVIRONMENTS:
        errors.append("release manifest environment catalogue differs from validator")

    for relative in FORBIDDEN_SOURCE_ROOTS:
        path = root / relative
        if _contains_files(path):
            errors.append(f"deferred V1 files remain under {relative}")

    for relative in RELEASE_CONFIG_ROOTS:
        config_root = root / relative
        if not config_root.is_dir():
            errors.append(f"missing release config root: {relative}")
            continue
        for path in sorted(config_root.rglob("*.yaml")):
            if "deploy" in path.relative_to(config_root).parts:
                deploy_config_count += 1
                try:
                    data = yaml.safe_load(path.read_text()) or {}
                except Exception as exc:
                    errors.append(
                        f"{path.relative_to(root)}: {type(exc).__name__}: {exc}"
                    )
                    continue
                if not isinstance(data, dict):
                    errors.append(f"{path.relative_to(root)}: deploy config is not a mapping")
                    continue
                required = (
                    {"exec_mode", "robot_type", "model_id"}
                    if "quadruped" in path.parts
                    else {"plan", "plan_path", "control_hz"}
                )
                missing = sorted(required.difference(data))
                if missing:
                    errors.append(
                        f"{path.relative_to(root)}: deploy config missing {missing}"
                    )
                continue
            config_count += 1
            try:
                config = ExperimentConfig.from_yaml(path)
            except Exception as exc:
                errors.append(f"{path.relative_to(root)}: {type(exc).__name__}: {exc}")
                continue
            method_counts[config.method] += 1
            environment_counts[config.env_name] += 1
            if config.method not in RELEASE_METHODS:
                errors.append(
                    f"{path.relative_to(root)}: method {config.method!r} is outside V1"
                )
            if config.env_name not in RELEASE_ENVIRONMENTS:
                errors.append(
                    f"{path.relative_to(root)}: environment {config.env_name!r} is outside V1"
                )
            allowed_backends = (
                {"torch"} if config.method in {"dpcc", "safediffuser"} else {"jax"}
            )
            if config.backend not in allowed_backends:
                errors.append(
                    f"{path.relative_to(root)}: backend {config.backend!r} is invalid "
                    f"for method {config.method!r}; expected {sorted(allowed_backends)}"
                )

    if not config_count:
        errors.append("no release configurations found")
    if errors:
        raise SystemExit("V1 release validation failed:\n- " + "\n- ".join(errors))
    return config_count, deploy_config_count, method_counts, environment_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    count, deploy_count, methods, environments = validate(args.root.resolve())
    print(
        f"validated {count} experiment configurations and "
        f"{deploy_count} deployment configurations"
    )
    print("methods:", ", ".join(f"{name}={methods[name]}" for name in sorted(methods)))
    print(
        "environments:",
        ", ".join(f"{name}={environments[name]}" for name in sorted(environments)),
    )


if __name__ == "__main__":
    main()
