"""Asset bank: on-disk catalog of generated meshes + their robotized SoftBodySpecs.

Layout
------
    <bank_root>/
    ├── manifest.json
    ├── meshes/
    │   └── <asset_id>.glb
    └── robotized/
        ├── <asset_id>.npz       # SoftBodySpec arrays
        └── <asset_id>.json      # report + provenance

The asset_id is a deterministic short hash of (prior_name + prior_version +
prompt + seed + index), so re-running build_asset_bank.py with the same
inputs reuses the same files (idempotent).

Schema is versioned (`schema_version` field) so we can evolve without
breaking older banks. Phase-3 ships v1.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .spec import SoftBodySpec
from .robotize.mesh import RobotizeReport, MeshRobotizeConfig


SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Deterministic asset ids
# ---------------------------------------------------------------------------


def asset_id(
    *,
    prior_name: str,
    prior_version: str,
    prompt: str,
    seed: int,
    index: int = 0,
    length: int = 12,
) -> str:
    """Stable short hash. Same inputs → same id forever."""
    h = hashlib.sha256()
    h.update(prior_name.encode("utf-8"))
    h.update(b"\0")
    h.update(prior_version.encode("utf-8"))
    h.update(b"\0")
    h.update(prompt.encode("utf-8"))
    h.update(b"\0")
    h.update(str(int(seed)).encode("ascii"))
    h.update(b"\0")
    h.update(str(int(index)).encode("ascii"))
    return h.hexdigest()[:length]


def prompt_id(text: str, length: int = 8) -> str:
    """Deterministic short id for a prompt — used as a folder bucket."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


# ---------------------------------------------------------------------------
# Spec serialization (numpy npz)
# ---------------------------------------------------------------------------


def save_spec_npz(spec: SoftBodySpec, path: str) -> None:
    """Persist a SoftBodySpec as a single .npz."""
    arrays: Dict[str, np.ndarray] = {
        "particles_x0": np.asarray(spec.particles_x0, dtype=np.float32),
        "actuator_id": np.asarray(spec.actuator_id, dtype=np.int32),
        "voxel_id": np.asarray(spec.voxel_id, dtype=np.int32),
        "fiber_dirs": np.asarray(spec.fiber_dirs, dtype=np.float32),
        "n_actuators": np.asarray(spec.n_actuators, dtype=np.int32),
        "n_voxels": np.asarray(spec.n_voxels, dtype=np.int32),
        "voxel_dims": np.asarray(spec.voxel_dims, dtype=np.int32),
    }
    if spec.E_per_particle is not None:
        arrays["E_per_particle"] = np.asarray(spec.E_per_particle, dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_spec_npz(path: str) -> SoftBodySpec:
    """Reload a SoftBodySpec from disk."""
    z = np.load(path, allow_pickle=False)
    return SoftBodySpec(
        particles_x0=np.asarray(z["particles_x0"], dtype=np.float32),
        actuator_id=np.asarray(z["actuator_id"], dtype=np.int32),
        voxel_id=np.asarray(z["voxel_id"], dtype=np.int32),
        fiber_dirs=np.asarray(z["fiber_dirs"], dtype=np.float32),
        n_actuators=int(z["n_actuators"]),
        n_voxels=int(z["n_voxels"]),
        voxel_dims=tuple(int(v) for v in np.asarray(z["voxel_dims"]).tolist()),
        E_per_particle=(
            np.asarray(z["E_per_particle"], dtype=np.float32)
            if "E_per_particle" in z.files
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Manifest dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PromptEntry:
    id: str
    text: str
    n_assets: int = 0


@dataclass
class AssetEntry:
    """One row in the manifest's `assets` list."""

    id: str
    prompt_id: str
    seed: int
    index: int
    mesh_path: Optional[str] = None         # relative to bank_root
    robotized_path: Optional[str] = None    # relative to bank_root
    robotized: bool = False
    report: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AssetEntry":
        return cls(
            id=str(d["id"]),
            prompt_id=str(d["prompt_id"]),
            seed=int(d["seed"]),
            index=int(d.get("index", 0)),
            mesh_path=d.get("mesh_path"),
            robotized_path=d.get("robotized_path"),
            robotized=bool(d.get("robotized", False)),
            report=dict(d.get("report") or {}),
        )


@dataclass
class BankManifest:
    schema_version: int = SCHEMA_VERSION
    bank_name: str = "unnamed"
    created: str = ""                                # ISO 8601 UTC
    prior_name: str = ""
    prior_version: str = ""
    robotize_config: Dict[str, Any] = field(default_factory=dict)
    prompts: List[PromptEntry] = field(default_factory=list)
    assets: List[AssetEntry] = field(default_factory=list)

    # ----- io ----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "bank_name": self.bank_name,
            "created": self.created,
            "prior": {"name": self.prior_name, "version": self.prior_version},
            "robotize_config": self.robotize_config,
            "prompts": [asdict(p) for p in self.prompts],
            "assets": [asdict(a) for a in self.assets],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BankManifest":
        if int(d.get("schema_version", 1)) > SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema_version={d['schema_version']} newer than "
                f"current code SCHEMA_VERSION={SCHEMA_VERSION}; "
                f"upgrade genedynamics.learning.priors.morphology.asset_bank"
            )
        prior = d.get("prior") or {}
        return cls(
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            bank_name=str(d.get("bank_name", "unnamed")),
            created=str(d.get("created", "")),
            prior_name=str(prior.get("name", "")),
            prior_version=str(prior.get("version", "")),
            robotize_config=dict(d.get("robotize_config") or {}),
            prompts=[PromptEntry(**p) for p in (d.get("prompts") or [])],
            assets=[AssetEntry.from_dict(a) for a in (d.get("assets") or [])],
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "BankManifest":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# AssetBank — high-level handle the CLIs use
# ---------------------------------------------------------------------------


@dataclass
class AssetBank:
    """In-memory + on-disk bank handle.

    Use `AssetBank.create(...)` for a fresh bank, or `AssetBank.open(root)`
    to attach to an existing one.
    """

    root: str
    manifest: BankManifest

    # ----- factories ---------------------------------------------------

    @classmethod
    def create(
        cls,
        root: str,
        bank_name: str,
        prior_name: str,
        prior_version: str,
        prompts: List[str],
        robotize_config: Optional[MeshRobotizeConfig] = None,
    ) -> "AssetBank":
        os.makedirs(os.path.join(root, "meshes"), exist_ok=True)
        os.makedirs(os.path.join(root, "robotized"), exist_ok=True)
        manifest = BankManifest(
            bank_name=bank_name,
            created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            prior_name=prior_name,
            prior_version=prior_version,
            robotize_config=(
                dataclasses.asdict(robotize_config) if robotize_config else {}
            ),
            prompts=[PromptEntry(id=prompt_id(p), text=p) for p in prompts],
            assets=[],
        )
        bank = cls(root=root, manifest=manifest)
        bank.save()
        return bank

    @classmethod
    def open(cls, root: str) -> "AssetBank":
        return cls(root=root, manifest=BankManifest.load(os.path.join(root, "manifest.json")))

    # ----- io ----------------------------------------------------------

    def save(self) -> None:
        self.manifest.save(os.path.join(self.root, "manifest.json"))

    def add_mesh(
        self,
        mesh,
        prompt: str,
        seed: int,
        index: int = 0,
    ) -> AssetEntry:
        """Persist ``mesh`` and append an AssetEntry. Idempotent on asset_id."""
        aid = asset_id(
            prior_name=self.manifest.prior_name,
            prior_version=self.manifest.prior_version,
            prompt=prompt,
            seed=seed,
            index=index,
        )
        # Skip if we already have this asset (replays should be free).
        for a in self.manifest.assets:
            if a.id == aid:
                return a
        # Trimesh exports based on extension.
        rel_mesh = os.path.join("meshes", f"{aid}.glb")
        mesh_path = os.path.join(self.root, rel_mesh)
        os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
        mesh.export(mesh_path)
        entry = AssetEntry(
            id=aid,
            prompt_id=prompt_id(prompt),
            seed=int(seed),
            index=int(index),
            mesh_path=rel_mesh,
        )
        self.manifest.assets.append(entry)
        # Update prompt's n_assets counter.
        for p in self.manifest.prompts:
            if p.id == entry.prompt_id:
                p.n_assets += 1
                break
        return entry

    def add_robotized(
        self,
        asset: AssetEntry,
        spec: Optional[SoftBodySpec],
        report: RobotizeReport,
    ) -> None:
        """Persist a SoftBodySpec next to its mesh and write the report."""
        rel_npz = os.path.join("robotized", f"{asset.id}.npz")
        rel_json = os.path.join("robotized", f"{asset.id}.json")
        if spec is not None:
            save_spec_npz(spec, os.path.join(self.root, rel_npz))
            asset.robotized_path = rel_npz
            asset.robotized = True
        else:
            asset.robotized = False
        asset.report = {
            "success": report.success,
            "failure_reasons": list(report.failure_reasons),
            "n_filled_cells": int(report.n_filled_cells),
            "n_dropped_for_connectivity": int(report.n_dropped_for_connectivity),
            "actuator_coverage": float(report.actuator_coverage),
            "n_actuators_used": int(report.n_actuators_used),
            "body_diameter": float(report.body_diameter),
        }
        with open(os.path.join(self.root, rel_json), "w") as f:
            json.dump(asset.report, f, indent=2)

    # ----- queries -----------------------------------------------------

    def successful_assets(self) -> List[AssetEntry]:
        return [a for a in self.manifest.assets if a.robotized]

    def robotization_success_rate(self) -> float:
        if not self.manifest.assets:
            return 0.0
        ok = sum(1 for a in self.manifest.assets if a.robotized)
        return ok / len(self.manifest.assets)

    def load_spec(self, asset: AssetEntry) -> SoftBodySpec:
        if not asset.robotized or not asset.robotized_path:
            raise ValueError(f"asset {asset.id} not robotized")
        return load_spec_npz(os.path.join(self.root, asset.robotized_path))
