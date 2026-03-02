from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path
    checkpoints_dir: Path
    configs_dir: Path
    meta_path: Path
    manifest_path: Path
    external_uris_path: Path
    normalizer_path: Path

    @staticmethod
    def from_root(root: str | Path) -> "ArtifactLayout":
        root = Path(root)
        return ArtifactLayout(
            root=root,
            checkpoints_dir=root / "checkpoints",
            configs_dir=root / "configs",
            meta_path=root / "meta.json",
            manifest_path=root / "manifest.json",
            external_uris_path=root / "external_uris.json",
            normalizer_path=root / "normalizer.pkl",
        )


class DPCCArtifactStore:
    """
    Minimal artifact store for DPCC with an extensible layout.

    Structure:
      root/
        checkpoints/
          best.pt
          checkpoint_{step}.pt
        configs/
          train.yaml
          plan.yaml
          indices.yaml
        normalizer.pkl
        meta.json
        manifest.json
        external_uris.json
    """

    def __init__(self, root: str | Path):
        self.layout = ArtifactLayout.from_root(root)

    def ensure_dirs(self) -> None:
        self.layout.root.mkdir(parents=True, exist_ok=True)
        self.layout.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.layout.configs_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, state_dict: Any, name: str = "best.pt") -> Path:
        if torch is None:
            raise ImportError("PyTorch is required to save checkpoints.")
        self.ensure_dirs()
        ckpt_path = self.layout.checkpoints_dir / name
        torch.save(state_dict, ckpt_path)
        return ckpt_path

    def save_checkpoint_step(self, state_dict: Any, step: int) -> Path:
        name = f"checkpoint_{int(step)}.pt"
        return self.save_checkpoint(state_dict, name)

    def load_checkpoint(self, name: str = "best.pt", map_location: str | None = "cpu") -> Any:
        if torch is None:
            raise ImportError("PyTorch is required to load checkpoints.")
        ckpt_path = self.layout.checkpoints_dir / name
        return torch.load(ckpt_path, map_location=map_location)

    def save_normalizer(self, normalizer: Any) -> Path:
        self.ensure_dirs()
        with open(self.layout.normalizer_path, "wb") as f:
            pickle.dump(normalizer, f)
        return self.layout.normalizer_path

    def load_normalizer(self) -> Any:
        with open(self.layout.normalizer_path, "rb") as f:
            return pickle.load(f)

    def save_config(self, config: Dict[str, Any], name: str) -> Path:
        self.ensure_dirs()
        path = self.layout.configs_dir / name
        if yaml is None:
            raise ImportError("PyYAML is required to save config files.")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return path

    def load_config(self, name: str) -> Dict[str, Any]:
        path = self.layout.configs_dir / name
        if yaml is None:
            raise ImportError("PyYAML is required to load config files.")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def save_meta(self, meta: Dict[str, Any]) -> Path:
        self.ensure_dirs()
        payload = dict(meta)
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        payload.setdefault("schema_version", "dpcc.artifact.v1")
        with open(self.layout.meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return self.layout.meta_path

    def load_meta(self) -> Dict[str, Any]:
        if not self.layout.meta_path.exists():
            return {}
        with open(self.layout.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_manifest(self, manifest: Dict[str, Any]) -> Path:
        self.ensure_dirs()
        payload = dict(manifest)
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        with open(self.layout.manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return self.layout.manifest_path

    def load_manifest(self) -> Dict[str, Any]:
        if not self.layout.manifest_path.exists():
            return {}
        with open(self.layout.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def register_external_uri(self, name: str, uri: str, metadata: Optional[Dict[str, Any]] = None) -> Path:
        self.ensure_dirs()
        payload = {"uri": uri, "metadata": metadata or {}, "updated_at": datetime.now(timezone.utc).isoformat()}
        all_uris = self.list_external_uris()
        all_uris[name] = payload
        with open(self.layout.external_uris_path, "w", encoding="utf-8") as f:
            json.dump(all_uris, f, indent=2, sort_keys=True)
        return self.layout.external_uris_path

    def list_external_uris(self) -> Dict[str, Any]:
        if not self.layout.external_uris_path.exists():
            return {}
        with open(self.layout.external_uris_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_all(
        self,
        *,
        state_dict: Any,
        normalizer: Any,
        train_config: Dict[str, Any],
        plan_config: Dict[str, Any],
        indices: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
        manifest: Optional[Dict[str, Any]] = None,
        checkpoint_name: str = "best.pt",
    ) -> Dict[str, Path]:
        paths = {
            "checkpoint": self.save_checkpoint(state_dict, checkpoint_name),
            "normalizer": self.save_normalizer(normalizer),
            "train_config": self.save_config(train_config, "train.yaml"),
            "plan_config": self.save_config(plan_config, "plan.yaml"),
            "indices": self.save_config(indices, "indices.yaml"),
        }
        if meta is not None:
            paths["meta"] = self.save_meta(meta)
        else:
            paths["meta"] = self.save_meta({})
        if manifest is not None:
            paths["manifest"] = self.save_manifest(manifest)
        return paths

    def load_all(
        self,
        *,
        checkpoint_name: str = "best.pt",
        map_location: str | None = "cpu",
    ) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        state_dict = self.load_checkpoint(checkpoint_name, map_location=map_location)
        normalizer = self.load_normalizer()
        train_config = self.load_config("train.yaml")
        plan_config = self.load_config("plan.yaml")
        indices = self.load_config("indices.yaml")
        meta = self.load_meta()
        manifest = self.load_manifest()
        external_uris = self.list_external_uris()
        return state_dict, normalizer, train_config, plan_config, indices, meta, manifest, external_uris

