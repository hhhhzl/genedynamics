"""
Configuration types for SoftZoo runtime and environment creation.

Separates runtime config (paths, device, logging) from env config
(task-specific parameters). Supports YAML/dict loading for experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SoftZooRuntimeConfig:
    """
    Runtime configuration for SoftZoo integration.

    Attributes:
        project_root: Project root path
        use_renderer: Whether to enable renderer
        device: torch_cpu | torch_gpu | numpy
        ti_arch: Optional Taichi backend override (cpu | metal | cuda | vulkan)
        ti_device_memory_fraction: Taichi GPU memory fraction (0-1)
        out_dir: Output directory for logs/checkpoints
        suppress_init_print: Suppress env init prints
    """

    project_root: Optional[Path] = None
    use_renderer: bool = False
    device: str = "torch_cpu"
    ti_arch: Optional[str] = None
    ti_device_memory_fraction: Optional[float] = None
    out_dir: str = "/tmp/softzoo"
    suppress_init_print: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SoftZooRuntimeConfig":
        """Load from dict (e.g. YAML)."""
        kwargs = {}
        for k, v in data.items():
            if k in cls.__dataclass_fields__:
                if k == "project_root" and v is not None:
                    v = Path(v)
                kwargs[k] = v
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        out = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if k == "project_root" and v is not None:
                v = str(v)
            if v is not None and k != "extra":
                out[k] = v
        out.update(self.extra)
        return out


@dataclass
class SoftZooEnvConfig:
    """
    Environment creation configuration.

    Combines task spec identifiers with runtime overrides.
    Used by make_softzoo_env().
    """

    task_id: str = "crawling_ground"
    env_config_file: str = "ground.yaml"
    env_type: str = "land_environment"
    designer_type: str = "annotated_pcd"
    pcd_name: str = "Caterpillar"
    pcd_path: Optional[str] = None
    n_actuators: int = 10
    actuation_omega: tuple = field(default_factory=lambda: (20.0, 100.0))
    max_steps: int = 200
    fidelity_level: int = 2
    cfg_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SoftZooEnvConfig":
        """Load from dict."""
        kwargs = {}
        for k, v in data.items():
            if k in cls.__dataclass_fields__:
                if k == "actuation_omega" and isinstance(v, (list, tuple)) and len(v) >= 2:
                    v = (float(v[0]), float(v[1]))
                kwargs[k] = v
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        out = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if v is not None and k != "extra":
                out[k] = v
        out.update(self.extra)
        return out
