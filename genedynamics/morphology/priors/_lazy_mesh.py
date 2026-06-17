"""Shared lazy mesh-prior adapter.

The Table-3 mesh priors (Hunyuan3D-2, TRELLIS, CraftsMan3D, MeshFlow, DiffGS)
all follow the same contract as `triposg.py`: import always succeeds; the
constructor / first `.sample()` surfaces a `MissingDependencyError` with an
install hint when the upstream package is absent; once installed, a
`from_pretrained` pipeline is called and its output coerced to trimesh. This
module factors that boilerplate into `LazyMeshPrior` + `register_lazy_mesh_prior`
so each concrete adapter is ~10 lines (model name, candidate module/class names,
default checkpoint, install hint). Upstream APIs are unstable, so we try a few
historical module/class names and pin the exact one at install time.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, List, Optional, Sequence

from .base import MorphologyPrior, PriorMetadata, MissingDependencyError
from . import register as _register


def _coerce_to_trimesh(out: Any):
    """Best-effort conversion of an upstream pipeline output to trimesh.Trimesh."""
    import trimesh
    if isinstance(out, trimesh.Trimesh):
        return out
    for attr in ("mesh", "meshes", "trimesh"):
        m = getattr(out, attr, None)
        if isinstance(m, trimesh.Trimesh):
            return m
        if isinstance(m, list) and m and isinstance(m[0], trimesh.Trimesh):
            return m[0]
    if isinstance(out, dict):
        for key in ("mesh", "meshes", "trimesh"):
            v = out.get(key)
            if isinstance(v, trimesh.Trimesh):
                return v
            if isinstance(v, list) and v and isinstance(v[0], trimesh.Trimesh):
                return v[0]
    if isinstance(out, (list, tuple)) and len(out) == 2:
        v, f = out
        try:
            return trimesh.Trimesh(vertices=v, faces=f, process=False)
        except Exception:
            return None
    return None


class LazyMeshPrior(MorphologyPrior):
    """Generic lazy wrapper around a `from_pretrained`-style text/image→mesh
    pipeline. Construction probes the upstream import so the failure is attached
    early; weights load lazily on the first `.sample()`."""

    def __init__(
        self,
        *,
        name: str,
        modules: Sequence[str],
        classes: Sequence[str],
        default_ckpt: str,
        install_hint: str,
        device: str = "cuda",
        guidance_scale: float = 7.0,
        num_inference_steps: int = 50,
        checkpoint: Optional[str] = None,
    ):
        self._name = name
        self._modules = tuple(modules)
        self._classes = tuple(classes)
        self._default_ckpt = default_ckpt
        self._hint = install_hint
        self.device = device
        self.guidance_scale = float(guidance_scale)
        self.num_inference_steps = int(num_inference_steps)
        self.checkpoint = checkpoint
        self._pipeline = None
        mod = self._try_import()  # probe early
        self.metadata = PriorMetadata(
            name=name,
            version=getattr(mod, "__version__", "unversioned"),
            output_format="mesh",
        )

    def _try_import(self):
        last: Optional[Exception] = None
        for m in self._modules:
            try:
                return importlib.import_module(m)
            except ImportError as exc:
                last = exc
        raise MissingDependencyError(
            f"{self._name} not importable (tried {self._modules}). "
            f"Install upstream first: {self._hint}",
            install_hint=self._hint,
        ) from last

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return
        mod = self._try_import()
        loaders = []
        for attr in self._classes:
            cls = getattr(mod, attr, None)
            if cls is not None and hasattr(cls, "from_pretrained"):
                loaders.append(cls)
        if not loaders:
            raise MissingDependencyError(
                f"No {self._name} pipeline class with .from_pretrained() in "
                f"{self._modules}. Upstream API may have changed; pin a version.",
                install_hint=self._hint,
            )
        ckpt = self.checkpoint or os.environ.get(
            f"{self._name.upper()}_CHECKPOINT", self._default_ckpt
        )
        self._pipeline = loaders[0].from_pretrained(ckpt).to(self.device)

    def sample(self, prompt: str, n: int = 1, *, seed: int = 0, **kwargs: Any) -> List[Any]:
        try:
            import trimesh  # noqa: F401
        except ImportError as exc:
            raise MissingDependencyError(
                "trimesh is required to materialize prior output",
                install_hint="pip install trimesh",
            ) from exc
        self._ensure_pipeline()
        merged = dict(guidance_scale=self.guidance_scale,
                      num_inference_steps=self.num_inference_steps)
        merged.update(kwargs)
        meshes: List[Any] = []
        for i in range(int(n)):
            out = self._pipeline(prompt=prompt, seed=int(seed) + i, **merged)
            mesh = _coerce_to_trimesh(out)
            if mesh is None:
                raise RuntimeError(
                    f"{self._name} returned {type(out).__name__} that this adapter "
                    f"cannot convert; extend _coerce_to_trimesh."
                )
            meshes.append(mesh)
        return meshes


def register_lazy_mesh_prior(
    *,
    name: str,
    modules: Sequence[str],
    classes: Sequence[str],
    default_ckpt: str,
    install_hint: str,
) -> None:
    """Register a LazyMeshPrior factory under ``name`` (UNCONDITIONALLY, so
    list_priors() shows it even when the upstream package is absent)."""

    def _factory(**kwargs):
        return LazyMeshPrior(
            name=name, modules=modules, classes=classes,
            default_ckpt=default_ckpt, install_hint=install_hint, **kwargs,
        )

    _register(name, _factory)
