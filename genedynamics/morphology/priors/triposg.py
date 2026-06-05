"""TripoSG adapter — writeup §2 main 3D foundation prior.

The actual TripoSG package lives under third_party/morphology_priors/triposg/
(see that directory's README for the clone instructions). This adapter is a
**lazy** wrapper: importing this module always succeeds, even when TripoSG
is not installed. The constructor or first `.sample()` call surfaces a
MissingDependencyError with the exact install hint.

The adapter does NOT vendor any TripoSG code; we only call the public API
(`TripoSGPipeline.from_pretrained` or whatever the upstream README documents).
"""

from __future__ import annotations

import importlib
import os
from typing import Any, List, Optional

from .base import MorphologyPrior, PriorMetadata, MissingDependencyError


_THIRD_PARTY_HINT = (
    "cd third_party/morphology_priors/triposg && "
    "git clone https://github.com/VAST-AI-Research/TripoSG.git . "
    "&& pip install -r requirements.txt"
)


def _try_import_triposg():
    """Look for an installed TripoSG package OR a third_party clone on path.

    Returns the imported module on success, raises MissingDependencyError
    otherwise. The lookup is widened to a few common module names since the
    upstream package layout has changed across versions.
    """
    candidates = ("triposg", "TripoSG", "triposg.pipelines")
    last_err: Optional[Exception] = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            last_err = exc
    raise MissingDependencyError(
        f"TripoSG not importable (tried {candidates}). "
        f"Install upstream first: {_THIRD_PARTY_HINT}",
        install_hint=_THIRD_PARTY_HINT,
    ) from last_err


def _get_version(mod) -> str:
    return getattr(mod, "__version__", "unversioned")


class TripoSGPrior(MorphologyPrior):
    """Wrapper around TripoSG's image/text-to-mesh pipeline.

    Construction is cheap (no model load); first `.sample()` call lazily
    instantiates the pipeline. Subsequent calls reuse it.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        checkpoint: Optional[str] = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 50,
    ):
        # Probe early so the failure message is attached to construction, not
        # to the first sampling call buried inside a worker process.
        mod = _try_import_triposg()
        self.metadata = PriorMetadata(
            name="triposg",
            version=_get_version(mod),
            output_format="mesh",
        )
        self.device = device
        self.checkpoint = checkpoint
        self.guidance_scale = float(guidance_scale)
        self.num_inference_steps = int(num_inference_steps)
        self._pipeline = None  # lazy

    # ------------------------------------------------------------------

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return
        mod = _try_import_triposg()
        # Upstream API is unstable across releases; try the documented entry
        # point first, fall back to a couple of historical variants. Always
        # raise MissingDependencyError on failure so callers get the same
        # error type as for "package not installed".
        loaders = []
        for attr in ("TripoSGPipeline", "Pipeline", "TextTo3DPipeline"):
            cls = getattr(mod, attr, None)
            if cls is not None and hasattr(cls, "from_pretrained"):
                loaders.append((attr, cls))
        if not loaders:
            raise MissingDependencyError(
                "Could not locate a TripoSG pipeline class with "
                ".from_pretrained() in the installed package. "
                "Upstream API may have changed; pin a known version.",
                install_hint=_THIRD_PARTY_HINT,
            )
        attr, cls = loaders[0]
        ckpt = self.checkpoint or os.environ.get(
            "TRIPOSG_CHECKPOINT",
            "VAST-AI/TripoSG",  # HuggingFace-style default; upstream may vary
        )
        self._pipeline = cls.from_pretrained(ckpt).to(self.device)

    def sample(
        self,
        prompt: str,
        n: int = 1,
        *,
        seed: int = 0,
        **kwargs: Any,
    ) -> List[Any]:
        """Generate ``n`` meshes from ``prompt``.

        Extra kwargs are forwarded to the pipeline; common ones:
          - guidance_scale (float): override constructor default
          - num_inference_steps (int): override constructor default
          - image (PIL.Image.Image): for image-conditioned variants
        """
        try:
            import trimesh  # noqa: F401 — ensure trimesh is present for output
        except ImportError as exc:
            raise MissingDependencyError(
                "trimesh is required to materialize TripoSG output",
                install_hint="pip install trimesh",
            ) from exc

        self._ensure_pipeline()
        merged = dict(
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
        )
        merged.update(kwargs)

        # The pipeline's output shape is upstream-dependent; we attempt the
        # most common contracts and convert to trimesh.Trimesh.
        meshes: List[Any] = []
        for i in range(int(n)):
            out = self._pipeline(
                prompt=prompt,
                seed=int(seed) + i,
                **merged,
            )
            mesh = _coerce_to_trimesh(out)
            if mesh is None:
                raise RuntimeError(
                    f"TripoSG returned an output of type {type(out).__name__} "
                    f"that this adapter does not know how to convert. "
                    f"Extend _coerce_to_trimesh in {__file__} for the new shape."
                )
            meshes.append(mesh)
        return meshes


def _coerce_to_trimesh(out: Any):
    """Best-effort: turn whatever TripoSG returns into a trimesh.Trimesh."""
    import trimesh
    if isinstance(out, trimesh.Trimesh):
        return out
    # HuggingFace-style pipeline output: object with .meshes or .mesh attr.
    for attr in ("mesh", "meshes"):
        m = getattr(out, attr, None)
        if isinstance(m, trimesh.Trimesh):
            return m
        if isinstance(m, list) and m and isinstance(m[0], trimesh.Trimesh):
            return m[0]
    # Dict outputs.
    if isinstance(out, dict):
        for key in ("mesh", "meshes", "trimesh"):
            v = out.get(key)
            if isinstance(v, trimesh.Trimesh):
                return v
            if isinstance(v, list) and v and isinstance(v[0], trimesh.Trimesh):
                return v[0]
    # (vertices, faces) tuple.
    if isinstance(out, tuple) and len(out) == 2:
        v, f = out
        try:
            return trimesh.Trimesh(vertices=v, faces=f, process=False)
        except Exception:
            return None
    return None


def _factory(**kwargs):
    return TripoSGPrior(**kwargs)


# Register UNCONDITIONALLY — lookup-time errors are deferred to construction.
# Listing the prior in `list_priors()` lets CLIs print "triposg (not installed)"
# instead of pretending it doesn't exist.
from . import register as _register
_register("triposg", _factory)
