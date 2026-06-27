"""MorphologyPrior protocol — interface every 3D generative prior must satisfy.

A prior is anything that takes a text prompt (or noise) and returns a list of
trimesh.Trimesh objects ready to be fed to the robotization pipeline. Concrete
implementations live in sibling files (random_shapes.py, triposg.py, ...) and
register themselves with the registry on import.

Industrial conventions:
- All priors are stateless once constructed; per-call randomness is driven by
  the caller-supplied seed.
- Heavy GPU resources (model weights) load lazily on first .sample() call so
  that just importing the module doesn't burn VRAM.
- Errors at construction (missing weights, missing optional dep) raise
  MissingDependencyError with a one-line install hint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol, runtime_checkable


class MissingDependencyError(ImportError):
    """Raised when a prior cannot be constructed because an optional dep
    (third_party clone, model weights, hardware) is missing.

    The .install_hint attribute should be a single short line the caller can
    print or log, e.g.::

        cd third_party/morphology_priors/triposg && git clone ...
    """

    def __init__(self, message: str, install_hint: str = ""):
        super().__init__(message)
        self.install_hint = install_hint


@dataclass(frozen=True)
class PriorMetadata:
    """Stable identifier for a prior, recorded in asset bank manifests so we
    can tell *which* prior produced an asset months later (writeup §13.1)."""

    name: str           # short id (registry key): "triposg", "random_shapes", ...
    version: str        # upstream package version or "unversioned"
    output_format: str  # "mesh" / "point_cloud" / "voxel" — informational only


@runtime_checkable
class MorphologyPrior(Protocol):
    """Interface for any 3D generative prior used by the robotization pipeline.

    Implementations must be importable with no side effects beyond registering
    themselves; heavy work (weight load, GPU init) belongs inside `sample()`.
    """

    metadata: PriorMetadata

    def sample(
        self,
        prompt: str,
        n: int = 1,
        *,
        seed: int = 0,
        **kwargs: Any,
    ) -> List[Any]:
        """Return ``n`` trimesh.Trimesh objects for the given prompt.

        Args
        ----
        prompt : str
            Free-form text condition. Subclasses interpret it however the
            underlying model expects (TripoSG: text or image; random_shapes:
            ignored).
        n : int
            Number of independent samples.
        seed : int
            Per-call RNG seed; identical seed + prompt + version must
            reproduce identical meshes (asset_id determinism).
        kwargs
            Prior-specific knobs (guidance scale, num_inference_steps, ...).
            Unknown keys must raise TypeError, not be silently dropped.

        Returns
        -------
        List[trimesh.Trimesh]
            Each entry is a watertight-ish mesh in arbitrary scale; the
            robotization layer will normalize and repair as needed.
        """
        ...
