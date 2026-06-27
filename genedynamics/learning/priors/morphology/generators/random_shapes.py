"""RandomShapesPrior — synthetic procedural prior.

Used for two things:
  1. Writeup §13.1 "Random valid prior" baseline — gives the optimizer a
     diverse but unstructured asset bank.
  2. CI / development surface — lets the entire Phase 3 pipeline run with
     no third_party clones, no model weights, no GPU.

Each sample is a random-axis-stretched primitive (sphere / cylinder / capsule
/ box) plus a random number of small "limb" bumps welded on. Output is
trimesh.Trimesh in unit-cube-ish bounds; the robotization layer rescales to
fit the MPM body box.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from .base import MorphologyPrior, PriorMetadata, MissingDependencyError

try:
    import trimesh
    _TRIMESH_OK = True
    _TRIMESH_VER = getattr(trimesh, "__version__", "unknown")
except ImportError:
    trimesh = None
    _TRIMESH_OK = False
    _TRIMESH_VER = "missing"


_SHAPE_KINDS = ("sphere", "cylinder", "capsule", "box")


class RandomShapesPrior(MorphologyPrior):
    """Procedural prior — no external deps beyond trimesh."""

    metadata = PriorMetadata(
        name="random_shapes",
        version=f"trimesh-{_TRIMESH_VER}",
        output_format="mesh",
    )

    def __init__(
        self,
        *,
        scale: float = 1.0,
        max_aspect: float = 3.0,
        n_limbs_range: tuple = (0, 4),
        limb_size_range: tuple = (0.05, 0.15),
    ):
        if not _TRIMESH_OK:
            raise MissingDependencyError(
                "trimesh is required for RandomShapesPrior",
                install_hint="pip install trimesh",
            )
        self.scale = float(scale)
        self.max_aspect = float(max_aspect)
        self.n_limbs_range = tuple(int(x) for x in n_limbs_range)
        self.limb_size_range = tuple(float(x) for x in limb_size_range)

    def sample(
        self,
        prompt: str,
        n: int = 1,
        *,
        seed: int = 0,
        **kwargs: Any,
    ) -> List[Any]:
        """Generate ``n`` procedural meshes. ``prompt`` is hashed into the seed
        so the same (prompt, seed, n) call is reproducible."""
        if kwargs:
            raise TypeError(
                f"RandomShapesPrior.sample got unexpected kwargs: {sorted(kwargs)}"
            )
        rng = np.random.default_rng(seed + (hash(prompt) & 0xFFFF))
        return [self._one(rng) for _ in range(int(n))]

    # ------------------------------------------------------------------

    def _one(self, rng: np.random.Generator):
        kind = rng.choice(_SHAPE_KINDS)
        # Aspect ratios per axis, biased to favor "long" bodies along x for
        # locomotion-friendly priors.
        ax = rng.uniform(1.0, self.max_aspect)
        ay = rng.uniform(0.5, 1.2)
        az = rng.uniform(0.5, 1.2)
        s = self.scale

        if kind == "sphere":
            mesh = trimesh.creation.icosphere(radius=0.5 * s, subdivisions=2)
        elif kind == "cylinder":
            mesh = trimesh.creation.cylinder(radius=0.5 * s, height=1.0 * s, sections=24)
        elif kind == "capsule":
            mesh = trimesh.creation.capsule(radius=0.4 * s, height=1.0 * s, count=(12, 12))
        else:  # "box"
            mesh = trimesh.creation.box(extents=(1.0 * s, 1.0 * s, 1.0 * s))

        # Per-axis stretch.
        mesh.apply_scale([ax, ay, az])

        # Optional limb bumps.
        n_limbs = int(rng.integers(self.n_limbs_range[0], self.n_limbs_range[1] + 1))
        if n_limbs > 0:
            base_extents = mesh.extents
            limbs = []
            for _ in range(n_limbs):
                r = rng.uniform(*self.limb_size_range) * s
                bump = trimesh.creation.icosphere(radius=r, subdivisions=1)
                # Place on the surface at a random direction.
                phi = rng.uniform(0, 2 * np.pi)
                theta = rng.uniform(0.2, np.pi - 0.2)
                radius_eff = 0.5 * float(np.linalg.norm(base_extents))
                pos = radius_eff * np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ], dtype=np.float64)
                bump.apply_translation(pos)
                limbs.append(bump)
            # Boolean union requires the optional `manifold3d` dep; fall back
            # to simple concatenation (which leaves overlap but the
            # robotization layer voxelizes the union of cells anyway).
            mesh = trimesh.util.concatenate([mesh, *limbs])

        # Center on origin so robotize.normalize is straightforward.
        mesh.apply_translation(-mesh.centroid)
        # Trimesh sometimes leaves degenerate faces from boolean unions; clean.
        mesh.process(validate=True)
        return mesh


# Register on import.
def _factory(**kwargs):
    return RandomShapesPrior(**kwargs)


from . import register as _register  # local import to avoid bootstrap cycle
_register("random_shapes", _factory)
