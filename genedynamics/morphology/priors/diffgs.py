"""DiffGS adapter — Table 3 prior (3D Gaussian Splatting diffusion).

DiffGS generates a 3D Gaussian splat set (centers μ, scales σ, opacities), not
a mesh. Two robotization routes:
  - native: feed the raw Gaussians to `morphology.robotize_gaussians`
    (gs_robotize) — the unified-robotization path added in Stage 6.
  - mesh:   convert the Gaussian density to a mesh (marching cubes) so it fits
    the mesh-returning MorphologyPrior contract used by the asset bank.

This adapter registers the mesh route (so DiffGS slots into the same Table-3
bank pipeline as the other priors); the upstream call yields Gaussians which
the pipeline wrapper is expected to surface as a mesh. Lazy:
get_prior("diffgs") raises MissingDependencyError until upstream is installed.
"""

from __future__ import annotations

from ._lazy_mesh import register_lazy_mesh_prior

register_lazy_mesh_prior(
    name="diffgs",
    modules=("diffgs", "DiffGS"),
    classes=("DiffGSPipeline", "Pipeline"),
    default_ckpt="diffgs/diffgs",
    install_hint=(
        "cd third_party/morphology_priors/diffgs && "
        "git clone https://github.com/StevenJ0X/DiffGS.git . && "
        "pip install -r requirements.txt    "
        "# raw Gaussians can instead feed morphology.robotize_gaussians"
    ),
)
