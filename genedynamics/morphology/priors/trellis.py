"""TRELLIS adapter — Table 3 prior (structured-latent 3D).

Lazy wrapper; get_prior("trellis") raises MissingDependencyError until upstream
is installed. TRELLIS exposes text- and image-conditioned pipelines; the base
adapter picks the first class with .from_pretrained().
"""

from __future__ import annotations

from ._lazy_mesh import register_lazy_mesh_prior

register_lazy_mesh_prior(
    name="trellis",
    modules=("trellis", "TRELLIS", "trellis.pipelines"),
    classes=("TrellisTextTo3DPipeline", "TrellisImageTo3DPipeline", "Pipeline"),
    default_ckpt="microsoft/TRELLIS-text-xlarge",
    install_hint=(
        "cd third_party/morphology_priors/trellis && "
        "git clone https://github.com/microsoft/TRELLIS.git . && "
        "pip install -r requirements.txt"
    ),
)
