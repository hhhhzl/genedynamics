"""CraftsMan3D adapter — Table 3 prior (native-3D diffusion mesh).

Lazy wrapper; get_prior("craftsman") raises MissingDependencyError until
upstream is installed.
"""

from __future__ import annotations

from ._lazy_mesh import register_lazy_mesh_prior

register_lazy_mesh_prior(
    name="craftsman",
    modules=("craftsman", "CraftsMan", "craftsman3d"),
    classes=("CraftsManPipeline", "Pipeline"),
    default_ckpt="craftsman3d/craftsman",
    install_hint=(
        "cd third_party/morphology_priors/craftsman && "
        "git clone https://github.com/wyysf-98/CraftsMan3D.git . && "
        "pip install -r requirements.txt"
    ),
)
