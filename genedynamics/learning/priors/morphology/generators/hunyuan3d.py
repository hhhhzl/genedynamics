"""Hunyuan3D-2 adapter — Table 3 prior (high-res mesh).

Lazy: importing always succeeds; get_prior("hunyuan3d") raises
MissingDependencyError until the upstream package is installed. Candidate
module/class names are best-effort and pinned at install time (upstream API
churns); the LazyMeshPrior base tries each in order.
"""

from __future__ import annotations

from ._lazy_mesh import register_lazy_mesh_prior

register_lazy_mesh_prior(
    name="hunyuan3d",
    modules=("hy3dgen", "hunyuan3d", "Hunyuan3D"),
    classes=("Hunyuan3DDiTFlowMatchingPipeline", "Hunyuan3DPipeline", "Pipeline"),
    default_ckpt="tencent/Hunyuan3D-2",
    install_hint=(
        "cd third_party/morphology_priors/hunyuan3d && "
        "git clone https://github.com/Tencent/Hunyuan3D-2.git . && "
        "pip install -r requirements.txt"
    ),
)
