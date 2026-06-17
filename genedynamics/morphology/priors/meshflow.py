"""MeshFlow adapter — Table 3 prior (CVPR'26: MeshVAE + flow-based DiT).

Artist-mesh generation; expected to have the highest robotization success of
the Table-3 priors (clean, watertight artist topology). Lazy wrapper;
get_prior("meshflow") raises MissingDependencyError until upstream is
installed. Module/class names are placeholders pending the upstream release.
"""

from __future__ import annotations

from ._lazy_mesh import register_lazy_mesh_prior

register_lazy_mesh_prior(
    name="meshflow",
    modules=("meshflow", "MeshFlow"),
    classes=("MeshFlowPipeline", "MeshFlowDiTPipeline", "Pipeline"),
    default_ckpt="meshflow/meshflow-base",
    install_hint=(
        "cd third_party/morphology_priors/meshflow && "
        "git clone <MeshFlow upstream repo> . && pip install -r requirements.txt"
    ),
)
