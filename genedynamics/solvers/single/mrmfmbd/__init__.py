"""
MRMFMBD (Soft-robot S1+S3) solver.

Multi-Resolution, Multi-Fidelity Model-Based Diffusion.
Posterior bridge for theta=(x, phi) co-design.
S1: Mode marginalization. S3: Multi-fidelity ladder.
"""

from .mrmfmbd import MRMFMBDSolver
from .types import FidelityLevel, ModeRegime, MRMFMBDResult
from .protocols import FidelitySimulator, ModeMarginalizer
from .theta_prior import ThetaParametrization, ThetaPrior, ThetaPriorConfig
from .backends import MRMFMBDPosteriorBackendJax, PosteriorBridgeConfig

# S1/S3 systems
from .s1_mode_system import (
    ModeSpec,
    ModeSystemConfig,
    ModeMarginalizerS1,
    default_mode_system_config,
    compute_marginal_log_likelihood,
    compute_responsibilities,
)
from .s3_fidelity_system import (
    FidelityLevelSpec,
    FidelitySystemConfig,
    default_fidelity_system_config,
    FidelityLadderS3,
    FidelityLadderType,
    create_fidelity_ladder,
    UpgradeRule,
    ScoreGapUpgradeRule,
    ESSUpgradeRule,
    CompositeUpgradeRule,
)

try:
    from .tasks import get_task_spec, list_task_ids, list_domains
except ImportError:
    get_task_spec = None  # type: ignore
    list_task_ids = None  # type: ignore
    list_domains = None  # type: ignore

__all__ = [
    "MRMFMBDSolver",
    "FidelityLevel",
    "ModeRegime",
    "MRMFMBDResult",
    "FidelitySimulator",
    "ModeMarginalizer",
    "ThetaParametrization",
    "ThetaPrior",
    "ThetaPriorConfig",
    "MRMFMBDPosteriorBackendJax",
    "PosteriorBridgeConfig",
    # S1
    "ModeSpec",
    "ModeSystemConfig",
    "ModeMarginalizerS1",
    "compute_marginal_log_likelihood",
    "compute_responsibilities",
    "default_mode_system_config",
    # S3
    "FidelityLevelSpec",
    "FidelitySystemConfig",
    "FidelityLadderS3",
    "FidelityLadderType",
    "create_fidelity_ladder",
    "UpgradeRule",
    "ScoreGapUpgradeRule",
    "ESSUpgradeRule",
    "CompositeUpgradeRule",
    "default_fidelity_system_config",
    "get_task_spec",
    "list_task_ids",
    "list_domains",
]
