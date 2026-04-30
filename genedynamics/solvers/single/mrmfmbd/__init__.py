"""
MRMFMBD (soft-robot co-design (mode marginalization + fidelity ladder)) solver.

Multi-Resolution, Multi-Fidelity Model-Based Diffusion.
Posterior bridge for theta=(x, phi) co-design.
Mode marginalization. Multi-fidelity ladder.
"""

from .mrmfmbd import MRMFMBDSolver
from .types import FidelityLevel, ModeRegime, MRMFMBDResult
from .protocols import FidelitySimulator, ModeMarginalizer
from .theta_prior import ThetaParametrization, ThetaPrior, ThetaPriorConfig
from .backends import MRMFMBDPosteriorBackendJax, PosteriorBridgeConfig

# Mode + Fidelity systems
from .mode_system import (
    ModeSpec,
    ModeSystemConfig,
    ModeMarginalizer,
    default_mode_system_config,
    compute_marginal_log_likelihood,
    compute_responsibilities,
)
from .fidelity_system import (
    FidelityLevelSpec,
    FidelitySystemConfig,
    default_fidelity_system_config,
    BlockFidelityLadder,
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
    # Mode system
    "ModeSpec",
    "ModeSystemConfig",
    "ModeMarginalizer",
    "compute_marginal_log_likelihood",
    "compute_responsibilities",
    "default_mode_system_config",
    # Fidelity system
    "FidelityLevelSpec",
    "FidelitySystemConfig",
    "BlockFidelityLadder",
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
