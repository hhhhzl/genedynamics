"""
genemetry -- Constraint geometry layer for generative dynamics on manifolds.

Provides backend-agnostic abstractions for:

- **Constraint manifolds**: local geometry from SDF / obstacle data.
- **Retraction operators**: project trajectories onto the feasible set.
- **Gate policies**: decide when to activate geometric corrections.
- **Task modulators**: adapt geometry to task context (stepping, etc.).
- **Geometry ops**: low-level primitives (active-row extraction,
  complement projection).
- **Window policies**: horizon-local windowing schemes.
- **Probe pipelines**: mini-batch probe sampling + retraction.
- **Constrained steps**: AGP and related constrained gradient steps.
- **Refinement pipelines**: post-hoc window-local trajectory refinement.
- **Schedule overlays**: hardness / constraint / diffusion parameter
  modulation based on constraint penalty state.

Quick start
-----------
::

    from genedynamics.genemetry import (
        SdfManifold, SteppingRetraction, CfsRetraction,
        MultimodalGate, ProbeModulator,
        SlidingWindow, ProbePipeline, AgpStep,
        LocalCfsRetraction, WindowRefinement,
        ScheduleOverlay, OverlayConfig, resolve_overlay_config,
    )

    manifold  = SdfManifold(backend="jax")
    gate      = MultimodalGate(backend="jax", multi_scale=0.25)
    retract   = SteppingRetraction(backend="jax", stone_centers=..., ...)
    window    = SlidingWindow(window_size=16, window_stride=8)
    probe     = ProbePipeline(backend="jax", tail_mix=0.5)
    agp       = AgpStep(backend="numpy", active_topk=8)
    refine    = WindowRefinement(backend="numpy", window_policy=window, ...)
    overlay   = ScheduleOverlay(config=resolve_overlay_config(cfg), backend="jax")

Backend implementations are auto-registered when the sub-packages are
imported (via ``try``/``except`` so missing backends are silently
skipped).
"""

# -- Core types and ABCs -----------------------------------------------
from genedynamics.genemetry.types import (
    GeometryBundle,
    GateDecision,
    RetractionResult,
    ProbeResult,
    RefinementResult,
    StepResult,
    OverlayParams,
)
from genedynamics.genemetry.base import (
    ConstraintManifold,
    RetractionOperator,
    GatePolicy,
    TaskModulator,
    WindowPolicy,
    ProbeSampler,
    ConstrainedStep,
    RefinementPipeline,
    ScheduleOverlayBase,
)
from genedynamics.genemetry.registry import (
    get_genemetry_registry,
    register_genemetry,
    GenemetryRegistry,
)

# -- Orchestrators (public API) ----------------------------------------
# Single-step orchestrators
from genedynamics.genemetry.manifold.sdf import SdfManifold
from genedynamics.genemetry.modulation.null import NullModulator
from genedynamics.genemetry.modulation.probe import ProbeModulator
from genedynamics.genemetry.retraction.cfs import CfsRetraction
from genedynamics.genemetry.retraction.stepping import SteppingRetraction
from genedynamics.genemetry.gate.multimodal import MultimodalGate
# Windowed refinement
from genedynamics.genemetry.window.sliding import SlidingWindow
from genedynamics.genemetry.pipeline.probe import ProbePipeline
from genedynamics.genemetry.pipeline.refine import WindowRefinement
from genedynamics.genemetry.step.agp import AgpStep
from genedynamics.genemetry.retraction.local_cfs import LocalCfsRetraction
# Schedule overlay
from genedynamics.genemetry.schedule.overlay import ScheduleOverlay
from genedynamics.genemetry.schedule.config import (
    OverlayConfig,
    resolve_overlay_config,
)

# -- Auto-register available backends ----------------------------------
# Each sub-package __init__ handles its own backend imports; importing
# the sub-packages here triggers registration.
import genedynamics.genemetry.ops  
import genedynamics.genemetry.manifold  
import genedynamics.genemetry.retraction  
import genedynamics.genemetry.gate  
import genedynamics.genemetry.window  
import genedynamics.genemetry.pipeline  
import genedynamics.genemetry.step  
import genedynamics.genemetry.schedule  

__all__ = [
    # Types
    "GeometryBundle",
    "GateDecision",
    "RetractionResult",
    "ProbeResult",
    "RefinementResult",
    "StepResult",
    "OverlayParams",
    # ABCs
    "ConstraintManifold",
    "RetractionOperator",
    "GatePolicy",
    "TaskModulator",
    "WindowPolicy",
    "ProbeSampler",
    "ConstrainedStep",
    "RefinementPipeline",
    "ScheduleOverlayBase",
    # Registry
    "get_genemetry_registry",
    "register_genemetry",
    "GenemetryRegistry",
    # Orchestrators — single-step
    "SdfManifold",
    "NullModulator",
    "ProbeModulator",
    "CfsRetraction",
    "SteppingRetraction",
    "MultimodalGate",
    # Orchestrators — windowed refinement
    "SlidingWindow",
    "ProbePipeline",
    "WindowRefinement",
    "AgpStep",
    "LocalCfsRetraction",
    # Orchestrators — schedule overlay
    "ScheduleOverlay",
    "OverlayConfig",
    "resolve_overlay_config",
]
