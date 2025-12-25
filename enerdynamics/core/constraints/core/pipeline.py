"""
High-performance constraint pipeline with JIT compilation and batching.

This module provides HighPerformanceConstraintPipeline, which orchestrates:
1. Constraint convexification (terms -> convex constraints)
2. Constraint enforcement (operators: QP, projection, etc.)
3. Scheduling (parameter annealing)

Performance optimizations:
- JIT compilation: Automatic JIT for JAX backends
- Batch processing: Vectorized operations via vmap
- Lazy evaluation: Only compute when needed
- Backend optimization: Auto-select best backend

Architecture:
    Pipeline = Scheduler -> Convexifier -> Operator
    Input: nominal trajectory, reference trajectory, schedule state
    Output: repaired trajectory, info dict
"""

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

import numpy as np

# Optional JAX imports
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

from enerdynamics.core.types import Trajectory, State, Action
from .array_interface import BackendArray, BackendType, ensure_backend
from .registry import get_registry, UnifiedRegistry
from .types import ScheduleState, ScheduleParams, ConvexConstraint, OperatorInfo
from .cache import ConstraintCache, ParamCache, CacheKey


@dataclass
class PipelineConfig:
    """
    Configuration for high-performance constraint pipeline.
    
    Attributes:
        backend: Preferred backend ("jax", "numpy", "torch", "rust")
        use_jit: Enable JIT compilation (JAX only)
        use_batch: Enable batch processing
        batch_size: Batch size (None = auto-detect)
        cache_constraints: Cache convexified constraints
        cache_params: Cache scheduler parameters
        verbose: Print performance info
    """
    backend: BackendType = "jax"
    use_jit: bool = True
    use_batch: bool = True
    batch_size: Optional[int] = None
    cache_constraints: bool = True
    cache_params: bool = True
    verbose: bool = False


class HighPerformanceConstraintPipeline:
    """
    High-performance constraint pipeline with JIT compilation and batching.
    
    This pipeline orchestrates the constraint system:
    1. Scheduler: Generates schedule parameters
    2. Convexifier: Builds convex constraints from terms
    3. Operator: Enforces constraints (QP, projection, etc.)
    
    Performance features:
    - JIT compilation: Automatic JIT for JAX backends
    - Batch processing: Vectorized operations via vmap
    - Smart caching: Cache constraints and parameters
    - Auto backend selection: Choose best available backend
    
    Example:
        >>> pipeline = HighPerformanceConstraintPipeline(
        ...     convexifier_name="cfs",
        ...     operator_name="per_step_qp",
        ...     scheduler_name="cosine_anneal",
        ...     config=PipelineConfig(backend="jax", use_jit=True)
        ... )
        >>> repaired, info = pipeline.apply(nominal, ref, state)
    """
    
    def __init__(
        self,
        convexifier_name: str,
        operator_name: str,
        scheduler_name: str,
        config: PipelineConfig,
        registry: Optional[UnifiedRegistry] = None,
        **kwargs
    ):
        """
        Initialize high-performance constraint pipeline.
        
        Args:
            convexifier_name: Convexifier name (e.g., "cfs", "cbf")
            operator_name: Operator name (e.g., "per_step_qp", "projection")
            scheduler_name: Scheduler name (e.g., "cosine_anneal")
            config: Pipeline configuration
            registry: Registry instance (uses global if None)
            **kwargs: Additional arguments passed to components
        """
        self.config = config
        self.registry = registry or get_registry()
        self.kwargs = kwargs
        
        # Auto-select backend if not specified
        if config.backend is None:
            config.backend = self.registry.auto_select_backend(
                "convexifier", convexifier_name
            ) or "numpy"
        
        # Get implementations
        self.convexifier = self._get_convexifier(convexifier_name, config.backend)
        self.operator = self._get_operator(operator_name, config.backend)
        self.scheduler = self._get_scheduler(scheduler_name)
        
        # JIT compilation (if enabled and JAX available)
        self._jit_compiled = False
        if config.use_jit and config.backend == "jax" and JAX_AVAILABLE:
            self._jit_compile()
        
        # Caching (use proper cache classes)
        if config.cache_constraints:
            self._constraint_cache = ConstraintCache(size=1000)
        else:
            self._constraint_cache = None
        
        if config.cache_params:
            self._param_cache = ParamCache()
        else:
            self._param_cache = None
        
        if config.verbose:
            print(f"[Pipeline] Initialized with backend={config.backend}, "
                  f"JIT={self._jit_compiled}, batch={config.use_batch}")
    
    def _get_convexifier(self, name: str, backend: str):
        """Get convexifier implementation."""
        impl_class = self.registry.get("convexifier", name, backend)
        if impl_class is None:
            raise ValueError(
                f"Convexifier '{name}' with backend '{backend}' not found. "
                f"Available backends: {self.registry.list_backends('convexifier', name)}"
            )
        return impl_class(**self.kwargs)
    
    def _get_operator(self, name: str, backend: str):
        """Get operator implementation."""
        impl_class = self.registry.get("operator", name, backend)
        if impl_class is None:
            raise ValueError(
                f"Operator '{name}' with backend '{backend}' not found. "
                f"Available backends: {self.registry.list_backends('operator', name)}"
            )
        return impl_class(**self.kwargs)
    
    def _get_scheduler(self, name: str):
        """Get scheduler implementation."""
        # Try registry first
        impl_class = self.registry.get("scheduler", name, "numpy")
        if impl_class is not None:
            return impl_class(**self.kwargs)
        
        # Fallback: try to import from schedulers module
        try:
            if name == "cosine_anneal":
                from enerdynamics.core.constraints.schedulers import CosineAnnealScheduler
                return CosineAnnealScheduler(**self.kwargs)
            else:
                # Try generic import
                from enerdynamics.core.constraints.schedulers import get_scheduler
                return get_scheduler(name, **self.kwargs)
        except:
            raise ValueError(
                f"Scheduler '{name}' not found. "
                f"Available schedulers: {self.registry.list_components('scheduler')}"
            )
    
    def _jit_compile(self):
        """JIT compile key functions (JAX only)."""
        if not JAX_AVAILABLE or self.config.backend != "jax":
            return
        
        try:
            # JIT compile convexifier if it has build_constraints method
            if hasattr(self.convexifier, 'build_constraints'):
                original_build = self.convexifier.build_constraints
                
                # Check if it's already JIT compiled
                if not hasattr(original_build, '_jax_jit'):
                    @jax.jit
                    def jit_build(ref, params, state):
                        return original_build(ref, params, state)
                    
                    self.convexifier.build_constraints = jit_build
                    if self.config.verbose:
                        print("[Pipeline] JIT compiled convexifier.build_constraints")
            
            # JIT compile operator if it has apply method
            if hasattr(self.operator, 'apply'):
                original_apply = self.operator.apply
                
                # Check if it's already JIT compiled
                if not hasattr(original_apply, '_jax_jit'):
                    @jax.jit
                    def jit_apply(nominal, constraints, params, state):
                        return original_apply(nominal, constraints, params, state)
                    
                    self.operator.apply = jit_apply
                    if self.config.verbose:
                        print("[Pipeline] JIT compiled operator.apply")
            
            # Note: _pipeline_step_jax will be defined if needed
            # JIT compilation of full pipeline step is done lazily
            
            self._jit_compiled = True
            if self.config.verbose:
                print("[Pipeline] JIT compilation complete")
        except Exception as e:
            if self.config.verbose:
                print(f"[Pipeline] JIT compilation failed: {e}")
            self._jit_compiled = False
    
    def _pipeline_step_jax(
        self,
        nominal_states: jnp.ndarray,  # (H+1, state_dim)
        nominal_actions: jnp.ndarray,  # (H, action_dim)
        ref_states: jnp.ndarray,  # (H+1, state_dim)
        params: ScheduleParams,
        state: ScheduleState
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        JIT-compiled pipeline step (JAX arrays).
        
        This is the core function that gets JIT compiled for maximum performance.
        Works directly with JAX arrays to avoid conversion overhead.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available")
        
        # Convert JAX arrays to Trajectory objects for convexifier/operator
        # (This is necessary because current interfaces expect Trajectory)
        # TODO: Future optimization - make convexifier/operator accept JAX arrays directly
        
        # Create temporary Trajectory from ref_states
        ref_traj = Trajectory(
            states=[State(ref_states[i]) for i in range(ref_states.shape[0])],
            actions=[],  # Not needed for convexification
            info={}
        )
        
        # Build constraints (JIT compiled)
        constraints = self.convexifier.build_constraints(
            ref_traj, params, state
        )
        
        # Create temporary Trajectory from nominal states/actions
        nominal_traj = Trajectory(
            states=[State(nominal_states[i]) for i in range(nominal_states.shape[0])],
            actions=[Action(nominal_actions[i]) for i in range(nominal_actions.shape[0])],
            info={}
        )
        
        # Apply operator (JIT compiled)
        repaired_traj, _ = self.operator.apply(
            nominal_traj, constraints, params, state
        )
        
        # Convert repaired actions back to JAX array
        repaired_actions = jnp.stack([jnp.asarray(a.data) for a in repaired_traj.actions])
        
        # Return repaired actions and constraint matrix A
        A_jax = jnp.asarray(constraints.A) if hasattr(constraints.A, '__array__') else constraints.A
        return repaired_actions, A_jax
    
    def apply(
        self,
        nominal: Trajectory,
        ref: Trajectory,
        state: ScheduleState
    ) -> Tuple[Trajectory, Dict[str, Any]]:
        """
        Apply constraint pipeline to single trajectory.
        
        Args:
            nominal: Nominal trajectory to repair
            ref: Reference trajectory for convexification
            state: Schedule state (diffusion step info)
            
        Returns:
            Tuple of (repaired trajectory, info dict)
        """
        if self.config.use_batch:
            # Wrap in batch for consistency
            result, info = self.apply_batch([nominal], [ref], state)
            return result[0], info
        
        return self._apply_single(nominal, ref, state)
    
    def apply_batch(
        self,
        nominals: List[Trajectory],
        refs: List[Trajectory],
        state: ScheduleState
    ) -> Tuple[List[Trajectory], Dict[str, Any]]:
        """
        Apply constraint pipeline to batch of trajectories.
        
        This is the high-performance path that uses vectorization.
        
        Args:
            nominals: List of nominal trajectories
            refs: List of reference trajectories
            state: Schedule state
            
        Returns:
            Tuple of (list of repaired trajectories, info dict)
        """
        if len(nominals) != len(refs):
            raise ValueError(f"Batch size mismatch: {len(nominals)} nominals vs {len(refs)} refs")
        
        if not self.config.use_batch or len(nominals) == 1:
            # Fallback to single trajectory processing
            results = [self._apply_single(n, r, state) for n, r in zip(nominals, refs)]
            repaired = [r[0] for r in results]
            info = self._merge_info([r[1] for r in results])
            return repaired, info
        
        # Batch processing
        if self.config.backend == "jax" and JAX_AVAILABLE and self._jit_compiled:
            return self._apply_batch_jax(nominals, refs, state)
        else:
            # Parallel processing for other backends
            return self._apply_batch_parallel(nominals, refs, state)
    
    def _apply_single(
        self,
        nominal: Trajectory,
        ref: Trajectory,
        state: ScheduleState
    ) -> Tuple[Trajectory, Dict[str, Any]]:
        """Apply pipeline to single trajectory."""
        # 1. Get schedule parameters
        params = self._get_params(state)
        
        # 2. Build convex constraints
        constraints = self._get_constraints(ref, params, state)
        
        # 3. Apply operator
        repaired, info = self.operator.apply(nominal, constraints, params, state)
        
        return repaired, info
    
    def _apply_batch_jax(
        self,
        nominals: List[Trajectory],
        refs: List[Trajectory],
        state: ScheduleState
    ) -> Tuple[List[Trajectory], Dict[str, Any]]:
        """
        Apply pipeline to batch using JAX vmap (high-performance path).
        
        This uses vmap to vectorize the entire pipeline across the batch,
        enabling GPU acceleration and maximum performance.
        """
        try:
            # Get schedule parameters (same for all in batch)
            params = self._get_params(state)
            
            # Convert trajectories to JAX arrays
            # Extract states and actions
            nominal_states_list = [np.stack([np.asarray(s) for s in n.states]) for n in nominals]
            nominal_actions_list = [np.stack([np.asarray(a) for a in n.actions]) for n in nominals]
            ref_states_list = [np.stack([np.asarray(s) for s in r.states]) for r in refs]
            
            # Pad to same length (if needed)
            max_H = max(len(n.states) for n in nominals)
            state_dim = nominal_states_list[0].shape[1] if nominal_states_list else 4
            action_dim = nominal_actions_list[0].shape[1] if nominal_actions_list else 2
            
            # Stack into batch arrays
            batch_size = len(nominals)
            nominal_states_batch = np.zeros((batch_size, max_H, state_dim), dtype=np.float32)
            nominal_actions_batch = np.zeros((batch_size, max_H - 1, action_dim), dtype=np.float32)
            ref_states_batch = np.zeros((batch_size, max_H, state_dim), dtype=np.float32)
            
            for i, (ns, na, rs) in enumerate(zip(nominal_states_list, nominal_actions_list, ref_states_list)):
                H = len(ns)
                nominal_states_batch[i, :H] = ns
                nominal_actions_batch[i, :H-1] = na
                ref_states_batch[i, :H] = rs
            
            # Convert to JAX
            nominal_states_jax = jnp.asarray(nominal_states_batch)
            nominal_actions_jax = jnp.asarray(nominal_actions_batch)
            ref_states_jax = jnp.asarray(ref_states_batch)
            
            # Use vmap for true vectorization
            # vmap over batch dimension (axis 0)
            # params and state are shared across batch (None in in_axes)
            if self.config.use_jit and self._jit_compiled:
                # JIT compile the vmap'd function
                batched_step = jax.jit(
                    jax.vmap(
                        self._pipeline_step_jax,
                        in_axes=(0, 0, 0, None, None)  # vmap over first 3 args, keep params/state constant
                    )
                )
            else:
                batched_step = jax.vmap(
                    self._pipeline_step_jax,
                    in_axes=(0, 0, 0, None, None)
                )
            
            # Apply batched pipeline step
            repaired_actions_batch_jax, _ = batched_step(
                nominal_states_jax,
                nominal_actions_jax,
                ref_states_jax,
                params,
                state
            )
            
            # Convert back to numpy
            repaired_actions_batch = np.asarray(repaired_actions_batch_jax)
            
            # Convert back to trajectories
            repaired_list = []
            info_list = []
            for i, nominal in enumerate(nominals):
                H = len(nominal.states)
                repaired_actions = np.asarray(repaired_actions_batch[i, :H-1])
                
                repaired = Trajectory(
                    states=nominal.states,
                    actions=[repaired_actions[t] for t in range(H-1)],
                    info=nominal.info
                )
                repaired_list.append(repaired)
                
                # Create dummy info (can be enhanced)
                from enerdynamics.core.constraints.core.types import OperatorInfo
                info_list.append(OperatorInfo(success=True, time=0.0))
            
            merged_info = self._merge_info(info_list)
            return repaired_list, merged_info
            
        except Exception as e:
            # Fallback to parallel processing
            if self.config.verbose:
                print(f"[Pipeline] JAX batch failed, falling back to parallel: {e}")
            return self._apply_batch_parallel(nominals, refs, state)
    
    def _apply_batch_parallel(
        self,
        nominals: List[Trajectory],
        refs: List[Trajectory],
        state: ScheduleState
    ) -> Tuple[List[Trajectory], Dict[str, Any]]:
        """Apply pipeline to batch using parallel processing."""
        # Simple parallel processing (can be upgraded with multiprocessing)
        results = [self._apply_single(n, r, state) for n, r in zip(nominals, refs)]
        repaired = [r[0] for r in results]
        info = self._merge_info([r[1] for r in results])
        return repaired, info
    
    def _get_params(self, state: ScheduleState) -> ScheduleParams:
        """Get schedule parameters (with caching)."""
        if self._param_cache is not None:
            cached = self._param_cache.get(state.k, state.K)
            if cached is not None:
                return cached
        
        params = self.scheduler.params(state)
        
        if self._param_cache is not None:
            self._param_cache.set(state.k, state.K, params)
        
        return params
    
    def _get_constraints(
        self,
        ref: Trajectory,
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """Get convex constraints (with caching)."""
        if self._constraint_cache is not None:
            # Use proper cache key
            cache_key = CacheKey.from_data(ref, params, state)
            cached = self._constraint_cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Build constraints
        constraints = self.convexifier.build_constraints(ref, params, state)
        
        if self._constraint_cache is not None:
            cache_key = CacheKey.from_data(ref, params, state)
            self._constraint_cache.set(cache_key, constraints)
        
        return constraints
    
    def _merge_info(self, info_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge info dictionaries from batch processing."""
        if not info_list:
            return {}
        
        merged = {}
        for key in info_list[0].keys():
            values = [info.get(key) for info in info_list]
            # Aggregate based on type
            if all(isinstance(v, (int, float)) for v in values if v is not None):
                merged[key] = {
                    "mean": np.mean([v for v in values if v is not None]),
                    "min": np.min([v for v in values if v is not None]),
                    "max": np.max([v for v in values if v is not None]),
                }
            else:
                merged[key] = values
        
        return merged
    
    def clear_cache(self):
        """Clear all caches."""
        if self._constraint_cache is not None:
            self._constraint_cache.clear()
        if self._param_cache is not None:
            self._param_cache.clear()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HighPerformanceConstraintPipeline("
            f"backend={self.config.backend}, "
            f"JIT={self._jit_compiled}, "
            f"batch={self.config.use_batch})"
        )

