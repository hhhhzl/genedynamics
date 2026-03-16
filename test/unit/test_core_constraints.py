"""
Unit tests for core constraint system infrastructure.

Tests cover:
- BackendArray: Array interface and conversions
- UnifiedRegistry: Registry operations
- Types: Data structure validation
- Cache: Caching functionality
- Stats: Statistical extraction
- Pipeline: Basic pipeline operations
"""

import pytest
import numpy as np

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from genedynamics.core.constraints.core import (
    BackendArray,
    BackendType,
    ensure_backend,
    UnifiedRegistry,
    get_registry,
    register,
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
    ConstraintStats,
    OperatorInfo,
    CacheKey,
    ConstraintCache,
    ParamCache,
    compute_violation,
    compute_min_sdf,
    compute_feasible_rate,
    compute_ess,
    extract_stats,
    topK_selection,
    PipelineConfig,
)


class TestBackendArray:
    """Tests for BackendArray."""
    
    def test_numpy_creation(self):
        """Test creating BackendArray from numpy array."""
        arr = np.array([1, 2, 3], dtype=np.float32)
        backend_arr = BackendArray(arr, backend="numpy")
        
        assert backend_arr.backend == "numpy"
        assert np.array_equal(backend_arr.data, arr)
        assert np.array_equal(backend_arr.to_numpy(), arr)
    
    def test_numpy_to_jax(self):
        """Test converting numpy to JAX."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        arr = np.array([1, 2, 3], dtype=np.float32)
        backend_arr = BackendArray(arr, backend="numpy")
        
        jax_arr = backend_arr.to_jax()
        assert isinstance(jax_arr, type(jnp.array([1])))
        assert np.allclose(np.asarray(jax_arr), arr)
    
    def test_conversion_caching(self):
        """Test that conversions are cached."""
        arr = np.array([1, 2, 3], dtype=np.float32)
        backend_arr = BackendArray(arr, backend="numpy")
        
        # First conversion
        jax_arr1 = backend_arr.to_jax() if JAX_AVAILABLE else None
        
        # Second conversion should return cached
        jax_arr2 = backend_arr.to_jax() if JAX_AVAILABLE else None
        
        if JAX_AVAILABLE:
            # Should be the same object (cached)
            assert jax_arr1 is jax_arr2
    
    def test_shape_and_dtype(self):
        """Test shape and dtype properties."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        backend_arr = BackendArray(arr, backend="numpy")
        
        assert backend_arr.shape() == (2, 2)
        assert backend_arr.dtype() == np.float32
    
    def test_ensure_backend(self):
        """Test ensure_backend convenience function."""
        arr = np.array([1, 2, 3], dtype=np.float32)
        
        # Already in numpy
        result = ensure_backend(arr, "numpy")
        assert result.backend == "numpy"
        
        # Convert to JAX
        if JAX_AVAILABLE:
            result = ensure_backend(arr, "jax")
            assert result.backend == "jax"


class TestUnifiedRegistry:
    """Tests for UnifiedRegistry."""
    
    def test_registry_creation(self):
        """Test creating registry."""
        registry = UnifiedRegistry()
        assert len(registry.list_module_types()) == 0
    
    def test_register_and_get(self):
        """Test registering and getting components."""
        registry = UnifiedRegistry()
        
        class TestImpl:
            pass
        
        registry.register("convexifier", "cfs", "numpy", TestImpl)
        
        impl = registry.get("convexifier", "cfs", "numpy")
        assert impl == TestImpl
    
    def test_list_backends(self):
        """Test listing available backends."""
        registry = UnifiedRegistry()
        
        class TestImpl:
            pass
        
        registry.register("convexifier", "cfs", "numpy", TestImpl)
        registry.register("convexifier", "cfs", "jax", TestImpl)
        
        backends = registry.list_backends("convexifier", "cfs")
        assert "numpy" in backends
        assert "jax" in backends
    
    def test_auto_select_backend(self):
        """Test automatic backend selection."""
        registry = UnifiedRegistry()
        
        class TestImpl:
            pass
        
        registry.register("convexifier", "cfs", "numpy", TestImpl)
        registry.register("convexifier", "cfs", "jax", TestImpl)
        
        # Should prefer JAX
        backend = registry.auto_select_backend("convexifier", "cfs")
        assert backend == "jax"
        
        # With preferred
        backend = registry.auto_select_backend("convexifier", "cfs", preferred="numpy")
        assert backend == "numpy"
    
    def test_decorator_registration(self):
        """Test decorator-based registration."""
        registry = UnifiedRegistry()
        
        @register("operator", "test", "numpy")
        class TestOperator:
            pass
        
        impl = get_registry().get("operator", "test", "numpy")
        assert impl == TestOperator


class TestTypes:
    """Tests for type definitions."""
    
    def test_schedule_state(self):
        """Test ScheduleState."""
        state = ScheduleState(k=10, K=100)
        
        assert state.k == 10
        assert state.K == 100
        assert state.progress == 0.1
    
    def test_schedule_state_validation(self):
        """Test ScheduleState validation."""
        with pytest.raises(ValueError):
            ScheduleState(k=-1, K=100)
        
        with pytest.raises(ValueError):
            ScheduleState(k=10, K=0)
        
        with pytest.raises(ValueError):
            ScheduleState(k=101, K=100)
    
    def test_schedule_params(self):
        """Test ScheduleParams."""
        params = ScheduleParams(margin=0.1, rho=2.0)
        
        assert params.margin == 0.1
        assert params.rho == 2.0
        assert params.get("margin") == 0.1
    
    def test_schedule_params_extra(self):
        """Test ScheduleParams extra parameters."""
        params = ScheduleParams()
        params.set("custom_param", 42)
        
        assert params.get("custom_param") == 42
    
    def test_convex_constraint(self):
        """Test ConvexConstraint."""
        A = np.array([[1, 0], [0, 1]], dtype=np.float32)
        b = np.array([0, 0], dtype=np.float32)
        
        constraint = ConvexConstraint(A=A, b=b, meta={"per_step": True})
        
        assert constraint.is_per_step()
        assert constraint.get_constraint_type() == "linear"
    
    def test_constraint_stats(self):
        """Test ConstraintStats."""
        stats = ConstraintStats(
            violation=0.5,
            min_sdf=0.1,
            feasible_rate=0.8
        )
        
        assert stats.violation == 0.5
        assert stats.min_sdf == 0.1
        assert stats.feasible_rate == 0.8
        
        # Test to_dict
        stats_dict = stats.to_dict()
        assert stats_dict["violation"] == 0.5
    
    def test_operator_info(self):
        """Test OperatorInfo."""
        info = OperatorInfo(
            success=True,
            violation_before=1.0,
            violation_after=0.5,
            iterations=10
        )
        
        assert info.success
        assert info.violation_before == 1.0
        assert info.violation_after == 0.5
        assert info.iterations == 10


class TestCache:
    """Tests for caching system."""
    
    def test_cache_key(self):
        """Test CacheKey."""
        ref = np.array([1, 2, 3])
        params = ScheduleParams(margin=0.1)
        state = ScheduleState(k=10, K=100)
        
        key = CacheKey.from_data(ref, params, state)
        
        assert isinstance(key, CacheKey)
        assert key == CacheKey.from_data(ref, params, state)
    
    def test_constraint_cache(self):
        """Test ConstraintCache."""
        cache = ConstraintCache(size=10)
        
        ref = np.array([1, 2, 3])
        params = ScheduleParams()
        state = ScheduleState(k=10, K=100)
        key = CacheKey.from_data(ref, params, state)
        
        A = np.array([[1, 0]], dtype=np.float32)
        b = np.array([0], dtype=np.float32)
        constraint = ConvexConstraint(A=A, b=b)
        
        # Set
        cache.set(key, constraint)
        
        # Get
        cached = cache.get(key)
        assert cached is not None
        assert np.array_equal(cached.A, A)
        
        # Stats
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hit_rate"] > 0
    
    def test_constraint_cache_lru(self):
        """Test LRU eviction in ConstraintCache."""
        cache = ConstraintCache(size=2)
        
        # Add 3 items (should evict first)
        for i in range(3):
            key = CacheKey.from_data(np.array([i]), ScheduleParams(), ScheduleState(k=i, K=10))
            constraint = ConvexConstraint(A=np.array([[1]]), b=np.array([0]))
            cache.set(key, constraint)
        
        assert len(cache) == 2
    
    def test_param_cache(self):
        """Test ParamCache."""
        cache = ParamCache()
        
        params = ScheduleParams(margin=0.1)
        cache.set(10, 100, params)
        
        cached = cache.get(10, 100)
        assert cached is not None
        assert cached.margin == 0.1


class TestStats:
    """Tests for statistical extraction."""
    
    def test_compute_violation(self):
        """Test violation computation."""
        violations = [0.0, 0.5, 1.0, 0.2]
        
        max_violation = compute_violation(violations, reduction="max")
        assert max_violation == 1.0
        
        mean_violation = compute_violation(violations, reduction="mean")
        assert mean_violation == 0.425
    
    def test_compute_min_sdf(self):
        """Test minimum SDF computation."""
        sdfs = [0.5, 0.1, 0.3, 0.2]
        
        min_sdf = compute_min_sdf(sdfs)
        assert min_sdf == 0.1
    
    def test_compute_feasible_rate(self):
        """Test feasible rate computation."""
        feasible = [True, True, False, True]
        
        rate = compute_feasible_rate(feasible)
        assert rate == 0.75
    
    def test_compute_ess(self):
        """Test ESS computation."""
        weights = [1.0, 2.0, 1.0, 2.0]
        
        ess = compute_ess(weights, normalized=False)
        assert ess > 0
        assert ess <= len(weights)
    
    def test_extract_stats(self):
        """Test comprehensive stats extraction."""
        violations = [0.0, 0.5, 1.0]
        sdfs = [0.5, 0.1, 0.3]
        feasible = [True, True, False]
        
        stats = extract_stats(
            violations=violations,
            sdfs=sdfs,
            feasible_flags=feasible
        )
        
        assert stats.violation == 1.0
        assert stats.min_sdf == 0.1
        assert stats.feasible_rate == pytest.approx(2/3)
    
    def test_topK_selection(self):
        """Test top-K selection."""
        values = [1.0, 5.0, 2.0, 4.0, 3.0]
        
        mask = topK_selection(values, K=3, largest=True)
        assert np.sum(mask) == 3
        assert mask[1] == True  # 5.0 should be selected
        assert mask[3] == True  # 4.0 should be selected


class TestPipeline:
    """Tests for pipeline (basic functionality)."""
    
    def test_pipeline_config(self):
        """Test PipelineConfig."""
        config = PipelineConfig(
            backend="jax",
            use_jit=True,
            use_batch=True
        )
        
        assert config.backend == "jax"
        assert config.use_jit
        assert config.use_batch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


