"""
Performance benchmarks for constraint system.

This module provides benchmarks to measure and compare performance
across different configurations and backends.
"""

import pytest
import numpy as np
import time
from typing import List

from enerdynamics.core.types import Trajectory
from enerdynamics.core.constraints.core import (
    HighPerformanceConstraintPipeline,
    PipelineConfig,
    ScheduleState,
    benchmark_pipeline,
    compare_backends,
    PerformanceProfiler,
)


@pytest.mark.performance
class TestConstraintBenchmarks:
    """Performance benchmarks for constraint system."""
    
    def create_test_trajectories(self, num_trajectories: int, horizon: int = 10) -> List[Trajectory]:
        """Create test trajectories for benchmarking."""
        trajectories = []
        for _ in range(num_trajectories):
            states = [np.random.randn(4).astype(np.float32) for _ in range(horizon + 1)]
            actions = [np.random.randn(2).astype(np.float32) for _ in range(horizon)]
            trajectories.append(Trajectory(states=states, actions=actions))
        return trajectories
    
    def test_single_vs_batch(self):
        """Benchmark single vs batch processing."""
        # Create test data
        nominals = self.create_test_trajectories(10)
        refs = nominals
        state = ScheduleState(k=10, K=100)
        
        # Create pipeline
        pipeline = HighPerformanceConstraintPipeline(
            convexifier_name="cfs",
            operator_name="per_step_qp",
            scheduler_name="cosine_anneal",
            config=PipelineConfig(backend="numpy", use_batch=False),
            obstacles=None  # Mock obstacles
        )
        
        # Benchmark single processing
        profiler = PerformanceProfiler()
        for nominal, ref in zip(nominals, refs):
            with profiler.time("single"):
                _ = pipeline.apply(nominal, ref, state)
        
        single_stats = profiler.get_stats("single")
        
        # Benchmark batch processing
        pipeline_batch = HighPerformanceConstraintPipeline(
            convexifier_name="cfs",
            operator_name="per_step_qp",
            scheduler_name="cosine_anneal",
            config=PipelineConfig(backend="numpy", use_batch=True),
            obstacles=None
        )
        
        profiler.reset()
        with profiler.time("batch"):
            _ = pipeline_batch.apply_batch(nominals, refs, state)
        
        batch_stats = profiler.get_stats("batch")
        
        # Batch should be faster (or at least not much slower)
        total_single_time = single_stats["total"]
        batch_time = batch_stats["mean"]
        
        print(f"Single processing: {total_single_time:.4f}s total")
        print(f"Batch processing: {batch_time:.4f}s")
        print(f"Speedup: {total_single_time / batch_time:.2f}x")
    
    def test_cache_effectiveness(self):
        """Test that caching improves performance."""
        from enerdynamics.core.constraints.core import ConstraintCache, CacheKey, ConvexConstraint
        from enerdynamics.core.constraints.core.types import ScheduleParams
        
        cache = ConstraintCache(size=100)
        
        # Create test data
        ref = np.array([1, 2, 3], dtype=np.float32)
        params = ScheduleParams(margin=0.1)
        state = ScheduleState(k=10, K=100)
        
        A = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([0.0], dtype=np.float32)
        constraint = ConvexConstraint(A=A, b=b)
        
        key = CacheKey.from_data(ref, params, state)
        
        # First access (cache miss)
        start = time.perf_counter()
        for _ in range(100):
            cached = cache.get(key)
            if cached is None:
                cache.set(key, constraint)
        time_with_cache = time.perf_counter() - start
        
        # Without cache (simulated)
        start = time.perf_counter()
        for _ in range(100):
            # Simulate constraint building (expensive)
            _ = ConvexConstraint(A=A, b=b)
        time_without_cache = time.perf_counter() - start
        
        print(f"With cache: {time_with_cache:.4f}s")
        print(f"Without cache: {time_without_cache:.4f}s")
        print(f"Cache speedup: {time_without_cache / time_with_cache:.2f}x")
    
    def test_jit_compilation(self):
        """Test JIT compilation benefits (if JAX available)."""
        try:
            import jax
            JAX_AVAILABLE = True
        except ImportError:
            JAX_AVAILABLE = False
            pytest.skip("JAX not available")
        
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        # Create test data
        nominals = self.create_test_trajectories(5)
        refs = nominals
        state = ScheduleState(k=10, K=100)
        
        # Pipeline without JIT
        pipeline_no_jit = HighPerformanceConstraintPipeline(
            convexifier_name="cfs",
            operator_name="per_step_qp",
            scheduler_name="cosine_anneal",
            config=PipelineConfig(backend="jax", use_jit=False, use_batch=True),
            obstacles=None
        )
        
        # Warmup
        _ = pipeline_no_jit.apply_batch(nominals, refs, state)
        
        # Benchmark without JIT
        profiler = PerformanceProfiler()
        for _ in range(10):
            with profiler.time("no_jit"):
                _ = pipeline_no_jit.apply_batch(nominals, refs, state)
        
        no_jit_stats = profiler.get_stats("no_jit")
        
        # Pipeline with JIT
        pipeline_jit = HighPerformanceConstraintPipeline(
            convexifier_name="cfs",
            operator_name="per_step_qp",
            scheduler_name="cosine_anneal",
            config=PipelineConfig(backend="jax", use_jit=True, use_batch=True),
            obstacles=None
        )
        
        # Warmup (includes JIT compilation)
        _ = pipeline_jit.apply_batch(nominals, refs, state)
        
        # Benchmark with JIT
        profiler.reset()
        for _ in range(10):
            with profiler.time("jit"):
                _ = pipeline_jit.apply_batch(nominals, refs, state)
        
        jit_stats = profiler.get_stats("jit")
        
        print(f"Without JIT: {no_jit_stats['mean']:.4f}s")
        print(f"With JIT: {jit_stats['mean']:.4f}s")
        if jit_stats['mean'] > 0:
            print(f"JIT speedup: {no_jit_stats['mean'] / jit_stats['mean']:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])


