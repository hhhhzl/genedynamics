"""
Performance profiling and optimization utilities.

This module provides tools for:
- Performance profiling
- Benchmarking
- Optimization suggestions
- Memory profiling
"""

import time
from typing import Dict, Any, List, Optional, Callable
from contextlib import contextmanager
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class PerformanceProfiler:
    """
    Performance profiler for constraint pipeline.
    
    Tracks execution times, memory usage, and performance metrics.
    """
    
    def __init__(self):
        """Initialize profiler."""
        self.timings: Dict[str, List[float]] = {}
        self.counts: Dict[str, int] = {}
        self.memory_usage: List[float] = []
    
    @contextmanager
    def time(self, name: str):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)
            self.counts[name] = self.counts.get(name, 0) + 1
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a timed operation.
        
        Args:
            name: Operation name
            
        Returns:
            Dictionary with mean, min, max, total, count
        """
        if name not in self.timings:
            return {}
        
        times = self.timings[name]
        return {
            "mean": float(np.mean(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "total": float(np.sum(times)),
            "count": len(times),
            "std": float(np.std(times))
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all timed operations."""
        return {name: self.get_stats(name) for name in self.timings.keys()}
    
    def reset(self):
        """Reset all profiling data."""
        self.timings.clear()
        self.counts.clear()
        self.memory_usage.clear()
    
    def summary(self) -> str:
        """Get summary string of profiling results."""
        lines = ["Performance Profiling Summary:"]
        lines.append("=" * 50)
        
        for name in sorted(self.timings.keys()):
            stats = self.get_stats(name)
            lines.append(
                f"{name}: "
                f"mean={stats['mean']:.4f}s, "
                f"min={stats['min']:.4f}s, "
                f"max={stats['max']:.4f}s, "
                f"count={stats['count']}"
            )
        
        return "\n".join(lines)


def benchmark_pipeline(
    pipeline: Any,
    nominals: List[Any],
    refs: List[Any],
    state: Any,
    num_runs: int = 10,
    warmup_runs: int = 2
) -> Dict[str, Any]:
    """
    Benchmark pipeline performance.
    
    Args:
        pipeline: Pipeline instance
        nominals: List of nominal trajectories
        refs: List of reference trajectories
        state: Schedule state
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs (for JIT compilation)
        
    Returns:
        Dictionary with benchmark results
    """
    profiler = PerformanceProfiler()
    
    # Warmup runs (for JIT compilation)
    for _ in range(warmup_runs):
        with profiler.time("warmup"):
            _ = pipeline.apply_batch(nominals, refs, state)
    
    # Benchmark runs
    for _ in range(num_runs):
        with profiler.time("benchmark"):
            _ = pipeline.apply_batch(nominals, refs, state)
    
    # Get statistics
    stats = profiler.get_all_stats()
    
    return {
        "stats": stats,
        "batch_size": len(nominals),
        "num_runs": num_runs,
        "warmup_runs": warmup_runs,
        "summary": profiler.summary()
    }


def compare_backends(
    pipeline_factory: Callable,
    nominals: List[Any],
    refs: List[Any],
    state: Any,
    backends: List[str] = ["numpy", "jax"],
    num_runs: int = 10
) -> Dict[str, Any]:
    """
    Compare performance across different backends.
    
    Args:
        pipeline_factory: Function that creates pipeline for given backend
        nominals: List of nominal trajectories
        refs: List of reference trajectories
        state: Schedule state
        backends: List of backends to compare
        num_runs: Number of runs per backend
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for backend in backends:
        try:
            pipeline = pipeline_factory(backend)
            benchmark = benchmark_pipeline(pipeline, nominals, refs, state, num_runs)
            results[backend] = benchmark["stats"]["benchmark"]
        except Exception as e:
            results[backend] = {"error": str(e)}
    
    # Compute speedup
    if "numpy" in results and "jax" in results:
        numpy_time = results["numpy"].get("mean", 0)
        jax_time = results["jax"].get("mean", 0)
        if jax_time > 0:
            speedup = numpy_time / jax_time
            results["speedup"] = speedup
    
    return results


def profile_memory(operation: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Profile memory usage of an operation.
    
    Args:
        operation: Function to profile
        *args: Arguments for operation
        **kwargs: Keyword arguments for operation
        
    Returns:
        Dictionary with memory statistics
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run operation
        result = operation(*args, **kwargs)
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "memory_before_mb": mem_before,
            "memory_after_mb": mem_after,
            "memory_delta_mb": mem_after - mem_before,
            "result": result
        }
    except ImportError:
        return {"error": "psutil not available"}


