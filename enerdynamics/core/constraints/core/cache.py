"""
Intelligent caching system for constraint pipeline.

This module provides caching utilities to avoid redundant computation:
- ConstraintCache: LRU cache for convexified constraints
- ParamCache: Cache for scheduler parameters (with precomputation)
- CacheKey: Efficient cache key generation

Performance optimizations:
- LRU eviction: Automatic eviction of least recently used items
- Hash-based keys: Fast O(1) lookup
- Precomputation: Pre-compute all parameters for known schedule
- Memory management: Configurable cache sizes
"""

from typing import Dict, Optional, Tuple, Any, Hashable
from collections import OrderedDict
import hashlib
import pickle

from .types import ScheduleState, ScheduleParams, ConvexConstraint


class CacheKey:
    """
    Efficient cache key for constraints.
    
    This generates hash-based keys from reference trajectories and parameters,
    enabling fast cache lookups while avoiding deep equality checks.
    
    Performance:
    - Fast hash computation (uses array hashing when possible)
    - Collision-resistant (uses multiple hash components)
    - Memory efficient (stores only hash, not full data)
    """
    
    def __init__(
        self,
        ref_hash: int,
        params_hash: int,
        state_hash: int,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize cache key.
        
        Args:
            ref_hash: Hash of reference trajectory
            params_hash: Hash of schedule parameters
            state_hash: Hash of schedule state
            extra: Additional hash components
        """
        self.ref_hash = ref_hash
        self.params_hash = params_hash
        self.state_hash = state_hash
        self.extra = extra or {}
    
    def __hash__(self) -> int:
        """Compute hash of cache key."""
        return hash((
            self.ref_hash,
            self.params_hash,
            self.state_hash,
            tuple(sorted(self.extra.items()))
        ))
    
    def __eq__(self, other) -> bool:
        """Check equality."""
        if not isinstance(other, CacheKey):
            return False
        return (
            self.ref_hash == other.ref_hash and
            self.params_hash == other.params_hash and
            self.state_hash == other.state_hash and
            self.extra == other.extra
        )
    
    @classmethod
    def from_data(
        cls,
        ref: Any,
        params: ScheduleParams,
        state: ScheduleState,
        extra: Optional[Dict[str, Any]] = None
    ) -> "CacheKey":
        """
        Create cache key from data.
        
        Args:
            ref: Reference trajectory (or any hashable data)
            params: Schedule parameters
            state: Schedule state
            extra: Additional data to include in key
            
        Returns:
            CacheKey instance
        """
        # Hash reference (try to use array hash if possible)
        ref_hash = cls._hash_data(ref)
        
        # Hash parameters
        params_hash = hash(params)
        
        # Hash state
        state_hash = hash(state)
        
        return cls(ref_hash, params_hash, state_hash, extra)
    
    @staticmethod
    def _hash_data(data: Any) -> int:
        """
        Hash data efficiently.
        
        Tries multiple strategies:
        1. If has __hash__ and is hashable: use it
        2. If has tobytes (numpy/JAX array): hash bytes
        3. If has id: use id
        4. Otherwise: pickle and hash
        """
        # Try direct hash
        try:
            if isinstance(data, (int, float, str, tuple)):
                return hash(data)
        except TypeError:
            pass
        
        # Try array tobytes
        if hasattr(data, 'tobytes'):
            try:
                return hash(data.tobytes())
            except:
                pass
        
        # Try id
        try:
            return id(data)
        except:
            pass
        
        # Fallback: pickle and hash
        try:
            pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            return int(hashlib.md5(pickled).hexdigest(), 16)
        except:
            # Last resort: use id
            return id(data)


class ConstraintCache:
    """
    LRU cache for convexified constraints.
    
    This cache stores the results of convexification (terms -> convex constraints)
    to avoid redundant computation when reference trajectories haven't changed.
    
    Performance:
    - O(1) lookup and insertion
    - Automatic eviction of least recently used items
    - Configurable size limit
    - Memory efficient (stores only constraints, not full trajectories)
    
    Example:
        >>> cache = ConstraintCache(size=1000)
        >>> key = CacheKey.from_data(ref, params, state)
        >>> if key in cache:
        ...     constraints = cache[key]
        ... else:
        ...     constraints = convexifier.build_constraints(ref, params, state)
        ...     cache[key] = constraints
    """
    
    def __init__(self, size: int = 1000):
        """
        Initialize constraint cache.
        
        Args:
            size: Maximum number of cached constraints
        """
        if size <= 0:
            raise ValueError(f"Cache size must be positive, got {size}")
        
        self.size = size
        self._cache: OrderedDict[CacheKey, ConvexConstraint] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: CacheKey) -> Optional[ConvexConstraint]:
        """
        Get cached constraint.
        
        Args:
            key: Cache key
            
        Returns:
            Cached constraint, or None if not found
        """
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        
        self._misses += 1
        return None
    
    def set(self, key: CacheKey, value: ConvexConstraint) -> None:
        """
        Set cached constraint.
        
        Args:
            key: Cache key
            value: Constraint to cache
        """
        if key in self._cache:
            # Update existing entry (move to end)
            self._cache.move_to_end(key)
        else:
            # Check size limit
            if len(self._cache) >= self.size:
                # Evict least recently used
                self._cache.popitem(last=False)
        
        self._cache[key] = value
    
    def clear(self) -> None:
        """Clear all cached constraints."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def __contains__(self, key: CacheKey) -> bool:
        """Check if key is in cache."""
        return key in self._cache
    
    def __len__(self) -> int:
        """Get number of cached items."""
        return len(self._cache)
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


class ParamCache:
    """
    Cache for scheduler parameters with precomputation support.
    
    This cache stores scheduler parameters to avoid redundant computation
    when the same schedule state is queried multiple times.
    
    Performance:
    - O(1) lookup
    - Precomputation: Can pre-compute all parameters for known schedule
    - Memory efficient: Only stores parameters, not full state
    
    Example:
        >>> cache = ParamCache()
        >>> cache.precompute(scheduler, total_steps=100)
        >>> params = cache.get(k=10, K=100)  # Fast lookup
    """
    
    def __init__(self):
        """Initialize parameter cache."""
        self._cache: Dict[Tuple[int, int], ScheduleParams] = {}
        self._precomputed = False
        self._total_steps: Optional[int] = None
    
    def get(self, k: int, K: int) -> Optional[ScheduleParams]:
        """
        Get cached parameters.
        
        Args:
            k: Current diffusion step
            K: Total diffusion steps
            
        Returns:
            Cached parameters, or None if not found
        """
        return self._cache.get((k, K))
    
    def set(self, k: int, K: int, params: ScheduleParams) -> None:
        """
        Set cached parameters.
        
        Args:
            k: Current diffusion step
            K: Total diffusion steps
            params: Parameters to cache
        """
        self._cache[(k, K)] = params
    
    def precompute(
        self,
        scheduler: Any,
        total_steps: int,
        start_step: int = 0
    ) -> None:
        """
        Precompute all parameters for a schedule.
        
        This is useful when the total number of steps is known in advance,
        allowing all parameters to be computed once and cached.
        
        Args:
            scheduler: Scheduler instance (must have params(state) method)
            total_steps: Total number of diffusion steps
            start_step: Starting step (default: 0)
        """
        if not hasattr(scheduler, 'params'):
            raise ValueError("Scheduler must have params(state) method")
        
        self._total_steps = total_steps
        self._precomputed = True
        
        # Precompute all steps
        for k in range(start_step, total_steps + 1):
            state = ScheduleState(k=k, K=total_steps)
            params = scheduler.params(state)
            self.set(k, total_steps, params)
    
    def clear(self) -> None:
        """Clear all cached parameters."""
        self._cache.clear()
        self._precomputed = False
        self._total_steps = None
    
    def __len__(self) -> int:
        """Get number of cached parameters."""
        return len(self._cache)
    
    @property
    def is_precomputed(self) -> bool:
        """Check if parameters are precomputed."""
        return self._precomputed
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "precomputed": self._precomputed,
            "total_steps": self._total_steps,
        }

