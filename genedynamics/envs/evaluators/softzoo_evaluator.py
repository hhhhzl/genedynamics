"""
SoftZoo rollout evaluator.

High-performance batch evaluation of design+controller pairs with
mode and fidelity support. Supports caching, optional parallelism,
and failure handling.
"""

from __future__ import annotations

import hashlib
import time
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .protocols import (
    RolloutBatchRequest,
    RolloutBatchResult,
    RolloutRequest,
    RolloutResult,
    RolloutEvaluator,
)


@dataclass
class SoftZooEvaluatorConfig:
    """
    Configuration for SoftZooRolloutEvaluator.

    Attributes:
        max_workers: Max parallel workers (0 = sequential)
        cache_size: LRU cache size (0 = disabled)
        max_retries: Retries per failed rollout
        timeout_per_rollout: Timeout in seconds (None = no limit)
        max_tasks_per_child: mp.Pool worker recycle count.
            None = auto (1 for CPU, unlimited for CUDA — Taichi 1.7.4 can
            re-init across rollouts on CUDA but silently segfaults on CPU
            when a 2nd env is created in the same process).
        ti_arch: Taichi backend (used to auto-pick max_tasks_per_child)
        reward_shaping_weight: Extra bonus added to per-rollout return:
            shaped = env_return + w * (final_com_x - init_com_x).
            0.0 disables. Defeats the per_step_velocity "wiggle-in-place"
            attractor by rewarding actual net forward displacement.
    """

    max_workers: int = 0
    cache_size: int = 0
    max_retries: int = 1
    timeout_per_rollout: Optional[float] = None
    raise_on_failure: bool = True
    max_tasks_per_child: Optional[int] = None
    ti_arch: Optional[str] = None
    reward_shaping_weight: float = 0.0
    reward_shaping_only: bool = False  # if True, ignore env_return; use only w·(final_com - init_com)
    extra: Dict[str, Any] = field(default_factory=dict)


def _theta_hash(x: np.ndarray, phi: np.ndarray) -> str:
    """Stable hash for (x, phi) for caching."""
    data = np.concatenate([np.asarray(x).ravel(), np.asarray(phi).ravel()])
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


# Persistent per-worker env cache (CUDA-only optimization).
# Key: (task_id, mode_id, fidelity_level, runtime_config_sig). On key match
# we skip the 5-7s softzoo/Taichi JIT build and just call env.reset(design)
# with the new morphology (~10 ms). On mismatch we close the old env +
# `ti.reset()` before building a new one (verified safe on CUDA by probe T5).
#
# Module-global because mp.Pool workers each import this module fresh on
# spawn; the cache lives for the worker's lifetime.
_WORKER_ENV_CACHE: Dict[Tuple, Any] = {}


def _runtime_config_sig(kw: Optional[Dict[str, Any]]) -> Tuple:
    if not kw:
        return ()
    return tuple(sorted(
        (k, v) for k, v in kw.items()
        if k in ("ti_arch", "device", "ti_device_memory_fraction")
    ))


def _run_single_rollout(
    task_id: str,
    x: np.ndarray,
    phi: np.ndarray,
    mode_id: int,
    fidelity_level: int,
    seed: int,
    num_repeats: int,
    record: bool,
    project_root: Optional[str],
    runtime_config_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[float, bool, int, float, Optional[str], Optional[Dict[str, np.ndarray]]]:
    """
    Run a single rollout in a subprocess (for true parallelism).

    Returns:
        (mean_return, success, num_steps, wall_time, failure_code, trajectory)
    """
    import sys
    import collections
    import collections.abc
    # Python 3.10+: attrdict imports Mapping etc from collections
    for _name in ("Mapping", "MutableMapping", "Sequence"):
        if not hasattr(collections, _name):
            setattr(collections, _name, getattr(collections.abc, _name))
    # macOS: ossaudiodev is Linux-only; stub for skvideo/moviepy deps
    if "ossaudiodev" not in sys.modules:
        import types

        class _OssStub(types.ModuleType):
            def __getattr__(self, name):
                return 0

        _stub = _OssStub("ossaudiodev")
        _stub.OSSAudioError = type("OSSAudioError", (Exception,), {})
        _stub.SNDCTL_COPR_SENDMSG = 0
        sys.modules["ossaudiodev"] = _stub

    import numpy as np

    from genedynamics.envs.external.softzoo.bootstrap import ensure_softzoo_on_path
    from genedynamics.envs.external.softzoo.adapters import (
        make_softzoo_env,
        encode_morphology,
        encode_controller,
    )
    from genedynamics.envs.external.softzoo.task_registry import get_task_spec
    from genedynamics.envs.external.softzoo.config import SoftZooRuntimeConfig

    ensure_softzoo_on_path(project_root)

    runtime_config = None
    if runtime_config_kwargs:
        from pathlib import Path
        kw = {k: v for k, v in runtime_config_kwargs.items() if k in ("ti_arch", "device", "ti_device_memory_fraction")}
        runtime_config = SoftZooRuntimeConfig(
            project_root=Path(project_root) if project_root else None,
            **kw,
        )

    task_spec = get_task_spec(task_id)
    modes = task_spec.modes
    fidelities = task_spec.fidelity_levels

    mode_spec = modes[mode_id] if modes and mode_id < len(modes) else None
    fid_spec = (
        fidelities[fidelity_level]
        if fidelities and fidelity_level < len(fidelities)
        else None
    )

    t0 = time.perf_counter()
    returns_list: List[float] = []
    success = False
    num_steps = 0
    trajectory: Optional[Dict[str, np.ndarray]] = None
    failure_code: Optional[str] = None

    shaping_w = 0.0
    shaping_only = False
    if runtime_config_kwargs:
        shaping_w = float(runtime_config_kwargs.get("reward_shaping_weight", 0.0) or 0.0)
        shaping_only = bool(runtime_config_kwargs.get("reward_shaping_only", False))
    fwd_dir = np.asarray(getattr(task_spec, "forward_direction", (1.0, 0.0, 0.0)), dtype=np.float32)

    def _com_x(env_obj) -> float:
        """Current COM projected on forward direction."""
        try:
            s = env_obj.sim.solver.current_s
            x_p = env_obj.design_space.get_x(s)
            mean_p = x_p.mean(0)
            if hasattr(mean_p, "numpy"):
                mean_p = np.asarray(mean_p.numpy(), dtype=np.float32)
            else:
                mean_p = np.asarray(mean_p, dtype=np.float32)
            return float(np.dot(mean_p[:3], fwd_dir))
        except Exception:
            return 0.0

    # Resolve env-reuse policy. Disabled on CPU (re-init segfaults in same
    # process); enabled on CUDA or when explicitly requested.
    arch = (runtime_config_kwargs or {}).get("ti_arch", "").lower() if runtime_config_kwargs else ""
    reuse_env = (arch == "cuda") and (runtime_config_kwargs or {}).get("env_reuse", True) is not False
    rc_sig = _runtime_config_sig(runtime_config_kwargs)
    cache_key = (task_id, int(mode_id), int(fidelity_level), rc_sig) if reuse_env else None

    env = None
    try:
        if reuse_env and cache_key in _WORKER_ENV_CACHE:
            env = _WORKER_ENV_CACHE[cache_key]
        else:
            # Tear down any previously cached env with a different key.
            if reuse_env and _WORKER_ENV_CACHE:
                for old_env in list(_WORKER_ENV_CACHE.values()):
                    try:
                        old_env.close()
                    except Exception:
                        pass
                _WORKER_ENV_CACHE.clear()
                try:
                    import taichi as ti
                    ti.reset()
                except Exception:
                    pass
            env = make_softzoo_env(
                task_spec=task_spec,
                fidelity_spec=fid_spec,
                mode_spec=mode_spec,
                runtime_config=runtime_config,
            )
            if reuse_env:
                _WORKER_ENV_CACHE[cache_key] = env

        controller = encode_controller(phi, task_spec, env)
        design = encode_morphology(x, task_spec, env=env)

        for rep in range(num_repeats):
            np.random.seed(seed + rep)
            obs = env.reset(design)
            controller.reset()
            ep_return = 0.0
            step_count = 0
            init_com_fwd = _com_x(env)

            for _ in range(task_spec.max_steps):
                act = controller(env.sim.solver.current_s, obs)
                obs, reward, done, info = env.step(act)
                ep_return += float(reward)
                step_count += 1
                if done:
                    break

            # B3: shaped reward.
            #   reward_shaping_only=True  -> ep_return = w * (final - init)  (pure
            #     net displacement; no per_step_velocity contribution; defeats
            #     wiggle attractor by construction since w·0 = 0).
            #   reward_shaping_only=False -> ep_return += w * (final - init)
            #     (additive bonus on top of env's per_step_velocity sum).
            if shaping_w != 0.0 or shaping_only:
                final_com_fwd = _com_x(env)
                disp = final_com_fwd - init_com_fwd
                if shaping_only:
                    ep_return = shaping_w * disp
                else:
                    ep_return = ep_return + shaping_w * disp

            returns_list.append(ep_return)
            if rep == 0:
                num_steps = step_count
                success = done and ep_return > 0
                if record and "trajectory" in info:
                    trajectory = {
                        k: np.asarray(v) if hasattr(v, "numpy") else np.asarray(v)
                        for k, v in info["trajectory"].items()
                    }

        if not reuse_env and env is not None:
            env.close()
    except Exception as e:
        failure_code = str(type(e).__name__) + ": " + str(e)[:64]
        returns_list = [0.0] * max(1, num_repeats)

    wall_time = time.perf_counter() - t0
    mean_ret = float(np.mean(returns_list)) if returns_list else 0.0
    std_ret = float(np.std(returns_list)) if len(returns_list) > 1 else 0.0

    return (mean_ret, success, num_steps, wall_time, failure_code, trajectory)


class _ResultCache:
    """LRU cache for rollout results."""

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._cache: Dict[str, RolloutResult] = {}
        self._order: List[str] = []

    def get(self, key: str) -> Optional[RolloutResult]:
        if self._max_size <= 0 or key not in self._cache:
            return None
        self._order.remove(key)
        self._order.append(key)
        return self._cache[key]

    def put(self, key: str, result: RolloutResult) -> None:
        if self._max_size <= 0:
            return
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self._max_size:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        self._cache[key] = result
        self._order.append(key)


class SoftZooRolloutEvaluator:
    """
    High-performance SoftZoo rollout evaluator.

    Evaluates batches of (x, phi) pairs with mode and fidelity support.
    Supports optional result caching and process-based parallelism.
    """

    def __init__(
        self,
        *,
        config: Optional[SoftZooEvaluatorConfig] = None,
        project_root: Optional[str] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.config = config or SoftZooEvaluatorConfig(**kwargs)
        self._project_root = project_root
        self._runtime_config = dict(runtime_config or self.config.extra.get("runtime_config") or {})
        # Mirror select config fields into runtime_config so they travel to workers.
        if self.config.ti_arch and "ti_arch" not in self._runtime_config:
            self._runtime_config["ti_arch"] = self.config.ti_arch
        if self.config.reward_shaping_weight:
            self._runtime_config["reward_shaping_weight"] = float(self.config.reward_shaping_weight)
        if self.config.reward_shaping_only:
            self._runtime_config["reward_shaping_only"] = True
        self._cache = _ResultCache(self.config.cache_size)
        # Persistent process pool: created on first parallel call, kept alive
        # across batches so workers don't repay Taichi/SoftZoo init cost
        # (which is several seconds per worker).
        # Pool worker recycle policy:
        #   CPU: maxtasksperchild=1 — Taichi 1.7.4 + softzoo segfault on 2nd
        #        env creation in same process.
        #   CUDA: maxtasksperchild=None (persistent) — verified safe; saves
        #        ~7s Taichi JIT per rollout.
        self._pool: Optional[Any] = None
        self._pool_workers: int = 0
        self._pool_maxtasks: Optional[int] = -1  # sentinel: never configured

    def _resolve_max_tasks(self) -> Optional[int]:
        cfg_val = self.config.max_tasks_per_child
        if cfg_val is not None:
            return cfg_val
        arch = (self.config.ti_arch or self._runtime_config.get("ti_arch") or "").lower()
        return None if arch == "cuda" else 1

    def _get_pool(self, workers: int):
        maxtasks = self._resolve_max_tasks()
        if (
            self._pool is not None
            and self._pool_workers == workers
            and self._pool_maxtasks == maxtasks
        ):
            return self._pool
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass
        ctx = _mp.get_context("spawn")
        self._pool = ctx.Pool(processes=workers, maxtasksperchild=maxtasks)
        self._pool_workers = workers
        self._pool_maxtasks = maxtasks
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                try:
                    self._pool.terminate()
                except Exception:
                    pass
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def evaluate_batch(
        self,
        request: RolloutBatchRequest,
        *,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> RolloutBatchResult:
        """
        Evaluate a batch of rollout requests.

        Groups requests by (task_id, mode_id, fidelity_level) for env reuse
        when running sequentially. With parallel=True, uses ProcessPoolExecutor.
        """
        workers = max_workers if max_workers is not None else self.config.max_workers
        parallel = parallel and workers > 0
        use_cache = use_cache and self.config.cache_size > 0

        results: List[Optional[RolloutResult]] = [None] * len(request.requests)
        cache_keys: List[Optional[str]] = [None] * len(request.requests)
        to_eval: List[int] = []

        for i, req in enumerate(request.requests):
            if use_cache:
                key = _theta_hash(req.morphology_params, req.controller_params)
                key = f"{request.task_id}:{req.mode_id}:{req.fidelity_level}:{key}"
                cache_keys[i] = key
                cached = self._cache.get(key)
                if cached is not None:
                    results[i] = cached
                    continue
            to_eval.append(i)

        t0 = time.perf_counter()

        if to_eval:
            if parallel and workers > 0:
                eval_list = self._eval_parallel(request, to_eval, workers)
            else:
                eval_list = self._eval_sequential(request, to_eval)

            for j, idx in enumerate(to_eval):
                res = eval_list[j]
                results[idx] = res
                if use_cache and cache_keys[idx]:
                    self._cache.put(cache_keys[idx], res)

        wall_total = time.perf_counter() - t0
        result_list = [r for r in results if r is not None]
        assert len(result_list) == len(request.requests), "Missing results"
        return self._aggregate(request, result_list, wall_total)

    def _eval_sequential(
        self,
        request: RolloutBatchRequest,
        indices: List[int],
    ) -> List[RolloutResult]:
        """Evaluate requests sequentially, reusing env per (task, mode, fidelity)."""
        out: List[RolloutResult] = []
        for idx in indices:
            req = request.requests[idx]
            mean_ret, success, num_steps, wall_time, failure_code, trajectory = _run_single_rollout(
                task_id=request.task_id,
                x=req.morphology_params,
                phi=req.controller_params,
                mode_id=req.mode_id,
                fidelity_level=req.fidelity_level,
                seed=req.seed,
                num_repeats=req.num_repeats,
                record=req.record,
                project_root=self._project_root,
                runtime_config_kwargs=self._runtime_config if self._runtime_config else None,
            )
            std_ret = 0.0  # Would need multiple repeats to compute
            if failure_code and self.config.raise_on_failure:
                raise RuntimeError(
                    f"SoftZoo rollout failed (task={request.task_id}, mode={req.mode_id}, "
                    f"fidelity={req.fidelity_level}, seed={req.seed}): {failure_code}"
                )
            out.append(
                RolloutResult(
                    return_=mean_ret,
                    success=success,
                    num_steps=num_steps,
                    mean_return=mean_ret,
                    std_return=std_ret,
                    failure_code=failure_code,
                    wall_time=wall_time,
                    trajectory=trajectory,
                )
            )
        return out

    def _eval_parallel(
        self,
        request: RolloutBatchRequest,
        indices: List[int],
        max_workers: int,
    ) -> List[RolloutResult]:
        """Evaluate requests in parallel via recycled mp.Pool workers."""
        out: List[Optional[RolloutResult]] = [None] * len(indices)
        pool = self._get_pool(max_workers)
        async_results = []
        for pos, idx in enumerate(indices):
            req = request.requests[idx]
            ar = pool.apply_async(
                _run_single_rollout,
                args=(
                    request.task_id,
                    req.morphology_params,
                    req.controller_params,
                    req.mode_id,
                    req.fidelity_level,
                    req.seed,
                    req.num_repeats,
                    req.record,
                    self._project_root,
                    self._runtime_config if self._runtime_config else None,
                ),
            )
            async_results.append((pos, ar))

        for pos, ar in async_results:
            try:
                mean_ret, success, num_steps, wall_time, failure_code, trajectory = ar.get()
                if failure_code and self.config.raise_on_failure:
                    raise RuntimeError(failure_code)
                out[pos] = RolloutResult(
                    return_=mean_ret,
                    success=success,
                    num_steps=num_steps,
                    mean_return=mean_ret,
                    std_return=0.0,
                    failure_code=failure_code,
                    wall_time=wall_time,
                    trajectory=trajectory,
                )
            except Exception as e:
                if self.config.raise_on_failure:
                    raise
                out[pos] = RolloutResult(
                    return_=0.0,
                    success=False,
                    num_steps=0,
                    failure_code=str(e)[:64],
                    wall_time=0.0,
                )
        return [o for o in out if o is not None]

    def _aggregate(
        self,
        request: RolloutBatchRequest,
        results: List[RolloutResult],
        wall_time_total: float = 0.0,
    ) -> RolloutBatchResult:
        """Aggregate results into RolloutBatchResult."""
        returns = np.array([r.return_ for r in results], dtype=np.float32)
        successes = np.array([r.success for r in results], dtype=bool)
        failure_codes = [r.failure_code for r in results]
        mean_return = float(np.mean(returns)) if returns.size > 0 else 0.0
        std_return = float(np.std(returns)) if returns.size > 1 else 0.0
        return RolloutBatchResult(
            results=results,
            returns=returns,
            successes=successes,
            mean_return=mean_return,
            std_return=std_return,
            wall_time_total=wall_time_total,
            failure_codes=failure_codes,
        )

    def evaluate_single(
        self,
        morphology_params: np.ndarray,
        controller_params: np.ndarray,
        *,
        task_id: str = "crawling_ground",
        mode_id: int = 0,
        fidelity_level: int = 2,
        seed: int = 0,
        num_repeats: int = 1,
        record: bool = False,
        **kwargs: Any,
    ) -> RolloutResult:
        """Evaluate a single design+controller pair."""
        req = RolloutBatchRequest(
            task_id=task_id,
            requests=[
                RolloutRequest(
                    morphology_params=morphology_params,
                    controller_params=controller_params,
                    mode_id=mode_id,
                    fidelity_level=fidelity_level,
                    seed=seed,
                    num_repeats=num_repeats,
                    record=record,
                )
            ],
        )
        batch = self.evaluate_batch(req, parallel=False)
        return batch.results[0]
