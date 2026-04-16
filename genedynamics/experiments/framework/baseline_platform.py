"""
Baseline experiment platform.

Task-agnostic: runs any registered baseline on any registered task domain.
Supports per-seed result caching (for resume), logging, and reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .baseline import BaselineConfig, BaselineResult
from .baseline_registry import get_baseline
from .task_domain_provider import get_task_domain_provider
from ..common.engineering import (
    CheckpointManager,
    CheckpointState,
    ExperimentLogger,
    set_seed,
)


@dataclass
class BaselineExperimentConfig:
    """
    Configuration for baseline comparison experiment.

    Attributes:
        baseline_name: Registered baseline (e.g. 'mrmfmbd')
        task_domain: Task domain (e.g. 'softzoo', '3dgs')
        task_id: Task identifier within domain
        seeds: Seeds to run
        output_dir: Output directory
        cache_dir: Optional per-seed result cache directory. Used to resume
            long multi-seed runs; nothing model-related is stored here (no
            weights, no checkpoints — most of our baselines including MBD are
            zero-shot samplers). Defaults to ``{output_dir}/run_cache``.
        baseline_params: Extra params for baseline
        evaluator_params: Extra params for evaluator creation
    """

    baseline_name: str
    task_domain: str = "softzoo"
    task_id: str = "crawling_ground"
    seeds: List[int] = field(default_factory=lambda: [0])
    output_dir: str = "results/baseline"
    cache_dir: Optional[str] = None
    scheduler_config: Optional[Dict[str, Any]] = None
    method_params: Dict[str, Any] = field(default_factory=dict)
    baseline_params: Dict[str, Any] = field(default_factory=dict)
    evaluator_params: Dict[str, Any] = field(default_factory=dict)
    save_gif: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "baseline_name": self.baseline_name,
            "task_domain": self.task_domain,
            "task_id": self.task_id,
            "seeds": list(self.seeds),
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "method_params": self.method_params,
            "evaluator_params": self.evaluator_params,
        }
        if self.scheduler_config:
            d["scheduler_config"] = self.scheduler_config
        if self.baseline_params:
            d["baseline_params"] = self.baseline_params
        return d

    def config_hash(self) -> str:
        """Stable hash for reproducibility."""
        data = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def validate(self) -> List[str]:
        """Validate config. Returns list of error messages (empty if valid)."""
        errs: List[str] = []
        if not self.baseline_name:
            errs.append("baseline_name is required")
        if not self.task_domain:
            errs.append("task_domain is required")
        if not self.task_id:
            errs.append("task_id is required")
        if not self.seeds:
            errs.append("seeds must be non-empty")
        return errs


class BaselineExperimentPlatform:
    """
    Platform for running baseline comparison experiments.

    Task-agnostic: works with any task domain (softzoo, 3dgs, quadruped, etc.)
    and any baseline (mrmfmbd, cmaes, etc.). Resolves evaluator and task_spec
    via task domain provider registry.
    """

    def __init__(
        self,
        config: BaselineExperimentConfig,
        *,
        project_root: Optional[Path] = None,
        use_checkpointing: bool = True,
        use_logging: bool = True,
        resume: bool = True,
    ):
        self.config = config
        self._resume = resume
        self._project_root = Path(project_root) if project_root else Path.cwd()
        self._output_dir = Path(config.output_dir)
        if not self._output_dir.is_absolute():
            self._output_dir = self._project_root / self._output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

        cache_dir = config.cache_dir or str(self._output_dir / "run_cache")
        # CheckpointManager here is just a seed-level result cache for resume;
        # no model weights / learned parameters are ever saved by this platform.
        self._cache = (
            CheckpointManager(Path(cache_dir), prefix="baseline") if use_checkpointing else None
        )
        self._logger = (
            ExperimentLogger(self._output_dir / "experiment.jsonl") if use_logging else None
        )

    def _create_scheduler(self) -> Optional[Any]:
        """Create scheduler from scheduler_config if present."""
        sc = self.config.scheduler_config
        if not sc:
            return None
        from ..common.constraints import create_scheduler_from_config
        return create_scheduler_from_config(sc, backend_name="jax")

    def _resolve_method_params(self) -> Dict[str, Any]:
        """Merge method_params and legacy baseline_params."""
        merged = dict(self.config.baseline_params)
        merged.update(self.config.method_params)
        return merged

    def run_single(self, seed: int) -> Dict[str, Any]:
        """Run single experiment with given seed."""
        provider = get_task_domain_provider(self.config.task_domain)
        task_spec = provider.get_task_spec(self.config.task_id)
        evaluator = provider.create_evaluator(
            str(self._project_root),
            max_workers=self.config.evaluator_params.get("max_workers", 0),
            cache_size=self.config.evaluator_params.get("cache_size", 64),
            **{k: v for k, v in self.config.evaluator_params.items() if k not in ("max_workers", "cache_size")},
        )

        scheduler = self._create_scheduler()
        method_params = self._resolve_method_params()

        set_seed(seed)
        if self._logger:
            self._logger.log("run_start", seed=seed, config_hash=self.config.config_hash())

        baseline = get_baseline(self.config.baseline_name)

        x_dim = getattr(task_spec, "x_dim", 3)
        phi_dim = getattr(task_spec, "phi_dim", 4)

        # Override x_dim if evaluator was built with non-default voxel_dims.
        voxel_dims = self.config.evaluator_params.get("voxel_dims")
        if voxel_dims is not None:
            x_dim = int(voxel_dims[0]) * int(voxel_dims[1]) * int(voxel_dims[2])

        # Forward voxel_dims from evaluator_params into method_params so the
        # baseline can apply morphology symmetry without knowing the evaluator.
        if voxel_dims is not None and "voxel_dims" not in method_params:
            method_params["voxel_dims"] = list(voxel_dims)
        bl_config = BaselineConfig(
            task_id=self.config.task_id,
            seed=seed,
            extra=method_params,
            scheduler=scheduler,
        )

        t0 = time.perf_counter()
        result = None
        try:
            result = baseline.run(
                bl_config,
                evaluator,
                task_spec,
                x_dim=x_dim,
                phi_dim=phi_dim,
            )
            wall_time = time.perf_counter() - t0
            out = {
                "seed": seed,
                "result": result.to_dict(),
                "wall_time": wall_time,
                "config_hash": self.config.config_hash(),
            }
            # Persist per-seed result IMMEDIATELY (before evaluator.close()).
            # CUDA Taichi pool teardown can deadlock against JAX's CUDA context
            # in the parent process; if that happens we still keep the result.
            seed_path = self._output_dir / f"results_seed_{seed}.json"
            try:
                with open(seed_path, "w") as f:
                    json.dump(self._serialize_results([out])[0], f, indent=2)
            except Exception:
                pass
            if self._logger:
                self._logger.log("run_complete", seed=seed, return_=result.return_, wall_time=wall_time)
                self._logger.flush()
        finally:
            # Best-effort pool cleanup with a short hard deadline. CUDA Taichi
            # workers occasionally hang in driver futexes; we'd rather leak a
            # subprocess than block the main job indefinitely.
            close_fn = getattr(evaluator, "close", None)
            if callable(close_fn):
                import threading
                done_evt = threading.Event()
                def _do_close():
                    try:
                        close_fn()
                    except Exception:
                        pass
                    finally:
                        done_evt.set()
                t = threading.Thread(target=_do_close, daemon=True)
                t.start()
                done_evt.wait(timeout=30.0)
        return out

    def run_all(self) -> List[Dict[str, Any]]:
        """Run all seeds, return list of results. Resumes from per-seed cache if enabled."""
        cfg_hash = self.config.config_hash()
        completed: Dict[int, Dict[str, Any]] = {}
        if self._resume and self._cache:
            state = self._cache.load_latest(key=cfg_hash)
            if state and state.config_hash == cfg_hash and state.payload:
                completed = {
                    int(k): v for k, v in state.payload.get("results_by_seed", {}).items()
                }

        results: List[Dict[str, Any]] = []
        for seed in self.config.seeds:
            if seed in completed:
                results.append(completed[seed])
                if self._logger:
                    self._logger.log("run_skipped_resume", seed=seed, reason="cache")
                continue
            out = self.run_single(seed)
            results.append(out)
            if self._cache:
                by_seed = {r["seed"]: r for r in results}
                self._cache.save(
                    CheckpointState(
                        step=len(results),
                        seed=seed,
                        config_hash=cfg_hash,
                        wall_time=0.0,
                        payload={
                            "results_by_seed": by_seed,
                            "completed_seeds": list(by_seed.keys()),
                        },
                    ),
                    key=cfg_hash,
                )
            # Write rolling results.json after every seed so we never lose
            # data if a later seed (or post-run teardown) hangs or crashes.
            try:
                out_path = self._output_dir / "results.json"
                with open(out_path, "w") as f:
                    json.dump(self._serialize_results(results), f, indent=2)
            except Exception:
                pass

        out_path = self._output_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(self._serialize_results(results), f, indent=2)
        return results

    def _serialize_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert results to JSON-serializable form. Keep full bridge_history
        so we can plot MBD reward convergence curves."""
        return list(results)
