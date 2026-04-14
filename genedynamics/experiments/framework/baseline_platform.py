"""
Baseline experiment platform.

Task-agnostic: runs any registered baseline on any registered task domain.
Supports checkpointing, logging, and reproducibility.
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
        checkpoint_dir: Optional checkpoint directory
        baseline_params: Extra params for baseline
        evaluator_params: Extra params for evaluator creation
    """

    baseline_name: str
    task_domain: str = "softzoo"
    task_id: str = "crawling_ground"
    seeds: List[int] = field(default_factory=lambda: [0])
    output_dir: str = "results/baseline"
    checkpoint_dir: Optional[str] = None
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
            "checkpoint_dir": self.checkpoint_dir,
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

        ckpt_dir = config.checkpoint_dir or str(self._output_dir / "checkpoints")
        self._checkpoint = (
            CheckpointManager(Path(ckpt_dir), prefix="baseline") if use_checkpointing else None
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
        bl_config = BaselineConfig(
            task_id=self.config.task_id,
            seed=seed,
            extra=method_params,
            scheduler=scheduler,
        )

        x_dim = getattr(task_spec, "x_dim", 3)
        phi_dim = getattr(task_spec, "phi_dim", 4)

        t0 = time.perf_counter()
        try:
            result = baseline.run(
                bl_config,
                evaluator,
                task_spec,
                x_dim=x_dim,
                phi_dim=phi_dim,
            )
        finally:
            # Release ProcessPoolExecutor workers so successive seeds don't
            # stack Taichi-allocated memory (each worker holds several GB).
            close_fn = getattr(evaluator, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        wall_time = time.perf_counter() - t0

        out = {
            "seed": seed,
            "result": result.to_dict(),
            "wall_time": wall_time,
            "config_hash": self.config.config_hash(),
        }
        if self._logger:
            self._logger.log("run_complete", seed=seed, return_=result.return_, wall_time=wall_time)
            self._logger.flush()
        return out

    def run_all(self) -> List[Dict[str, Any]]:
        """Run all seeds, return list of results. Resumes from checkpoint if enabled."""
        cfg_hash = self.config.config_hash()
        completed: Dict[int, Dict[str, Any]] = {}
        if self._resume and self._checkpoint:
            state = self._checkpoint.load_latest(key=cfg_hash)
            if state and state.config_hash == cfg_hash and state.payload:
                completed = {
                    int(k): v for k, v in state.payload.get("results_by_seed", {}).items()
                }

        results: List[Dict[str, Any]] = []
        for seed in self.config.seeds:
            if seed in completed:
                results.append(completed[seed])
                if self._logger:
                    self._logger.log("run_skipped_resume", seed=seed, reason="checkpoint")
                continue
            out = self.run_single(seed)
            results.append(out)
            if self._checkpoint:
                by_seed = {r["seed"]: r for r in results}
                self._checkpoint.save(
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

        out_path = self._output_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(self._serialize_results(results), f, indent=2)
        return results

    def _serialize_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert results to JSON-serializable form. Truncates large metadata."""
        out = []
        for r in results:
            rcopy = dict(r)
            if "result" in rcopy and isinstance(rcopy["result"], dict):
                res = rcopy["result"]
                if (
                    "bridge_history" in res
                    and isinstance(res["bridge_history"], list)
                    and len(res["bridge_history"]) > 20
                ):
                    bh = res["bridge_history"]
                    res = dict(res)
                    res["bridge_history"] = bh[:5] + ["...truncated..."] + bh[-5:]
                    rcopy["result"] = res
            out.append(rcopy)
        return out
