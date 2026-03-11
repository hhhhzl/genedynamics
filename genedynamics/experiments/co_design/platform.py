"""
Co-design experiment platform.

Runs experiments with any registered baseline. Supports checkpointing,
logging, and reproducibility.
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
from .registry import get_baseline
from .engineering import CheckpointManager, ExperimentLogger, config_hash, set_seed


@dataclass
class CoDesignExperimentConfig:
    """
    Configuration for co-design experiment run.

    Attributes:
        baseline_name: Registered baseline (e.g. 'mrmfmbd')
        task_id: Task identifier
        seeds: Seeds to run
        output_dir: Output directory
        checkpoint_dir: Optional checkpoint directory
        baseline_params: Extra params for baseline
    """

    baseline_name: str
    task_id: str = "crawling_ground"
    seeds: List[int] = field(default_factory=lambda: [0])
    output_dir: str = "results/co_design"
    checkpoint_dir: Optional[str] = None
    baseline_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "task_id": self.task_id,
            "seeds": list(self.seeds),
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "baseline_params": self.baseline_params,
        }

    def config_hash(self) -> str:
        """Stable hash for reproducibility."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:12]


class CoDesignExperimentPlatform:
    """
    Platform for running co-design experiments with any baseline.

    Handles evaluator creation, task resolution, baseline dispatch,
    and result aggregation.
    """

    def __init__(
        self,
        config: CoDesignExperimentConfig,
        *,
        project_root: Optional[Path] = None,
        use_checkpointing: bool = True,
        use_logging: bool = True,
    ):
        self.config = config
        self._project_root = Path(project_root) if project_root else Path.cwd()
        self._output_dir = Path(config.output_dir)
        if not self._output_dir.is_absolute():
            self._output_dir = self._project_root / self._output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

        ckpt_dir = config.checkpoint_dir or str(self._output_dir / "checkpoints")
        self._checkpoint = CheckpointManager(Path(ckpt_dir), prefix="co_design") if use_checkpointing else None
        self._logger = ExperimentLogger(self._output_dir / "experiment.jsonl") if use_logging else None

    def run_single(self, seed: int) -> Dict[str, Any]:
        """Run single experiment with given seed."""
        from genedynamics.envs.evaluators import SoftZooRolloutEvaluator, SoftZooEvaluatorConfig
        from genedynamics.solvers.single.mrmfmbd.tasks import get_task_spec

        set_seed(seed)
        if self._logger:
            self._logger.log("run_start", seed=seed, config_hash=self.config.config_hash())

        task_spec = get_task_spec(self.config.task_id)
        evaluator = SoftZooRolloutEvaluator(
            config=SoftZooEvaluatorConfig(max_workers=0, cache_size=64),
            project_root=str(self._project_root),
        )

        baseline = get_baseline(self.config.baseline_name)
        bl_config = BaselineConfig(
            task_id=self.config.task_id,
            seed=seed,
            extra=self.config.baseline_params,
        )

        t0 = time.perf_counter()
        result = baseline.run(
            bl_config,
            evaluator,
            task_spec,
            x_dim=task_spec.x_dim,
            phi_dim=task_spec.phi_dim,
        )
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
        """Run all seeds, return list of results."""
        results: List[Dict[str, Any]] = []
        for seed in self.config.seeds:
            results.append(self.run_single(seed))

        # Save aggregated results
        out_path = self._output_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(
                self._serialize_results(results),
                f,
                indent=2,
            )
        return results

    def _serialize_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert results to JSON-serializable form."""
        out = []
        for r in results:
            rcopy = dict(r)
            if "result" in rcopy and hasattr(rcopy["result"], "get"):
                res = rcopy["result"]
                if "metadata" in res:
                    meta = res["metadata"]
                    # Simplify metadata for JSON
                    if "bridge_history" in meta and len(meta["bridge_history"]) > 20:
                        meta["bridge_history"] = meta["bridge_history"][:5] + ["..."] + meta["bridge_history"][-5:]
            out.append(rcopy)
        return out
