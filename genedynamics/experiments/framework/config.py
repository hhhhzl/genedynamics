"""
Configuration management for experiments.

This module provides the ExperimentConfig class for managing experiment
parameters and loading from YAML/JSON files.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from numbers import Integral
import copy
import json

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries without mutating either input.

    Dictionaries merge key-wise. Scalars and lists are replaced by the child
    value, which makes seed/suite lists deterministic under ``base:`` config
    inheritance.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class ExperimentConfig:
    """
    Unified experiment configuration.
    
    This class holds all configuration parameters for running experiments,
    including environment settings, method parameters, obstacle configuration,
    and output settings.
    """
    
    # Experiment identification (required fields first)
    name: str
    output_dir: Path
    
    # Environment configuration (required fields first)
    env_name: str  # e.g., "single_integrator_box_2d", "double_integrator_box_2d"
    
    # Method configuration (required fields first)
    method: str  # e.g., "edoc", "mppi", "cem"
    
    # Optional fields (with defaults) come after required fields
    env_params: Dict[str, Any] = field(default_factory=dict)
    execution_env_params: Dict[str, Any] = field(default_factory=dict)
    method_params: Dict[str, Any] = field(default_factory=dict)

    # Named task/physics conditions.  When non-empty, ExperimentRunner expands
    # suite x seed while preserving the established level_<name>/seed_<n>
    # result layout.  Existing obstacle-level configs leave this empty.
    suites: List[Dict[str, Any]] = field(default_factory=list)
    
    # Obstacle configuration. Defaults to a single dummy level so configs
    # that don't sweep obstacles (e.g. 3DGS reconstruction) can omit them.
    obstacle_levels: List[int] = field(default_factory=lambda: [0])
    obstacle_config: Dict[str, Any] = field(default_factory=dict)
    
    # Experiment execution
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    n_steps: int = 100
    backend: str = "jax"
    device: str = "cpu"
    
    # Metrics and visualization
    metrics: List[str] = field(default_factory=lambda: ["ssr", "obstacle_density", "nonconvexity"])
    visualizations: List[str] = field(default_factory=lambda: ["trajectory", "diffusion", "energy_reward"])
    auto_report: bool = True  # Generate report after run_all
    
    # Advanced configurations
    constraint_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None  # New scheduler system
    visualization_config: Optional[Dict[str, Any]] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization: convert output_dir to Path if needed."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # If output_dir is relative, resolve it relative to project root
        if not self.output_dir.is_absolute():
            # Find project root (directory containing setup.py, pyproject.toml, or .git)
            project_root = self._find_project_root()
            if project_root:
                self.output_dir = (project_root / self.output_dir).resolve()
    
    @staticmethod
    def _find_project_root() -> Optional[Path]:
        """
        Find project root directory by looking for setup.py, pyproject.toml, or .git.
        
        Returns:
            Path to project root, or None if not found
        """
        current = Path(__file__).resolve()
        # Start from this file and go up: genedynamics/experiments/framework/config.py
        # -> genedynamics/experiments/framework -> genedynamics/experiments -> genedynamics -> project_root
        for parent in [current.parent.parent.parent.parent, current.parent.parent.parent]:
            if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists() or (parent / ".git").exists():
                return parent
        
        # Fallback: try current working directory
        cwd = Path.cwd()
        if (cwd / "setup.py").exists() or (cwd / "pyproject.toml").exists() or (cwd / ".git").exists():
            return cwd
        
        return None
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'ExperimentConfig':
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            ExperimentConfig instance
            
        Raises:
            ImportError: If PyYAML is not installed
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to load YAML configuration files. "
                "Install it with: pip install pyyaml"
            )
        data = cls._load_yaml_dict(Path(path).resolve(), seen=set())
        
        # Convert output_dir string to Path if present
        if 'output_dir' in data and isinstance(data['output_dir'], str):
            output_dir = Path(data['output_dir'])
            # If relative, resolve relative to project root
            if not output_dir.is_absolute():
                project_root = cls._find_project_root()
                if project_root:
                    output_dir = (project_root / output_dir).resolve()
            data['output_dir'] = output_dir
        
        return cls(**data)

    @classmethod
    def _load_yaml_dict(cls, path: Path, seen: Set[Path]) -> Dict[str, Any]:
        """Load a YAML dictionary and resolve its relative ``base:`` chain."""
        path = path.resolve()
        if path in seen:
            chain = " -> ".join(str(p) for p in [*seen, path])
            raise ValueError(f"Cyclic experiment config base chain: {chain}")
        if not path.exists():
            raise FileNotFoundError(f"Experiment config base not found: {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Experiment config must be a mapping: {path}")
        data = copy.deepcopy(data)
        base_ref = data.pop('base', None)
        if base_ref is None:
            return data
        if not isinstance(base_ref, str) or not base_ref.strip():
            raise ValueError(f"Experiment config base must be a path string: {path}")
        base_path = (path.parent / base_ref).resolve()
        base_data = cls._load_yaml_dict(base_path, seen | {path})
        return deep_merge(base_data, data)
    
    @classmethod
    def from_json(cls, path: Path) -> 'ExperimentConfig':
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to JSON configuration file
            
        Returns:
            ExperimentConfig instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert output_dir string to Path if present
        if 'output_dir' in data and isinstance(data['output_dir'], str):
            output_dir = Path(data['output_dir'])
            # If relative, resolve relative to project root
            if not output_dir.is_absolute():
                project_root = cls._find_project_root()
                if project_root:
                    output_dir = (project_root / output_dir).resolve()
            data['output_dir'] = output_dir
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation of configuration
        """
        result = asdict(self)
        # Convert Path to string for JSON serialization
        result['output_dir'] = str(self.output_dir)
        return result

    def use_development_output_root(self, output_root: Path) -> None:
        """Mirror the canonical project-relative output under an isolated root.

        This keeps development seeds out of formal result directories without
        introducing a second config tree.  The canonical path and run class are
        retained in metadata and therefore persisted in manifests/results.
        """
        project_root = self._find_project_root()
        if project_root is None:
            raise ValueError("Cannot isolate output without a project root")
        project_root = project_root.resolve()
        canonical = self.output_dir.resolve()
        try:
            relative = canonical.relative_to(project_root)
        except ValueError as exc:
            raise ValueError(
                f"Canonical output directory is outside project root: {canonical}"
            ) from exc

        root = Path(output_root).expanduser()
        if not root.is_absolute():
            root = project_root / root
        root = root.resolve()
        isolated = (root / relative).resolve()
        if isolated == canonical:
            raise ValueError(
                "Development output root resolves to the canonical output path"
            )

        self.output_dir = isolated
        self.metadata = deep_merge(self.metadata, {
            "run_class": "development",
            "formal_seeds": False,
            "canonical_output_dir": str(canonical),
            "development_output_root": str(root),
        })
    
    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
            
        Raises:
            ImportError: If PyYAML is not installed
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to save YAML configuration files. "
                "Install it with: pip install pyyaml"
            )
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, path: Path) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to save JSON file
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        if not self.name:
            errors.append("Configuration name cannot be empty")
        
        if not self.env_name:
            errors.append("Environment name cannot be empty")
        
        if not self.method:
            errors.append("Method name cannot be empty")
        
        if not self.seeds:
            errors.append("At least one seed must be specified")
        if isinstance(self.n_steps, bool) or not isinstance(self.n_steps, Integral) or self.n_steps <= 0:
            errors.append("n_steps must be a positive integer")

        suite_names = []
        for index, suite in enumerate(self.suites):
            if not isinstance(suite, dict):
                errors.append(f"Suite {index} must be a mapping")
                continue
            name = suite.get("name")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"Suite {index} needs a non-empty string name")
                continue
            if "/" in name or "\\" in name or name in {".", ".."}:
                errors.append(f"Suite name is not path-safe: {name!r}")
            suite_names.append(name)
            forbidden = sorted(set(suite).intersection({
                "seeds", "backend", "device", "output_dir",
            }))
            if forbidden:
                errors.append(
                    f"Suite {name!r} cannot override protocol fields: "
                    + ", ".join(forbidden)
                )
            if "n_steps" in suite:
                steps = suite["n_steps"]
                if isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0:
                    errors.append(f"Suite {name!r} n_steps must be a positive integer")
            for key in ("env_params", "execution_env_params", "method_params"):
                if key in suite and not isinstance(suite[key], dict):
                    errors.append(f"Suite {name!r} field {key!r} must be a mapping")
            suite_method = suite.get("method_params", {})
            if isinstance(suite_method, dict):
                budget_keys = sorted(
                    set(suite_method).intersection({
                        "Nsample", "Hsample", "Hnode", "Ndiffuse",
                        "Ndiffuse_init", "n_steps",
                    })
                )
                if budget_keys:
                    errors.append(
                        f"Suite {name!r} cannot change fairness budget: "
                        + ", ".join(budget_keys)
                    )
        duplicates = sorted({name for name in suite_names if suite_names.count(name) > 1})
        if duplicates:
            errors.append(f"Duplicate suite names: {', '.join(duplicates)}")
        
        return errors

    def for_suite(self, suite: Dict[str, Any]) -> 'ExperimentConfig':
        """Resolve one task condition, including its optional episode length.

        Sampling budgets remain shared.  Episode duration may differ between
        tasks, but the protocol auditor still pairs it across methods within
        each suite.
        """
        name = str(suite["name"])
        steps = suite.get("n_steps", self.n_steps)
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0:
            raise ValueError(f"Suite {name!r} n_steps must be a positive integer")
        cfg = copy.deepcopy(self)
        cfg.n_steps = int(steps)
        cfg.suites = []
        cfg.obstacle_levels = [name]
        cfg.env_params = deep_merge(self.env_params, suite.get("env_params", {}))
        if suite.get("level") is not None:
            cfg.env_params["level"] = suite["level"]
        cfg.execution_env_params = deep_merge(
            self.execution_env_params,
            suite.get("execution_env_params", {}),
        )
        cfg.method_params = deep_merge(
            self.method_params,
            suite.get("method_params", {}),
        )
        cfg.metadata = deep_merge(self.metadata, {
            "suite": name,
            "suite_level": suite.get("level", name),
        })
        return cfg


__all__ = ["ExperimentConfig", "deep_merge"]
