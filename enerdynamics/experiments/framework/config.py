"""
Configuration management for experiments.

This module provides the ExperimentConfig class for managing experiment
parameters and loading from YAML/JSON files.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


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
    method_params: Dict[str, Any] = field(default_factory=dict)
    
    # Obstacle configuration
    obstacle_levels: List[int] = field(default_factory=lambda: list(range(11)))
    obstacle_config: Dict[str, Any] = field(default_factory=dict)
    
    # Experiment execution
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    backend: str = "jax"
    device: str = "cpu"
    
    # Metrics and visualization
    metrics: List[str] = field(default_factory=lambda: ["ssr", "obstacle_density", "nonconvexity"])
    visualizations: List[str] = field(default_factory=lambda: ["trajectory", "diffusion", "energy_reward"])
    
    # Advanced configurations
    constraint_config: Optional[Dict[str, Any]] = None
    visualization_config: Optional[Dict[str, Any]] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization: convert output_dir to Path if needed."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
    
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
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert output_dir string to Path if present
        if 'output_dir' in data and isinstance(data['output_dir'], str):
            data['output_dir'] = Path(data['output_dir'])
        
        return cls(**data)
    
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
            data['output_dir'] = Path(data['output_dir'])
        
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
        
        if not self.obstacle_levels:
            errors.append("At least one obstacle level must be specified")
        
        return errors

