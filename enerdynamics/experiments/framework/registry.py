"""
Plugin registry for managing and discovering plugins.

This module provides a centralized registry for all plugin types,
enabling dynamic plugin discovery and registration.
"""

from typing import Dict, Any, Optional, Type, List
from collections import defaultdict


class PluginRegistry:
    """
    Centralized registry for experiment framework plugins.
    
    Supports registration and discovery of plugins by type and name.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._plugins: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._plugin_types = {
            'method': 'MethodPlugin',
            'environment': 'EnvironmentPlugin',
            'metric': 'MetricsPlugin',
            'visualization': 'VisualizationPlugin',
            'obstacle_generator': 'ObstacleGeneratorPlugin',
        }
    
    def register(self, plugin: Any, plugin_type: str, name: Optional[str] = None) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance implementing the appropriate interface
            plugin_type: Type of plugin ('method', 'environment', 'metric', etc.)
            name: Optional custom name (uses plugin.name if not provided)
            
        Raises:
            ValueError: If plugin_type is invalid
        """
        if plugin_type not in self._plugin_types:
            available = ', '.join(self._plugin_types.keys())
            raise ValueError(
                f"Invalid plugin_type: {plugin_type}. "
                f"Available types: {available}"
            )
        
        plugin_name = name if name is not None else plugin.name
        self._plugins[plugin_type][plugin_name] = plugin
    
    def get_plugin(self, plugin_type: str, name: str) -> Any:
        """
        Get a plugin by type and name.
        
        Args:
            plugin_type: Type of plugin
            name: Plugin name
            
        Returns:
            Plugin instance
            
        Raises:
            KeyError: If plugin not found
            ValueError: If plugin_type is invalid
        """
        if plugin_type not in self._plugin_types:
            available = ', '.join(self._plugin_types.keys())
            raise ValueError(
                f"Invalid plugin_type: {plugin_type}. "
                f"Available types: {available}"
            )
        
        if name not in self._plugins[plugin_type]:
            available = ', '.join(self._plugins[plugin_type].keys())
            raise KeyError(
                f"Plugin '{name}' of type '{plugin_type}' not found. "
                f"Available: {available}"
            )
        
        return self._plugins[plugin_type][name]
    
    def list_plugins(self, plugin_type: Optional[str] = None) -> List[str]:
        """
        List available plugins.
        
        Args:
            plugin_type: Optional filter by plugin type
            
        Returns:
            List of plugin names (or dict mapping type to names if plugin_type is None)
        """
        if plugin_type is not None:
            if plugin_type not in self._plugin_types:
                return []
            return list(self._plugins[plugin_type].keys())
        else:
            return {
                ptype: list(plugins.keys())
                for ptype, plugins in self._plugins.items()
            }
    
    def has_plugin(self, plugin_type: str, name: str) -> bool:
        """
        Check if a plugin is registered.
        
        Args:
            plugin_type: Type of plugin
            name: Plugin name
            
        Returns:
            True if plugin is registered, False otherwise
        """
        if plugin_type not in self._plugins:
            return False
        return name in self._plugins[plugin_type]

