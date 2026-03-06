"""
Localization plugins for real robot deployment.

Provides pose/velocity from ROS2, Vicon, or mock. Plugins implement
BaseLocalizationPlugin and return (qpos, qvel) in world frame.
"""

import importlib.util
import os
import pkgutil
import threading
from typing import Any, Dict, List, Optional, Type

import importlib

from genedynamics.deploy.localization.base_plugin import BaseLocalizationPlugin

_plugin_registry: Dict[str, Optional[Type[BaseLocalizationPlugin]]] = {}
_registry_lock = threading.Lock()


def get_available_plugins() -> List[str]:
    """List registered plugin names."""
    with _registry_lock:
        return list(_plugin_registry.keys())


def discover_builtin_plugins() -> None:
    """Auto-discover built-in plugins in this package."""
    plugin_path = os.path.dirname(__file__)
    for _finder, name, ispkg in pkgutil.iter_modules([plugin_path]):
        if name not in _plugin_registry and name != "base_plugin":
            _plugin_registry[name] = None


def register_plugin(
    name: str,
    plugin_cls: Optional[Type[BaseLocalizationPlugin]] = None,
    module_path: Optional[str] = None,
) -> None:
    """Register a localization plugin."""
    with _registry_lock:
        if name in _plugin_registry:
            raise ValueError(f"Plugin '{name}' is already registered.")
        if plugin_cls:
            if not issubclass(plugin_cls, BaseLocalizationPlugin):
                raise TypeError("Plugin must inherit from BaseLocalizationPlugin.")
            _plugin_registry[name] = plugin_cls
        elif module_path:
            spec = importlib.util.spec_from_file_location(name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            plugin_cls = getattr(module, "BaseLocalizationPlugin", None)
            if not plugin_cls or not issubclass(plugin_cls, BaseLocalizationPlugin):
                raise TypeError("Module must define BaseLocalizationPlugin subclass.")
            _plugin_registry[name] = plugin_cls
        else:
            raise ValueError("Provide plugin_cls or module_path.")


def load_plugin(plugin_name: str) -> Optional[Type[BaseLocalizationPlugin]]:
    """Load plugin by name. Returns plugin class (instantiate with config)."""
    with _registry_lock:
        plugin_cls = _plugin_registry.get(plugin_name)
        if plugin_cls is None:
            try:
                module = importlib.import_module(f".{plugin_name}", package=__package__)
                candidates = [
                    attr for attr in vars(module).values()
                    if isinstance(attr, type)
                    and issubclass(attr, BaseLocalizationPlugin)
                    and attr is not BaseLocalizationPlugin
                ]
                if len(candidates) == 1:
                    plugin_cls = candidates[0]
                    _plugin_registry[plugin_name] = plugin_cls
                else:
                    return None
            except ImportError:
                return None
        return plugin_cls


discover_builtin_plugins()

# Alias: "mock" -> mock_plugin
from genedynamics.deploy.localization.mock_plugin import MockLocalizationPlugin
register_plugin("mock", MockLocalizationPlugin)

__all__ = [
    "BaseLocalizationPlugin",
    "get_available_plugins",
    "register_plugin",
    "load_plugin",
]
