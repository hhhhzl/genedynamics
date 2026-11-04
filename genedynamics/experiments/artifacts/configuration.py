"""Relative base chains, deterministic mapping merge, and cycle detection."""

from pathlib import Path
import yaml


def merge_mapping(base, override):
    result = dict(base)
    for name, value in override.items():
        result[name] = merge_mapping(result[name], value) if isinstance(value, dict) and isinstance(result.get(name), dict) else value
    return result


def load_configuration(path, _ancestors=()):
    path = Path(path).resolve()
    if path in _ancestors:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    content = yaml.safe_load(path.read_text())
    if not isinstance(content, dict):
        raise ValueError("Configuration root must be a mapping")
    bases = content.pop("base", [])
    if isinstance(bases, str):
        bases = [bases]
    result = {}
    for base in bases:
        result = merge_mapping(result, load_configuration(path.parent / base, _ancestors + (path,)))
    return merge_mapping(result, content)
