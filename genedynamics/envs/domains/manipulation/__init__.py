"""Manipulation domain envs — registered on import (backend imports guarded).

Franka-Panda surface-contact scanning (`panda_brax`). brax raises at
import on CPU-only installs, so the factory is registered under try/except and is
simply absent there (matching the other domains' lazy pattern)."""

try:
    from genedynamics.core.registry.environments import register_environment_factory
    from . import panda_brax as _panda

    def _make_panda_surface_scan(level="plane", **kw):
        return _panda.PandaSurfaceScanEnv(_panda.PandaSurfaceScanConfig(level=level, **kw))

    register_environment_factory("manipulator_surface_scan", _make_panda_surface_scan)
    # convenience surface-family aliases (descriptive names): plane / cylinder
    # (analytic), convex / bumpy (NURBS), unseen (NURBS + domain randomization).
    for _fam in ("plane", "cylinder", "convex", "bumpy", "unseen"):
        register_environment_factory(
            f"manipulator_surface_scan_{_fam}",
            (lambda lv: lambda **kw: _make_panda_surface_scan(level=lv, **kw))(_fam))
except Exception:
    pass
