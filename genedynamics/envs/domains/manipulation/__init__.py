"""Manipulation domain envs — registered on import (backend imports guarded).

Robot-parameterized surface-contact scanning (`surface_scan_brax`). brax raises at
import on CPU-only installs, so the factory is registered under try/except and is
simply absent there (matching the other domains' lazy pattern)."""

try:
    from genedynamics.core.registry.environments import register_environment_factory
    from . import surface_scan_brax as _scan

    def _make_surface_scan(level="plane", **kw):
        return _scan.SurfaceScanEnv(_scan.SurfaceScanConfig(level=level, **kw))

    register_environment_factory("manipulator_surface_scan", _make_surface_scan)
    # convenience surface-family aliases (descriptive names): plane / cylinder
    # (analytic), convex / bumpy (NURBS), unseen (NURBS + domain randomization).
    for _fam in ("plane", "cylinder", "convex", "bumpy", "unseen"):
        register_environment_factory(
            f"manipulator_surface_scan_{_fam}",
            (lambda lv: lambda **kw: _make_surface_scan(level=lv, **kw))(_fam))
except Exception:
    pass

try:
    from genedynamics.core.registry.environments import register_environment_factory
    from . import peg_insert_brax as _insert

    def _make_peg_insert(level="wide", **kw):
        return _insert.PegInsertEnv(_insert.PegInsertConfig(level=level, **kw))

    register_environment_factory("manipulator_peg_insert", _make_peg_insert)
except Exception:
    pass
