"""Domain-grouped environment subpackages (quadruped / humanoid / drone / toy).

Importing this package populates the environment registry via the
``@register_environment`` decorators co-located with each env class.

Every domain import is wrapped in a broad ``try/except`` because some backends
(e.g. brax) raise at import time when their optional deps are absent (verified:
``import genedynamics.envs.quadruped_brax`` raises AttributeError in CPU-only
envs). A failed import simply leaves that backend's names absent from the
registry — matching the previous lazy if/elif behaviour — without breaking
``import genedynamics.envs``.
"""

for _mod in ("toy", "drone", "quadruped", "humanoid"):
    try:
        __import__(f"genedynamics.envs.domains.{_mod}")
    except Exception:
        pass
del _mod
