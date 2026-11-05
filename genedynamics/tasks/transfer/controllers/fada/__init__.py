"""Task-facing FADA actors with lazy PyTorch imports."""

__all__ = ["PublicFADATaskActor"]


def __getattr__(name):
    if name == "PublicFADATaskActor":
        from .base import PublicFADATaskActor

        return PublicFADATaskActor
    raise AttributeError(name)
