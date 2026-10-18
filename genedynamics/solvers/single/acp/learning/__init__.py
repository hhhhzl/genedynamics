"""Authentic ACP native training and dataset integration."""

__all__ = ["ACPTrainer"]


def __getattr__(name):
    if name == "ACPTrainer":
        from .trainer import ACPTrainer
        return ACPTrainer
    raise AttributeError(name)
