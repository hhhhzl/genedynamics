"""Explicit JAX device selection without importing a physics engine."""
import re
import jax


def select_device(request):
    match=re.fullmatch(r"(cpu|gpu|cuda)(?::([0-9]+))?",request)
    if not match: raise ValueError("device must be cpu[:ordinal], gpu[:ordinal] or cuda[:ordinal]")
    platform,ordinal=match[1],int(match[2] or 0)
    try: devices=jax.devices("gpu" if platform=="cuda" else platform)
    except RuntimeError as error:
        raise RuntimeError(f"requested device {request} is unavailable; no CPU fallback") from error
    if ordinal>=len(devices): raise RuntimeError(f"requested device {request} is unavailable; no CPU fallback")
    return devices[ordinal]
