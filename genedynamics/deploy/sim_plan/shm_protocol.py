"""
Shared memory protocol for Sim/Plan separation.

Names and layout aligned with dial-mpc for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class ShmNames:
    """Shared memory segment names."""
    time: str = "time_shm"
    state: str = "state_shm"
    acts: str = "acts_shm"
    refs: str = "refs_shm"
    plan_time: str = "plan_time_shm"
    tau: str = "tau_shm"


def create_sim_shm(
    nx: int,
    nu: int,
    n_acts: int,
    names: Optional[ShmNames] = None,
) -> dict:
    """
    Create shared memory for sim process (publisher of state, consumer of acts).

    Returns dict with buffers: time_shared, state_shared, acts_shared, refs_shared,
    plan_time_shared, tau_shared, and the SharedMemory objects for cleanup.
    """
    names = names or ShmNames()
    time_shm = shared_memory.SharedMemory(name=names.time, create=True, size=32)
    time_shared = np.ndarray(1, dtype=np.float32, buffer=time_shm.buf)
    time_shared[0] = 0.0

    state_shm = shared_memory.SharedMemory(name=names.state, create=True, size=nx * 32)
    state_shared = np.ndarray((nx,), dtype=np.float32, buffer=state_shm.buf)

    acts_shm = shared_memory.SharedMemory(name=names.acts, create=True, size=n_acts * nu * 32)
    acts_shared = np.ndarray((n_acts, nu), dtype=np.float32, buffer=acts_shm.buf)

    refs_shm = shared_memory.SharedMemory(name=names.refs, create=True, size=n_acts * nu * 3 * 32)
    refs_shared = np.ndarray((n_acts, nu, 3), dtype=np.float32, buffer=refs_shm.buf)
    refs_shared[:] = 0.0

    plan_time_shm = shared_memory.SharedMemory(name=names.plan_time, create=True, size=32)
    plan_time_shared = np.ndarray(1, dtype=np.float32, buffer=plan_time_shm.buf)
    plan_time_shared[0] = -0.05

    tau_shm = shared_memory.SharedMemory(name=names.tau, create=True, size=n_acts * nu * 32)
    tau_shared = np.ndarray((n_acts, nu), dtype=np.float32, buffer=tau_shm.buf)

    return {
        "time_shm": time_shm,
        "state_shm": state_shm,
        "acts_shm": acts_shm,
        "refs_shm": refs_shm,
        "plan_time_shm": plan_time_shm,
        "tau_shm": tau_shm,
        "time_shared": time_shared,
        "state_shared": state_shared,
        "acts_shared": acts_shared,
        "refs_shared": refs_shared,
        "plan_time_shared": plan_time_shared,
        "tau_shared": tau_shared,
    }


def create_plan_shm(
    nx: int,
    nu: int,
    n_acts: int,
    names: Optional[ShmNames] = None,
) -> dict:
    """
    Create shared memory for plan process (consumer of state, publisher of acts).

    Plan process attaches to existing shm created by sim. So this is "attach" not create.
    """
    return attach_plan_shm(nx, nu, n_acts, names)


def attach_sim_shm(
    nx: int,
    nu: int,
    n_acts: int,
    names: Optional[ShmNames] = None,
) -> dict:
    """Attach to existing sim shm (for plan process attaching to state)."""
    names = names or ShmNames()
    time_shm = shared_memory.SharedMemory(name=names.time, create=False, size=32)
    state_shm = shared_memory.SharedMemory(name=names.state, create=False, size=nx * 32)
    plan_time_shm = shared_memory.SharedMemory(name=names.plan_time, create=False, size=32)
    return {
        "time_shm": time_shm,
        "state_shm": state_shm,
        "plan_time_shm": plan_time_shm,
        "time_shared": np.ndarray(1, dtype=np.float32, buffer=time_shm.buf),
        "state_shared": np.ndarray((nx,), dtype=np.float32, buffer=state_shm.buf),
        "plan_time_shared": np.ndarray(1, dtype=np.float32, buffer=plan_time_shm.buf),
    }


def attach_plan_shm(
    nx: int,
    nu: int,
    n_acts: int,
    names: Optional[ShmNames] = None,
) -> dict:
    """Attach to shm created by sim (plan process: read state, write acts)."""
    names = names or ShmNames()
    time_shm = shared_memory.SharedMemory(name=names.time, create=False, size=32)
    state_shm = shared_memory.SharedMemory(name=names.state, create=False, size=nx * 32)
    acts_shm = shared_memory.SharedMemory(name=names.acts, create=False, size=n_acts * nu * 32)
    refs_shm = shared_memory.SharedMemory(name=names.refs, create=False, size=n_acts * nu * 3 * 32)
    plan_time_shm = shared_memory.SharedMemory(name=names.plan_time, create=False, size=32)
    tau_shm = shared_memory.SharedMemory(name=names.tau, create=False, size=n_acts * nu * 32)

    return {
        "time_shm": time_shm,
        "state_shm": state_shm,
        "acts_shm": acts_shm,
        "refs_shm": refs_shm,
        "plan_time_shm": plan_time_shm,
        "tau_shm": tau_shm,
        "time_shared": np.ndarray(1, dtype=np.float32, buffer=time_shm.buf),
        "state_shared": np.ndarray((nx,), dtype=np.float32, buffer=state_shm.buf),
        "acts_shared": np.ndarray((n_acts, nu), dtype=np.float32, buffer=acts_shm.buf),
        "refs_shared": np.ndarray((n_acts, nu, 3), dtype=np.float32, buffer=refs_shm.buf),
        "plan_time_shared": np.ndarray(1, dtype=np.float32, buffer=plan_time_shm.buf),
        "tau_shared": np.ndarray((n_acts, nu), dtype=np.float32, buffer=tau_shm.buf),
    }
