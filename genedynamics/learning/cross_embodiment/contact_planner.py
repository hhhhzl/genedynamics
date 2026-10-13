"""Solver-style interface for a calibrated, finite-candidate contact MPC.

The policy callback consumes (runner, observed state, policy carry) and returns
(commands, next carry, reference). A provider owns candidate/horizon execution.
Only NumPy/CPU is qualified; a GPU provider must batch candidates and scan time
inside this call, rather than moving a per-world Python loop into orchestration.
"""


class ContactPlanner:
    def __init__(self, config, mechanism, candidates, *, horizon_chunks=3,
                 backend="numpy", device="cpu"):
        if backend != "numpy" or device != "cpu":
            raise NotImplementedError("contact candidate MPC currently qualifies numpy/cpu only")
        if isinstance(horizon_chunks, bool) or not isinstance(horizon_chunks, int) or horizon_chunks < 1:
            raise ValueError("positive integer horizon required")
        if not candidates:
            raise ValueError("nonempty frozen candidate bank required")
        from .backends.numpy.contact_planner import NumpyContactPlanner
        self._impl = NumpyContactPlanner(config, mechanism, candidates, horizon_chunks)
        self.backend, self.device = backend, device

    def select(self, observed_state, policy_carry, policy):
        return self._impl.select(observed_state, policy_carry, policy)

    def statistics(self):
        return self._impl.statistics()
