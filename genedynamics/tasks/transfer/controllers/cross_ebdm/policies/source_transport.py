"""Ours' frozen source-prior/transport inference through the public CPU contract.

The legacy fused simulation solver remains a historical implementation. This
actor receives public arrays only; the executor, mechanism and delay queue are
never passed to it. Legacy transports are migration diagnostics until recalibrated.
"""
from copy import deepcopy
import numpy as np

from genedynamics.tasks.transfer.observation import PublicControllerMemory
from genedynamics.core.control.transfer import orientation_error
from genedynamics.solvers.single.cross_ebdm.learning.learned import FrozenRegressor
from genedynamics.solvers.single.cross_ebdm.learning.reference_transport import ReferenceTransport
from genedynamics.tasks.transfer.behavior import PhaseGaussianPrior, reference_commands


class PublicCrossEBDMPolicy:
    def __init__(self, spec, prior, *, modes=(0,), transport=None, seed=0):
        if not isinstance(prior, PhaseGaussianPrior):
            raise TypeError("public actor requires a genuine frozen phase source prior")
        self.spec, self.prior, self.transport = spec, prior, transport
        self.reset(modes=modes, seed=seed)

    @classmethod
    def load(cls, spec, prior_path, *, transport_path=None, **kwargs):
        prior = PhaseGaussianPrior(FrozenRegressor.load(prior_path, backend="native", device="cpu"))
        transport = None if transport_path is None else ReferenceTransport.load(transport_path, backend="native", device="cpu")
        return cls(spec, prior, transport=transport, **kwargs)

    def reset(self, *, modes=None, seed=None):
        modes = getattr(self, "_reset_modes", (0,)) if modes is None else modes
        seed = getattr(self, "_reset_seed", 0) if seed is None else seed
        self._reset_modes, self._reset_seed = tuple(modes), seed
        self.observer = PublicControllerMemory(self.spec, modes)
        self.rng = np.random.default_rng(seed)
        self.commands = []

    def get_state(self):
        return deepcopy(dict(observer=self.observer.get_state(), rng=self.rng.bit_generator.state, commands=self.commands))

    def restore(self, state):
        self.observer.restore(state["observer"])
        self.rng.bit_generator.state = deepcopy(state["rng"])
        self.commands = deepcopy(state["commands"])

    def reorder(self, indices):
        self.observer.reorder(indices)
        self.commands = [row[indices].copy() for row in self.commands]

    def act(self, public):
        self.observer.update(public)
        memory = self.observer.memory
        if not self.commands:
            noise = self.rng.standard_normal((len(memory.phase), 5)).astype(np.float32)
            reference = self.prior.reference(memory, noise)
            if self.spec.alias == "T3":
                commands = reference_commands(reference, public["reference_position"], public["reference_rotation"],
                    public["tcp_position"], public["tcp_rotation"], integral_bias=memory.tracking_integral,
                    dt=self.spec.config.control_dt)
            else:
                p, r = public["tcp_position"], public["tcp_rotation"]
                rp, rr, goal_r = public["reference_position"], public["reference_rotation"], public["goal_rotation"]
                commands = np.zeros((len(p), 1, 13), dtype=np.float32)
                commands[:, 0, :3] = np.clip(4*(reference[:, :3]-rp)+3*(reference[:, :3]-p)+memory.tracking_integral[:, :3], -.12, .12)
                commands[:, 0, 3:6] = np.clip(np.stack([3*orientation_error(np, a, c)+3*orientation_error(np, b, c)
                    for a, b, c in zip(rr, r, goal_r)])+memory.tracking_integral[:, 3:], -.6, .6)
                commands[:, :, 6:12] = 1.
            if self.transport is not None:
                commands = self.transport.apply(commands, phase=memory.phase)
            self.commands = [row.copy() for row in commands.swapaxes(0, 1)]
        return self.commands.pop(0)
