"""Thin frontend: configure once, validate capabilities, delegate whole chunks."""
from genedynamics.tasks.transfer.runtime import (
    ObstacleTransferConfig, ExecutionSpec, MechanismSpec, SkillSpec,
    mechanism_specs,
)
from .registry import backend_registry


class TransferWorkcell:
    def __init__(self, *, execution=None, task=None, skill=None, mechanism=None):
        self.execution = execution or ExecutionSpec()
        self.task = task or ObstacleTransferConfig()
        self.skill = skill or SkillSpec()
        self.mechanism = mechanism if mechanism is not None else MechanismSpec(payload_mass=self.task.nominal_payload_mass)
        if any(value.delay_steps > self.task.max_delay_steps for value in mechanism_specs(self.mechanism)):
            raise ValueError("mechanism delay exceeds declared ring-buffer capacity")
        self._impl = backend_registry.create(
            self.execution.backend, execution=self.execution, task=self.task,
            skill=self.skill, mechanism=self.mechanism,
        )
        self._impl.capabilities().require(self.execution.execution_mode)

    def capabilities(self):
        return self._impl.capabilities()

    def reset_batch(self, request, *, previous=None, mask=None, noise_batch=None):
        return self._impl.reset_batch(request, previous=previous, mask=mask, noise_batch=noise_batch)

    def collect_chunk(self, state, commands):
        self.capabilities().require("collect")
        return self._impl.collect_chunk(state, commands)

    def evaluate_chunk(self, state, commands):
        self.capabilities().require("evaluate")
        return self._impl.evaluate_chunk(state, commands)

    def train_step(self, train_state, batch, noise_batch):
        self.capabilities().require("train")
        return self._impl.train_step(train_state, batch, noise_batch)

    def synchronize(self, value):
        self._impl.synchronize(value)

    def describe(self):
        return self._impl.describe()
