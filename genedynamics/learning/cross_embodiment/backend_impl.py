"""Method-specific backend interface, following the existing solver pattern."""
from typing import Any, Protocol
from .contracts import ChunkResult, ExecutionState, RandomRequest
from .core.capabilities import Capabilities


class ExecutionBackend(Protocol):
    """World batches preserve mechanism order and may run in device microbatches.

    Construction accepts one broadcast mechanism or a fixed-size mechanism
    tuple; partial resets retain that binding. Microbatching never resets state
    or changes sample identity, randomness, command timing, or terminal masks.
    """
    def capabilities(self) -> Capabilities: ...
    def reset_batch(self, request: RandomRequest, *, previous=None, mask=None, noise_batch=None) -> ExecutionState: ...
    def collect_chunk(self, state: ExecutionState, commands: Any) -> ChunkResult: ...
    def evaluate_chunk(self, state: ExecutionState, commands: Any) -> ChunkResult: ...
    def synchronize(self, value: Any) -> None: ...
    def describe(self) -> dict: ...


class TrainingBackend(Protocol):
    """Optional future capability; scene execution does not pretend to train."""
    def train_step(self, train_state: Any, batch: Any, noise_batch: Any) -> Any: ...
    def fit_context(self, probe_batch: Any, initial_context: Any, request: RandomRequest) -> Any: ...


class TwinResponseBackend(Protocol):
    """Structured diagnostic response, independent of the surrogate learner.

    Initial observed coordinates/controller carry and commanded interventions
    belong to the record; a provider must build its own hypothesis model. The
    current NumPy provider owns MuJoCo/SciPy. Future device providers retain the
    frontend contract while batching hypotheses and scanning command time.
    """
    def predict_probe(self, record: Any) -> Any: ...
    @property
    def capabilities(self) -> Any: ...
