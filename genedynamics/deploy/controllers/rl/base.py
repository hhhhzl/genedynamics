"""Reusable RL controller framework.

This module provides the four building blocks every third-party policy
adapter implements (described in §14 of the deploy refactor plan):

* :class:`PolicyArtifact`  — where weights live and how to load them
* :class:`ObsBuilder`      — ``RobotState`` + ``Intent`` → obs vector
* :class:`ActionMapper`    — policy output → :class:`ControlCommand`
* :class:`RLController`    — :class:`Controller` protocol implementation

Adding a new policy means writing one ``ObsBuilder``, one ``ActionMapper``,
optionally subclassing :class:`RLController` to bind them, and registering
the result. The pipeline / IO / safety / observers all stay untouched.

Decimation, history rolling, last-action feedback, and frame-by-frame
diagnostics are handled in :class:`RLController` so per-policy code stays
small.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
)
from genedynamics.deploy.interfaces.robot_io import RobotIO

__all__ = [
    "PolicyArtifact",
    "ObsBuilder",
    "ActionMapper",
    "InferenceFn",
    "RLController",
]


# ---------------------------------------------------------------------------
# Artifact + protocols
# ---------------------------------------------------------------------------


@dataclass
class PolicyArtifact:
    """Where a third-party policy lives on disk and how to load it.

    Attributes:
        ckpt_path: Path to the weights file (``.pt`` / ``.onnx`` / ``.pkl`` / …).
        train_cfg_path: Path to the training config the adapter parses to
            recover obs scaling, action scaling, joint order, Kp/Kd, default pose.
        framework: ``"torch"`` / ``"onnx"`` / ``"jax"`` / ``"tflite"``. The
            adapter uses this to dispatch in :meth:`RLController._load_policy`.
        meta: Free-form metadata (e.g. training run id, commit hash) carried
            into ``ControlCommand.extras["rl"]`` for diagnostics.
    """

    ckpt_path: Path
    train_cfg_path: Optional[Path] = None
    framework: str = "torch"
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ObsBuilder(Protocol):
    """Build the obs vector for the policy.

    Implementations capture the EXACT obs layout from the upstream training
    repo: field order, units, frame, normalization, history stacking. They
    are stateful (history buffer, last_action latch) and must be reset at
    episode boundaries.
    """

    obs_dim: int

    def reset(self) -> None: ...

    def build(self, state: RobotState, intent: Intent) -> np.ndarray:
        """Return a single observation vector of shape ``(obs_dim,)``."""
        ...

    def remember_last_action(self, action: np.ndarray) -> None:
        """Latch the most recently executed action for the next obs build.

        Most sim2real RL policies feed the previous action back into the
        observation. The :class:`RLController` calls this on every act()
        after the action mapper produces its command.
        """
        ...


@runtime_checkable
class ActionMapper(Protocol):
    """Map raw policy output to a :class:`ControlCommand`.

    Owns DoF permutation (policy order ↔ robot order), default-pose offset,
    action scaling, and clipping. Stateless from the controller's
    perspective — all internal data is set at construction.
    """

    action_dim: int

    def map(self, action: np.ndarray) -> ControlCommand: ...


# ---------------------------------------------------------------------------
# Inference function abstraction
# ---------------------------------------------------------------------------


#: An inference closure: takes a single obs vector ``(obs_dim,)`` and returns
#: a single action vector ``(action_dim,)``. The framework dispatches via
#: :class:`PolicyArtifact.framework`; for tests / passthrough adapters, you
#: can construct an :class:`RLController` directly with an inline callable.
InferenceFn = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class RLController:
    """Generic RL controller. Subclass to bind a specific policy stack.

    Handles the cross-cutting concerns:

    * **Decimation** — policies typically run at 50 Hz while control runs
      at 200 Hz. The controller latches the last command between policy
      ticks so the underlying PD loop is fed every step.
    * **Obs history** — delegated to :class:`ObsBuilder` (which owns the
      buffer) but the controller signals reset boundaries.
    * **Last-action feedback** — after every policy call, the produced
      action is fed back into :meth:`ObsBuilder.remember_last_action`.
    * **Telemetry** — every :class:`ControlCommand` carries a
      ``"rl"`` extras dict with the policy method, last obs/action, and
      step counter.

    Args:
        spec: Robot spec (typically a :class:`G1RobotSpec`).
        artifact: :class:`PolicyArtifact`. Required if ``inference_fn`` is
            not supplied (the framework will load weights from disk).
        obs_builder: :class:`ObsBuilder` instance.
        action_mapper: :class:`ActionMapper` instance.
        policy_hz: Frequency at which the policy is queried. Below the
            control loop frequency the controller decimates with last-
            command latching.
        control_hz: Frequency of the outer pipeline / IO loop.
        runtime: Tag matching the inference framework (``"torch"`` /
            ``"jax"`` / ``"numpy"``). Validated by the pipeline against
            the active IO's ``array_runtime``.
        inference_fn: Optional pre-built inference closure. When provided,
            ``artifact`` is not loaded — useful for tests and for adapters
            that want full control over policy construction.
    """

    produces: Tuple[str, ...] = ("joint_pos",)

    def __init__(
        self,
        spec: Any,
        *,
        obs_builder: ObsBuilder,
        action_mapper: ActionMapper,
        policy_hz: float,
        control_hz: float,
        runtime: str = "torch",
        artifact: Optional[PolicyArtifact] = None,
        inference_fn: Optional[InferenceFn] = None,
    ) -> None:
        if artifact is None and inference_fn is None:
            raise ValueError("RLController requires either an artifact or an inference_fn")

        self.spec = spec
        self.runtime = runtime
        self.obs_builder = obs_builder
        self.action_mapper = action_mapper
        self.artifact = artifact
        self._infer = inference_fn or self._load_policy(artifact)

        if policy_hz <= 0 or control_hz <= 0:
            raise ValueError("policy_hz and control_hz must be positive")
        self.policy_hz = float(policy_hz)
        self.control_hz = float(control_hz)
        self._decimation = max(1, int(round(self.control_hz / self.policy_hz)))

        self._tick: int = 0
        self._last_cmd: Optional[ControlCommand] = None
        self._last_obs: Optional[np.ndarray] = None
        self._last_action: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Controller protocol
    # ------------------------------------------------------------------

    def reset(self, io: Optional[RobotIO] = None) -> None:
        if io is not None:
            self.spec = io.spec
        self.obs_builder.reset()
        self._tick = 0
        self._last_cmd = None
        self._last_obs = None
        self._last_action = None

    def act(self, state: RobotState, intent: Intent) -> ControlCommand:
        if self._tick % self._decimation == 0 or self._last_cmd is None:
            obs = self.obs_builder.build(state, intent)
            action = self._call_inference(obs)
            cmd = self.action_mapper.map(action)
            cmd = self._attach_telemetry(cmd, obs, action)
            self.obs_builder.remember_last_action(action)
            self._last_obs = obs
            self._last_action = action
            self._last_cmd = cmd
        self._tick += 1
        return self._last_cmd

    # ------------------------------------------------------------------
    # Inference dispatch
    # ------------------------------------------------------------------

    def _call_inference(self, obs: np.ndarray) -> np.ndarray:
        action = self._infer(obs)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_mapper.action_dim:
            raise ValueError(
                f"Inference returned action of shape {action.shape}, "
                f"expected ({self.action_mapper.action_dim},)"
            )
        return action

    def _load_policy(self, artifact: PolicyArtifact) -> InferenceFn:
        """Load weights from disk according to ``artifact.framework``.

        Subclasses may override to add custom framework support; the default
        handles ``torch`` (jit), ``onnx`` (onnxruntime) and ``jax`` (pickle).
        """
        if artifact is None:
            raise ValueError("PolicyArtifact required to load policy weights")

        framework = artifact.framework.lower()
        ckpt = Path(artifact.ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {ckpt}")

        if framework == "torch":
            return _make_torch_inference(ckpt)
        if framework == "onnx":
            return _make_onnx_inference(ckpt)
        if framework == "jax":
            return _make_jax_inference(ckpt)
        raise ValueError(f"Unsupported policy framework: {artifact.framework!r}")

    def _attach_telemetry(
        self,
        cmd: ControlCommand,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> ControlCommand:
        """Annotate ``cmd.extras["rl"]`` with the most recent inference state."""
        rl_extras = {
            "obs": obs,
            "action": action,
            "tick": self._tick,
            "decimation": self._decimation,
            "policy_hz": self.policy_hz,
            "framework": self.artifact.framework if self.artifact else "callable",
        }
        new_extras = dict(cmd.extras)
        new_extras["rl"] = rl_extras
        cmd.extras = new_extras
        return cmd


# ---------------------------------------------------------------------------
# Framework-specific inference loaders
# ---------------------------------------------------------------------------


def _make_torch_inference(ckpt: Path) -> InferenceFn:
    import torch  # local import — torch is heavy

    model = torch.jit.load(str(ckpt))
    model.eval()

    @torch.no_grad()
    def infer(obs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        y = model(x)
        return y.squeeze(0).cpu().numpy()

    return infer


def _make_onnx_inference(ckpt: Path) -> InferenceFn:
    import onnxruntime as ort  # local import

    session = ort.InferenceSession(str(ckpt))
    input_name = session.get_inputs()[0].name

    def infer(obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        y = session.run(None, {input_name: x})[0]
        return np.asarray(y).reshape(-1)

    return infer


def _make_jax_inference(ckpt: Path) -> InferenceFn:
    import pickle

    import jax  # local import

    with open(ckpt, "rb") as f:
        params = pickle.load(f)
    fn = params if callable(params) else None
    if fn is None:
        raise ValueError(
            "JAX policy checkpoint must be a callable; pickled params alone "
            "are not enough — wrap them in a closure before saving."
        )
    fn_jit = jax.jit(fn)

    def infer(obs: np.ndarray) -> np.ndarray:
        return np.asarray(fn_jit(np.asarray(obs, dtype=np.float32)))

    return infer
