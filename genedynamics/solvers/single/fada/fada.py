"""FADA closed-loop policy frontend; training and adaptation are separate APIs."""

from __future__ import annotations

from collections.abc import Mapping


class FADASolver:
    """Execute the original Planner-IDM and optionally an adapted IDM.

    ``act`` expects public, right-aligned ``history_obs``, ``history_act`` (previous
    actions aligned with each observation), ``current_command`` and
    ``history_valid_mask``. Tensor dimensions and physical units are task-owned;
    an arm/mobile adapter is an adapted baseline, not native humanoid reproduction.
    ``predict`` returns the complete action horizon; ``act`` executes a prefix.
    """

    def __init__(self, model, *, open_loop_deploy_steps=1, **backend_options):
        from .backends.fada_torch import FADABackendTorch

        self.backend = FADABackendTorch(model, **backend_options)
        self.open_loop_deploy_steps = int(open_loop_deploy_steps)
        if not 1 <= self.open_loop_deploy_steps <= model.pred_horizon:
            raise ValueError("open_loop_deploy_steps must lie in [1, pred_horizon]")
        self.reset()

    @property
    def model(self):
        return self.backend.model

    @property
    def metadata(self):
        return self.backend.metadata

    def predict(self, history_obs, history_act, current_command, *, history_valid_mask, return_obs=False):
        return self.backend.predict(
            history_obs, history_act, current_command,
            history_valid_mask=history_valid_mask, return_obs=return_obs,
        )

    def reset(self):
        """Discard cached open-loop actions at each episode or batch reset."""
        self._actions = None
        self._index = 0

    def act(self, public_observation):
        if not isinstance(public_observation, Mapping):
            raise TypeError("FADA act requires an explicit public history mapping")
        if self._actions is None or self._index >= self.open_loop_deploy_steps:
            self._actions = self.predict(
                public_observation["history_obs"], public_observation["history_act"],
                public_observation["current_command"],
                history_valid_mask=public_observation["history_valid_mask"],
            )
            self._index = 0
        action = self._actions[:, self._index, :]
        self._index += 1
        return action

    @classmethod
    def from_checkpoint(cls, payload, *, device="cpu", metadata=None, open_loop_deploy_steps=1,
                        task_action_binding=False):
        from .backends.fada_torch import FADABackendTorch

        adapter = payload.get("genedynamics_task_adapter", {}) if isinstance(payload, Mapping) else {}
        decoder = adapter.get("action_decoder", {"kind": "continuous13D"})
        if decoder.get("kind", "continuous13D") != "continuous13D" and not task_action_binding:
            raise ValueError("FADA checkpoint requires its policy-owned task action decoder; load with SourceTaskPolicy.from_checkpoint(task=...) instead of an unbound solver")
        backend = FADABackendTorch.from_checkpoint(payload, device=device, metadata=metadata)
        instance = cls.__new__(cls)
        instance.backend = backend
        instance.open_loop_deploy_steps = int(open_loop_deploy_steps)
        if not 1 <= instance.open_loop_deploy_steps <= backend.model.pred_horizon:
            raise ValueError("open_loop_deploy_steps must lie in [1, pred_horizon]")
        instance.reset()
        return instance


__all__ = ["FADASolver"]
