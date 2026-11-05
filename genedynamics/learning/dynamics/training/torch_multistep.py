"""Masked autoregressive model fitting without task-specific model assumptions."""

import torch


class MultiStepModelTrainer:
    def __init__(self, model, step, error, *, learning_rate=2e-4, gradient_clip=1.0):
        self.model, self.step, self.error = model, step, error
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.gradient_clip = gradient_clip
        self.updates = 0

    def loss(self, states, next_states, actions, contexts, mask):
        if states.shape[:2] != mask.shape or mask.sum() <= 0:
            raise ValueError(
                "Multistep loss requires aligned states and a nonempty loss mask"
            )
        current = states[:, 0]
        losses = []
        for index in range(actions.shape[1]):
            current = self.step(current, actions[:, index], contexts[:, index])
            losses.append(self.error(current, next_states[:, index]))
        error = torch.stack(losses, dim=1)
        return (error * mask).sum() / mask.sum()

    def update(self, states, next_states, actions, contexts, mask):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(states, next_states, actions, contexts, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("Model loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Model gradient is non-finite")
        self.optimizer.step()
        self.updates += 1
        return {
            "loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm),
            "updates": self.updates,
        }
