"""Clipped PPO updates; application collectors own transition/reset semantics."""

import torch


def generalized_advantage(
    rewards, values, next_values, terminated, ended, *, discount=0.99, trace_decay=0.95
):
    """Timeouts bootstrap the pre-reset terminal value; recursion stops at every reset."""
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[-1])
    for index in reversed(range(len(rewards))):
        delta = (
            rewards[index]
            + discount * (~terminated[index]).to(values.dtype) * next_values[index]
            - values[index]
        )
        carry = (
            delta + discount * trace_decay * (~ended[index]).to(values.dtype) * carry
        )
        advantage[index] = carry
    return advantage, advantage + values


class PPOTrainer:
    def __init__(
        self, model, *, learning_rate=3e-4, clip_ratio=0.2, entropy_weight=0.0
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.clip_ratio, self.entropy_weight, self.updates = (
            clip_ratio,
            entropy_weight,
            0,
        )

    def update(
        self,
        features,
        raw_actions,
        old_log_probability,
        returns,
        advantages,
        *,
        epochs=4,
        minibatch=64,
    ):
        if min(epochs, minibatch, len(features)) < 1:
            raise ValueError("PPO requires nonempty data and positive update budgets")
        normalized = (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp(min=1e-6)
        losses = []
        for _ in range(epochs):
            for indices in torch.randperm(len(features), device=features.device).split(
                minibatch
            ):
                distribution, value = self.model.distribution(features[indices])
                log_probability = distribution.log_prob(raw_actions[indices]).sum(-1)
                ratio = (log_probability - old_log_probability[indices]).exp()
                actor_loss = -torch.minimum(
                    ratio * normalized[indices],
                    ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio)
                    * normalized[indices],
                ).mean()
                critic_loss = ((value - returns[indices]) ** 2).mean()
                loss = (
                    actor_loss
                    + 0.5 * critic_loss
                    - self.entropy_weight * distribution.entropy().sum(-1).mean()
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                losses.append(float(loss.detach()))
        self.updates += 1
        return {"loss": sum(losses) / len(losses), "updates": self.updates}
