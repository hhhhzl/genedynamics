"""Torch derivatives preserve the model/controller input graph."""

import torch


class TorchRuntime:
    name = "torch"
    autodiff = True

    def __init__(self, device="cpu", dtype="float32"):
        self.device, self.dtype, self.xp = (
            torch.device(device),
            getattr(torch, dtype),
            torch,
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

    def array(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def to_numpy(self, value):
        return value.detach().cpu().numpy()

    def scalar(self, value):
        return float(value.detach())

    def copy(self, value):
        return value.detach().clone()

    def clip(self, value, lower, upper):
        return torch.clamp(value, min=lower, max=upper)

    def eye(self, size, like):
        return torch.eye(size, dtype=like.dtype, device=like.device)

    def normal(self, shape, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(
            shape, dtype=self.dtype, device=self.device, generator=generator
        )

    def value_and_grad(self, function, value):
        argument = value.detach().clone().requires_grad_(True)
        result = function(argument)
        (gradient,) = torch.autograd.grad(result, argument)
        return result.detach(), gradient.detach()

    def jacobian(self, function, value):
        return torch.autograd.functional.jacobian(
            function, value, vectorize=True
        ).detach()
