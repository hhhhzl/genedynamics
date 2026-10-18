from dataclasses import dataclass


@dataclass(frozen=True)
class Capabilities:
    backend: str
    device: str
    physics_backend: str
    operations: frozenset
    compiled_chunk: bool = False
    device_resident: bool = False
    autodiff_inputs: bool = False
    autodiff_parameters: bool = False

    def require(self, operation):
        if operation not in self.operations:
            raise NotImplementedError(f"{self.backend}/{self.device} has no {operation} capability")
