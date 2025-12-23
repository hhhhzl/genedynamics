"""
PyTorch backend implementation with device support.

This backend provides PyTorch tensor operations and support
for neural network-based dynamics and energy models.
Supports CPU and GPU devices.
"""

from typing import Any, Callable, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class TorchBackend:
    """
    PyTorch backend implementation.
    
    This backend provides PyTorch tensor operations and support
    for neural network-based dynamics and energy models.
    Supports CPU and GPU devices.
    """
    
    name = "torch"
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize PyTorch backend.
        
        Args:
            device: Device to use ("cpu" or "gpu")
            
        Raises:
            ImportError: If PyTorch is not installed
            RuntimeError: If GPU is requested but not available
        """
        if not HAS_TORCH:
            raise ImportError(
                "Torch backend requires PyTorch to be installed. "
                "Install with: pip install torch"
            )
        
        self.torch = torch
        self.device = device
        
        # Set up device
        if device == "gpu":
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                self.device = "cpu"
                self.device_obj = torch.device("cpu")
            else:
                self.device_obj = torch.device("cuda")
        elif device == "webgpu":
            # WebGPU support would require additional setup
            # For now, fall back to CPU
            print("Warning: WebGPU not yet implemented, falling back to CPU")
            self.device = "cpu"
            self.device_obj = torch.device("cpu")
        else:
            self.device_obj = torch.device("cpu")
            self.device = "cpu"
    
    def tensor(self, x: Any, dtype: Optional[Any] = None) -> Any:
        """
        Convert input to PyTorch tensor.
        
        Args:
            x: Input data (numpy array, list, etc.)
            dtype: Optional data type
            
        Returns:
            PyTorch tensor on the configured device
        """
        if dtype is None:
            return torch.as_tensor(x, device=self.device_obj)
        return torch.as_tensor(x, dtype=dtype, device=self.device_obj)
    
    def concat(self, xs: list, axis: int = 0) -> Any:
        """
        Concatenate PyTorch tensors.
        
        Args:
            xs: List of tensors to concatenate
            axis: Axis along which to concatenate
            
        Returns:
            Concatenated tensor
        """
        return torch.cat(xs, dim=axis)
    
    def randn(self, shape: tuple, rng_key: Optional[Any] = None) -> Any:
        """
        Generate random normal samples using PyTorch.
        
        Args:
            shape: Shape of output tensor
            rng_key: Optional random generator or seed
            
        Returns:
            Random tensor on the configured device
        """
        if rng_key is not None:
            generator = torch.Generator(device=self.device_obj)
            if isinstance(rng_key, int):
                generator.manual_seed(rng_key)
            else:
                generator.manual_seed(hash(rng_key) % 2**32)
            return torch.randn(shape, generator=generator, device=self.device_obj)
        return torch.randn(shape, device=self.device_obj)
    
    def vmap(self, fn: Callable) -> Callable:
        """
        Vectorize function (PyTorch doesn't have vmap, use manual batching).
        
        Args:
            fn: Function to vectorize
            
        Returns:
            Same function (PyTorch requires manual batching)
        """
        # PyTorch doesn't have vmap, so we return the function as-is
        # Users should manually batch operations
        return fn
    
    def jit(self, fn: Callable) -> Callable:
        """
        JIT compile function using TorchScript.
        
        Args:
            fn: Function to compile
            
        Returns:
            Compiled function
        """
        return torch.jit.script(fn)
    
    def create_rng(self, seed: int) -> Any:
        """
        Create a random number generator from seed.
        
        Args:
            seed: Random seed
            
        Returns:
            PyTorch Generator
        """
        generator = torch.Generator(device=self.device_obj)
        generator.manual_seed(seed)
        return generator
    
    def split_rng(self, rng: Any, num: int = 2) -> Any:
        """
        Split RNG generator (creates new generators with different seeds).
        
        Args:
            rng: Random generator
            num: Number of generators to create
            
        Returns:
            Tuple of generators
        """
        # For PyTorch, we create new generators with sequential seeds
        generators = []
        if isinstance(rng, torch.Generator):
            # Get current seed and create new generators
            # Note: PyTorch generators don't expose seed directly,
            # so we create new ones with sequential seeds
            base_seed = hash(rng) % 2**32
            for i in range(num):
                gen = torch.Generator(device=self.device_obj)
                gen.manual_seed(base_seed + i)
                generators.append(gen)
        else:
            # If rng is an int seed, create generators from it
            base_seed = rng if isinstance(rng, int) else hash(rng) % 2**32
            for i in range(num):
                gen = torch.Generator(device=self.device_obj)
                gen.manual_seed(base_seed + i)
                generators.append(gen)
        
        return tuple(generators) if len(generators) > 1 else generators[0]
    
    def uniform(self, rng: Any, shape: tuple, minval: float = 0.0, maxval: float = 1.0) -> Any:
        """
        Generate uniform random samples.
        
        Args:
            rng: Random generator or seed
            shape: Shape of output tensor
            minval: Minimum value
            maxval: Maximum value
            
        Returns:
            Random tensor with uniform distribution
        """
        if isinstance(rng, torch.Generator):
            generator = rng
        else:
            generator = torch.Generator(device=self.device_obj)
            seed = rng if isinstance(rng, int) else hash(rng) % 2**32
            generator.manual_seed(seed)
        
        return torch.empty(shape, device=self.device_obj).uniform_(minval, maxval, generator=generator)
    
    def normal(self, rng: Any, shape: tuple, mean: float = 0.0, std: float = 1.0) -> Any:
        """
        Generate normal random samples.
        
        Args:
            rng: Random generator or seed
            shape: Shape of output tensor
            mean: Mean value
            std: Standard deviation
            
        Returns:
            Random tensor with normal distribution
        """
        if isinstance(rng, torch.Generator):
            generator = rng
        else:
            generator = torch.Generator(device=self.device_obj)
            seed = rng if isinstance(rng, int) else hash(rng) % 2**32
            generator.manual_seed(seed)
        
        return torch.normal(mean, std, size=shape, generator=generator, device=self.device_obj)
    
    def to_device(self, x: Any, device: str = None) -> Any:
        """
        Move tensor to specified device.
        
        Args:
            x: Tensor to move
            device: Target device ("cpu" or "gpu")
            
        Returns:
            Tensor on target device
        """
        if device is None:
            device = self.device
        
        if device == "gpu":
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            target_device = torch.device("cpu")
        
        if isinstance(x, torch.Tensor):
            return x.to(target_device)
        return x
    
    def get_device(self) -> str:
        """
        Get current device.
        
        Returns:
            Device name ("cpu" or "gpu")
        """
        return self.device
    
    def register_dynamics_kernel(self, key: str, fn: Callable) -> None:
        """
        Register a high-performance dynamics kernel (optional).
        
        Args:
            key: Unique identifier for the kernel
            fn: Kernel function
        """
        # PyTorch doesn't need explicit kernel registration
        pass
    
    def register_constraint_kernel(self, key: str, fn: Callable) -> None:
        """
        Register a high-performance constraint kernel (optional).
        
        Args:
            key: Unique identifier for the kernel
            fn: Kernel function
        """
        # PyTorch doesn't need explicit kernel registration
        pass

