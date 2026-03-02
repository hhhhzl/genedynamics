"""
Rendering backend protocol for visualization systems.

This module defines the RenderBackend protocol that rendering systems
(Matplotlib, MuJoCo viewer, Isaac Sim, etc.) should implement.
"""

from typing import Protocol, Optional, Any, runtime_checkable
import numpy as np


@runtime_checkable
class RenderBackend(Protocol):
    """
    Protocol for rendering backends.
    
    Rendering backends handle visualization of environments, including:
    - Real-time rendering (human mode)
    - Image/video capture (rgb_array mode)
    - Depth rendering (depth mode)
    - Offline rendering (save to file)
    
    Examples:
        - MatplotlibRenderer: 2D visualization using matplotlib
        - MujocoRenderer: MuJoCo's built-in viewer
        - IsaacSimRenderer: NVIDIA Isaac Sim rendering
        - PyBulletRenderer: PyBullet's GUI
        - RustRenderer: Custom Rust rendering (via FFI)
    """
    
    name: str  # "matplotlib", "mujoco", "isaac", "pybullet", "rust", etc.
    
    def render(
        self,
        state: Optional[np.ndarray] = None,
        mode: str = "human",
        **kwargs
    ) -> Optional[Any]:
        """
        Render the environment.
        
        Args:
            state: Optional state to render (if None, uses current state)
            mode: Rendering mode
                - "human": Display to screen (interactive)
                - "rgb_array": Return RGB image array
                - "depth": Return depth image array
                - "rgbd": Return RGB + depth
            **kwargs: Additional rendering parameters
            
        Returns:
            Rendered output (depends on mode):
            - "human": None (displays to screen)
            - "rgb_array": np.ndarray of shape (H, W, 3)
            - "depth": np.ndarray of shape (H, W)
            - "rgbd": dict with "rgb" and "depth" keys
        """
        ...
    
    def save_frame(self, filepath: str, state: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Save current frame to file.
        
        Args:
            filepath: Path to save image/video frame
            state: Optional state to render
            **kwargs: Additional rendering parameters
        """
        ...
    
    def save_video(self, filepath: str, states: list, **kwargs) -> None:
        """
        Save sequence of states as video.
        
        Args:
            filepath: Path to save video file
            states: List of states to render
            **kwargs: Additional rendering parameters (fps, etc.)
        """
        ...
    
    def set_camera(self, position: np.ndarray, target: np.ndarray, **kwargs) -> None:
        """
        Set camera position and target.
        
        Args:
            position: Camera position (3D)
            target: Camera target/look-at point (3D)
            **kwargs: Additional camera parameters (fov, etc.)
        """
        ...
    
    def close(self) -> None:
        """Close renderer and free resources."""
        ...


class DummyRenderBackend:
    """
    Dummy render backend for testing or environments without rendering.
    
    This backend does nothing and is useful for:
    - Testing environments that don't need visualization
    - Headless environments
    - Development before integrating actual renderer
    """
    
    name = "dummy"
    
    def render(
        self,
        state: Optional[np.ndarray] = None,
        mode: str = "human",
        **kwargs
    ) -> Optional[Any]:
        """No-op."""
        if mode == "rgb_array":
            # Return dummy image
            return np.zeros((100, 100, 3), dtype=np.uint8)
        return None
    
    def save_frame(self, filepath: str, state: Optional[np.ndarray] = None, **kwargs) -> None:
        """No-op."""
        pass
    
    def save_video(self, filepath: str, states: list, **kwargs) -> None:
        """No-op."""
        pass
    
    def set_camera(self, position: np.ndarray, target: np.ndarray, **kwargs) -> None:
        """No-op."""
        pass
    
    def close(self) -> None:
        """No-op."""
        pass
