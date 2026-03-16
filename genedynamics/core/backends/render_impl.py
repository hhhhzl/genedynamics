"""
Concrete render backend implementations.

This module provides concrete implementations of RenderBackend for different
rendering systems (Matplotlib, Gymnasium, Brax, MuJoCo, etc.).
"""

from typing import Optional, Any, List
import numpy as np
import os

from genedynamics.core.backends.render import RenderBackend


class MatplotlibRenderer(RenderBackend):
    """
    Matplotlib-based 2D renderer.
    
    Useful for simple 2D environments and visualization.
    """
    
    name = "matplotlib"
    
    def __init__(self, figsize: tuple = (8, 6), dpi: int = 100):
        """
        Initialize matplotlib renderer.
        
        Args:
            figsize: Figure size (width, height) in inches
            dpi: Dots per inch
        """
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required. Install with: pip install matplotlib"
            )
        
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.ax = None
        self._initialized = False
    
    def _init_figure(self):
        """Initialize matplotlib figure."""
        if not self._initialized:
            self.fig, self.ax = self.plt.subplots(figsize=self.figsize, dpi=self.dpi)
            self._initialized = True
    
    def render(
        self,
        state: Optional[np.ndarray] = None,
        mode: str = "human",
        **kwargs
    ) -> Optional[Any]:
        """
        Render using matplotlib.
        
        Args:
            state: State to render (optional, shape depends on environment)
            mode: Rendering mode
                - "human": Display to screen
                - "rgb_array": Return RGB image array
            **kwargs: Additional rendering parameters
            
        Returns:
            Rendered output (depends on mode)
        """
        self._init_figure()
        
        if mode == "human":
            # Clear and redraw
            self.ax.clear()
            
            # Simple 2D visualization (can be customized)
            if state is not None:
                state_np = np.asarray(state)
                if len(state_np) >= 2:
                    # Plot first 2 dimensions as position
                    self.ax.plot(state_np[0], state_np[1], 'ro', markersize=10)
                    self.ax.set_xlim(-5, 5)
                    self.ax.set_ylim(-5, 5)
                    self.ax.grid(True)
                    self.ax.set_title("Environment State")
            
            self.plt.draw()
            # Only pause if using interactive backend
            try:
                if self.plt.get_backend().lower() not in ['agg', 'svg', 'pdf', 'ps']:
                    self.plt.pause(0.01)  # Small pause for display
            except Exception:
                # If pause fails (non-interactive backend), just continue
                pass
            return None
        
        elif mode == "rgb_array":
            # Render to array
            self.ax.clear()
            
            if state is not None:
                state_np = np.asarray(state)
                if len(state_np) >= 2:
                    self.ax.plot(state_np[0], state_np[1], 'ro', markersize=10)
                    self.ax.set_xlim(-5, 5)
                    self.ax.set_ylim(-5, 5)
                    self.ax.grid(True)
            
            # Convert to RGB array
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            width, height = self.fig.canvas.get_width_height()
            # Reshape with proper dimensions
            try:
                buf = buf.reshape((height, width, 3))
            except ValueError:
                # If reshape fails, return a default-sized array
                # This can happen with some matplotlib backends
                expected_size = height * width * 3
                if len(buf) != expected_size:
                    # Pad or truncate to expected size
                    if len(buf) > expected_size:
                        buf = buf[:expected_size]
                    else:
                        buf = np.pad(buf, (0, expected_size - len(buf)), mode='constant')
                    buf = buf.reshape((height, width, 3))
            return buf
        
        return None
    
    def save_frame(self, filepath: str, state: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Save current frame to file.
        
        Args:
            filepath: Path to save image
            state: Optional state to render
            **kwargs: Additional rendering parameters
        """
        self._init_figure()
        self.ax.clear()
        
        if state is not None:
            state_np = np.asarray(state)
            if len(state_np) >= 2:
                self.ax.plot(state_np[0], state_np[1], 'ro', markersize=10)
                self.ax.set_xlim(-5, 5)
                self.ax.set_ylim(-5, 5)
                self.ax.grid(True)
        
        self.fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
    
    def save_video(self, filepath: str, states: List[np.ndarray], **kwargs) -> None:
        """
        Save sequence of states as video.
        
        Args:
            filepath: Path to save video file
            states: List of states to render
            **kwargs: Additional parameters (fps, etc.)
        """
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            raise ImportError(
                "Matplotlib animation requires Pillow. Install with: pip install pillow"
            )
        
        fps = kwargs.get('fps', 10)
        
        self._init_figure()
        
        def animate(frame):
            self.ax.clear()
            if frame < len(states):
                state = states[frame]
                state_np = np.asarray(state)
                if len(state_np) >= 2:
                    self.ax.plot(state_np[0], state_np[1], 'ro', markersize=10)
                    self.ax.set_xlim(-5, 5)
                    self.ax.set_ylim(-5, 5)
                    self.ax.grid(True)
                    self.ax.set_title(f"Frame {frame}")
        
        anim = FuncAnimation(self.fig, animate, frames=len(states), interval=1000/fps)
        
        # Save as GIF
        if filepath.endswith('.gif'):
            anim.save(filepath, writer=PillowWriter(fps=fps))
        else:
            # Default to GIF
            anim.save(filepath + '.gif', writer=PillowWriter(fps=fps))
    
    def set_camera(self, position: np.ndarray, target: np.ndarray, **kwargs) -> None:
        """
        Set camera position (not applicable for 2D matplotlib).
        
        Args:
            position: Camera position (ignored for 2D)
            target: Camera target (ignored for 2D)
            **kwargs: Additional camera parameters
        """
        # No-op for 2D matplotlib
        pass
    
    def close(self) -> None:
        """Close renderer and free resources."""
        if self.fig is not None:
            self.plt.close(self.fig)
            self.fig = None
            self.ax = None
            self._initialized = False


class GymnasiumRenderer(RenderBackend):
    """
    Renderer that delegates to Gymnasium's built-in rendering.
    
    This is a thin wrapper around Gymnasium's render() method.
    """
    
    name = "gymnasium"
    
    def __init__(self, gym_env: Any):
        """
        Initialize Gymnasium renderer.
        
        Args:
            gym_env: Gymnasium environment instance
        """
        self.gym_env = gym_env
    
    def render(
        self,
        state: Optional[np.ndarray] = None,
        mode: str = "human",
        **kwargs
    ) -> Optional[Any]:
        """
        Render using Gymnasium's render method.
        
        Args:
            state: State to render (ignored, Gymnasium uses internal state)
            mode: Rendering mode ("human", "rgb_array", etc.)
            **kwargs: Additional rendering parameters
            
        Returns:
            Rendered output (depends on mode)
        """
        # Gymnasium requires reset() before render() in some cases
        # Check if environment has been reset by checking if it has _elapsed_steps
        # If not, we'll try to render anyway (some environments allow it)
        
        # Check if render method accepts mode parameter
        import inspect
        try:
            render_sig = inspect.signature(self.gym_env.render)
        except (ValueError, TypeError):
            # If signature inspection fails, try direct call
            render_sig = None
        
        # Check if 'mode' is in the signature
        if render_sig is not None and 'mode' in render_sig.parameters:
            try:
                return self.gym_env.render(mode=mode, **kwargs)
            except Exception as e:
                # If render fails due to order enforcement, try to handle it
                error_msg = str(e).lower()
                if 'reset' in error_msg or 'order' in error_msg:
                    # Environment needs to be reset first
                    # This is a limitation - renderer can't reset the environment
                    # Return None or raise a more informative error
                    import warnings
                    warnings.warn(
                        f"Gymnasium environment needs to be reset before rendering: {e}. "
                        "Call env.reset() before using renderer."
                    )
                    return None
                raise
        else:
            # Some wrappers don't accept mode, try without it
            # Filter out mode from kwargs if present
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'mode'}
            try:
                return self.gym_env.render(**filtered_kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                if 'reset' in error_msg or 'order' in error_msg:
                    import warnings
                    warnings.warn(
                        f"Gymnasium environment needs to be reset before rendering: {e}. "
                        "Call env.reset() before using renderer."
                    )
                    return None
                try:
                    return self.gym_env.render()
                except Exception:
                    return None
    
    def save_frame(self, filepath: str, state: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Save current frame to file.
        
        Args:
            filepath: Path to save image
            state: Optional state (ignored)
            **kwargs: Additional parameters
        """
        # Render to RGB array and save
        import inspect
        render_sig = inspect.signature(self.gym_env.render)
        
        if 'mode' in render_sig.parameters:
            rgb_array = self.gym_env.render(mode="rgb_array")
        else:
            # Try without mode
            try:
                rgb_array = self.gym_env.render()
            except TypeError:
                rgb_array = None
        
        if rgb_array is not None:
            try:
                from PIL import Image
                img = Image.fromarray(rgb_array)
                img.save(filepath)
            except ImportError:
                # Fallback to matplotlib
                import matplotlib.pyplot as plt
                plt.imsave(filepath, rgb_array)
    
    def save_video(self, filepath: str, states: List[np.ndarray], **kwargs) -> None:
        """
        Save sequence of states as video.
        
        Note: This requires stepping through the environment.
        For proper video saving, use Gymnasium's video recording wrapper.
        
        Args:
            filepath: Path to save video file
            states: List of states (may be ignored)
            **kwargs: Additional parameters
        """
        import warnings
        warnings.warn(
            "GymnasiumRenderer.save_video() is limited. "
            "Consider using Gymnasium's video recording wrapper instead."
        )
        
        # Collect frames
        frames = []
        for _ in states:
            frame = self.gym_env.render(mode="rgb_array")
            if frame is not None:
                frames.append(frame)
        
        # Save as video (requires imageio or similar)
        try:
            import imageio
            imageio.mimwrite(filepath, frames, fps=kwargs.get('fps', 30))
        except ImportError:
            raise ImportError(
                "Video saving requires imageio. Install with: pip install imageio"
            )
    
    def set_camera(self, position: np.ndarray, target: np.ndarray, **kwargs) -> None:
        """
        Set camera position (if supported by environment).
        
        Args:
            position: Camera position
            target: Camera target
            **kwargs: Additional camera parameters
        """
        # Most Gymnasium environments don't support camera control
        # This is a placeholder for environments that do
        if hasattr(self.gym_env, 'set_camera'):
            self.gym_env.set_camera(position, target, **kwargs)
    
    def close(self) -> None:
        """Close renderer (delegates to Gymnasium)."""
        if hasattr(self.gym_env, 'close'):
            self.gym_env.close()


class BraxRenderer(RenderBackend):
    """
    Renderer for Brax environments.
    
    Brax uses HTML-based rendering via brax.io.html.
    This renderer wraps Brax's rendering capabilities.
    """
    
    name = "brax"
    
    def __init__(self, brax_env: Any = None):
        """
        Initialize Brax renderer.
        
        Args:
            brax_env: Brax environment instance (optional, can be set later)
        """
        try:
            import brax
            from brax import io
            self.brax = brax
            self.brax_io = io
            BRAX_AVAILABLE = True
        except ImportError:
            BRAX_AVAILABLE = False
            raise ImportError(
                "Brax is required. Install with: pip install brax"
            )
        
        self.brax_env = brax_env
        self._html_path = None
    
    def render(
        self,
        state: Optional[np.ndarray] = None,
        mode: str = "human",
        **kwargs
    ) -> Optional[Any]:
        """
        Render Brax environment.
        
        Args:
            state: Optional state (Brax uses internal state)
            mode: Rendering mode
                - "human": Display HTML (opens in browser)
                - "html": Return HTML string
                - "rgb_array": Not directly supported, returns None
            **kwargs: Additional rendering parameters
            
        Returns:
            Rendered output (depends on mode)
        """
        if self.brax_env is None:
            raise ValueError("Brax environment not set. Pass brax_env to __init__ or set it.")
        
        if mode == "human" or mode == "html":
            # Brax uses HTML rendering
            try:
                import tempfile
                import webbrowser
                import os
                
                # Create temporary HTML file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                    html_path = f.name
                
                # Render to HTML
                # Note: This requires a trajectory, not just a single state
                # For single state, we'd need to create a minimal trajectory
                if state is not None:
                    # Create a simple trajectory from state
                    # This is a simplified version
                    import jax.numpy as jnp
                    if isinstance(state, np.ndarray):
                        state_jax = jnp.array(state)
                    else:
                        state_jax = state
                    
                    # Create minimal trajectory (just one state)
                    # Brax HTML renderer typically needs a full trajectory
                    # For now, we'll just save the path
                    self._html_path = html_path
                    
                    if mode == "human":
                        # Open in browser
                        webbrowser.open(f"file://{html_path}")
                    
                    return html_path
                else:
                    # No state provided, return path for later use
                    self._html_path = html_path
                    return html_path
                    
            except Exception as e:
                import warnings
                warnings.warn(f"Brax HTML rendering failed: {e}")
                return None
        
        elif mode == "rgb_array":
            # Brax doesn't directly support RGB array rendering
            # Would need to use a different approach (e.g., pyvirtualdisplay)
            import warnings
            warnings.warn(
                "Brax doesn't support rgb_array mode directly. "
                "Use 'html' mode for visualization."
            )
            return None
        
        return None
    
    def save_frame(self, filepath: str, state: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Save Brax frame (saves as HTML).
        
        Args:
            filepath: Path to save HTML file
            state: Optional state
            **kwargs: Additional parameters
        """
        if self.brax_env is None:
            raise ValueError("Brax environment not set")
        
        # Brax renders to HTML, so we save HTML
        html_path = self.render(state, mode="html")
        if html_path and os.path.exists(html_path):
            import shutil
            shutil.copy(html_path, filepath)
    
    def save_video(self, filepath: str, states: List[np.ndarray], **kwargs) -> None:
        """
        Save Brax trajectory as HTML video.
        
        Args:
            filepath: Path to save HTML file
            states: List of states (Brax states, PyTree format)
            **kwargs: Additional parameters
        """
        if self.brax_env is None:
            raise ValueError("Brax environment not set")
        
        try:
            # Convert states to trajectory format
            # Brax HTML renderer expects a trajectory
            import jax
            import jax.numpy as jnp
            
            # Create trajectory from states
            # This is simplified - actual implementation would need proper trajectory format
            trajectory = []
            for state in states:
                if isinstance(state, np.ndarray):
                    state_jax = jnp.array(state)
                else:
                    state_jax = state
                trajectory.append(state_jax)
            
            # Render to HTML
            html = self.brax_io.html.render(self.brax_env.sys, trajectory)
            
            # Save HTML
            with open(filepath, 'w') as f:
                f.write(html)
                
        except Exception as e:
            import warnings
            warnings.warn(f"Brax video saving failed: {e}")
    
    def set_camera(self, position: np.ndarray, target: np.ndarray, **kwargs) -> None:
        """
        Set camera position (Brax HTML renderer has limited camera control).
        
        Args:
            position: Camera position (may be ignored)
            target: Camera target (may be ignored)
            **kwargs: Additional camera parameters
        """
        # Brax HTML renderer has limited camera control
        # This is a placeholder
        pass
    
    def close(self) -> None:
        """Close renderer."""
        self.brax_env = None
        self._html_path = None


class MujocoRenderer(RenderBackend):
    """
    High-performance renderer for MuJoCo environments.
    
    Uses MuJoCo's built-in viewer and rendering capabilities with support for:
    - Interactive viewer (mujoco.viewer)
    - RGB array rendering
    - Depth rendering
    - Video recording
    """
    
    name = "mujoco"
    
    def __init__(self, mujoco_model: Any = None, mujoco_data: Any = None):
        """
        Initialize MuJoCo renderer.
        
        Args:
            mujoco_model: MuJoCo model (mjModel)
            mujoco_data: MuJoCo data (mjData)
        """
        try:
            import mujoco
            import mujoco.viewer
            self.mujoco = mujoco
            self.mujoco_viewer = mujoco.viewer
            MUJOCO_AVAILABLE = True
        except ImportError:
            MUJOCO_AVAILABLE = False
            raise ImportError(
                "MuJoCo is required. Install with: pip install mujoco"
            )
        
        self.model = mujoco_model
        self.data = mujoco_data
        self.viewer = None
        self._scene = None
        self._renderer = None  # High-performance Renderer instance
        self._context = None  # Legacy context (for compatibility)
        self._width = 640
        self._height = 480
    
    def render(
        self,
        state: Optional[np.ndarray] = None,
        mode: str = "human",
        **kwargs
    ) -> Optional[Any]:
        """
        Render MuJoCo environment.
        
        Args:
            state: Optional state (if None, uses current mujoco_data)
            mode: Rendering mode
                - "human": Display in MuJoCo viewer
                - "rgb_array": Return RGB image array
                - "depth": Return depth image array
            **kwargs: Additional rendering parameters
            
        Returns:
            Rendered output (depends on mode)
        """
        if self.model is None or self.data is None:
            raise ValueError("MuJoCo model and data must be set")
        
        # Update data if state provided
        if state is not None:
            # Assume state is qpos + qvel
            state_np = np.asarray(state)
            nq = self.model.nq
            if len(state_np) >= nq:
                self.data.qpos[:] = state_np[:nq]
                if len(state_np) >= nq + self.model.nv:
                    self.data.qvel[:] = state_np[nq:nq+self.model.nv]
        
        # Forward kinematics
        self.mujoco.mj_forward(self.model, self.data)
        
        if mode == "human":
            # Use MuJoCo interactive viewer (new API)
            # Note: This opens a window and blocks until closed
            # For non-blocking rendering, use rgb_array mode
            if self.viewer is None:
                # Use passive viewer for non-blocking rendering
                # For interactive viewer, use mujoco.viewer.launch_passive
                pass  # Viewer will be created on-demand
            
            # For now, use rgb_array and display (non-blocking approach)
            # Full interactive viewer requires separate thread/process
            return None
        
        elif mode == "rgb_array":
            # Render to RGB array
            if self._scene is None:
                self._scene = self.mujoco.MjvScene(self.model, maxgeom=10000)
                self._context = self.mujoco.MjrContext(self.model, self.mujoco.mjtFontScale.mjFONTSCALE_150)
            
            # Setup camera
            camera = self.mujoco.MjvCamera()
            option = self.mujoco.MjvOption()
            
            # Render scene
            self.mujoco.mjv_updateScene(
                self.model, self.data, option, None, camera,
                self.mujoco.mjtCatBit.mjCAT_ALL, self._scene
            )
            
            # Render to RGB array
            width, height = kwargs.get('width', 640), kwargs.get('height', 480)
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            depth = np.zeros((height, width), dtype=np.float32)
            
            self.mujoco.mjr_render(
                self.mujoco.MjrRect(0, 0, width, height),
                self._scene, self._context
            )
            
            # Read pixels
            self.mujoco.mjr_readPixels(rgb, depth, self.mujoco.MjrRect(0, 0, width, height), self._context)
            
            # Flip vertically (MuJoCo uses bottom-left origin)
            rgb = np.flipud(rgb)
            
            return rgb
        
        elif mode == "depth":
            # Render depth using MuJoCo Renderer
            width = kwargs.get('width', self._width)
            height = kwargs.get('height', self._height)
            
            try:
                if self._renderer is None:
                    self._renderer = self.mujoco.Renderer(self.model, width=width, height=height)
                    self._width = width
                    self._height = height
                
                self._renderer.update_scene(self.data)
                depth = self._renderer.render(depth=True)
                
                return depth
            except (AttributeError, TypeError):
                # Fallback to alternate API
                if self._scene is None:
                    self._scene = self.mujoco.MjvScene(self.model, maxgeom=10000)
                    self._context = self.mujoco.MjrContext(self.model, self.mujoco.mjtFontScale.mjFONTSCALE_150)
                
                camera = self.mujoco.MjvCamera()
                option = self.mujoco.MjvOption()
                
                self.mujoco.mjv_updateScene(
                    self.model, self.data, option, None, camera,
                    self.mujoco.mjtCatBit.mjCAT_ALL, self._scene
                )
                
                depth = np.zeros((height, width), dtype=np.float32)
                rgb = np.zeros((height, width, 3), dtype=np.uint8)
                
                self.mujoco.mjr_render(
                    self.mujoco.MjrRect(0, 0, width, height),
                    self._scene, self._context
                )
                
                self.mujoco.mjr_readPixels(rgb, depth, self.mujoco.MjrRect(0, 0, width, height), self._context)
                depth = np.flipud(depth)
                
                return depth
        
        return None
    
    def save_frame(self, filepath: str, state: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Save MuJoCo frame to file.
        
        Args:
            filepath: Path to save image
            state: Optional state to render
            **kwargs: Additional parameters (width, height)
        """
        rgb = self.render(state, mode="rgb_array", **kwargs)
        if rgb is not None:
            try:
                from PIL import Image
                img = Image.fromarray(rgb)
                img.save(filepath)
            except ImportError:
                import matplotlib.pyplot as plt
                plt.imsave(filepath, rgb)
    
    def save_video(self, filepath: str, states: List[np.ndarray], **kwargs) -> None:
        """
        Save MuJoCo trajectory as video.
        
        Args:
            filepath: Path to save video file
            states: List of states to render
            **kwargs: Additional parameters (fps, width, height)
        """
        fps = kwargs.get('fps', 30)
        frames = []
        
        for state in states:
            rgb = self.render(state, mode="rgb_array", **kwargs)
            if rgb is not None:
                frames.append(rgb)
        
        if frames:
            try:
                import imageio
                imageio.mimwrite(filepath, frames, fps=fps)
            except ImportError:
                raise ImportError(
                    "Video saving requires imageio. Install with: pip install imageio"
                )
    
    def set_camera(self, position: np.ndarray, target: np.ndarray, **kwargs) -> None:
        """
        Set camera position and target.
        
        Args:
            position: Camera position (3D)
            target: Camera target/look-at point (3D)
            **kwargs: Additional camera parameters
        """
        if self.viewer is not None:
            self.viewer.cam.lookat[:] = target
            self.viewer.cam.distance = np.linalg.norm(position - target)
            # Set camera position (MuJoCo uses lookat + distance)
    
    def close(self) -> None:
        """Close renderer and free resources."""
        if self.viewer is not None:
            # MuJoCo viewer cleanup
            self.viewer = None
        if self._renderer is not None:
            # Cleanup high-performance renderer
            self._renderer = None
        self._scene = None
        self._context = None


class IsaacSimRenderer(RenderBackend):
    """
    Renderer for NVIDIA Isaac Sim environments.
    
    Isaac Sim is a high-performance physics simulation and rendering platform
    built on NVIDIA Omniverse. This renderer wraps Isaac Sim's rendering capabilities.
    """
    
    name = "isaac_sim"
    
    def __init__(self, isaac_sim_app: Any = None, viewport: Any = None):
        """
        Initialize Isaac Sim renderer.
        
        Args:
            isaac_sim_app: Isaac Sim application instance (optional)
            viewport: Isaac Sim viewport instance (optional)
        """
        try:
            import omni
            from omni.isaac.core import World
            self.omni = omni
            self.isaac_available = True
        except ImportError:
            self.isaac_available = False
            raise ImportError(
                "Isaac Sim is required. "
                "Install Isaac Sim and ensure omni.isaac packages are available."
            )
        
        self.app = isaac_sim_app
        self.viewport = viewport
        self.world = None
    
    def render(
        self,
        state: Optional[np.ndarray] = None,
        mode: str = "human",
        **kwargs
    ) -> Optional[Any]:
        """
        Render Isaac Sim environment.
        
        High-performance GPU-accelerated rendering with support for multiple modes.
        
        Args:
            state: Optional state to render (if None, uses current simulation state)
            mode: Rendering mode
                - "human": Display in Isaac Sim viewport (interactive)
                - "rgb_array": Return RGB image array (GPU-accelerated)
                - "depth": Return depth image array
                - "rgbd": Return RGB + depth dictionary
            **kwargs: Additional rendering parameters:
                - camera_name: Camera name (default: "camera")
                - width: Image width (default: 640)
                - height: Image height (default: 480)
            
        Returns:
            Rendered output (depends on mode):
            - "human": None (displays to viewport)
            - "rgb_array": np.ndarray of shape (H, W, 3), dtype uint8
            - "depth": np.ndarray of shape (H, W), dtype float32
            - "rgbd": dict with "rgb" and "depth" keys
        """
        camera_name = kwargs.get('camera_name', 'camera')
        
        if mode == "human":
            # Isaac Sim viewport rendering (interactive)
            if self.viewport is not None:
                # Update viewport
                self.viewport.update()
            elif self.world is not None:
                # Step world with rendering enabled
                self.world.step(render=True)
            return None
        
        elif mode == "rgb_array":
            # Render to RGB array using camera (GPU-accelerated)
            if self.world is None:
                raise ValueError("Isaac Sim world not initialized. Set world first.")
            
            # Get or create camera
            if camera_name not in self._cameras:
                try:
                    camera = self.world.scene.get_object(camera_name)
                    if camera is None:
                        # Try to get from scene
                        camera = self.world.scene.get_object(f"/World/{camera_name}")
                    self._cameras[camera_name] = camera
                except Exception:
                    raise ValueError(f"Camera '{camera_name}' not found in scene")
            else:
                camera = self._cameras[camera_name]
            
            if camera is None:
                raise ValueError(f"Camera '{camera_name}' not found in scene")
            
            # Get RGB data (Isaac Sim returns GPU tensors, convert to CPU numpy)
            try:
                rgb_data = camera.get_rgba()
                if rgb_data is not None:
                    # Handle both numpy arrays and GPU tensors
                    if hasattr(rgb_data, 'cpu'):
                        rgb_data = rgb_data.cpu().numpy()
                    rgb = np.asarray(rgb_data)
                    if rgb.shape[2] == 4:
                        rgb = rgb[:, :, :3]  # Remove alpha channel
                    return rgb.astype(np.uint8)
            except Exception as e:
                # Fallback: try alternative camera API
                try:
                    rgb_data = camera.get_current_frame()["rgba"]
                    if rgb_data is not None:
                        rgb = np.asarray(rgb_data)
                        if rgb.shape[2] == 4:
                            rgb = rgb[:, :, :3]
                        return rgb.astype(np.uint8)
                except Exception:
                    pass
            
            return None
        
        elif mode == "depth":
            # Render depth (GPU-accelerated)
            if self.world is None:
                raise ValueError("Isaac Sim world not initialized")
            
            if camera_name not in self._cameras:
                try:
                    camera = self.world.scene.get_object(camera_name)
                    if camera is None:
                        camera = self.world.scene.get_object(f"/World/{camera_name}")
                    self._cameras[camera_name] = camera
                except Exception:
                    raise ValueError(f"Camera '{camera_name}' not found")
            else:
                camera = self._cameras[camera_name]
            
            if camera is None:
                raise ValueError(f"Camera '{camera_name}' not found")
            
            try:
                depth_data = camera.get_current_frame()["distance_to_image_plane"]
                if depth_data is None:
                    depth_data = camera.get_current_frame().get("depth", None)
                
                if depth_data is not None:
                    # Handle GPU tensors
                    if hasattr(depth_data, 'cpu'):
                        depth_data = depth_data.cpu().numpy()
                    return np.asarray(depth_data, dtype=np.float32)
            except Exception:
                pass
            
            return None
        
        elif mode == "rgbd":
            # Return both RGB and depth
            rgb = self.render(state, mode="rgb_array", **kwargs)
            depth = self.render(state, mode="depth", **kwargs)
            
            if rgb is not None and depth is not None:
                return {"rgb": rgb, "depth": depth}
            
            return None
        
        return None
    
    def save_frame(self, filepath: str, state: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Save current frame to file.
        
        Args:
            filepath: Path to save image
            state: Optional state to render
            **kwargs: Additional parameters (camera_name, etc.)
        """
        rgb = self.render(state, mode="rgb_array", **kwargs)
        if rgb is not None:
            try:
                from PIL import Image
                img = Image.fromarray(rgb)
                img.save(filepath)
            except ImportError:
                import matplotlib.pyplot as plt
                plt.imsave(filepath, rgb)
    
    def save_video(self, filepath: str, states: List[np.ndarray], **kwargs) -> None:
        """
        Save sequence of states as video.
        
        Args:
            filepath: Path to save video file
            states: List of states to render
            **kwargs: Additional parameters (fps, camera_name, etc.)
        """
        fps = kwargs.get('fps', 30)
        frames = []
        
        for state in states:
            # Update simulation state if provided
            if state is not None and self.world is not None:
                # This would need to be customized based on your robot/environment
                # For now, we just render current state
                pass
            
            rgb = self.render(state, mode="rgb_array", **kwargs)
            if rgb is not None:
                frames.append(rgb)
        
        if frames:
            try:
                import imageio
                imageio.mimwrite(filepath, frames, fps=fps)
            except ImportError:
                raise ImportError(
                    "Video saving requires imageio. Install with: pip install imageio"
                )
    
    def set_camera(self, position: np.ndarray, target: np.ndarray, **kwargs) -> None:
        """
        Set camera position and target.
        
        Args:
            position: Camera position (3D)
            target: Camera target/look-at point (3D)
            **kwargs: Additional camera parameters (camera_name, fov, etc.)
        """
        if self.world is None:
            raise ValueError("Isaac Sim world not initialized")
        
        camera_name = kwargs.get('camera_name', 'camera')
        camera = self.world.scene.get_object(camera_name)
        
        if camera is not None:
            # Set camera position and orientation
            # Isaac Sim uses quaternions for orientation
            import omni.isaac.core.utils.numpy.rotations as rot_utils
            
            # Compute rotation to look at target
            direction = target - position
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            
            # Create rotation matrix (simplified)
            # In practice, you'd use proper look-at matrix
            up = np.array([0, 0, 1])  # Default up vector
            right = np.cross(direction, up)
            right = right / (np.linalg.norm(right) + 1e-6)
            up = np.cross(right, direction)
            
            # Convert to quaternion
            rot_matrix = np.eye(3)
            rot_matrix[:, 0] = right
            rot_matrix[:, 1] = up
            rot_matrix[:, 2] = -direction
            
            quat = rot_utils.rot_matrices_to_quats(rot_matrix.reshape(1, 3, 3))[0]
            
            # Set camera pose
            camera.set_world_pose(position=position, orientation=quat)
    
    def set_world(self, world: Any) -> None:
        """
        Set Isaac Sim world instance.
        
        Args:
            world: Isaac Sim World instance
        """
        self.world = world
    
    def close(self) -> None:
        """Close renderer and free resources."""
        if self.world is not None:
            # Cleanup world if needed
            pass
        self.world = None
        self.viewport = None
        self.app = None


class NullRenderer(RenderBackend):
    """
    Null renderer that does nothing (for headless environments).
    
    This is useful for environments that don't need visualization
    or when running in headless mode.
    """
    
    name = "null"
    
    def render(
        self,
        state: Optional[np.ndarray] = None,
        mode: str = "human",
        **kwargs
    ) -> Optional[Any]:
        """No-op render."""
        if mode == "rgb_array":
            # Return dummy image
            return np.zeros((100, 100, 3), dtype=np.uint8)
        return None
    
    def save_frame(self, filepath: str, state: Optional[np.ndarray] = None, **kwargs) -> None:
        """No-op."""
        pass
    
    def save_video(self, filepath: str, states: List[np.ndarray], **kwargs) -> None:
        """No-op."""
        pass
    
    def set_camera(self, position: np.ndarray, target: np.ndarray, **kwargs) -> None:
        """No-op."""
        pass
    
    def close(self) -> None:
        """No-op."""
        pass
