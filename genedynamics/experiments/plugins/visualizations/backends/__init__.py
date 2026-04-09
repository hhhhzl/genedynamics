"""
Renderer backends for trajectory visualizations.

Each backend renders a `SceneIR` + list of `RobotPoseIR` to a matplotlib
figure (or other surface). New backends slot in here without touching
the visualization plugins or the env classes.

Currently available:
  * matplotlib_top : top-down 2D, byte-equivalent to the original draw
                     code. Default.

Planned:
  * matplotlib_iso : isometric pseudo-3D in matplotlib (no new deps).
  * pyrender_mesh  : true 3D using G1 STL meshes from mujoco_menagerie.
"""
