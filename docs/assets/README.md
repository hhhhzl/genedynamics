# Documentation media

## Motion wall

The README uses one composed animation instead of independently sized GIFs:

- `showcase.gif`: a looping six-panel motion wall, for GitHub and the docs.
- `showcase.mp4`: the 1440-pixel video export, without GIF palette limitations.
- `showcase-poster.png`: a representative still from the same composition.
- `architecture.svg`: the matching, editable platform diagram.

All motion comes from project renders. The wall covers MGA surface scanning
and insertion, EB-MBD D3IL avoidance, 2GO quadruped footholds and humanoid
corridor motion, and MBD trajectory generation. The original three README
GIFs remain inputs; the D3IL 2D view appears as an inset in its 3D panel.

These are simulation and planning visualizations. Playback is rescaled to a
common 12-second composition; it does not represent real-time execution speed.
The plot display palette is adjusted to match the wall. Geometry and trajectory
positions are unchanged. Humanoid walking/push is absent from the showcase.

The compact additional inputs are under `showcase_sources/`. Their original
run paths, checksums, crop bounds, and sampling settings are recorded in
`showcase_sources/manifest.json`. Regeneration needs only these checked-in
inputs, not the private result tree or a simulator:

```bash
python -m pip install Pillow imageio-ffmpeg
python scripts/visualizations/build_readme_showcase.py
```

The compositor uses system Arial, DejaVu Sans, or Liberation Sans. Install one
of these fonts on a build machine. System font differences may affect text
metrics; the checked-in exports are the reviewed versions.

## Original renders

- `mga_surface_scan.gif`: `reports/mga/paper_figures/gifs/successful/mga_surface_hybrid_stripes.gif`.
- `mga_peg_insert.gif`: `reports/mga/paper_figures/gifs/successful/mga_peg_id_wide.gif`.
- `d3il_avoiding.gif`: `results/d3il_avoiding/ebmbd/level_0/seed_0/trajectory/trajectory_best_exec.gif`.
- `2go_corridor.png`: governed humanoid corridor motion strip.
- `2go_stepping_stones.png`: multi-modal stepping-stone plan.
