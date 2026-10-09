# MuJoCo Menagerie assets

The submodule pins official upstream commit
`a03e87bf13502b0b48ebbf2808928fd96ebf9cf3`. The former pin,
`e5146679f3cfcb327cf759fc9706f5bb0236bd5e`, was a local commit that the
official remote cannot serve, which prevented clean recursive checkouts.

`mujoco_menagerie-go2-mjx.patch` preserves that local commit's only change:
13 Go2 MJX collision geometries explicitly use spheres with a single radius.
Robot meshes, dynamics parameters, and the G1 model are unchanged. Upstream model
licenses remain in the submodule.

From the repository root, initialize the pinned assets and apply the patch:

```bash
bash scripts/setup/setup_mujoco_menagerie.sh
```

The command is safe to repeat. It checks both forward and reverse application,
leaves an already patched checkout as-is, and refuses conflicting edits or a
different revision. It never resets or cleans an existing checkout. A fresh
checkout will show the patched XML as a local submodule modification; the
submodule HEAD remains at the official pin. An existing checkout at the former
local commit is recognized and preserved.

To check an independently initialized copy, pass its directory:

```bash
bash scripts/setup/setup_mujoco_menagerie.sh /path/to/mujoco_menagerie
```
