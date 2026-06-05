# AR digital-twin conventions (ICD)

The **normative** contract shared by every renderer (Swift/RealityKit,
Unity/AR Foundation, WebXR) and every producer. The two engines interoperate
because they share **this document + `world.fbs`** — nothing else. Keep this in
sync with `world.fbs`.

## 1. Frame of record (tracker-agnostic)
- One **shared world frame** named by `WorldSnapshot.frame` (default `"world"`),
  provided by *any* `BaseLocalizationPlugin` (mocap / SLAM / VIO / fiducial /
  shared anchor). `WorldSnapshot.tracker` records which.
- Convention of the world/scene frames: **right-handed, meters & radians,
  `z` up, `x` forward, `y` left.**
- The only structural requirement: the robot **and** the AR devices localize
  into this same frame (see ARCHITECTURE §4b for per-tracker co-registration).
- Child frames (`<robot>/base`, `<arm>/link_k`) reference parents (TF-style);
  publish their transforms as entities/poses.

## 2. Per-engine basis transform (world → engine)
Each renderer applies its own map; **verify the sign with one known point.**
- **Unity** (left-handed, y-up, +z forward): `unity = (-y, z, x)`; world yaw
  about +z → Unity rotation about +y of `-yaw`. (See `FrameRegistration.cs`.)
- **RealityKit** (right-handed, y-up, −z forward): `rk = (-y, z, -x)`.
- **three.js / WebXR** (right-handed, y-up, −z forward): same as RealityKit.

## 3. Pose & geometry
- `Pose.p` = position in `Entity.frame` (m); `Pose.q` = unit quaternion
  `(w, x, y, z)`.
- `Geometry` is in the entity's **local** frame; `pose` places/orients it:
  - `Box`/`Capsule`: `half_extents` (m). Box local axes follow the frame; use
    `pose.q` for orientation (e.g. a scene-axis-aligned box rotated into world).
  - `Sphere`: `radius`.
  - `Cylinder`/`Capsule`: `radius` + `height` (axis = local up / world +z).
  - `Usd`/`Urdf`: `asset_uri` (+ `joints` for URDF).
  - `Path`: `polyline` points in `Entity.frame`.
- Assets: **USD/USDZ** for meshes (RealityKit-native, Unity-importable);
  **URDF** as the robot source (Unity URDF-Importer; URDF→USD for RealityKit).

## 4. Entity identity & revisions
- `Entity.id` is **stable across frames** — renderers key their GameObjects /
  RealityKit entities by it (create / update / destroy).
- `Entity.rev` is the producer's monotonic revision at the entity's last change.
- **Keyframe** (`is_keyframe=true`): the full, self-contained entity set; the
  client replaces its world with it. `removed_ids` is empty.
- **Delta** (`is_keyframe=false`): only entities whose `rev` exceeds the client's
  cursor, plus `removed_ids` deleted since then. The client must have a keyframe
  first; on reconnect/gap, request a keyframe.

## 5. Channel discipline (P2/P3)
- **High-rate poses** → best-effort / drop-old (datagrams; P3: NATS core).
- **Low-rate scene/config** → reliable (P3: NATS JetStream).
- P2 carries both over the existing `scene_server` WebSocket (binary FlatBuffers
  frames). **NATS only enters at P3 (edgecloud).**

## 6. Versioning
- `schema_version`: `1` = legacy corridor contract; `2` = this entity model.
- Add fields at the END of tables (FlatBuffers forward/backward compatible);
  never renumber/reorder. Bump `schema_version` only on breaking changes.

## 7. Units & colors
- Lengths m, angles rad, time `stamp_ns` in nanoseconds (producer wall clock).
- `color_rgba`: packed RGBA8 (`0xRRGGBBAA`).
