# AR clients — corridor shared digital twin

Renders the **same** obstacle scene the robot avoids, as holograms registered
to the **Vicon world frame**, on Vision Pro and phones/tablets from one Unity
codebase (AR Foundation → ARKit / ARCore / visionOS PolySpatial).

```
scene_server (ws://host:8765/ws)  ──JSON contract──►  SceneClient.cs
        ▲ same single obstacle source the robot uses                │ parse
        │                                                           ▼
   robot side (replan + governor)                         CorridorRenderer.cs
                                                                    │ place in
                                                          FrameRegistration.cs
                                                          (T_world_scene + Vicon↔Unity)
                                                                    ▼
                                                      holographic obstacles, aligned
```

## Why this is "perception" for the robot
The robot never sees holograms. Both the robot and the AR client read the
**same obstacle list in the same Vicon world frame** (see the project's
`SceneSource`/`scene_server`). The robot "perceives" the obstacles by feeding
that list to its governor (`body_sdf_scene`); the AR client renders the same
list for the human. So AR-registration error is **cosmetic only** — it never
affects whether the robot avoids the obstacle (that depends solely on Vicon +
the shared list). Keep that in mind when budgeting registration accuracy.

## The contract (what `/ws` streams)
```jsonc
{
  "schema_version": 1, "frame": "vicon_world", "units": "m_rad",
  "revision": 0, "scene_preset": "zone_d",
  "T_world_scene": { "x": 1.2, "y": 0.4, "yaw": 0.0 },   // SE(2): scene -> world
  "corridor": { "corridor_width": 1.6, "corridor_length": 4.0,
                "start_pos": [0.5,0.0], "goal_pos": [3.5,0.0],
                "wall_y_min": -0.8, "wall_y_max": 0.8 },
  "obstacles": [ { "name":"ball_L1","shape":"sphere","cx":2.0,"cy":0.25,
                   "radius":0.06,"z_min":0.0,"z_max":2.0,
                   "x_min":1.94,"x_max":2.06,"y_min":0.19,"y_max":0.31,
                   "qc_clip_sign":0.0 }, ... ],
  "robot_pose_world": null,            // optional live Vicon base pose
  "plan_xy_scene": null,               // optional planned path (scene frame)
  "seq": 42, "server_time": 1.0
}
```
Obstacle `shape` ∈ `box | sphere | qc`. Coordinates are in the **scene frame**
(meters): apply `T_world_scene` to get world, then the Vicon↔Unity axis map.

## Coordinate frames
- **Scene / world (genedynamics, Vicon):** right-handed, `x` = forward
  (corridor length), `y` = lateral (**left** positive), `z` = up. Meters.
- **Unity:** left-handed, `x` = right, `y` = up, `z` = forward.
- **Map world→Unity (under the world-origin anchor):**
  `unity = (-y_world, z_world, x_world)` (one axis flip for RH→LH).
  Yaw about world `+z` becomes a rotation about Unity `+y` of `-yaw`.
  All of this lives in `FrameRegistration.cs`.

## Registration (tie the AR session to the Vicon origin)
Place a **world-origin anchor** in the AR session at the known Vicon origin,
then render obstacles as its children using the world→Unity map above.

- **Phone / tablet (recommended first):** print a fiducial, measure its Vicon
  pose, and detect it with ARKit `ARReferenceImage` / ARCore `AugmentedImage`
  (AR Foundation `ARTrackedImageManager`). The tracked image pose + its known
  Vicon pose give the world-origin anchor. Accurate to ~cm, easiest to demo.
- **Vision Pro (most robust):** rigid Vicon marker cluster on the headset →
  Vicon streams the headset world pose; one-time hand-eye calibration ties the
  visionOS world-anchor frame to Vicon. (Fallback: a visionOS world anchor
  dropped at a Vicon-surveyed point.)

`FrameRegistration` exposes a single `WorldOriginAnchor` Transform; swap the
registration source without touching the renderer.

## Files
| File | Role |
|---|---|
| `unity/Scripts/SceneContract.cs` | `[Serializable]` data model mirroring the contract |
| `unity/Scripts/SceneClient.cs` | `ClientWebSocket` subscriber; raises `OnContract` |
| `unity/Scripts/FrameRegistration.cs` | `T_world_scene` + Vicon↔Unity map; world-origin anchor |
| `unity/Scripts/CorridorRenderer.cs` | instantiate/update obstacle + wall GameObjects |

## Setup (Unity 2022.3+ / 6)
1. New Unity project; install **AR Foundation** + **ARKit XR Plugin** +
   **ARCore XR Plugin** (and **PolySpatial** for visionOS).
2. Add an `AR Session` + `AR Session Origin` (XR Origin). Add
   `ARTrackedImageManager` and a reference-image library with your fiducial.
3. Create an empty `CorridorTwin` GameObject; attach `SceneClient`,
   `FrameRegistration`, `CorridorRenderer`. Wire references in the Inspector.
4. Set `SceneClient.Url` to `ws://<server-host>:8765/ws`.
5. Build to the device. Point at the fiducial (phone) / start Vicon streaming
   (Vision Pro); obstacles appear locked to the physical space.

> The robot side is unaffected by any of this — it consumes the same contract
> directly. This client is purely the human-facing visualization layer.
