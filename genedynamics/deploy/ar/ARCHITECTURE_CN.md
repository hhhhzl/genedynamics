# General AR Architecture: Multi-Robot Digital Twins

Engine-independent, with Swift/RealityKit **and** Unity/AR Foundation; supports Vision Pro, phones, tablets, HoloLens, Quest, and browsers; targets high performance and edge/distributed integration. **Extend** existing `deploy/ar` (`SceneSource` + `scene_server`) rather than replacing it.

> See also `ARCHITECTURE.md`. This document additionally explains **delegating the P3 distributed layer to the existing `edgecloud` module**. Its filename is retained for compatibility with existing links.
>
> The **future compatibility roadmap** in `future.md` covers other OpenXR headsets, Unreal, ROS2, transports, non-AR consumers, and itemized implemented/reachable/out-of-scope status.

## 0. One Principle

Everything follows from **four independently replaceable layers**. Compatibility, performance, and distribution depend on the implementation chosen for each layer.

```
  ┌────────────┐   ┌──────────┐   ┌─────────────┐   ┌──────────────┐
  │  STATE     │ → │  SCHEMA  │ → │  TRANSPORT  │ → │   RENDER     │
  │ World-model│   │ Neutral  │   │ Pub/sub +   │   │ Thin engine  │
  │ aggregator │   │ contract │   │ edge gateway│   │ adapters     │
  └────────────┘   └──────────┘   └─────────────┘   └──────────────┘
   Robots/obstacles/ FlatBuffers    Delegated to edgecloud Swift/RealityKit
   poses/plans/goals IDL + ICD      NATS bus + gateways    Unity/ARF
   (shared world)   (compatibility) + controller           WebXR
```

- **Engine interoperability:** share only **SCHEMA** (one IDL generates Swift/C#) and **ICD conventions**; clients otherwise remain independent views.
- **Performance:** **TRANSPORT + SCHEMA**, using zero-copy binary data, deltas/keyframes, datagrams, and client interpolation.
- **Edge/distribution:** delegate the entire **TRANSPORT** layer to `edgecloud` (§6).
- **Generality/multiple robots:** **STATE** uses entities rather than a hard-coded corridor.

---

## 1. STATE: Generalize `SceneSource` into a World Model

Currently `SceneSource` holds one corridor. Generalize to **`WorldState`**, a set of entities, each `{id, type, pose, frame, geometry, meta, rev}`. Obstacles, walls, robot bases, arm links, goals, paths, and **robot occlusion proxies** are all entities.

- Small **producer adapters** feed `WorldState`: corridor obstacles (`SceneSource`), robot poses from any tracker, joints, planners, ROS2 topics. `SceneSource` becomes one producer rather than the whole state.
- Per-entity `rev` enables delta encoding.
- Reference frame: **a shared world frame supplied by a pluggable tracker**, via `BaseLocalizationPlugin`. Vicon, OptiTrack, SLAM, VIO, and fiducials are interchangeable backends; see frame registration below. The only structural requirement is that **both robots and AR devices localize in the same world**. Child frames (`<robot>/base`, `<arm>/link_k`) reference parents as in TF; multiple robots/arms then need only their transforms.

> Multiple robots/arms are a **STATE** extension: N producers write IDs into one `WorldState`. No new architecture is needed.

---

## 2. SCHEMA: One IDL, Generated Multi-Language Compatibility

Define the contract once in **FlatBuffers** IDL; `flatc` generates **Swift, C#, Python, TS** structures. FlatBuffers offers **zero-copy reads**, avoiding per-frame pose parsing, and supports schema evolution.

```fbs
// deploy/ar/schema/world.fbs (sketch)
namespace twin;
struct Vec3 { x:float; y:float; z:float; }
struct Quat { w:float; x:float; y:float; z:float; }
struct Pose  { p:Vec3; q:Quat; }                  // Expressed in `frame`
enum GeomKind:byte { Box, Sphere, Cylinder, Capsule, Usd, Urdf, Path }
enum EntityType:byte { Obstacle, Wall, Robot, ArmLink, Goal, Path, Occluder }
table Geometry { kind:GeomKind; half_extents:Vec3; radius:float; height:float;
                 asset_uri:string; joints:[float]; polyline:[Vec3]; }
table Entity   { id:string(key); type:EntityType; frame:string; pose:Pose;
                 geom:Geometry; color_rgba:uint; rev:uint; meta:string; }
table WorldSnapshot { schema_version:uint; site:string; stamp_ns:ulong;
                      frame:string;   // Root-world frame name, e.g. "world"
                      tracker:string; // Source: "vicon"|"optitrack"|"slam"|"vio"|"fiducial"|...
                      units:string; is_keyframe:bool;
                      entities:[Entity]; removed_ids:[string]; }
```

Provide a same-schema **JSON projection** for debugging, browsers, and retained `scene_server`. JSON is the human/debug view; FlatBuffers is the hot path.

**ICD conventions, `schema/conventions.md`,** complete compatibility:
- One tracker-independent shared world, **meters/radians, right-handed, z up, x forward**.
- Per-client basis conversion, checked against known measured points:
  - Unity, left-handed/y-up: `unity = (-y, z, x)`, already in `FrameRegistration.cs`.
  - RealityKit, right-handed/y-up/−z forward: `rk ≈ (-y, z, -x)`.
  - three.js/WebXR uses the RealityKit mapping.
- Stable cross-frame `id`, `rev` semantics, **USD/USDZ** meshes/**URDF** robots, colors, and units.

---

## 3. TRANSPORT: Delegate Performance and Distribution to `edgecloud`

**The entire layer belongs to the existing `edgecloud` module; do not rebuild it in enerdynamics.** Boundaries/mappings are in §6. Retain two principles:

- **Separate channels:** high-frequency poses use best-effort/drop-old NATS core; low-frequency scenes/configuration use reliable NATS JetStream. Decouple rates.
- **Payload:** zero-copy FlatBuffers, deltas and periodic keyframes, without retransmitting static scenes.

---

## 4. RENDER: Thin Engine Adapters for Both Ecosystems

Each adapter is a few hundred lines: connect to gateway, read FlatBuffers without copies, apply its basis mapping, instantiate/update by `id`, and interpolate/extrapolate client-side, rendering90 Hz from30 Hz state. Clients share generated schema and ICD only.

- **Unity/AR Foundation:** existing `CorridorRenderer`/`FrameRegistration` scaffolding; replace JSON with FlatBuffers, generalize entities, add `URDF-Importer`. One codebase targets iOS/Android/visionOS(PolySpatial)/HoloLens/Quest(OpenXR).
- **Swift/RealityKit:** adapter uses URLSession WebSocket to `scene_server` inP2, NATS-WS fromP3, FlatBuffers-Swift, RealityKit entities, and `rk=(-y,z,-x)`. Reuse the existing Vision Pro framework.
- **WebXR/three.js:** installation-free Android/browser view.
- **Real-robot occlusion (Q3):** publish an `Occluder` at the tracked robot world pose using URDF/proxy mesh. Engines render invisible depth-only/no-color material, allowing the real robot to occlude virtual walls correctly.

### 4a. Meta/Quest Targets: OpenXR and Meta XR SDK

Default headsets are **Meta Quest3/3S/Pro**, with passthrough MR,6-DoF inside-out tracking, and color passthrough, enabling world-locked holograms as on Vision Pro. **Ray-Ban/Ray-Ban Display are not targets:** monocular HUD, no third-party6-DoF or spatial-rendering runtime, hence no world-locked holograms. This is a hardware/SDK boundary, not engine choice.

**ARCore and Quest are different runtimes, not a migration sequence.** ARCore serves Android phones; Quest uses **OpenXR + Meta XR SDK**, not ARCore. Treat runtimes as leaves of one tree: **STATE/SCHEMA/TRANSPORT and the entire RENDER adapter (`WorldRenderer` + `WorldClient` + `EntityState` + `(-y,z,x)`) remain unchanged**. Quest is Android, so WebSocket/WebTransport endpoints also remain. Only two differences from phones:

1. **XR provider plugin:** replace ARCore XR Plugin with Meta OpenXR features/Meta XR SDK and target Quest. These are Unity player settings, not renderer code; `WorldRenderer.cs` imports no XR package.
2. **Registration:** change the provider driving `FrameRegistration.WorldOriginAnchor`. Phones detect surveyed fiducials with `ARTrackedImageManager`; OpenXR headsets use vendor-neutral **`OpenXRAnchorProvider`** (`clients/unity/Scripts/OpenXRAnchorProvider.cs`), attaching Vicon markers and writing the same anchor Transform. `QuestAnchorProvider` is a thin Quest subclass; **Pico/Magic Leap2/Android XR/Vive XR use `OpenXRAnchorProvider` directly**. Registration is identical; only player-setting OpenXR feature groups differ.

**Registration using existing Vicon, corresponding to the Vision Pro approach in §4b:** attach rigid markers to the headset. Vicon provides world pose; local Quest tracking provides Unity pose. Solve `anchor = headset_unity ∘ L⁻¹`, where `L` is Vicon pose through the same `(-y,z,x)` mapping. `FrameRegistration.WorldToAnchorLocalRot` is the shared rendering/registration source of truth. Solve per Vicon sample and EMA-smooth the quasistatic anchor, filtering jitter while continually realigning to Vicon truth so session holograms remain aligned; one-shot fiducials may drift. Quest's low-latency tracking handles fast head motion; Vicon network delay feeds slow drift correction only. No fiducial/passthrough-camera permission is needed, and robot/headset naturally share Vicon world.

> **Mounting offsets belong upstream.** Marker→head rigid calibration is a producer/calibration responsibility in `producers/tracker_pose.py`, as for robot bases. Unity consumes already-world-frame poses through `SetHeadsetPoseWorld(...)`, independent of transport: dedicated pose WS or a `headset/<id>` (`viewer`) entity on the existing stream; see §1/`populate_headset`. Enable `LockWhenConverged` for unstable Vicon headset streams.

> Other targets retain the same Unity code: iOS/Android via ARKit/ARCore AR Foundation, visionOS via PolySpatial, and all OpenXR headsets through `OpenXRAnchorProvider`. Only XR feature groups and, for phones, registration providers vary.

---

## 4b. Frame Sources and Registration: Tracker-Independent

The design assumes only **one world frame in which both robots and AR devices can localize**. Its source is pluggable: `deploy/localization/BaseLocalizationPlugin.get_state() -> (qpos, qvel)`. Coordinate glue `Se2Transform.scene_base_from_world(qpos, qvel)` accepts any backend. **Vicon is a backend, not an assumption.**

Only two components vary with tracker; STATE/SCHEMA/TRANSPORT/RENDER, coordinate glue, and occlusion proxies remain:
1. One `BaseLocalizationPlugin` per localization source.
2. A registration mechanism putting robot and AR device into the same frame.

| Tracker | Localization backend | AR registration into the shared frame | Accuracy |
|---|---|---|---|
| Optical capture: Vicon/OptiTrack/Qualisys | NatNet/RT/shm plugin | Markers provide direct world pose, simplest route | mm–cm |
| LiDAR SLAM/AMCL | ROS2 odom plugin | Relocalize AR device in the same map, or fiducial at known map pose | cm–dm, with drift |
| VIO/onboard odometry | New plugin | Shared fiducial anchors drifting VIO | Short-term cm, then drift |
| External camera + AprilTag/ArUco | Fiducial plugin | Both solve against the same fiducial set | Nearby cm |
| AR device tracking + robot detection | Device estimates relative robot pose; no global backend | Device frame becomes world; robot plans attach to it | More fragile |
| Shared spatial anchors: ARCore Cloud/ARKit collaboration | Robot joins through fiducials | Multiple AR devices share one anchor frame | cm–dm |

**Accuracy is a parameter, not a structural assumption.** Set planning `collision_margin` and governor `m_track` from tracker accuracy+latency. Motion capture(~cm) permits tighter margins; SLAM/VIO needs larger margins or wider initial scenes. The architecture stays identical: change plugin and margins.

## 5. Performance

1. **Zero-copy serialization:** FlatBuffers avoids client pose parsing.
2. **Deltas/keyframes:** send changed entities and periodic snapshots, not repeated static scenes.
3. **Separate channels:** high-rate poses on best-effort/drop-old datagrams; low-rate scenes on reliable streams.
4. **Client interpolation/extrapolation:** dead-reckon between frames, rendering90 Hz from30 Hz networking; render and network rates are independent.
5. **Same-host shared memory:** avoid serialization between robot and local gateway/NATS edge endpoint.
6. **Interest management/LOD:** spatial/subject subscriptions and reduced detail for distant/numerous entities support multiple robots.
7. **QUIC/WebSocket endpoints:** lower latency than raw TCP.
8. **PTP/NTP synchronization:** synchronize edges; messages carry `stamp_ns` for interpolation/multisource fusion.

---

## 6. Edge and Distributed Deployment Through `edgecloud`

The existing Go `edgecloud` (`WorkGetBetter/edgeforest/edgecloud`) is an edge-orchestration/control plane providing P3's distributed foundation:

| P3 need | Delegate from enerdynamics to edgecloud | Existing edgecloud capability |
|---|---|---|
| Pub/sub bus | **NATS**, replacing the original Zenoh proposal | Already uses `tasks.submit`, `node.score.*`, `wasm.metrics.*`, etc. |
| Interest management/routing | **NATS subjects/wildcards**, `twin.{site}.{robot}.pose` | Native subject routing |
| High-rate/reliable split | NATS **core** for drop-old poses, **JetStream** for reliable scenes | NATS |
| Edge hierarchy | NATS **leaf nodes**, one edge per site | Go edge/gateway: http/mqtt/media and spool queues |
| AR endpoint fan-out | **NATS over WebSocket** directly to Unity C#/Swift/browser nats.ws, or via edge/gateway | Gateway protocol layer |
| Orchestration/scheduling/observability | **controller** deploys/scales gateways across edges, Prom/Grafana | controller,edgectl/schedctl,Prometheus/Grafana |

**Responsibility boundary:**
```
enerdynamics/deploy/ar ── FlatBuffers WorldSnapshot on NATS subjects ──► edgecloud
  STATE (WorldState)                                                    NATS(core+JetStream)
  SCHEMA (world.fbs)                                                     edge/gateway(fan-out)
  RENDER adapters ◄──────── NATS-WS / edge gateway subscriptions ─────── controller(orchestration/observability)
     │
     └─ World-origin registration with any tracker (§4b)
```
- **enerdynamics owns** STATE+SCHEMA+RENDER and calls edgecloud for NATS publish/subscribe.
- **edgecloud owns** TRANSPORT,distribution,edge orchestration: NATS,gateways,controller,monitoring.
- Interface: **NATS subject + FlatBuffers payload**, defined by `world.fbs`.
- **ROS2 interoperability:** a small **NATS↔ROS2 bridge**, or robots publishing NATS directly, can attach to existing ROS2 IO.

This keeps distribution/edge infrastructure in edgecloud and AR focused on robots/rendering, while reusing fleet,scheduling,WASM,observability for multiple robots/arms.

---

## 7. `deploy/ar` Modules and Migration

```
deploy/ar/
  schema/
    world.fbs            # Compatibility IDL
    conventions.md       # ICD: frames/units/engine mappings/registration/assets
    generated/           # flatc → swift/ csharp/ python/ ts/
  world_state.py         # Entities and per-entity rev; generalizes SceneSource
  producers/
    corridor.py          # SceneSource producer
    tracker_pose.py      # Robot base/occluder and viewer-headset world poses from BaseLocalizationPlugin
    ros2.py              # ROS2 topics → entities
  encoders/
    json_codec.py        # Debug/browser/current contract
    flatbuffers_codec.py # Hot path; deltas/keyframes
  transport/
    nats_pub.py          # Publish WorldSnapshot to edgecloud NATS: delegation boundary
    ws_json.py           # Existing scene_server retained for debug/browser
  registration/          # Tracker-independent world-origin helpers, including Se2Transform
  clients/
    unity/   # AR Foundation(phone/visionOS) + WorldRenderer/WorldClient
             # OpenXRAnchorProvider.cs: vendor-neutral registration for
             # Quest/Pico/ML2/Android XR; QuestAnchorProvider.cs: Quest subclass (§4a)
    swift/   # RealityKit(Vision Pro)
    web/     # WebXR/three.js
```

> Quest/Meta XR setup—packages,player settings,scene connections,deployment—is in `clients/README.md`, “Setup — Meta Quest.” Registration principles are in §4a.

**Incremental migration, preserving existing value:**
- **P0, complete:** `SceneSource`/`scene_server`(WS+JSON),`Se2Transform`,Unity scaffolding.
- **P1, generalization:** `world.fbs`/`conventions.md`,entity-based`WorldState`,`SceneSource` producer,both JSON/FlatBuffers. Unity/Swift read entities.
- **P2, performance on existing `scene_server`, without NATS:** binary FlatBuffers WS payloads,deltas/keyframes,client interpolation,and optional WebTransport datagrams. NATS begins inP3 only.
- **P3, distributed through edgecloud:** NATS+edge/gateway+controller,subject interests,multi-robot ROS2↔NATS integration.

> Existing `scene_server`/contract is forward-compatible: it becomes the `ws_json` debug gateway/JSON projection of `world.fbs`. Extend beyond it rather than rewriting it.

---

## 8. Decisions to Confirm
- Serialization: **FlatBuffers**, zero-copy/multilanguage, versus Cap'n Proto or Protobuf.
- Bus: **NATS, already in edgecloud**, replacing the original Zenoh proposal.
- Endpoints: existing WebSocket `scene_server` inP2 without NATS; NATS-over-WebSocket fromP3, optionally WebTransport.
- **Pluggable tracker:** motion capture,SLAM,VIO,fiducials,shared anchors; no mandatory tracker. One `BaseLocalizationPlugin` per source (§4b).
- Robot models: **URDF**, via Unity URDF-Importer and conversion to USD for RealityKit.
- edgecloud integration: confirm NATS subject naming,JetStream stream setup,and whether edge/gateway needs a dedicated AR protocol or direct NATS-WS suffices.
