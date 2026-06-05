# General AR architecture — multi-robot digital twin

Engine-agnostic (Swift/RealityKit **and** Unity/AR Foundation), multi-device
(Vision Pro / phone / tablet / HoloLens / Quest / web), high-performance, and
edge/distributed-ready. Extends today's `deploy/ar` (`SceneSource` +
`scene_server`) instead of replacing it.

## 0. The one principle

Everything below follows from **decoupling four layers**, each independently
swappable. Compatibility, performance, and distribution are all just
"which implementation of which layer."

```
  ┌────────────┐   ┌──────────┐   ┌─────────────┐   ┌──────────────┐
  │  STATE     │ → │  SCHEMA  │ → │  TRANSPORT  │ → │   RENDER     │
  │ world model│   │ neutral  │   │ pub/sub +   │   │ per-engine   │
  │ aggregator │   │ contract │   │ gateways    │   │ thin adapter │
  └────────────┘   └──────────┘   └─────────────┘   └──────────────┘
   robots,obstacles  FlatBuffers     Zenoh/DDS        Swift/RealityKit
   poses,plans,goals  IDL + ICD       backbone +       Unity/ARF
   (world frame)     (the compat       WebTransport     WebXR
                      core)            last-mile gw
```

- **Two engines interoperate** because they share only the **Schema** layer
  (one IDL → generated Swift + C# code) + the **conventions** (ICD). Nothing
  else is shared. Each engine is an independent view client.
- **Extreme performance** lives in **Transport** + **Schema** (zero-copy
  binary, delta+keyframe, datagrams, interpolation).
- **Edge/distributed** lives in **Transport** (Zenoh/DDS backbone, edge
  gateways, interest management).
- **Generality (multi-robot)** lives in **State** (entities, not a corridor).

---

## 1. STATE — the world model (generalize `SceneSource`)

Today `SceneSource` holds one corridor scene. Generalize to a **`WorldState`**:
a flat set of **entities**, each `{id, type, pose, frame, geometry, meta, rev}`.
An obstacle, a wall, a robot base, an arm link, a goal, a planned path, and a
**robot occluder proxy** are all just entities.

- **Producers** feed `WorldState` (each is a small adapter): corridor obstacles
  (`SceneSource`), robot poses (from any tracker), joint states, planners, ROS2 topics.
  `SceneSource` becomes *one* producer, not the whole thing.
- `WorldState` tracks per-entity `rev` so the encoder can emit **deltas**.
- Frame of record: a **single shared world frame from a pluggable tracker**
  (the `BaseLocalizationPlugin` abstraction — Vicon / OptiTrack / SLAM / VIO /
  fiducial are interchangeable backends; see "Frame of record & registration —
  tracker-agnostic" below). The ONLY structural requirement is that the robot
  **and** the AR devices localize into the **same** world frame. Child frames
  (`<robot>/base`, `<arm>/link_k`) reference parents (TF-style) → robots/arms
  come "for free" once their transforms are published.

> Multi-robot/arm scaling is a State concern: N robots = N producers publishing
> into one `WorldState` keyed by id. No new architecture.

---

## 2. SCHEMA — the compatibility core (one IDL, many languages)

This is what makes Swift + Unity interoperate. Define the contract **once** as
a **FlatBuffers** IDL; `flatc` generates **Swift, C#, Python, TS** structs.
FlatBuffers is **zero-copy on read** (clients read poses every render frame
without a parse step) and supports schema evolution.

```fbs
// deploy/ar/schema/world.fbs  (sketch)
namespace twin;
struct Vec3 { x:float; y:float; z:float; }
struct Quat { w:float; x:float; y:float; z:float; }
struct Pose  { p:Vec3; q:Quat; }                  // in `frame`
enum GeomKind:byte { Box, Sphere, Cylinder, Capsule, Usd, Urdf, Path }
enum EntityType:byte { Obstacle, Wall, Robot, ArmLink, Goal, Path, Occluder }
table Geometry { kind:GeomKind; half_extents:Vec3; radius:float; height:float;
                 asset_uri:string; joints:[float]; polyline:[Vec3]; }
table Entity   { id:string(key); type:EntityType; frame:string; pose:Pose;
                 geom:Geometry; color_rgba:uint; rev:uint; meta:string; }
table WorldSnapshot { schema_version:uint; site:string; stamp_ns:ulong;
                      frame:string;   // root world frame name, e.g. "world"
                      tracker:string; // source: "vicon"|"optitrack"|"slam"|"vio"|"fiducial"|...
                      units:string; is_keyframe:bool;
                      entities:[Entity]; removed_ids:[string]; }
```

Ship a **JSON projection** of the same schema for debugging / browser / our
existing `scene_server` (keep it). JSON = the human/debug view; FlatBuffers =
the hot path.

**Conventions (the ICD — `schema/conventions.md`)** — the other half of compat:
- Frame: a single shared world frame (tracker-agnostic; see next section),
  **meters/radians, right-handed, z-up, x-forward**.
- Per-engine basis map (each client applies its own; verify with a test point):
  - Unity (LH, y-up): `unity = (-y, z, x)` — already in `FrameRegistration.cs`.
  - RealityKit (RH, y-up, −z fwd): `rk ≈ (-y, z, -x)`.
  - three.js/WebXR (RH, y-up, −z fwd): same as RealityKit.
- Stable `id` (cross-frame entity identity), `rev` semantics, asset format
  (**USD/USDZ** mesh, **URDF** robots), color/units.

---

## 3. TRANSPORT — backbone + last-mile gateways (perf + distribution)

> **DECISION (see `ARCHITECTURE_CN.md` §6):** this whole layer is **delegated to
> the external `edgecloud` module** (`WorkGetBetter/edgeforest/edgecloud`), which
> already provides the backbone (**NATS**, not Zenoh), edge gateways, and fleet
> orchestration. enerdynamics only *publishes FlatBuffers `WorldSnapshot` to NATS
> subjects* and lets edgecloud distribute. The Zenoh discussion below is the
> generic design that edgecloud's NATS now fills (NATS core = drop-old poses,
> JetStream = reliable scene, subjects = interest mgmt, leaf nodes = edge tiers,
> NATS-over-WebSocket = AR last-mile).

Split into two hops; this is the crux of both "extreme perf" and "edge".

### 3a. Backbone (robot/edge side): Zenoh (or DDS/ROS2)
**Eclipse Zenoh** is the recommendation: pub/sub/query, **shared-memory**
intra-host, efficient WAN, **interest-based routing**, edge-native, and it
**bridges ROS2/DDS** (`zenoh-bridge-ros2dds`, or ROS2's `rmw_zenoh`). Your
deploy stack already has ROS2 IO, so producers can publish to ROS2 → Zenoh, or
to Zenoh directly.
- Key-space encodes the world: `twin/{site}/{robot}/pose`,
  `twin/{site}/obstacles/{id}`, `twin/{site}/arm/{id}/joints` …
- **Interest management = subscribe by key/region** → only relevant data moves.
  This is how you scale to many robots without flooding clients.
- QoS: best-effort + keep-last-1 for high-rate poses; reliable for scene/config.

### 3b. Last-mile (AR client side): WebTransport gateway
Mobile/headset/browser can't speak DDS/Zenoh well (no good iOS/Android DDS,
WiFi multicast issues). So an **edge gateway** bridges backbone → a
client-friendly protocol:
- **WebTransport (HTTP/3 / QUIC)** — low latency, **datagrams** for poses
  (drop-old, no head-of-line blocking), **streams** for scene/keyframes.
  Native on Android/Chromium + web; fall back to **WebSocket** where WebTransport
  is unavailable (iOS today). Both carry **FlatBuffers** payloads.
- The gateway is **stateless + horizontally scalable**: run one per edge site,
  each fanning out to its local clients. Today's `scene_server` is the simplest
  gateway (WS+JSON); add `ws_binary` and `webtransport` gateways beside it.

```
robots/arms ─Zenoh(shm/WAN)─► [edge gateway: zenoh→WebTransport/WS + FlatBuffers]
   tracker ──┘ interest-routed        │ datagrams=poses  streams=scene/keyframe
   planners ──┘                       ├──► Vision Pro (Swift/RealityKit)
                                      ├──► Android tablet (Unity/ARF)
                                      └──► browser (WebXR)         ← all share schema
```

---

## 4. RENDER — per-engine thin adapters (the two ecosystems)

Each engine = a few-hundred-LOC adapter that: connects to a gateway,
deserializes FlatBuffers (zero-copy), applies its basis map, instantiates/
updates entities by `id`, runs **client-side interpolation/extrapolation**
(render at 90 Hz from 30 Hz state). Shared between them: only the generated
schema code + the ICD.

- **Unity/AR Foundation** — have the scaffold (`CorridorRenderer` +
  `FrameRegistration`); swap JSON→FlatBuffers, generalize to entities, add
  `URDF-Importer` for robots. One codebase → iOS/Android/visionOS(PolySpatial)/
  HoloLens·Quest(OpenXR).
- **Swift/RealityKit** — parallel adapter (URLSessionWebSocketTask/WebTransport
  + FlatBuffers-Swift + RealityKit entities + `rk=(-y,z,-x)`). Reuse your
  existing Vision Pro framework here.
- **WebXR/three.js** — zero-install Android/web view.
- **Occlusion of the real robot** (your Q3): publish a `Occluder` entity at the
  robot's tracked world pose (URDF/proxy mesh); each engine renders it with an
  invisible depth-write material → real robot correctly occludes virtual walls.

---

## 4b. Frame of record & registration — tracker-agnostic

The design assumes **exactly one thing: a single shared world frame that both the
robot and the AR devices are localized in.** WHO provides that frame is a
pluggable backend — `deploy/localization/BaseLocalizationPlugin.get_state() ->
(qpos, qvel)` — and the `Se2Transform.scene_base_from_world(qpos, qvel)` glue
consumes any of them. **Vicon is one backend, not an assumption.**

Only two things vary by tracker; everything else (STATE / SCHEMA / TRANSPORT /
RENDER, the frame glue, the occluder proxy) is identical:
1. **Localization backend** — one `BaseLocalizationPlugin` per source.
2. **Co-registration** — how the robot **and** the AR device end up in the *same*
   frame.

| Tracker | localization backend | how the AR device joins the same frame | accuracy |
|---|---|---|---|
| Optical mocap (Vicon / OptiTrack / Qualisys) | NatNet / RT / shm plugin | marker on the device → mocap gives its world pose (easiest) | mm–cm |
| LiDAR SLAM / AMCL | ROS2 odom plugin | AR device relocalizes in the same map, or a fiducial at a known map pose | cm–dm, drift |
| VIO / onboard odom | new plugin | anchor to a shared fiducial (VIO drifts) | cm short / drift |
| External cam + AprilTag/ArUco | fiducial plugin | both solve against the same fiducials | cm near |
| AR-device own tracking + detect robot | (no global) device computes relative robot pose | world frame = device frame; tie the plan to it | fragile |
| Shared spatial anchors (ARCore Cloud / ARKit collaborative) | robot ties in via fiducial | multiple AR devices share one anchor frame | cm–dm |

**Accuracy is a parameter, not a structural assumption.** Set the planning
`collision_margin` and the governor `m_track` relative to the tracker's accuracy
+ latency: tight (~cm) for mocap, looser (or easier scenes) for SLAM/VIO. The
architecture is identical across all of them — swap the plugin, set the margin.

## 5. Performance tactics ("极致")

1. **Zero-copy serialization** — FlatBuffers; clients read poses without parse.
2. **Delta + keyframe** — never resend the static scene; send changed entities +
   periodic keyframe. (`is_keyframe`, `rev`, `removed_ids`.)
3. **Channel split** — high-rate poses on **datagrams (best-effort, drop-old)**;
   low-rate scene/config on **reliable streams**. Decouples the two rates.
4. **Client interpolation/extrapolation** — dead-reckon poses between updates →
   smooth 90 Hz render from 30 Hz network. Render rate ⟂ network rate.
5. **Shared memory** intra-host (Zenoh/DDS) for the robot→gateway hop — no
   serialization cost on the same box.
6. **Interest management + LOD** — subscribe per spatial region / key-space;
   level-of-detail for far/many entities. Scales to many robots.
7. **QUIC/HTTP-3** last-mile — lower latency than TCP/WebSocket, no HOL blocking.
8. **PTP/NTP time sync** across edges; every message stamped (`stamp_ns`) for
   correct interpolation and multi-source fusion.

---

## 6. Edge & distribution

- **Backbone is inherently distributed** (Zenoh/DDS = peer-to-peer, no central
  broker required); add brokers/routers only where topology needs them.
- **Hierarchical edge gateways** — one per site/zone; clients hit the nearest.
  Stateless → autoscale.
- **Interest-routed** — a client in zone 3 only receives zone-3 data; the WAN
  carries only what's subscribed.
- **ROS2 interop** — `zenoh-bridge-ros2dds` / `rmw_zenoh` so any ROS2 robot
  joins without changes; ties into the deploy stack's existing ROS2 IO.
- **Fault isolation** — a downed gateway/edge doesn't take down others; the
  backbone reroutes.

---

## 7. Mapping onto `deploy/ar` (module layout + migration)

```
deploy/ar/
  schema/
    world.fbs            # the compatibility core (IDL)
    conventions.md       # ICD: frame, units, per-engine axis maps, registration, assets
    generated/           # flatc → swift/ csharp/ python/ ts/
  world_state.py         # general WorldState (entities + per-entity rev)   [generalizes SceneSource]
  producers/
    corridor.py          # SceneSource as a producer
    tracker_pose.py      # robot/headset world poses from any BaseLocalizationPlugin (vicon/optitrack/slam/vio/fiducial)
    ros2.py              # bridge ROS2 topics → entities
  encoders/
    json_codec.py        # debug/browser (today's contract)
    flatbuffers_codec.py # hot path; delta + keyframe
  gateways/
    ws_json.py           # == today's scene_server (keep for debug/browser)
    ws_binary.py         # WebSocket + FlatBuffers
    webtransport.py      # HTTP/3 datagrams + streams
    zenoh_gw.py          # backbone bridge (edge/distributed)
  registration/          # world-origin helpers, tracker-agnostic (Se2Transform lives here)
  clients/
    unity/   
    swift/   
    web/ # per-engine adapters
```

**Migration (each phase ships value, nothing thrown away):**
- **P0 (done):** `SceneSource` + `scene_server` (WS+JSON), `Se2Transform`, Unity scaffold.
- **P1 generalize:** `world.fbs` + `conventions.md`; `WorldState` (entities);
  `SceneSource`→producer; emit JSON **and** FlatBuffers. Unity/Swift adapters read entities.
- **P2 perf (still on the existing `scene_server`, NO NATS):** switch the WS
  payload from JSON to **FlatBuffers** (binary frames); delta+keyframe; client
  interpolation; (optional) add a WebTransport datagram channel. **NATS belongs
  to P3 — don't pull it in here.**
- **P3 distributed:** Zenoh backbone + edge gateways + interest management; ROS2 bridge for multi-robot.

> Today's `scene_server`/contract is **forward-compatible**: it becomes the
> `ws_json` debug gateway and the JSON projection of `world.fbs`. You never
> rewrite it — you grow past it.

---

## 8. Decisions to confirm
- Serialization: **FlatBuffers** (zero-copy, multi-lang) vs Cap'n Proto vs Protobuf.
- Backbone: **delegate to `edgecloud` → NATS** (decided; see §3 and `ARCHITECTURE_CN.md` §6).
- Last-mile: P2 uses the existing `scene_server` (WebSocket, **no NATS**); from
  P3, **NATS-over-WebSocket** (+ optional WebTransport).
- Tracker: **pluggable** (mocap / SLAM / VIO / fiducial / shared anchors) — NOT
  assumed; one `BaseLocalizationPlugin` per source (see §4b).
- Robot model source: **URDF** (→ Unity URDF-Importer; → USD for RealityKit).
