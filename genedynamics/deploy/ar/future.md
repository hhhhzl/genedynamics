# Future ecosystems — what this architecture can still reach

A compatibility roadmap, not a TODO list. It reads off the four-layer decoupling
in `ARCHITECTURE.md` §0 (STATE → SCHEMA → TRANSPORT → RENDER): every "can we
support X?" is really "which implementation of which layer is X?". Pair this with
`ARCHITECTURE.md` (design) and `schema/conventions.md` (the normative ICD).

**Status legend:** ✅ exists in-tree · 🟡 reachable (thin adapter / project config,
no new architecture) · 🔴 not feasible for *world-locked holograms* (hardware/SDK
limit, not an engine choice).

---

## 0. The two questions that decide everything

A new ecosystem is gated by exactly two things:

1. **Can it do it at all?** → Does the device give **6-DoF** *and* can it
   **co-register into the shared world frame** (`ARCHITECTURE.md` §4b)? If yes →
   full holographic twin. If 6-DoF but no easy registration → still yes, harder.
   If **3-DoF / no positional tracking** → 🔴 only non-registered 2D HUD, never
   world-locked obstacles. If **not an AR device at all** → it can still consume
   the stream as a *non-AR* viewer (§5).
2. **How expensive is it?** → Cost is **per render engine, not per device**: one
   Unity/OpenXR adapter covers every OpenXR headset; a new engine (Unreal) is a
   new adapter. The shared invariants — `world.fbs` + `conventions.md` + the
   single world frame — never change.

> Restated: brand doesn't decide feasibility, **6-DoF + co-registration** does;
> and the marginal build cost is **a thin RENDER adapter + a registration
> backend**, with STATE/SCHEMA/TRANSPORT reused.

---

## 1. RENDER — device & engine ecosystems

### 1a. The OpenXR universe (biggest multiplier)
One Unity + OpenXR codebase (`WorldRenderer`/`WorldClient` + `OpenXRAnchorProvider`)
reaches every 6-DoF OpenXR headset. Switching device = swap the vendor's OpenXR
**feature group** in Player Settings + attach `OpenXRAnchorProvider` — **no render
code changes** (`ARCHITECTURE.md` §4a).

| Device | runtime + engine | registration | status |
|---|---|---|---|
| Meta Quest 3 / 3S / Pro | OpenXR + Meta XR · Unity | Vicon marker (`OpenXRAnchorProvider`) | ✅ |
| Pico 4 / 4 Ultra / Neo | OpenXR (PICO feature) · Unity | same | 🟡 |
| HTC Vive XR Elite / Focus 3 / Focus Vision | OpenXR (Wave) · Unity | same | 🟡 |
| Magic Leap 2 | OpenXR (Android) · Unity | same | 🟡 |
| Varjo XR-3 / XR-4 | OpenXR (PC-tethered) · Unity | same | 🟡 high-fidelity |
| Lynx R1 | OpenXR · Unity | same | 🟡 |
| **Samsung Galaxy XR / Android XR** (Project Moohan) | OpenXR + Jetpack XR · Unity | shared anchor / marker | 🟡 emerging |
| Snapdragon Spaces glasses (Lenovo VRX/A3, TCL RayNeo X2…) | OpenXR (Spaces) · Unity | marker / Spaces anchor | 🟡 constrained 6-DoF |
| HoloLens 2 | OpenXR / MRTK · Unity | spatial anchor | 🟡 deployable but EOL |

### 1b. Apple ecosystem (two paths, both already in-tree)
| Target | path | status |
|---|---|---|
| iPhone / iPad | ARKit · Unity AR Foundation | ✅ |
| Apple Vision Pro | visionOS · Unity PolySpatial **or** the Swift/RealityKit client (`clients/swift/`) | ✅ |

### 1c. Google / Android phones
| Target | path | status |
|---|---|---|
| Android phone / tablet | ARCore · Unity AR Foundation | ✅ |

### 1d. Web (zero-install)
| Target | path | status |
|---|---|---|
| Browser / WebXR | three.js (`clients/web/`) | ✅ |
| Babylon.js / A-Frame / 8th Wall | alt web adapters reading the same JSON projection | 🟡 |

### 1e. Second render engine (covers the same devices, different stack)
The architecture is explicitly engine-agnostic — a new engine is just another
RENDER leaf reading `world.fbs` + applying its own basis map.
| Engine | reaches | status |
|---|---|---|
| **Unreal Engine** (OpenXR) | every §1a device, high-fidelity | 🟡 new adapter |
| Godot (OpenXR) | same | 🟡 new adapter |
| Niantic Lightship ARDK | phone shared-AR | 🟡 new adapter |

### 1f. Smart glasses with a display (need 6-DoF)
| Device | status |
|---|---|
| Xreal Ultra, Rokid Max/AR, TCL RayNeo X2 (via Snapdragon Spaces/Nebula 6-DoF) | 🟡 constrained holographic |
| Xreal Air (3-DoF only) | 🔴 head-locked HUD only |

### 1g. 🔴 The red line — HUD-only, no world-locking
Meta Ray-Ban Display, Even Realities G1, Brilliant Labs Frame, and any glasses
without exposed 6-DoF + a spatial-render runtime. Reachable **only** as a 2D info
channel (text / minimap / alerts) — they cannot pin obstacles to physical space.
Same conclusion as the original Ray-Ban discussion; not an engine problem.

---

## 2. STATE / producers — data-source & robotics ecosystems

The producer side is as pluggable as the render side (one `populate_*` per
source, all writing one `WorldState`).

| Ecosystem | how it joins | status |
|---|---|---|
| **ROS 2 / RViz** | `producers/ros2.py` — `MarkerArray`/`PoseStamped` → entities (RViz-compatible) | ✅ |
| ROS 2 over the backbone | `rmw_zenoh` / `zenoh-bridge-ros2dds`, or a NATS↔ROS2 bridge | 🟡 |
| Optical mocap (Vicon / OptiTrack / Qualisys) | `BaseLocalizationPlugin` + `tracker_pose.py` | ✅ Vicon; 🟡 others |
| LiDAR SLAM / AMCL, VIO / onboard odom | new `BaseLocalizationPlugin` | 🟡 markerless registration |
| AprilTag / ArUco fiducials | fiducial plugin | 🟡 |
| Shared spatial anchors (ARCore Cloud, ARKit collaborative, OpenXR anchors, Niantic VPS, Google Geospatial) | anchor-based co-registration | 🟡 multi-device |
| Simulators (MuJoCo, Isaac Sim/Lab, Gazebo, Genesis, Drake) | sim state → producer | 🟡 sim-in-the-loop twin |

**Asset ecosystems** (via `Geometry.asset_uri`): **USD/USDZ** (OpenUSD / NVIDIA
Omniverse / Apple / Unity) and **URDF** (robots) are first-class today; glTF and
MJCF are 🟡 straightforward additions.

---

## 3. TRANSPORT — distribution ecosystems

The wire contract is just "a `WorldSnapshot` on a subject/socket," so any pub/sub
or streaming transport can carry it.

| Transport | role | status |
|---|---|---|
| WebSocket (`transport/world_ws.py`) | P2 last-mile (JSON/binary frames) | ✅ |
| WebTransport / QUIC (HTTP/3) | low-latency datagrams for poses | 🟡 gateway |
| **NATS** (core + JetStream + NATS-over-WS) | P3 backbone, delegated to `edgecloud` | 🟡 (decided; §6) |
| Zenoh / DDS | alt robot-side backbone | 🟡 |
| MQTT | edge/IoT gateways | 🟡 |
| Kafka | analytics / replay fan-out | 🟡 |

---

## 4. SCHEMA — serialization ecosystems

One IDL (`world.fbs`) is the source of truth; projections follow.

| Format | use | status |
|---|---|---|
| FlatBuffers | zero-copy hot path | ✅ (`encoders/flatbuffers_codec.py`) |
| JSON (+ gzip binary) | debug / browser / today's stream | ✅ (`json_codec.py`, `binary_codec.py`) |
| Protobuf / Cap'n Proto / CBOR | other consumer ecosystems | 🟡 add a codec |
| ROS 2 `.msg` projection | native ROS2 consumers | 🟡 |
| glTF (assets) | web/engine asset interchange | 🟡 |

---

## 5. Non-AR consumers (the cheapest, often-missed class)

Anything that can read the entity stream is a viewer — and these need **no
registration** (they don't overlay reality), so they're near-zero effort:

- Desktop **three.js / web** 3-D playback of the live or recorded world.
- **RViz / Foxglove**-style monitoring dashboards.
- Unity / Unreal **spectator** views (operator's-eye view on a screen).
- **Another robot** consuming the world as a perception/coordination input.
- **Record & replay** (rosbag-like) of `WorldSnapshot` streams for debugging.

---

## 6. Build-order suggestion (highest leverage first)

1. ✅ `producers/ros2.py` — unlocks the whole ROS2/RViz robot ecosystem.
2. ✅ `OpenXRAnchorProvider` — one registrant for all OpenXR headsets.
3. 🟡 **Unreal adapter** — a second engine; high-fidelity + reuses every OpenXR device.
4. 🟡 **WebTransport gateway** — lower-latency last-mile than WebSocket.
5. 🟡 **NATS publish** — P3 backbone via `edgecloud` (§6 of `ARCHITECTURE.md`).
6. 🟡 **Per-vendor bring-up** — verify Pico / ML2 / Android XR (feature group only).
7. 🟡 **Markerless localization backend** (SLAM/VIO) — drop the Vicon-on-headset
   requirement for field use.
8. 🟡 **Non-AR desktop / Foxglove viewer** — cheapest reach, big debugging payoff.

---

## 7. The one invariant

Whatever you add, the single structural requirement never changes
(`conventions.md` §1): **the robot and the new device/consumer must localize into
the same shared world frame.** Everything else — engine, transport, serialization,
device — is "which implementation of which layer."
