# Swift / RealityKit adapter (Vision Pro / iOS)

Renders the generic WorldSnapshot stream (`transport/world_ws.py`) as RealityKit
entities, with **client-side interpolation** and **real-robot occlusion**. One of
two interchangeable view clients (the other is Unity); both consume the same
contract (`schema/world.fbs` / conventions.md) — that shared contract is the only
thing they have in common.

## Files (`Sources/CorridorTwin/`)
- `Contract.swift` — Codable models of the JSON contract.
- `EntityState.swift` — per-entity timestamped pose buffer + interpolation (simd).
- `WorldClient.swift` — `URLSessionWebSocketTask` subscriber (keyframe + deltas).
- `WorldRenderer.swift` — RealityKit draw: interpolated poses, world→RK axis map
  `(-y, z, -x)`, `OcclusionMaterial` for occluders.

## Wire it up
```swift
import RealityKit
import CorridorTwin

// 1. Place an anchor at the Vicon WORLD origin (your registration: marker on the
//    headset tracked by mocap, a world anchor at a surveyed point, or a fiducial).
let worldAnchor = AnchorEntity(world: .zero)   // set transform from registration
arView.scene.addAnchor(worldAnchor)

// 2. Renderer + client.
let renderer = WorldRenderer(worldAnchor: worldAnchor)
let client = WorldClient(url: URL(string: "ws://<server-host>:8766/")!)
client.onMessage = { [weak renderer] msg in renderer?.enqueue(msg) }  // bg thread → thread-safe enqueue
client.connect()

// 3. Drive interpolation every frame.
arView.scene.subscribe(to: SceneEvents.Update.self) { _ in
    renderer.update(now: CACurrentMediaTime())
}.store(in: &subscriptions)
```

## Notes
- **Interpolation:** `renderer.interpolationDelay` (default 0.08 s ≈ 1.5–2 network
  periods) renders between two real samples — smooth at display rate from ~30 Hz
  state. Poses are stamped with the client clock on apply (single-source robust;
  use `stamp_ns` only with PTP/multi-source fusion).
- **Occlusion (the real robot):** the producer sends an `occluder` entity at the
  robot's tracked pose; it's drawn with `OcclusionMaterial`, so the **real** robot
  correctly occludes virtual obstacles behind it. Precision comes from the tracker.
- **Axis map:** verify the sign once with a known pose (conventions.md §2).
- **`generateCylinder`** needs recent RealityKit (visionOS 2 / iOS 18); fall back
  to a box for older runtimes. `usd`/`urdf` geom → load the asset instead of a
  primitive (USDZ via `Entity.load`; URDF via a converter).
- **Perf path:** swap the JSON decode in `WorldClient` for the generated Swift
  FlatBuffers (`../../schema/generated/swift`) + FlatBuffers SPM, and set
  `world_ws(encode=flatbuffers_codec.encode)` — zero-copy reads, no JSON parse.
