# Corridor AR Hardware Experiment · Complete Operating Guide

Step-by-step instructions for running this repository's AR digital-twin pipeline on
**a real Unitree G1 with AR obstacles**. Obstacles are virtual and exist only in AR.
The robot uses Vicon plus obstacle geometry to perceive/avoid them; a person wearing
Vision Pro or holding an iPad sees aligned holographic obstacles.

> Default scene: **zone_d**, corridor length 4.0m, width 1.6m, start=(0.5,0),
> goal=(3.5,0), and 2 virtual spheres. Substitute corresponding values for other zones.

---

## 0. Machine Roles and Networking

| Role | Machine | Workload | Key software |
|---|---|---|---|
| **Motion-capture host** | Mocap computer | Motive NatNet stream, `natnet_ros2` driver, `natnet_shm_writer` writing `mocap_state_shm` | ROS2, `natnet_ros2`, repository writer |
| **Twin server** | Linux/Mac, optionally the robot host | `run_twin_server.py`, world_ws :8766 | Conda `fedguide`, `websockets` |
| **Robot host** | Linux connected to G1 | `run_real_g1.py`, governor and hardware IO | `unitree_sdk2py`, this repository |
| **Planner host** | Any host, optionally shared | `replan_from_scene.py`, 2GO in Docker | Docker `genedynamics/dev-cpu:torch` |
| **AR device** | Vision Pro / iPad | Twin rendering | Part B |
| **Apple build host** | **Apple Silicon Mac** | Xcode packages for Vision Pro/iPad | Xcode and optionally Unity |

Connect the four machines and AR device to one LAN through a gigabit switch. Record
the twin server IP as `TWIN_IP`, e.g. `192.168.1.50`; AR clients connect to
`ws://TWIN_IP:8766/`. Connect G1 directly to the robot host by Ethernet, default `eth0`.

---

## 1. One-Time Software Installation by Machine

**Motion-capture host (ROS2):** install ROS2, e.g. Humble, and `natnet_ros2`:

```bash
# In the ROS2 workspace
git clone https://github.com/L2S-lab/natnet_ros2 src/natnet_ros2
colcon build --packages-select natnet_ros2 && source install/setup.bash
python -m pip install numpy scipy        # Finite differences in natnet_shm_writer
```

> The repository's `natnet_shm_writer` runs only on this ROS2 host, writing NatNet
> poses to `mocap_state_shm`. Downstream twin/robot hosts need no rclpy and continue
> reading shared memory through `--localization vicon`. See A7.

**Twin server (fedguide):**

```bash
/opt/anaconda3/envs/fedguide/bin/python -m pip install "websockets>=12" "flatbuffers>=2.0"
```

**Planner host:** confirm the Docker image with
`docker images | grep genedynamics/dev-cpu`.

**Robot host:** install `unitree_sdk2py` following Unitree instructions, clone this
repository, and confirm `python -c "import unitree_sdk2py"` succeeds.

**Mac build host:** install the latest Xcode including visionOS SDK from the App Store;
run `xcodebuild -runFirstLaunch` once. Unity routes also need Unity Hub/Editor
(Part B options 2/3).

---

# Part A — Physical Setup and Vicon Floor Calibration

Goals: establish an unambiguous corridor scene frame on the floor; measure its pose
`T_world_scene=(x0,y0,yaw0)` in the Vicon world; stream robot and optional AR-device
poses into shared memory.

## A1 Hardware Checklist

Tape measure ≥5m, chalk line/laser level, masking tape in three colors, marker pen,
right-angle square, Vicon calibration wand, G1 base marker cluster, printed A3 fiducial
(A6), gantry/harness, and a physical emergency stop.

## A2 Vicon Camera Calibration

1. Perform wand calibration according to Vicon procedures until residuals meet requirements.
2. Set the floor to z=0 and establish a Vicon origin with the L-shaped calibration object.
   Vicon and corridor origins need not coincide; the A4 transform connects them.
3. Measure Vicon-to-control-loop latency, possibly later. First ensure stable marker
   visibility across the entire corridor without occlusion gaps.

## A3 Mark the Corridor Frame on the Floor

Scene convention: x forward along the corridor, y left, z up, right-handed, meters.
For zone_d: length 4.0m, half-width 0.8m, start=(0.5,0), goal=(3.5,0).

Choose a straight, flat, unobstructed area at least 5m×2.5m:

1. **Origin O**, scene (0,0): place a red tape cross; its center defines O.
2. **+x centerline:** use chalk/laser to mark a straight 4.0m line from O, then yellow
   tape. The endpoint is (4.0,0). Add labeled ticks every 0.5m: 0.0, 0.5, …, 4.0.
3. **+y direction:** use a square at O to mark the perpendicular toward the left when
   facing +x; label a short segment "+y".
4. **Start/goal:** green cross at x=0.5m, blue cross at x=3.5m on the centerline.
5. **Corridor walls**, reference only because walls are virtual: tape two 4.0m lines
   parallel to the centerline at `y=+0.8` and `y=-0.8`. Measure the 0.8m offset at
   multiple points to ensure parallelism.
6. **Robot heading:** add a +x arrow at the start cross; align robot toes/front with it.

> Verify a straight O→(4,0), start/goal on that line, constant 0.8m wall offsets,
> and +y to the left when facing +x.

## A4 Measure `T_world_scene`

`T_world_scene` places the scene in Vicon world coordinates:
`world = R(yaw0)·scene + (x0,y0)`. Probe two points in Vicon:

1. Place the wand tip precisely at O, scene (0,0); record world coordinates `(X0,Y0)`.
2. Place it at centerline x=1.0m, scene (1,0); record `(X1,Y1)`.
3. Compute `x0=X0`, `y0=Y0`, and `yaw0=atan2(Y1−Y0,X1−X0)` in radians.
4. Record:

   | Point | Scene coordinates | Vicon world coordinates |
   |---|---|---|
   | O | (0, 0) | (X0=____, Y0=____) |
   | P | (1, 0) | (X1=____, Y1=____) |
   | → | `x0`=____ `y0`=____ `yaw0`=____ rad | |

   All subsequent commands use these three numbers as **`TWS = "x0 y0 yaw0"`**.
   Example: O=(1.20,0.40), P=(2.20,0.40) gives yaw0=atan2(0,1)=0,
   hence `TWS="1.2 0.4 0.0"`.

## A5 Robot Markers / Rigid Body

1. Rigidly attach at least 4 markers to G1's pelvis/base, clear of arm occlusion while walking.
2. Define the cluster as a rigid body such as `g1_base` in Vicon, with origin at pelvis
   center and x pointing forward.
3. Stream `g1_base` 6DoF in real time.

## A6 Print a Fiducial for AR Registration

1. Print a high-contrast, textured, asymmetric ARKit/ARCore reference image at A3 size
   and mount flat on a rigid board. Measure its actual width in meters, e.g. 0.40m×0.28m.
2. Place it flat or upright beside the corridor, visible to Vicon and the AR device.
3. Probe its center and +x edge direction to compute `T_world_fiducial`, using A4's
   method: center supplies translation, edge supplies yaw; record the normal if upright.
   AR clients detect its device-frame pose and combine it with `T_world_fiducial` to
   recover the world origin; see registration in Part B.

## A7 Motion-Capture Input: NatNet / `natnet_ros2` and Shared-Memory Validation

> This setup uses OptiTrack/Motive NatNet. [`natnet_ros2`](https://github.com/L2S-lab/natnet_ros2)
> publishes rigid-body poses as `geometry_msgs/PoseStamped` on `/<rigid-body-name>/pose`,
> in meters. The repository's `natnet_shm_writer` bridges them into `mocap_state_shm`.
> ROS2 is confined to the mocap host; downstream readers continue using
> `--localization vicon`, without rclpy in fedguide/on the robot host.
> For Vicon DataStream rather than NatNet, use `ViconDemoWriter` in `vicon_shm_plugin.py`,
> requiring `pyvicon_datastream` and mm→m division by 1000. Everything else is unchanged.

**Check coordinate/unit conventions; errors rotate or scale the entire downstream system:**

- Motive Streaming: Up Axis=Z, units=meters. NatNet/ROS already send meters; the writer
  does not divide by 1000 again.
- World frame: right-handed, z up, x forward, y left (`ar/schema/conventions.md`, Section 1).
- Set G1 pelvis rigid-body pivot to pelvis center and +x to robot forward (A5), avoiding
  a further mounting offset. Body `g1_base` publishes `/g1_base/pose`.

1. **Start NatNet→shm on the mocap host in its ROS2 environment:**

   ```bash
   # ① Start the driver: serverIP=Motive, clientIP=this host, serverType=multicast/unicast
   ros2 launch natnet_ros2 natnet_ros2.launch.py    # Or configure config/initiate.yaml per its README
   # Verify pose publication
   ros2 topic echo /g1_base/pose --once
   # ② Bridge /g1_base/pose into mocap_state_shm
   python -m genedynamics.deploy.localization.natnet_shm_writer --rigid-body g1_base
   ```

   Success prints `subscribing /g1_base/pose … → shm 'mocap_state_shm'`, followed by
   periodic `pos=(…)`. Layout `q13d` is int64 utime (µs) plus 13 float64 values:
   pos3 + quat_xyzw4 + vel3 + omega3. Velocities are finite differences of poses;
   use `--vel-lpf 0.2` for noise or `--z-offset` for an elevated pivot.
   Quaternion order and byte layout already match `vicon_shm_plugin.py`.
2. **Verify reading on the robot host:**

   ```bash
   python -c "from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin; \
   p=ViconShmPlugin({}); import time; time.sleep(0.3); print('state=',p.get_state(),'health=',p.health())"
   ```

   Move the robot and repeat; the pose should change and `health()=ok`.

---

# Part B — Four AR Client Options and Build Instructions

All connect to the same `ws://TWIN_IP:8766/` and render the same WorldSnapshot;
only engine and packaging differ. Apple options 1/2/3 require Mac+Xcode.
WebXR option 4 needs no build and uses Android Chrome only.

| Option | Device | Engine | Build host |
|---|---|---|---|
| 1 | Vision Pro | Native Swift/RealityKit | Mac/Xcode |
| 2 | Vision Pro | Unity + PolySpatial | Unity → Mac/Xcode |
| 3 | iPad/iPhone | Unity + ARKit | Unity → Mac/Xcode |
| 4 | Android phone/tablet | WebXR/three.js webpage | No build; browser |

---

## Option 1: RealityKit → Vision Pro, Native Swift

Build on Mac/Xcode; install on Vision Pro.

### 1.1 Create the Project

1. Xcode → File → New → Project → visionOS → App, named `CorridorTwinAR`, Interface=SwiftUI.
2. Add the four `.swift` files `Contract / EntityState / WorldClient / WorldRenderer`
   from `genedynamics/deploy/ar/clients/swift/Sources/CorridorTwin/` to the target,
   or import the local SPM package via `Package.swift`.

### 1.2 Connect Client and Renderer in the App Entry

Create `ImmersiveView.swift`:

```swift
import SwiftUI
import RealityKit
import ARKit
import CorridorTwin   // If imported as an SPM package

struct ImmersiveView: View {
    @State private var renderer: WorldRenderer?
    @State private var client: WorldClient?
    let twinURL = URL(string: "ws://TWIN_IP:8766/")!         // Replace with the twin server IP
    // Fiducial pose in Vicon world from A6; initially approximate with translation:
    let fiducialWorld = SIMD3<Float>(/*X*/0, /*Y*/0, /*Z*/0) // Fill from A6

    var body: some View {
        RealityView { content in
            let worldAnchor = AnchorEntity(world: .zero)     // Start at origin; registration updates transform
            content.add(worldAnchor)
            let r = WorldRenderer(worldAnchor: worldAnchor)
            let c = WorldClient(url: twinURL)
            c.onMessage = { [weak r] m in r?.enqueue(m) }
            c.connect()
            renderer = r; client = c
            // Drive interpolation each frame
            _ = content.subscribe(to: SceneEvents.Update.self) { _ in
                r.update(now: CACurrentMediaTime())
            }
        }
        .task { await runImageTracking() }                   // Registration, see 1.3
    }

    // 1.3 Registration: ARKit image tracking places worldAnchor at the Vicon world origin
    func runImageTracking() async {
        let session = ARKitSession()
        // Add A6's fiducial and actual width to the app's ARReferenceImage resources
        guard let refs = try? ReferenceImage.loadReferenceImages(inGroupNamed: "Fiducials") else { return }
        let provider = ImageTrackingProvider(referenceImages: refs)
        try? await session.run([provider])
        for await update in provider.anchorUpdates where update.anchor.isTracked {
            let T_device_fiducial = update.anchor.originFromAnchorTransform   // Fiducial in device frame
            // worldOrigin_in_device = T_device_fiducial * inv(T_fiducial_world)
            // Use known Vicon-world fiducial pose (A6) to recover device-frame world origin.
            // Assemble a 4x4 from T_world_fiducial; translation-only approximation uses position.
            // renderer?.worldAnchor.transform = Transform(matrix: worldOrigin_in_device)
        }
    }
}
```

> Registration combines the detected fiducial with its known Vicon-world pose to place
> `worldAnchor` at the Vicon origin. A more stable alternative is to add Vicon markers
> to Vision Pro, obtain headset world pose directly, and perform one-time hand-eye
> calibration without depending on image tracking.

### 1.3 Build and Deploy

1. In Xcode target `CorridorTwinAR`, select Apple ID/Team under Signing & Capabilities.
2. Select Apple Vision Pro hardware, paired in developer mode, or visionOS Simulator
   for initial logic checks.
3. Cmd+R to build/run; trust the developer certificate on Vision Pro on first use.
4. Enter the immersive space and look at the fiducial to register. Obstacles should
   remain fixed on the floor.

---

## Option 2: Unity → Vision Pro, PolySpatial

Develop in Unity on any OS, export an Xcode project, and package on Mac/Xcode.
Install on Vision Pro.

### 2.1 Project and Packages

1. Create a Unity Hub 3D (URP) project, recommended for PolySpatial; install visionOS
   Build Support in the Editor.
2. Install AR Foundation, Apple visionOS XR Plugin, the `com.unity.polyspatial` packages
   requiring Unity Pro/Industry licensing, and Newtonsoft Json
   (`com.unity.nuget.newtonsoft-json`).
3. Enable visionOS in XR Plug-in Management; set PolySpatial to Mixed Reality,
   using an unbounded volume for large scenes.

### 2.2 Scripts and Scene

1. Copy `WorldEntities.cs / WorldClient.cs / WorldRenderer.cs / FrameRegistration.cs`
   from `clients/unity/Scripts/` into `Assets/Scripts/`.
2. Create the depth-only `Assets/CorridorTwin/Occluder.shader`:

   ```shader
   Shader "CorridorTwin/Occluder" {
     SubShader { Tags{"Queue"="Geometry-1"} ColorMask 0 ZWrite On Pass {} }
   }
   ```

   Create `OccluderMat` from this shader.
3. Create an empty `CorridorTwin` object with:
   - `WorldClient`: `Url = ws://TWIN_IP:8766/`.
   - `FrameRegistration`: assign a child empty object to `WorldOriginAnchor`, positioned
     at the Vicon origin after registration.
   - `WorldRenderer`: assign local `WorldClient` to `Client`, registration to `Frame`,
     `OccluderMat` to `OccluderMaterial`, and `InterpolationDelay`=0.08.
4. Add `ARTrackedImageManager` and a Reference Image Library containing A6's fiducial
   with actual dimensions. In `trackedImagesChanged`, set
   `FrameRegistration.WorldOriginAnchor` from the fiducial's Vicon-world pose,
   following option 1's transform logic.

### 2.3 Build

1. File → Build Settings → visionOS → Build; choose an output directory to create Xcode project.
2. Open it on Mac, choose Signing Team and Apple Vision Pro, then Cmd+R.

---

## Option 3: Unity → iPad/iPhone/Tablet, ARKit

Develop in Unity on any OS, export Xcode, and package on Mac/Xcode for iPad/iPhone.

### 3.1 Project and Packages

1. Create a Unity Hub 3D (URP) project with iOS Build Support.
2. Install AR Foundation, ARKit XR Plugin, and Newtonsoft Json.
3. Enable ARKit under Project Settings → XR Plug-in Management → iOS.

### 3.2 Scripts and Scene

1. Copy the four scripts and create `OccluderMat` as in option 2, including its depth shader.
2. Add AR Session and XR Origin (AR); add `WorldClient`, `FrameRegistration`, and
   `WorldRenderer` to an empty `CorridorTwin`, with assignments from 2.2.
3. Register using `ARTrackedImageManager` and A6's fiducial in a Reference Image Library
   with its measured width. In `trackedImagesChanged`:

   ```csharp
   // Pseudocode: known fiducial Vicon pose Twf (A6), detected AR-session pose Taf.
   // World origin in AR-session coordinates = Taf * inverse(Twf).
   frameRegistration.WorldOriginAnchor.SetPositionAndRotation(worldOriginPos, worldOriginRot);
   ```

### 3.3 Build

1. File → Build Settings → iOS → Build to generate an Xcode project.
2. Open on Mac, select Signing Team and the iPad/iPhone, then Cmd+R.
3. Trust the developer on first use; point the device at the fiducial to register
   and view obstacles.

> Devices must be supported by ARCore for Android or ARKit for iPad/iPhone;
> recent iPads support ARKit. [[Link](https://developers.google.com/ar/devices#ios)]

---

## Option 4: WebXR, Android Browser Without Installation

No build is needed: open a static webpage in Chrome on Android phones/tablets.
This is the easiest demonstration route, without App Store, Mac, or app installation.
Tradeoffs: immersive AR (`immersive-ar`) is supported only in Android Chrome here;
iOS Safari does not expose it, and visionOS Safari has partial support from visionOS 2+.
Registration accuracy is lower than the first three options.

Starter page: `genedynamics/deploy/ar/clients/web/index.html`, using three.js/WebXR,
WebSocket subscription, interpolation, and occluder holdout. Axis mapping matches
RealityKit: `(-y,z,-x)`.

### 4.1 HTTPS Is Required

WebXR requires a secure context. Except on `localhost`, serve the page via HTTPS
or the browser will not start AR. For LAN tablet access, start a self-signed HTTPS
static server in `clients/web/`:

```bash
cd genedynamics/deploy/ar/clients/web
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 -subj "/CN=twin"
python -c "import http.server,ssl,functools as f; \
s=http.server.HTTPServer(('0.0.0.0',8443), f.partial(http.server.SimpleHTTPRequestHandler)); \
c=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); c.load_cert_chain('cert.pem','key.pem'); \
s.socket=c.wrap_socket(s.socket,server_side=True); s.serve_forever()"
```

### 4.2 WebSocket Connections and Mixed Content

The page defaults to `ws://<page-host>:8766/`, the `run_twin_server` endpoint.
HTTPS pages may block plaintext `ws://` as mixed content. Options:

- For demonstrations, temporarily allow insecure content for the site in Android Chrome
  under Site Settings → Insecure content → Allow.
- For regular operation, add TLS to world_ws using `wss://` and an nginx/caddy reverse
  proxy terminating TLS; set page `WS_URL` to `wss://...`.

### 4.3 Run

1. Start `run_twin_server` as in C2.
2. Open `https://<TWIN_IP>:8443/` in tablet Chrome and accept the self-signed certificate warning.
3. Select “Start AR” (ARButton), grant camera access, and enter AR.
4. The starter page registers by tap-to-place: hit-test the floor and place the world origin.
   This is sufficient for a quick demo, not Vicon-level accuracy. For greater accuracy,
   replace registration in `index.html` with A6 fiducial tracking and set `worldOrigin`
   using `T_world_fiducial`, as in options 1–3, or use those clients directly.
5. Obstacles, robot, and occluder should remain fixed to the floor and update with the server.

> See `clients/web/README.md` for browser support, HTTPS/wss, registration, and FlatBuffers.

---

# Part C — Per-Experiment Startup Order

> `TWS` is A4's measured `"x0 y0 yaw0"`; `TWIN_IP` is the twin server IP.

## C1 Start Motion Capture and Shared Memory, ROS2 Host

1. Start Motive; confirm stable `g1_base` tracking, and the fiducial if a rigid body,
   without dropped frames. Streaming must be Z-up/meters.
2. Start `natnet_ros2`; confirm `ros2 topic echo /g1_base/pose --once` shows a pose.
3. Start `python -m genedynamics.deploy.localization.natnet_shm_writer --rigid-body g1_base`
   to populate `mocap_state_shm`.
4. On the robot host, perform A7 step 2's `ViconShmPlugin` check: `health()=ok`,
   with pose changing when the robot moves.

## C2 Start the Twin Server

```bash
/opt/anaconda3/envs/fedguide/bin/python scripts/tasks/robot/humanoid/run_twin_server.py \
    --preset zone_d --t-world-scene <TWS> --localization vicon --port 8766
```

- Expect `[world_ws] serving ws://0.0.0.0:8766/ ... N entities`.
- Verify JSON via browser or `wscat -c ws://TWIN_IP:8766/`, containing
  `ball_L1/ball_R1/wall_*/goal/start`, `g1/base`, and `g1/occluder`.
  `g1/base` pose must follow the moving robot.
- For an initial check without Vicon, use `--localization mock`; the robot stays at start.

## C3 Generate a Certified Plan, Planner Host/Docker

```bash
docker run --rm -v "$PWD:/work" -w /work genedynamics/dev-cpu:torch \
    python scripts/tasks/robot/humanoid/replan_from_scene.py --preset zone_d --out results/ar/run1
```

- Expect `calibrated: best_idx=… m_track=… (CERT)` and artifact
  `results/ar/run1/level_1/seed_0/trajectory/trajectory.json`.
- Skip this step if a certified plan already exists.

## C4 Start and Register the AR Client

1. Open the chosen Part B app.
2. Look/point the device at A6's fiducial to register `worldAnchor` at the Vicon origin.
3. Verify alignment using a known small object at the start cross: its AR counterpart
   should coincide within the safety margin; Vicon's 1–2cm is sufficient here.
   The empty floor should show 2 virtual spheres and corridor walls.

## C5 Robot: Dry Run, Suspended Run, Ground Run

1. Place the G1 base at the green start cross, facing the +x arrow; begin suspended
   or partially supported.
2. Dry-run communications and coordinates without movement:

   ```bash
   python scripts/tasks/robot/humanoid/run_real_g1.py \
       --plan results/ar/run1/level_1/seed_0/trajectory/trajectory.json \
       --preset zone_d --t-world-scene <TWS> --network-interface eth0 --dry-run
   ```

   Expect `frame glue check: world start → scene pose (0.500, 0.000)` and `wiring OK`.
3. Remove `--dry-run` for suspended walking. Check gait startup, no missed steps,
   and smooth governor commands. Keep the emergency stop in hand.
4. With safety layers and emergency stop ready, place the robot on the start mark and run:

   ```bash
   python scripts/tasks/robot/humanoid/run_real_g1.py \
       --plan results/ar/run1/level_1/seed_0/trajectory/trajectory.json \
       --preset zone_d --t-world-scene <TWS> --network-interface eth0
   ```

   The robot follows the certified plan; the governor avoids injected AR obstacles online.

## C6 Observation and Acceptance

- Terminal: final `Exec min body-SDF: +0.xxx → CERTIFIED`, endpoint distance, and fall status.
- AR: floor-locked obstacles, correct real-robot occlusion in front of virtual walls,
  and smooth interpolation without jumps.
- Data: per-step JSONL and npz under `results/g1_corridor/real/` for replay/comparison.
- Acceptance: `certified_safe=True`, no collisions, actual-versus-planned error below
  the margin, and AR alignment error below the margin.

---

# D. Safety

- Keep the physical emergency stop in hand throughout. Start ground runs with easy
  zone_a/zone_d, not zone_c's squeeze geometry.
- Spark sport-mode lateral speed is clamped to ±0.3 m/s; zone_d's ground-truth margin
  is approximately +0.08m. Leave sufficient space.
- Before ground operation, enable `joint_limit + torque_limit + self_collision`
  safety layers and fall detection.

# E. Troubleshooting

| Symptom | Check |
|---|---|
| `ViconShmPlugin` cannot read | Writer not started, shared-memory name differs from `mocap_state_shm`, or upstream `natnet_ros2` has no poses; first run `ros2 topic echo /g1_base/pose` |
| `natnet_shm_writer` receives no poses | Topic differs from `/<rigid-body-name>/pose` (check `ros2 topic list`); QoS mismatch (add `--reliable` for a reliable publisher); wrong serverIP/clientIP/multicast-unicast settings |
| Pose rotated 90° or upside down | Motive Streaming is not Z-up, or rigid-body pivot orientation is wrong (A5) |
| Position differs by factor 1000 | Vicon DataStream mm path lacks division by 1000; NatNet already sends meters, use `natnet_shm_writer` |
| No AR stream | `TWIN_IP`, port, firewall, or inactive `run_twin_server` |
| AR obstacles drift/misalign | Wrong fiducial dimensions or inaccurate A6 `T_world_fiducial` |
| Wrong robot starting position | Not at start or incorrect `--t-world-scene`; recheck A4 |
| Dry-run pose is not (0.5,0) | Incorrect `T_world_scene` or marker-body orientation |
| Plan reports `VIOL`, not certified | Use full `replan_from_scene`, not `--fast`; adjust governor `m_track` if needed, measured optimum ~0.08 |

# F. Quick Reference

| Item | Value |
|---|---|
| Twin server | `scripts/tasks/robot/humanoid/run_twin_server.py` → `ws://TWIN_IP:8766/` |
| Replanning | `scripts/tasks/robot/humanoid/replan_from_scene.py`, Docker |
| Hardware execution | `scripts/tasks/robot/humanoid/run_real_g1.py`; check with `--dry-run` first |
| Mocap bridge | `python -m genedynamics.deploy.localization.natnet_shm_writer --rigid-body g1_base`; NatNet→shm on ROS2 host |
| Coordinate transform | `genedynamics/deploy/localization/scene_frame_plugin.py`, world→scene |
| Contract/conventions | `genedynamics/deploy/ar/schema/{world.fbs, conventions.md}` |
| AR clients | `genedynamics/deploy/ar/clients/{swift, unity, web}/` |
| Simulation recheck | `scripts/tasks/robot/humanoid/run_sport_mode_zones.py`, Docker |
