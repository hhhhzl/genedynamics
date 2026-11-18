# G1 Corridor 2GO: Real-Robot and AR Deployment Manual

> **Rewritten 2026-06-23 from hardware bring-up measurements.** Standing up, standing, and walking have **passed hardware validation**. Full-corridor execution, arm folding, motion-capture feedback, and AR feedback remain **unverified**, as marked below. This is the complete setup-to-execution runbook.
>
> The selected, hardware-tested control route is **high-level sport, rather than low-level lowcmd**: legs use `LocoClient.Move` (sport's built-in RL gait), arms use `rt/arm_sdk`, and standing uses sport FSM `SetFsmId(4)→(200)`. Reference: CMU SPARK `g1_real_agent`.

---

## 0. Overview: Three Devices and One Principle

Actual topology:
```
[iPhone]    Unity→Xcode AR app ── WiFi ──► Mac twin service

[Mac]  Algorithms/control (native ~/g1 environment)
   └─ Ethernet en7 ─► [Switch] ─┬─► [G1 robot 192.168.123.161]
                                └─► [Motive host 192.168.0.77]
                                      Direct NatNet multicast, no ROS
```
> A single Mac `en7` interface with a /16 subnet reaches both robot (`.123.x`) and motion capture (`.0.x`) on one Ethernet segment. **Motion capture uses wired NatNet, not WiFi or ROS.** See [`localization/mocap.md`](localization/mocap.md) for detailed setup and pitfalls.

**Principle: a shared digital twin.** One `SceneSource` (the sole obstacle source), one **motion-capture world frame**, and one `T_world_scene`. **AR-rendered obstacles and governor-avoided obstacles occupy the same physical locations.** The robot “perceives” virtual obstacles through its motion-capture localization and shared geometry; there is no vision-perception stack, and obstacles are virtual.

**Device responsibilities:**

| Device | Runs |
|---|---|
| **Mac** | Control (`run_real_g1`), twin service (`run_twin_server`), **direct motion-capture parsing (`natnet_shm_writer --natnet`)**, and offline planning in Docker |
| **Motion-capture host** | **Motive only**, directly streaming NatNet multicast, without ROS or a bridge |
| **iPhone** | Unity ARKit app rendering from the Mac twin service |
| **G1** | Ethernet-connected to Mac; SDK drives high-level sport |

---

## 1. One-Time Native Mac Environment and Network Setup

### 1.1 Why Native Mac Rather Than Docker, fedguide, or the Onboard PC
Hardware control requires `unitree_sdk2py`, arm64 `mujoco`, and direct robot DDS. `fedguide` is Rosetta x86 and cannot run arm64 mujoco; Docker on Mac cannot provide the required robot DDS multicast; the onboard PC lacks Internet access and uses Python3.8. Use an independent **native arm64 Mac venv (`~/g1/venv`) and source-built CycloneDDS**.

### 1.2 Install Once
**CycloneDDS:** macOS has no wheel or brew formula for `cyclonedds==0.10.2`, which `unitree_sdk2py` pins, so build from source:
```bash
git clone --depth 1 -b 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds /tmp/cdds
cmake -S /tmp/cdds -B /tmp/cdds/build -DCMAKE_INSTALL_PREFIX=$HOME/g1/cyclonedds -DBUILD_IDLC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cdds/build --target install -j4    # Produces libddsc and idlc
```
**venv and dependencies:** native arm64 Homebrew Python3.12:
```bash
/opt/homebrew/bin/python3 -m venv $HOME/g1/venv
CYCLONEDDS_HOME=$HOME/g1/cyclonedds $HOME/g1/venv/bin/pip install "cyclonedds==0.10.2"
CYCLONEDDS_HOME=$HOME/g1/cyclonedds $HOME/g1/venv/bin/pip install "git+https://github.com/unitreerobotics/unitree_sdk2_python.git"
$HOME/g1/venv/bin/pip install "mujoco" "numpy<2" scipy "jax[cpu]" "websockets>=12" "flatbuffers>=2.0"
```
**Required environment variables for every run** (a shell alias is convenient):
```bash
export CYCLONEDDS_HOME=$HOME/g1/cyclonedds DYLD_LIBRARY_PATH=$HOME/g1/cyclonedds/lib
```

### 1.3 Network
- Connect Mac↔G1 by Ethernet. Configure the Mac USB adapter (**`en7`** on this machine) with a static address:
  ```bash
  sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.123.222 255.255.255.0   # Leave router blank
  ```
  Check `ping 192.168.123.161` (robot mainboard). Use `--network-interface en7`; `eth0` applies to the onboard PC, not this Mac.
- **Connect Mac WiFi to the lab network** to reach the motion-capture host and iPhone.

---

## 2. Validated and Pending Capabilities

✅ **Hardware-validated on 2026-06-23:**
- Mac↔G1 DDS: reads `rt/lowstate`, with35 motor channels; **0–28 are active joints**,29–34 hands/reserved; `mode_machine=5`.
- **Standing up:** `GetFsmId → SetFsmId(4)` (LockStand) `→ SetFsmId(200)` (MainMode, walking enabled).
  - ⚠️ The pip SDK's **`Start()`=SetFsmId(500)** and **`Squat2StandUp()`=706** use IDs rejected by this firmware (`Invalid FSM ID 7302`). Use generic **`SetFsmId` with4/200**, following SPARK.
- **Standing balance:** in MainMode, small autonomous balance steps are normal, not loss of control.
- **Walking:** `Move(vx)` produces steps. **Feet must bear weight** because sport uses contact gating; a fully suspended robot stands but does not walk. `vx=0.1` produces shuffling; `0.2+` produces clear forward steps.
- **Direct motion capture, 2026-06-24, without robot:** `Motive NatNet multicast → natnet_shm_writer --natnet → mocap_state_shm → ViconShmPlugin` reads G1 base pose end to end, with millimeter-level stability. **No ROS anywhere.** See [`localization/mocap.md`](localization/mocap.md).

⏳ **Pending validation:**
- Full `run_real_g1` corridor: stand-up, governor, walking, arm folding together.
- **arm_sdk folding** on hardware for the first time. If arms swing erratically, adjust `motor_cmd.mode`: SPARK leaves default0, official arm_sdk example uses1.
- **Motion-capture feedback:** robot and capture together with `run_real_g1 --localization vicon`, replacing mock's open-loop heading drift, including rigid-body +x calibration.
- AR feedback: iPhone registration and robot avoidance of virtual obstacles.

**Implemented deployment support** in `genedynamics/deploy/io/unitree_g1_io.py`: `_bring_up_sport()` follows the current FSM through the repaired full `…→4→200` sequence; `_send_arm_sdk()` uses `rt/arm_sdk` and `kNotUsedJoint=29` weight channel; `LowCmd_` uses its default factory and `mode_machine`. `run_real_g1` therefore needs no changes: reset stands up automatically, legs use Move, arms use arm_sdk.

---

## 3. Robot Motion Validation Ladder: Repeat After Each Power-On

> **Safety: this lab has no e-stop. Run all motion from your own terminal with Ctrl-C ready, a gantry/support arrangement, and a hand near the power switch.** Velocity commands expire automatically after1 second; stopping the script stops the robot in approximately1 s.

After environment/network setup and successful `ping .161`:

### 3.0 Read-Only Communication Check: No Motion
```bash
$HOME/g1/venv/bin/python - <<'PY'
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
ChannelFactoryInitialize(0, "en7")
got={}; ChannelSubscriber("rt/lowstate", LowState_).Init(lambda m: got.__setitem__('m',m), 10)
t=time.time()
while time.time()-t<10 and 'm' not in got: time.sleep(0.2)   # DDS startup may take tens of seconds after power-on
m=got.get('m')
print("lowstate:", "OK" if m else "None (robot still starting?)", "| mode_machine=", m.mode_machine if m else None)
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
lc=LocoClient(); lc.SetTimeout(3.0); lc.Init(); print("current FSM:", lc.GetFsmId())
PY
```
> After power-on, ping works before the `rt/lowstate` motion service starts. **Wait tens of seconds and retry if no state arrives.**

### 3.1 Stand Up: `~/g1/bringup_test.py`, Enter Between Steps
At startup the robot is in **ZeroTorque, FSM0, compliant**. Provide ground contact/clearance or gantry support:
```bash
$HOME/g1/venv/bin/python $HOME/g1/bringup_test.py
```
The script reads current FSM and follows `0→1→4→200`, requiring Enter at each step. **`SetFsmId(4)` makes the robot exert force and stand.** FSM200 means success. To relax: `… bringup_test.py --damp`.
> ⚠️ **Never apply `--damp` directly while the robot supports itself:** the legs will collapse. Lower it onto support or suspend it first.

### 3.2 Short Walk: `~/g1/move_test.py`
**Feet must bear weight** for contact gating. In MainMode, skip stand-up and issue Move:
```bash
$HOME/g1/venv/bin/python $HOME/g1/move_test.py --vx 0.2 --sec 3.0
```
Check for clear forward stepping rather than shuffling. If still not stepping, try `--vx 0.3`.

### 3.3 Full `run_real_g1`: Stand-Up, Governor, Walking, Arm Folding
```bash
$HOME/g1/venv/bin/python scripts/tasks/robot/humanoid/run_real_g1.py \
    --plan results/humanoid/corridor_2d/main/twogo_zone_a/level_1/seed_0/trajectory/trajectory.json \
    --preset zone_a --localization mock --network-interface en7 \
    --plan-speed 0.35 --max-steps 150
```
- `--localization mock` uses a fixed initial pose without capture, first checking the **complete software chain**: stand-up, governor, Move, arm_sdk.
- Reset automatically performs `4→200`; already being in MainMode is a no-op.
- **⚠️ Watch the first arm fold: smooth folding is expected; erratic swinging requires immediate Ctrl-C**, followed by arm_sdk mode inspection.

---

## 4. Direct NatNet Motion Capture: No ROS

> **Validated end to end on2026-06-24. Full instructions/pitfalls are in [`localization/mocap.md`](localization/mocap.md); essentials follow.**

Flow: `Motive (NatNet multicast, Z-up, Rigid Bodies) → Mac en7 → natnet_shm_writer --natnet → mocap_state_shm → --localization vicon`. **Motive sends NatNet directly; Mac parses directly; no ROS/UDP bridge.** Earlier `mocap_udp_bridge.py`/`natnet_ros2` routes are obsolete for this setup.

### 4.1 Network: Temporary, Repeat After Restart
```bash
sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.0.100 255.255.0.0  # /16 reaches capture and robot
sudo route -n add -host 239.255.42.99 -interface en7                          # Route multicast through Ethernet
ping 192.168.0.77   # Motive host
```
⚠️ **Do not assign Mac `.0.77`, which conflicts with Motive. The multicast route is required:** macOS otherwise defaults to WiFi and en7 receives nothing.

### 4.2 Motive: Persistent One-Time Settings
Settings → Streaming: **Enable**, **Local Interface=`192.168.0.77`** (the adapter toward Mac, critical), **Multicast**, **Rigid Bodies=ON**, **Up Axis=Z**. Create G1's rigid body with **+x forward and origin at the pelvis**; record its **Streaming ID, `5` here**.

### 4.3 Subscribe and Verify
```bash
# Keep this shared-memory writer running
PYTHONPATH=<repo> $HOME/g1/venv/bin/python -m genedynamics.deploy.localization.natnet_shm_writer \
    --natnet --local-ip 192.168.0.100 --server 192.168.0.77 --rigid-body-id 5
# Verify the exact path read by the robot
python -c "from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin; \
import time,numpy as np; p=ViconShmPlugin({}); time.sleep(0.3); print(np.round(p.get_state()[0],3))"
```
A stable stationary pose indicates readiness for feedback. Use `--localization vicon` for `run_twin_server`/`run_real_g1`.
> **Wireless:** select Motive `Unicast` and add `--unicast`, **for debugging only**. Hardware feedback uses Ethernet because WiFi jitter/loss degrades control. See [`localization/mocap.md`](localization/mocap.md) §5.

### 4.4 `T_world_scene`: Scene Versus Motion-Capture World
**The Motive world origin is already fixed and confirmed; do not change it.** Align the scene to it:
- **Direct alignment, simplest:** `TWS = "0 0 0"`, no measurement required. Locate Motive origin/+x, then place scene originO there and align corridor+x with Motive+x.
- **Two-point method when origins cannot coincide:** place a reflective marker atO and read `(X0,Y0)`, then at scene(1,0) and read `(X1,Y1)`. Set `x0=X0, y0=Y0, yaw0=atan2(Y1-Y0, X1-X0)`; pass `--t-world-scene x0 y0 yaw0` identically to twin and robot.
- Use a marker to read Motive coordinates, **not the camera-calibration wand**.

---

## 5. iPhone AR Through Unity→Xcode

### 5.1 Make the Robot Use AR Obstacles
The scene rendered by `run_twin_server` and consumed by `run_real_g1` **must be the same `SceneSource`**:
- **Static obstacles:** identical `--preset`, such as `zone_a`.
- **Edited/dynamic scene:** save the twin contract as JSON and use `run_real_g1 --scene-file <ar_scene.json>`.

The governor automatically avoids through `body_sdf_scene = corridor_scene_to_dict(scene)`, already connected in [run_real_g1.py](../../scripts/tasks/robot/humanoid/run_real_g1.py), lines77–81. **Perception is analytical: shared geometry + motion-capture pose → tightened command envelope and lateral steering. There is no vision stack.**

### 5.2 Register AR to the Robot's Starting Frame

> **AR registration error affects visual alignment, not obstacle avoidance.** The robot avoids using motion capture/shared obstacles, independently of AR alignment; see [ar/clients/README.md](ar/clients/README.md). Approximate visual registration is sufficient; perfection is unnecessary.

**The frame chain simplifies because Motive origin equals scene origin:**
- World origin=sceneO, +x along corridor, so **`T_world_scene` is identity (`"0 0 0"`)**.
- Contract obstacles are in scene=world coordinates. World→Unity mapping is **`(-y, z, x)`**, one right-to-left-handed conversion in [FrameRegistration.cs](ar/clients/unity/Scripts/FrameRegistration.cs).
- **AR only places `WorldOriginAnchor` at the motion-capture world origin's pose within the iPhone AR session.** Obstacles are its children and align together.

Two registration routes: **A is the current/fastest route; B is the more stable motion-capture-room upgrade.**

#### Method A: Fiducial Image Registration, Current Unity ARKit Route
Place a printed image at a known world pose; ARKit detects it and recovers world origin. With identityTWS, place it exactly atO.
1. Print a high-contrast, richly textured, **asymmetric** ARKit Reference Image on rigid A3 backing; measure width/height in meters.
2. Place it flat atO with image+x along corridor+x: **`T_world_fiducial` becomes identity**.
   - IfO is unavailable, use another known world pose. Easiest: add markers to the board, create a Motive rigid body, and read `T_world_fiducial` directly.
3. Unity project: AR Foundation + ARKit XR Plugin + Newtonsoft Json.
   - Add `AR Session`, `XR Origin (AR)`, `ARTrackedImageManager`, and a Reference Image Library containing the image/physical dimensions.
   - On empty `CorridorTwin`, attach `WorldClient` (`Url = ws://<Mac_WiFi_IP>:8766/`), `FrameRegistration` (assign an empty child to `WorldOriginAnchor`), and `WorldRenderer` (connect `Client`/`Frame`, assign occluder material).
4. Add an iPhone registration script; the repository currently has the OpenXR/Quest route only. In `trackedImagesChanged`, set the anchor from the recognized image:
   ```csharp
   // ARKitImageRegistration.cs: attach beside ARTrackedImageManager
   public FrameRegistration Frame;          // This script sets its WorldOriginAnchor
   public string FiducialName = "corridor_origin";
   void Apply(ARTrackedImage img) {
       if (img.referenceImage.name != FiducialName) return;
       if (img.trackingState != TrackingState.Tracking) return;
       // Fiducial at world origin O: image pose in AR approximates world-origin pose
       Frame.WorldOriginAnchor.SetPositionAndRotation(img.transform.position, img.transform.rotation);
       // If not at O: anchor = imgPose * inverse(T_world_fiducial)
   }
   ```
   - ⚠️ Check the **orientation convention**: ARKit's image axes may differ by a fixed rotation from world(x forward/y left/z up) and Unity `(-y,z,x)`, depending on placement/ARKit conventions. Run first and inspect whole-scene rotation; multiply a fixed correction if needed. Aligning image+x with corridor+x minimizes this.
5. Connect iPhone/Mac to the same WiFi, launch, and **briefly view the fiducial**; obstacles should lock to the floor.

#### Method B: Track iPhone as a Rigid Body, Recommended Stable Upgrade
No fiducial, continuous drift correction, direct world-origin registration:
1. Attach3–4 markers, create Motive body `ar_phone` with camera-centered local origin, and read its world pose through the §4 NatNet route.
2. Feed pose to twin `populate_headset` in [ar/producers/tracker_pose.py](ar/producers/tracker_pose.py), publishing a `viewer` entity (`headset/base`) in the same stream.
3. Use [OpenXRAnchorProvider.cs](ar/clients/unity/Scripts/OpenXRAnchorProvider.cs). It is **vendor-neutral and needs only Camera + FrameRegistration**, so iPhone's AR camera works: `HeadPoseSource`=AR camera, `Frame`=FrameRegistration. It solves `anchor = camera_unity ∘ L⁻¹` each frame and EMA-smooths, continuously aligning holograms to motion capture and correcting ARKit drift.
4. Connect `WorldRenderer.AnchorProvider`; set `LocalHeadsetId = headset/base` to hide the device's own proxy.
- Cost: an extra phone-motion-capture feed. Benefit: persistent alignment without drift, submillimeter precision, more stable than one-shot fiducials.

#### Verify Alignment
- At the green start mark, scene(0.5,0), AR start/centerline walls should match floor tape within **a few centimeters**.
- Walk along centerline; the AR centerline follows and the wall stands atx=1.7–2.0.
- Large offsets suggest skewed fiducial,wrong dimensions,missing orientation correction(A),or missing phone rigid-body feed(B).
- **Registration error is visual only; robot avoidance is unaffected.** A demonstration offset by a few centimeters can still run.

### 5.3 Virtual-Obstacle Collisions and Falls
- Virtual obstacles have **no physical contact**; holograms cannot physically trip the robot.
- Governor steering acts in advance: observed blind following penetrated−0.267 m; governor avoidance achieved+0.004 m.
- Insufficient avoidance lets the robot pass through the hologram, producing a body-SDF log violation (`certified_safe=False`), a logical failure rather than a physical collision.
- Actual falling risk comes from unstable gait/sim-to-real, independently of virtual obstacles.
- ⚠️ Tight scenes(zone_d/b) have execution-clearance limits around**+0.08 m** (`m_track≈0.08`). **Start with the widest zone_a.**
- This is AR's value for avoidance testing: virtual-collision failures are recorded without collision damage to the robot.

---

## 6. Motion-Capture Feedback Integration: Mock to Vicon

> Before a full corridor, isolate **motion capture→feedback**. `mock` is open-loop, using a fixed starting pose and drifting after turns; `vicon` uses actual pose. Another drift source is **rigid-body heading**, calibrated below.

### 6.1 Prerequisites: Network and Two Processes
- Set en7 **/16**: `sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.0.100 255.255.0.0`; both `ping 192.168.0.77` and `ping 192.168.123.161` must work, with §4.1 multicast routing.
- **TerminalA:** keep `natnet_shm_writer --natnet …` running.
- **TerminalB:** `ViconShmPlugin({}).get_state()` returns stable pose (§4.3).

### 6.2 Rigid-Body +x Heading Calibration: Required Initially
**Why:** rigid-body **+x must match robot forward**. A constant offsetα between streamed yaw/actual heading rotates body-frame velocity requests byα, causing diagonal motion/drift: a second source beyond mock open-loop drift.

Walk briefly and compare displacement heading with rigid-body yaw. Save as `~/g1/heading_check.py`, run with `PYTHONPATH=<repo>`:
```python
import time, math, numpy as np
from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin

def yaw_deg(pose):                       # pose=[x,y,z, qw,qx,qy,qz] (z-up)
    w,x,y,z = pose[3],pose[4],pose[5],pose[6]
    return math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))

p = ViconShmPlugin({})
def read():
    pose,_ = p.get_state(); return np.array(pose[:2]), yaw_deg(pose)

t0=time.time(); P=[]; Y=[]                # Stationary 2 s: initial position and yaw
while time.time()-t0<2.0:
    xy,yw=read(); P.append(xy); Y.append(yw); time.sleep(0.05)
p0=np.mean(P,axis=0); yaw0=float(np.mean(Y))
print(f"[Stationary] position=({p0[0]:+.3f},{p0[1]:+.3f})  yaw_world={yaw0:+.1f}°")
input("→ Walk robot forward ~0.5 m toward corridor +x (another terminal: move_test.py --vx 0.2 --sec 3), then press Enter…")

far=p0.copy(); d=0.0; t0=time.time()      # Select the frame farthest from the starting point
while time.time()-t0<1.0:
    xy,_=read()
    if np.linalg.norm(xy-p0)>d: d=np.linalg.norm(xy-p0); far=xy
    time.sleep(0.05)
disp=far-p0; dist=float(np.linalg.norm(disp))
if dist<0.15:
    print(f"⚠️ Traveled only {dist:.2f} m; too short for a reliable check. Repeat over at least 0.3 m."); raise SystemExit
heading=math.degrees(math.atan2(disp[1],disp[0]))
delta=(heading-yaw0+180)%360-180
print(f"[Walking] displacement=({disp[0]:+.3f},{disp[1]:+.3f}) |{dist:.2f}m|  heading={heading:+.1f}°")
print(f"Rigid-body +x versus robot-forward offset Δ = {delta:+.1f}°  →  " +
      ("✅ Aligned (|Δ|<10°)" if abs(delta)<10 else "⚠️ Misaligned: correct the Motive rigid-body orientation or include Δ in --t-world-scene yaw"))
print(f"If the robot currently faces scene +x, set --t-world-scene yaw to {math.radians(yaw0):+.3f} rad (={yaw0:+.1f}°)")
```
**Interpretation:** `|Δ|<~10°` is aligned; constantΔ means correct the rigid-body heading in Motive(preferred) or include it in `--t-world-scene` yaw; `Δ≈±90/180°` means rebuild the rigid-body orientation. Final `yaw0` feeds §6.3 registration.

### 6.3 Register `T_world_scene`: Place Robot at Scene Start
Physically place the robot at zone_a **(0.5,0), facing+x**, mapping world pose to scene(0.5,0,0):
- **(a) Motive origin=scene origin:** place the robot at the corresponding world point; use `--t-world-scene 0 0 0`.
- **(b) Robot cannot be moved:** from measured `(Xr,Yr,yaw_r)`, set `yaw0=yaw_r`, `(x0,y0)=(Xr,Yr) − R(yaw0)·(0.5,0)`; pass `--t-world-scene x0 y0 yaw0`.
- Formula/two-point details: §4.4.

### 6.4 Live Frame-Glue Check: Actual Capture, No Motor Motion
Read live capture through the same `SceneFrameLocalization` as `run_real_g1`:
```python
import time, math
from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin
from genedynamics.deploy.localization.scene_frame_plugin import SceneFrameLocalization
TWS={"x":0.0,"y":0.0,"yaw":0.0}          # Replace with your --t-world-scene values
loc=SceneFrameLocalization(ViconShmPlugin({}), TWS)
for _ in range(6):
    pose,_=loc.get_state(); w,x,y,z=pose[3],pose[4],pose[5],pose[6]
    yaw=math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    print(f"scene pose=({pose[0]:+.3f},{pose[1]:+.3f})  yaw_scene={yaw:+.1f}°")
    time.sleep(0.3)
```
At the start facing+x, expect approximately **(0.5,0),yaw_scene≈0°**. Otherwise checkTWS,placement,and rigid-body heading (§6.2).
> `run_real_g1 --dry-run` checks frame-glue **mathematics at a fixed start**, without shared memory. This script checks the actual live motion-capture→scene mapping; both are needed.

### 6.5 First Minimal Feedback Run: No Obstacles
After §6.2/§6.4 pass, run minimal feedback before AR/obstacles:
```bash
# Terminal A: keep the writer running (§4.3)
# Terminal B:
$HOME/g1/venv/bin/python scripts/tasks/robot/humanoid/run_real_g1.py \
    --plan results/humanoid/corridor_2d/main/twogo_zone_a/level_1/seed_0/trajectory/trajectory.json \
    --preset zone_a --localization vicon --t-world-scene 0 0 0 \
    --network-interface en7 --plan-speed 0.3 --max-steps 150
```
- **Safety:** gantry/manual support, Ctrl-C ready, hand near power, following §3.
- **Observe:** feedback follows real capture instead of mock's turn-related drift; `Pelvis z range` is measured height, no longer constant0.793.
- Once stable, proceed to §7 full zone_a with AR/center wall.

---

## 7. End-to-End zone_a in the Motion-Capture Room

**Geometry:** corridor4.0 m long,1.6 m wide(walls y=±0.8); start scene(0.5,0),goal(3.5,0); **one thin centerline wall** x∈[1.7,2.0],y≈0,requiring lateral avoidance.

> Complete §6 heading calibration/minimal feedback first. This adds AR and the center wall.

1. **Mark the floor frame:** red crossO(0,0); chalk/laser+x centerline4 m with0.5 m marks; green start crossx=0.5/forward arrow; blue goalx=3.5; obstacle intervalx=1.7–2.0 without a physical object; sidewall linesy=±0.8.
2. **TWS:** align sceneO with fixed Motive origin (`TWS="0 0 0"`) or measure by two points (§4.4).
3. **Place G1 in capture volume/start Motive:** NatNet streams pose automatically; record streaming ID,here`5`.
4. **Mac capture subscriber:** `natnet_shm_writer --natnet --local-ip 192.168.0.100 --server 192.168.0.77 --rigid-body-id 5`; see §4/[mocap.md](localization/mocap.md).
5. **Twin:** Mac `run_twin_server.py --preset zone_a --t-world-scene <TWS> --localization vicon`.
6. **iPhone:** connect `ws://<Mac>:8766/`,register fiducial; holographic corridor/center wall lock atx=1.85.
7. **Dry-run frame glue, no motors:** `run_real_g1.py --plan <…twogo_zone_a…trajectory.json> --preset zone_a --t-world-scene <TWS> --localization vicon --dry-run`; expect `world start → scene pose (0.500, 0.000)`.
8. **Hardware run:** robot on green cross facing+x; remove `--dry-run`,initially keep short `--plan-speed 0.35 --max-steps` execution. Robot laterally avoids center wall,AR shows avoidance,and occluder makes the real robot correctly occlude the wall.

> Existing plans: `results/humanoid/corridor_2d/main/twogo_zone_a/level_1/seed_*/trajectory/trajectory.json`. Replan/calibrate AR scenes in Docker via `replan_from_scene.py`/`governor_calibrate.py`. Measured best `m_track≈0.08`; uncalibrated run_real_g1 defaults to0.08.

---

## 8. Safety, Troubleshooting, and Quick Reference

### Safety Requirements
- **No e-stop:** run motion yourself with Ctrl-C ready,hand near power,and initial gantry/support/fall protection.
- Never suddenly Damp a self-supporting robot; support it first.
- Increase speed gradually;start with easy zone_a and sufficient clearance for tight zone_d/b.
- Watch the first arm_sdk run; immediate Ctrl-C for erratic arms.

### Troubleshooting
| Symptom | Check |
|---|---|
| Cannot `ping .161` | Cable/en7 static IP `192.168.123.222/24` |
| Ping works, no `rt/lowstate` | Robot still booting; wait tens of seconds |
| `SetFsmId(4)/(200)` rejected | Use4/200,not pip500/706; alternatively enter LockStand using remote/App |
| Standing, Move produces no steps | Feet not weight-bearing/contact gate,or vx too low; try0.2–0.3 |
| Mock run reports `Fell over` immediately | Mock formerly reported z=0,causing false fall detection; repaired to0.793 standing height. Hardware `vicon` uses actual height. “SparkRL policy diverged” verdict text is simulation wording,ignore on hardware |
| Erratic arms | `motor_cmd.mode`: SPARK0/official1; inspect `_send_arm_sdk` |
| No `--localization vicon` data | Writer absent,wrong Motive Local Interface,Rigid Bodies disabled,or missing multicast route `route add 239.255.42.99 en7`; use layered probes in [mocap.md](localization/mocap.md) §2 |
| AR drift/misalignment | Incorrect `T_world_scene` or measured fiducial world pose |
| Dry-run pose not(0.5,0) | Incorrect TWS,robot not at start,or rigid-body orientation error |

### Quick Reference
| Item | Value |
|---|---|
| Native Mac environment | `~/g1/venv`; `CYCLONEDDS_HOME=$HOME/g1/cyclonedds DYLD_LIBRARY_PATH=$HOME/g1/cyclonedds/lib` |
| Interface | Mac `en7`(192.168.123.222);robot`.161` |
| Stand-up FSM | `GetFsmId → SetFsmId(4) → SetFsmId(200)` (LockStand→MainMode) |
| Stand/walk scripts | `~/g1/bringup_test.py`, `~/g1/move_test.py` |
| Capture subscription | `natnet_shm_writer --natnet --local-ip 192.168.0.100 --server 192.168.0.77 --rigid-body-id 5`; see [mocap.md](localization/mocap.md) |
| Hardware entry | `scripts/tasks/robot/humanoid/run_real_g1.py`; check `--dry-run` first |
| Twin service | `scripts/tasks/robot/humanoid/run_twin_server.py` → `ws://<Mac>:8766/` |
| Hardware IO | `genedynamics/deploy/io/unitree_g1_io.py` (`_bring_up_sport`/`_send_arm_sdk`) |
| Plan | `results/humanoid/corridor_2d/main/twogo_zone_a/.../trajectory.json` |
| References | CMU SPARK `g1_real_agent` (`enerdynamics/spark/`);Unitree G1 SDK documentation |
