# Motion Capture (OptiTrack/Motive · NatNet) → Mac Setup and Troubleshooting Record

> **End-to-end test passed on 2026-06-24** (without the robot):
> `Motive(NatNet multicast)→ Mac en7 → natnet_shm_writer --natnet → mocap_state_shm → ViconShmPlugin`.
> The G1 base pose read through this path matched Motive exactly.
>
> This guide covers **connecting a Mac to motion capture, testing it, and encountered
> pitfalls**. For the complete hardware corridor workflow, see
> [`../REAL_DEPLOY_G1_CN.md`](../REAL_DEPLOY_G1_CN.md). This provides the detailed background
> for its Section 4.

---

## 0. Final Architecture and Three Key Findings

```
[Motive motion-capture host 192.168.0.77]
   Streaming: Enable / Local Interface=192.168.0.77 / Multicast 239.255.42.99
              data:1511 cmd:1510 / Rigid Bodies=ON / Up Axis=Z / meters
        │  NatNet multicast (UDP), direct, no ROS
        ▼  Ethernet (switch/cable)
[Mac  en7  192.168.0.100]
   natnet_shm_writer.py --natnet      ← Direct parsing, no ROS
        │  Write
        ▼
   mocap_state_shm  (q13d: utime + pos3 + quat_xyzw4 + vel3 + omega3)
        │  Read
        ▼
   ViconShmPlugin   == --localization vicon in run_real_g1.py / run_twin_server.py
```

**Three key findings:**

1. **No ROS.** The earlier assumption that `natnet_ros2` plus a UDP bridge was necessary
   was incorrect. Motive sends NatNet directly and the Mac parses it directly.
   The `mocap_udp_bridge.py` route is obsolete for this setup.
2. **The robot consumes the NatNet stream, not Motive's screen display.** The display
   is Y-up for people; the stream is Z-up for the robot. Axis conventions can differ
   (Section 6). **Always calibrate and validate against the stream.**
3. **Networking failures came from interface selection on the Mac/Motive sides, not code.**
   Once interfaces were correct, parsing was stable at millimeter scale.

---

## 1. Step-by-Step Setup from Scratch

> This procedure incorporates all encountered fixes. After restarting the Mac or
> reconnecting Ethernet, **repeat steps 2 and 3** because those settings are temporary.
> Motive settings in steps 4 and 5 persist in its configuration.

### Step 1 · Physical Connection

Mac USB Ethernet adapter (**`en7`** here) → switch → Motive host and optionally G1
on the same Ethernet segment.

- Motion-capture testing only: one cable between Mac and Motive suffices.
- Closed-loop robot operation: Mac, Motive (`192.168.0.77`), and G1 (`192.168.123.161`)
  share **one Ethernet segment/switch**. One Mac `en7` reaches both using /16 in step 2.

### Step 2 · Static Mac IP (Temporary; Reset After Restart)

```bash
# Motion capture only (/24 reaches 192.168.0.x):
sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.0.100 255.255.255.0

# Closed-loop robot operation (/16 reaches both .0.x mocap and .123.x robot):
sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.0.100 255.255.0.0
```

- ⚠️ **Never assign `192.168.0.77` to the Mac**: it belongs to Motive and causes an IP
  conflict, as encountered during setup. Use an unused address such as `.100`.
- Verify `ping 192.168.0.77` reaches Motive; for closed-loop operation,
  `ping 192.168.123.161` must also reach the robot.

### Step 3 · Route Multicast Through Ethernet (Temporary; Reset After Restart)

```bash
sudo route -n add -host 239.255.42.99 -interface en7
```

- **Why this matters:** macOS defaults multicast (`239.x`) to `en0` (WiFi). Without
  this route, **no NatNet packets arrive on en7**, even with correct IP and successful
  ping; this caused a long debugging delay.
- Our `--natnet` code also uses `local_ip` with `IP_ADD_MEMBERSHIP` to join the group
  on en7. Keep the explicit route as well.

### Step 4 · Motive Settings (Persistent; Configure Once)

Motive → Settings → **Streaming**:

| Setting | Value | Notes |
|---|---|---|
| **Enable** | ON | Enable NatNet |
| **Local Interface** | **`192.168.0.77`** | ⚠️ Select the interface facing the Mac. This was the final missing fix: the wrong interface yielded no packets at the Mac, only empty frames |
| **Transmission Type** | **Multicast** | Recommended for Ethernet; use Unicast for WiFi (Section 5) |
| **Rigid Bodies** | **ON** | ⚠️ OFF gives no rigid bodies/empty frames. Only this stream is needed; Markers/Skeletons can be disabled |
| **Up Axis** | **Z-Axis** | Matches our world frame (z-up, meters); no axis conversion needed |
| Multicast / ports | `239.255.42.99` / data `1511` cmd `1510` | NatNet defaults, matching code defaults |

### Step 5 · Create/Confirm the G1 Rigid Body and Record Its Streaming ID

1. Place G1 in the capture volume; its markers must appear in Motive's live view.
2. Select markers → right-click **Create Rigid Body**, e.g. `g1_base`.
3. **Align body +x with robot forward and set the origin at the pelvis center**;
   otherwise heading has a constant offset (Section 6).
4. Record the **Streaming ID** from rigid-body properties. This G1 uses **`5`**;
   pass it through `--rigid-body-id`.

### Step 6 · Run the Subscriber (Shared-Memory Writer)

```bash
REPO=/Users/zhilinhe/Desktop/hhhhzl/EduGetRicher/CMU/projects/enerdynamics
PYTHONPATH=$REPO $HOME/g1/venv/bin/python -m genedynamics.deploy.localization.natnet_shm_writer \
    --natnet --local-ip 192.168.0.100 --server 192.168.0.77 --rigid-body-id 5
```

- `--local-ip` is the Mac's motion-capture-network IP, used to join multicast on the
  correct interface; **required**.
- Omitting `--rigid-body-id` selects the first tracked rigid body, acceptable if only G1 is present.
- Add `--unicast` when Motive is set to Unicast.

### Step 7 · Validate Through the Robot's Actual Read Path

```bash
PYTHONPATH=$REPO $HOME/g1/venv/bin/python - <<'PY'
import time, numpy as np
from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin
p=None
for _ in range(60):
    try: p=ViconShmPlugin({}); break
    except Exception: time.sleep(0.1)
for _ in range(5):
    pose,twist=p.get_state()
    print("pose=",np.round(pose,3)," twist=",np.round(twist,3))
    time.sleep(0.25)
PY
```

A stationary robot should yield stable `pose=[x y z | qw qx qy qz]` and `twist≈0`.
**Closed-loop localization is now ready**; `run_real_g1.py --localization vicon`
reads this shared memory directly.

---

## 2. Layer-by-Layer Diagnostics

Validate each layer independently, in this order:

```bash
# (a) Network: can the Mac ping the Motive host?
ping -c3 192.168.0.77

# (b) Reception: are NatNet multicast packets arriving on en7? Count without parsing.
$HOME/g1/venv/bin/python - <<'PY'
import socket,struct,time
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind(("",1511)); s.setsockopt(socket.IPPROTO_IP,socket.IP_ADD_MEMBERSHIP,
    struct.pack("4s4s",socket.inet_aton("239.255.42.99"),socket.inet_aton("192.168.0.100")))
s.settimeout(3); n=0; t=time.time()
while time.time()-t<3:
    try: d,a=s.recvfrom(65536); n+=1
    except socket.timeout: break
print(f"Received {n} packets in 3 seconds" + ("" if n else "  ← 0 packets: check Motive Enable/Local Interface or the route in step 3"))
PY

# (c) Parsing: can rigid bodies be decoded? Run the writer and look for tracking rigid body id=…
```

- (a) fails → steps 1/2: cable, IP, mask, address conflicts.
- (a) passes but (b) receives 0 → step 3 multicast routing or step 4 Motive **Local Interface**.
- (b) receives packets but (c) finds 0 bodies → **Rigid Bodies=OFF** in step 4,
  no body created in step 5, or the robot is outside the capture volume.

---

## 3. NatNet 4.5 Frame Format: Why Initial Parsing Returned Zeros

In Motive's NatNet 4.5 `FrameOfData` (msgID=7), **each top-level section is
`[count int32][byteSize int32][payload]`**. The size prefix added from 3.0 lets clients
skip unknown sections. Initial old-style fixed-offset parsing interpreted size as
count, producing zero marker sets/rigid bodies. Correct parsing:

- Frame number (int32) → marker-set section (count+size, skip by size) → legacy/other
  markers (count+size, skip) → **rigid bodies (count+size, decode individually)**.
- Each rigid body is **38 bytes**:
  `id(i32) x,y,z(3f) qx,qy,qz,qw(4f) meanError(f) flags(i16)`;
  bit0 of `flags` indicates valid tracking.
- See `parse_frame_of_data` / `NatNetDirectReceiver` in
  [`natnet_shm_writer.py`](natnet_shm_writer.py).

---

## 4. Encountered Problems in Chronological Order

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | No `192.168.0.x` Mac interface; `ping 192.168.0.77` loses 100% | Mac has no motion-capture-subnet IP | Assign `192.168.0.100` to en7 in step 2 |
| 2 | IP still fails/conflicts after assignment | Mac was assigned `192.168.0.77`, Motive's own address | Use free `.100`, never `.77` |
| 3 | Correct IP and successful ping, but no NatNet packet on en7 | macOS routes multicast to en0/WiFi | Step 3: `route add -host 239.255.42.99 -interface en7` |
| 4 | Still no packets after adding route | Motive **Local Interface selects the wrong NIC** | Step 4: select `192.168.0.77`, facing the Mac; the final missing fix |
| 5 | Packets arrive but **markerset=0 / rigid bodies=0**, all empty frames | Streaming **Rigid Bodies=OFF**, and no body created | Enable Rigid Bodies in step 4 and create the G1 body in step 5 |
| 6 | Rigid-body section arrives but fixed-offset parsing gives zeros | NatNet 4.5 includes a `byteSize` section prefix (Section 3) | Parse/skip sections using count+size |
| 7 | Cannot identify G1 among multiple bodies | FrameOfData provides IDs, not names | Record streaming ID 5 in Motive and select it with `--rigid-body-id` |

---

## 5. Can Motion Capture Use WiFi?

**Yes for debugging; use Ethernet for closed-loop hardware operation.**

- **Position viewing/debugging, low rate and outside the control loop:** WiFi works.
  Set Motive **Transmission Type to `Unicast`** and use Mac `--unicast`; APs often
  drop/throttle multicast. Keep Mac/Motive on the same WiFi subnet or a routable network.
  Set `--server` to Motive's WiFi IP and `--local-ip` to the Mac's WiFi IP.
  **No code changes:** `--natnet --unicast` already supports this.
- **Hardware closed loop, with 120Hz localization feeding balance/walking:** WiFi
  is **not recommended**. Jitter, packet loss, and latency directly degrade control,
  causing drift/shaking. Use stable low-latency Ethernet for robot runs.
- **SSH:** it is remote login, not the pose-stream transport. “Connect over WiFi” means
  NatNet Unicast between Mac and Motive on WiFi. Forwarding ports 1511/1510 through
  an SSH tunnel can work but adds unnecessary latency.
- WiFi+Unicast is convenient at a desk; **formal robot operation uses Ethernet**.

---

## 6. Coordinate Frames: Motive Display vs NatNet Stream

- The **Motive display** is usually **Y-up**, while our **NatNet stream** is **Z-up**
  because **Up Axis=Z**. Conventions differ, approximately
  `(X,Y,Z)→(X,−Z,Y)` from Y-up to Z-up. An axis sign in displayed versus streamed
  (x,y) can therefore differ.
- Example: stream `pos=(0.184, 0.991, 0.791)` may look like `(-0.2, -1)` on screen.
  Matching magnitudes (≈0.2, ≈1.0) identify the same point here; signs differ by convention.
- **The robot uses only the stream.** Calibrate `T_world_scene` and determine heading
  from streamed (x,y,yaw), not screen readings.
- **Rigid-body +x must match robot forward** (step 5). Otherwise a constant yaw bias
  causes diagonal walking. Reset body orientation in Motive or incorporate the offset
  into the yaw of `--t-world-scene`.
- Before operation, **verify direction**: move a short distance along physical +x and
  check that streamed x increases while y stays approximately fixed. Adjust rigid-body
  orientation / TWS if they disagree.

---

## 7. Quick Reference

| Item | Value / command |
|---|---|
| Motive host | `192.168.0.77`; NatNet multicast `239.255.42.99`, data `1511`, cmd `1510` |
| Mac interface | `en7`; mocap-only `192.168.0.100/24`, closed-loop `192.168.0.100/16` to reach robot `.123.161` |
| Multicast route | `sudo route -n add -host 239.255.42.99 -interface en7`; repeat after restart |
| Required Motive settings | Enable / Local Interface=`192.168.0.77` / Multicast / **Rigid Bodies=ON** / Up Axis=Z |
| G1 rigid-body streaming ID | **5** |
| Start subscriber | `python -m genedynamics.deploy.localization.natnet_shm_writer --natnet --local-ip 192.168.0.100 --server 192.168.0.77 --rigid-body-id 5` |
| Wireless | Set Motive Unicast and add `--unicast`; debugging only, hardware closed loop uses Ethernet |
| Validate reads | `ViconShmPlugin({}).get_state()` (same as `--localization vicon`) |
| Code | `natnet_shm_writer.py` (`parse_frame_of_data`/`NatNetDirectReceiver`/`run_natnet`) |
