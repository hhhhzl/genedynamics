# Robotics Laboratory × edgecloud Deployment Blueprint

> Goal: build a laboratory LAN connecting computers and robots, and use **edgecloud**
> (`WorkGetBetter/edgeforest/edgecloud`) to distribute **deployment, testing, and experiment
> workloads** across cloud, edge, and devices. Provide compute visibility, remote control,
> easy migration/deployment, robot package management, and controllable security, while
> leaving room for collaboration, ROS2, and federated learning.
>
> This research summary addresses four questions: **① What are the requirements?
> ② Which can edgecloud solve, and how? ③ Which other features are reusable?
> ④ What should be requested from the edgecloud team?**
>
> ⚠️ **Maturity note:** findings come from edgecloud `docs/`, `test/README`, and source.
> **The control plane/scheduler are relatively mature, but edge-agent container execution,
> UI wiring, and collaborative-inference runtime are only partially complete.** Test-README
> checkmarks are optimistic. **Validate a minimal PoC before committing laboratory
> operations to it (Section 9).**

---

## 1. Requirements

| # | Requirement |
|---|---|
| R1 | Build a LAN connecting several machines, currently wired |
| R2 | Dispatch deployment/test/experiment workloads to edgecloud |
| R3 | See other nodes and compute capacity in a dashboard |
| R4 | Make validation easy |
| R5 | Remotely control edge nodes from an external Mac/cloud |
| R6 | Easily migrate deployments for experiments/tests |
| R7 | Easily deploy software such as the Unitree SDK |
| R8 | Manage robot packages such as SPARK and genedynamics |
| R9 | Convenient, safe control of multiple robots on one network |
| R10 | Collaborative algorithms |
| R11 | ROS2 interoperability |
| R12 | Federated-learning compatibility |
| R13 | Slim deployment with an explicit heavy/light boundary; avoid multi-GB payloads on every edge |
| R14 | Local developer-Mac access: temporary, third-party, or high-security nodes |
| R15 | Serverless deployment |

---

## 2. What edgecloud Provides

A Go controller, **Rust edge agent**, Go gateway, Rust/Go schedulers, and Next.js
dashboard. **NATS is the bus**, Postgres/MinIO store state/artifacts, and the control
plane runs on k3s. **Fully local/LAN deployment requires no public cloud.** This is
application-level orchestration, not underlying network configuration: it does not
configure WiFi or network interfaces.

---

## 3. Requirements × edgecloud: Feasibility and Approach

| # | Supported? | Approach |
|---|---|---|
| R1 Networking | ✅ Application layer | Run NATS on one node and point other agents at it. ⚠️ edgecloud does not connect WiFi (OS `nmcli` does), and has no overlay/VPN/NAT traversal. Nodes need the same subnet or an added VPN. Use a dedicated router/AP with controller, all edges, and Mac on one subnet; wired/WiFi can mix |
| R2 Workload dispatch | ✅ | `runs`/`tasks` API and CPU/GPU/memory-aware placement. ⚠️ Agent container execution is partial; k3s pods, preinstalled-runtime shell jobs, or WASM are safer |
| R3 Compute dashboard | ✅ Partial | Nodes page: CPU/memory/GPU/online/capabilities; metrics page: per-node resources/network. ⚠️ Some UI remains mocked; no topology diagram |
| R4 Validation | ✅ | Live SSE watch, Prometheus, Grafana |
| R5 Remote control | ✅ | REST API `:8080`, `edgectl`/`schedctl`, WAN-accessible; outbound agent registration can work behind NAT |
| R6 Deployment migration | ✅ | Full lifecycle: precheck→snapshot→transfer→install→health_gate→rollback |
| R7 Deployment, e.g. Unitree SDK | ⚠️ | Requires OCI containerization, not simply uploading a repository. k3s pods are more reliable. Use slim images (Section 6) |
| R8 Package management | ✅ | OCI images, versioned/canary/staged/rollback `templates`, MinIO artifacts, Python SDK; containerize first |
| R9 Multi-robot safety/control | ✅/⚠️ | Strong control-plane security: OIDC/RBAC/tenants/WORM audit/Istio mTLS/Kyverno. Default edge↔NATS is weak: plaintext/self-registration. Isolate privately or manually enable NATS TLS/auth (Sections 6, 8) |
| R10 Collaborative algorithms | ⚠️ | Control-plane orchestration exists: `internal/chain`/Knative, 3328 lines covering orchestrator/collab_planner/SLO scheduling/NATS↔CloudEvents. Edge DAG execution is unfinished; build it or request it (Section 7, Knative) |
| R11 ROS2 | ❌ | Entirely absent. Interim options: containerized ROS2 on k3s or gateway MQTT/HTTP bridges. Request a DDS↔NATS bridge |
| R12 Federated learning | ❌ | `federated-runs` means federated execution across regions with NO_EGRESS data governance, not FL; no weight aggregation. Build aggregation above it or request it |
| R13 Slim boundary | ✅ With work | Heavy planning/training on compute, light deployment control+SDK on robot, artifacts via MinIO. Capability labels keep heavy jobs off robots (Section 6) |
| R14 Mac access | ✅/⚠️/❌ | Trusted temporary LAN access: client is safest, or shell/WASM edge. Untrusted third-party access is currently unsafe. High-security access requires manual NATS hardening (Section 6) |
| R15 Serverless | ⚠️ | Control loops should stay resident/stateful/realtime. Planning/inference can be serverless: `replan` functions, WASM perception. Knative exists, but scale-to-zero remains TODO |

The networking/scheduling/observability/remote-access/migration/package-management
orchestration is available. Four robotics-specific gaps remain: native deployment R7,
collaborative runtime R10, ROS2 R11, and actual FL R12. Implement these or request them.

---

## 4. Installation on a Local LAN Without Cloud

| Role | Installation | Commands |
|---|---|---|
| **Controller node**, local machine or cloud VM | k3s server, controller image, NATS, Postgres, MinIO | `sudo bash infra/scripts/ops/onprem_install.sh` → `kubectl apply -k deployment/onprem`; API `:8080`, NATS `:4222` |
| **Edge nodes**, computers or robot onboard PCs | containerd+nerdctl, optional NATS, Rust **edge-agent** | `sudo bash infra/scripts/recipes/edge_min_stack.sh`; build agent with `cargo build --release`; self-register via `NATS_URL=nats://<controller>:4222 ./edgecloud-agent` |
| **Operator Mac/Linux** | Nothing required | Browser `http://<controller>:8080/ui`; alternatively `edgectl` with `CONTROLLER_API=http://<controller>:8080`, or `kubectl` |

- Robot onboard PC is an edge; the Mac is an operator client or CPU compute node (Section 6).
- Remote access requires securely exposing controller `:8080` or using a VPN.

---

## 5. Cloud–Edge–Device Pattern and Three Examples

**Pattern:** split into heavy/light multi-architecture images; label nodes `compute`/`gpu`
versus `robot`/`edge`; run heavy work on compute and store artifacts in MinIO; let slim
robot deployments fetch artifacts and run realtime locally; verify through SSE/dashboard;
migrate when needed. Commands are illustrative; actual flags follow `edgectl` and
`/api/v1/runs`.

**Example 1 · Generic RL:** heavy GPU `rl-trainer` produces `policy.onnx` in MinIO;
slim arm64 `rl-runtime` fetches it on the robot and performs local inference.

**Example 2 · gd:** heavy JAX/2GO `gd-planner` runs `replan_from_scene`, placing
`trajectory.json` in MinIO; slim `gd-deploy` runs `run_real_g1` onboard, with a 50Hz
`Move` governor and arm_sdk. IO handles the FSM 4→200 standing transition.

**Example 3 · SPARK:** heavy `spark-sim`, the current Ubuntu+CUDA+Torch image, runs
benchmarks/synthesis on compute. Slim arm64 `spark-agent`, first split out per Section 6,
runs `run_g1_safe_teleop_real` on the robot.

> **Latency boundary:** realtime 50–500Hz `lowcmd`/`Move` loops to onboard `.161` must
> execute locally at the robot edge. Compute sends low-rate plans/policies/parameters,
> not per-tick commands.

---

## 6. Key Design Decisions

### 6.1 Heavy/Light Split and Measured Sizes

- `gd/deploy` source is only 2.8M. Dependencies dominate: jaxlib+jax 271M, scipy 120M,
  numpy 56M, mujoco 53M; complete venv 691M. The governor uses pure NumPy; JAX/SciPy/
  MuJoCo are pulled by imports but unused at runtime.
- Slim-edge route: remove JAX by decoupling `deploy` from jax/envs and precomputing
  `body_sdf_scene` offline as an artifact; remove MuJoCo by exporting actuator specs
  as static JSON; remove unused SciPy. Remaining numpy+SDK+cyclonedds+source ≈62M.
- Python-library payload can reach ~62M, but complete images remain ~150–200M because
  CPython/NumPy set a floor. A true tens-of-MB single binary requires native Rust.
- SPARK is similar: its current `ubuntu24.04 + CUDA + torch` Dockerfile is x86_64,
  several GB, and cannot run on Jetson. Robot deployment needs a slim agent/base and arm64.

### 6.2 Serverless Boundary

Resident stateful control loops are not serverless. Planning/inference can be:
`replan` as a function and perception as WASM.

### 6.3 Rust vs C++ for the Edge Controller: Rust Recommended

Official `unitree_sdk2py` does not wrap the C++ SDK; it uses cyclonedds, IDL, and pure
RPC. The protocol is understood: topics, FSM 4-200, CRC, mode_machine, and arm_sdk
weights. Rust can use DDS directly (`cyclonedds-rs` or `rustdds`), IDL, and a thin RPC
rewrite without touching the C++ SDK. Benefits include memory safety for control loops,
alignment with edgecloud/SPARK Rust agents for integration, and a Cargo single binary.
Choose C++ primarily for a strongly C++ team or deep ROS2 integration.

### 6.4 Mac Role

- ✅ Operator/client: submit jobs and view dashboards.
- ✅ CPU compute node: agent shell invokes `docker run <linux-image>` for compute
  containers that produce artifacts. ⚠️ Docker-on-Mac cannot reach robot DDS and has
  no GPU; accept compute-only jobs with no robot interaction.
- ❌ Not the robot edge; the control loop always executes locally on the robot.

### 6.5 Temporary, Third-Party, and High-Security Access

- Trusted temporary LAN nodes are supported; client access is safest, or use shell/WASM
  edges that go offline after heartbeat expiry.
- Untrusted third-party nodes are currently unsafe: self-registration, no bootstrap
  token, plaintext NATS, and no subject isolation. Interim isolation uses a separate
  NATS account/tenant, NATS TLS, and Kata, or scoped client-only RBAC.
- High-security control-plane/sandbox features exist (OIDC/RBAC/tenants/WORM/Kata/
  NO_EGRESS), but edge↔NATS needs manually enabled mTLS/auth and per-node identities.

### 6.6 Two Edge Runtime Options, Usable Separately or Together

| | **Slim Rust agent (NATS)** | **K8s edge (SuperEdge/KubeEdge backend)** |
|---|---|---|
| Form | Single agent binary + container runtime | Robot as K8s node: kubelet, pods, addons |
| Size/overhead | Light, tens of MB | Heavy, k3s/k8s and components |
| **Disconnected autonomy** | ❌ Heartbeat timeout marks offline | ✅ Pods keep running without eviction; peer checks avoid false termination |
| **NAT traversal** | ❌ Direct NATS access required | ✅ Cloud-edge tunnels reach robots behind NAT |
| Multi-site groups | Manual | ✅ ServiceGroup keeps traffic within a site |
| edgecloud integration | Native agent self-registration | Shallow placement-level backend provider with stubs; deploy a real SuperEdge/KubeEdge cluster for autonomy/tunnels/ServiceGroup |
| Best fit | Resource-constrained, minimal control edge | Reliable sites needing disconnected autonomy, tunnel maintenance, and multi-robot groups |

> Use slim agents/Rust for small, fast control loops. For disconnection tolerance/NAT
> operations, wrap robots in K8s edge, preferably KubeEdge as the more active option,
> and let it preserve control/safety pods. Both approaches can coexist. The control
> loop remains local; autonomy preserves pods, not realtime latency.

---

## 7. Additional Reusable edgecloud Features

- **WASM (WasmEdge) + WASI-NN:** lightweight, signed, sandboxed edge inference for
  safety filtering/perception.
- **Cloud bursting (SkyPilot):** overflow large RL retraining to cloud GPUs on demand.
- **Knative serverless + `internal/chain` DAG:** use functions for replanning/perception/
  safety composition and scale-to-zero when idle; build SLO-aware perception/planning
  DAGs, including `collab_planner` collaboration (3328 lines). ⚠️ Backend is feature-gated,
  scale-to-zero remains TODO, and `chain` is most mature. Keep control loops outside it.
- **Templates with staged/blue-green rollout and rollback:** safely deploy new robot
  policies/controllers.
- **MinIO artifacts and lineage:** version model weights, `trajectory.json`, and
  checkpoints, with presigned distribution.
- **Prometheus + Grafana:** fleet telemetry, control-loop metrics, robot health.
- **WORM audit + RBAC + tenants:** traceable shared multi-user/multi-robot use.
- **NO_EGRESS governance / Kata confidential computing:** retain sensitive data locally
  and isolate untrusted code.
- **W3 desired state + reconciliation:** declaratively specify “robot X should run
  policy Y” and converge automatically.
- **Edge autonomy and cloud-edge tunnels via SuperEdge/KubeEdge:** prevent control/safety
  pod eviction during disconnection, use peer checks to avoid false termination, and
  reach robots behind NAT for maintenance. This requires heavier K8s-on-robot; see Section 6.6.
- 🔑 **AR and motion-capture digital twins share the same infrastructure:** AR transport
  already plans FlatBuffers WorldSnapshots over NATS subjects. edgecloud's NATS bus
  supplies that layer. Bridge `run_twin_server` to unify deployment, AR, and telemetry.
  Add a `NatNet→NATS` producer using P1/P2 `producers/`/`world_state`; publish mocap
  poses to a subject such as `world.mocap.g1` for robot, AR, and dashboard subscribers,
  without each consumer connecting separately to NatNet.
  - ⚠️ Two boundaries remain. First, edgecloud does not configure the physical network;
    IP/mask/multicast-route/Motive Local Interface issues in
    [`localization/mocap.md`](localization/mocap.md) still need one-time setup.
    NATS simplifies distribution, not physical connectivity. Second, control localization
    still uses the local direct Mac `NatNet→shm` path for minimum latency. NATS is only
    fan-out for multiple consumers, subnet crossing, monitoring, and AR; never insert
    it into the 120Hz balance/walking loop. Another NATS hop adds jitter/latency,
    as with WiFi; see Section 5.
  - Deployment form: preserve control localization
    `natnet_shm_writer --natnet → shm → --localization vicon`, optionally publishing
    another copy to NATS for AR/monitoring/other robots. This requires edgecloud ready
    and belongs to P3, deliberately deferred until then.

---

## 8. Requests for the edgecloud Team

**P0: Blocking robotics requirements**

1. **First-class ROS2/DDS:** ROS2 node templates and a DDS↔NATS bridge.
2. **Native-process / “run this repository” workload type**, beyond WASM/containers,
   or complete agent container execution for one-step Unitree SDK/SPARK/gd deployment.
3. **Secure node enrollment:** bootstrap tokens, server-issued identities, revocation
   instead of self-registration.
4. **NATS mTLS/auth and tenant-isolated subjects**, replacing plaintext/shared access.

**P1: Important**

5. **Actual FL aggregation primitives:** rounds, weight averaging, secure aggregation,
   or an FL coordinator template.
6. **Complete collaborative-inference runtime (WP8.2–8.6)** for multi-robot perception/planning.
7. **Overlay/NAT traversal:** automatic robot enrollment over WiFi/campus networks,
   NATS leafnode/WS, or built-in mesh. SuperEdge/KubeEdge tunnels partly address this
   already (Section 6.6); this request mainly targets slim-agent/NATS deployments.
8. **Temporary-node leases/TTL** for clean temporary/third-party entry and exit.
9. **Third-party node attestation admission**, using existing JWT/Kata-CC to establish
   trust before enrollment.

**P2: Usability**

10. **Robot-device scheduling:** robot device classes and realtime constraints.
11. **Docker-API runner backend** so Mac+Docker can act natively as compute nodes.
12. **Improve `edgectl`** for robot-workflow CLI/UX.

> Each request should include the laboratory scenario, current state, gap, and
> acceptance criteria, so the team can act directly.

---

## 9. Implementation Roadmap

1. **One-day PoC first:** run `onprem_install.sh` on one controller, agents on 1–2 nodes,
   and verify nodes/compute in the Mac dashboard. Dispatch a hello-world container
   and validate scheduling, migration, and SSE.
2. **Dedicated laboratory network:** router/AP with controller, edges, and Mac on one
   subnet, mixing Ethernet/WiFi and avoiding campus isolation.
3. **Implement heavy/light split:** decouple deployment from jax/envs and precompute
   specs/scenes as artifacts. Create `gd-deploy`/`gd-planner` Dockerfiles; split SPARK's
   slim arm64 `spark-agent`.
4. **Integrate and validate:** run all three Section 5 examples on sim-labeled nodes
   before migrating to robots.
5. **File requests:** convert Section 8 into formal issues for the edgecloud team.
6. **Optional native Rust edge controller:** rewrite governor+IO in Rust, align with
   edgecloud's Rust agent, and target a realtime single binary of tens of MB.

> Related documents: hardware bring-up in [`REAL_DEPLOY_G1_CN.md`](REAL_DEPLOY_G1_CN.md);
> AR twin architecture in [`ar/ARCHITECTURE_CN.md`](ar/ARCHITECTURE_CN.md).
