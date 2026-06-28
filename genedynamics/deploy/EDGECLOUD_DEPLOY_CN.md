# 机器人实验室 × edgecloud 部署蓝图

> 目标:自建一个实验室局域网,把几台机器 + 机器人连成一张网,通过 **edgecloud**(`WorkGetBetter/edgeforest/edgecloud`)把**部署 / 测试 / 实验任务**下放到云-边-端,实现:dashboard 看算力、远程控制、轻松迁移/部署、机器人包管理、安全可控,并为协同/ROS2/联邦学习留口子。
>
> 本文是对一轮调研的整理,回答四件事:**① 需求有哪些 ② 哪些能用 edgecloud 解决、怎么解决 ③ 还有哪些功能可以直接拿来用 ④ 哪些要向 edgecloud 团队提需求。**
>
> ⚠️ **成熟度提醒**:结论来自读 edgecloud 的 `docs/` + `test/README` + 代码。**控制面/调度成熟度高,但 edge agent 的容器执行、UI 接线、协同推理运行时是"部分完成"**,test README 的 ✅ 偏乐观。**正式压上实验室前先做最小 PoC 验真(见 §9)。**

---

## 1. 需求清单

| # | 需求 |
|---|---|
| R1 | 自建局域网,把几台机器(现在走网线)连起来 |
| R2 | 把部署/测试/实验任务下放到 edgecloud |
| R3 | dashboard 看到彼此 + 算力 |
| R4 | 轻松验证 |
| R5 | 外部 Mac/云远程控制 edge |
| R6 | 轻松迁移 deploy 做实验/测试 |
| R7 | 轻松部署(如 unitree SDK) |
| R8 | 机器人包管理(SPARK、genedynamics) |
| R9 | 多机器人同网,安全但方便控制 |
| R10 | 协同算法 |
| R11 | ROS2 互通 |
| R12 | 兼容联邦学习 |
| R13 | deploy 瘦身 / 肥瘦边界可定义(边缘别全吃几个 G) |
| R14 | 本地开发者 Mac 接入(临时 / 第三方 / 高安全节点) |
| R15 | serverless 方式部署 |

---

## 2. edgecloud 是什么

Go 控制器 + **Rust edge agent** + Go 网关 + Rust/Go 调度器 + Next.js dashboard;**NATS 做总线**,Postgres/MinIO 存状态与产物,控制面跑在 k3s。**完全支持纯本地/局域网部署,无需公有云。** 它是**应用层编排**,不管底层网络(不配 WiFi/网卡)。

---

## 3. 需求 × edgecloud:能不能 + 怎么解决

| # | 能否 | 怎么解决 |
|---|---|---|
| R1 连网 | ✅(应用层) | 一台起 NATS,其余 agent 指向它即可。⚠️ **edgecloud 不连 WiFi**(OS 层 `nmcli` 的事);**无 overlay/VPN/NAT 穿透** → 同子网才行,跨网段要自己加 VPN。**建议自架一个路由器/AP,控制器+所有 edge+Mac 同子网**,有线/WiFi 混用 |
| R2 下放任务 | ✅ | `runs`/`tasks` API + 调度器(按 CPU/GPU/内存打分放置)。⚠️ agent 容器执行"部分完成",最稳走 k3s-pod / 预装运行时+shell / WASM |
| R3 dashboard 看算力 | ✅(部分) | nodes 页(CPU/mem/GPU/在线/能力)+ metrics 页(每节点资源/网络)。⚠️ UI 部分还是 mock,无拓扑图 |
| R4 验证 | ✅ | SSE 实时 watch + Prometheus + Grafana |
| R5 远程控制 | ✅ | REST API(`:8080`)+ `edgectl`/`schedctl`,跨 WAN 可用;agent 出站注册(可在 NAT 后) |
| R6 迁移 deploy | ✅ | 完整迁移生命周期(precheck→snapshot→transfer→install→health_gate→rollback) |
| R7 部署(unitree SDK) | ⚠️ | **必须容器化(OCI)**,不能"丢个仓库就跑";走 k3s-pod 更稳。**用瘦镜像**(见 §6) |
| R8 包管理 | ✅ | OCI 镜像 + `templates`(语义版本/canary/灰度/回滚)+ MinIO 产物 + Python SDK。前提:先容器化 |
| R9 多机器人安全可控 | ✅/⚠️ | 控制面安全强(OIDC/RBAC/租户/WORM 审计/Istio mTLS/Kyverno);**但 edge↔NATS 默认弱**(明文+自注册)→ 靠私网隔离,或手动开 NATS TLS+auth(见 §6、§8) |
| R10 协同算法 | ⚠️ | **控制面编排已有**(`internal/chain`/Knative,3328 行:orchestrator/collab_planner/SLO 调度/NATS↔CloudEvents),**边缘侧 DAG 执行待完成** → 自建边缘执行或提需求(见 §7 Knative) |
| R11 ROS2 | ❌ | **完全没有**。临时:把 ROS2 容器化跑 k3s,或网关 MQTT/HTTP 桥 → 提需求(DDS↔NATS 桥) |
| R12 联邦学习 | ❌ | `federated-runs` 是**联邦"执行"(跨地域 + NO_EGRESS 数据治理),不是 FL**(无权重聚合) → 在它之上自建聚合,或提需求 |
| R13 瘦身/边界 | ✅(要做工) | 肥(规划/训练)在 compute,瘦(deploy 控制环+SDK)在机器人,产物经 MinIO;调度器用能力标签确保重活不落机器人(见 §6) |
| R14 Mac 接入 | ✅/⚠️/❌ | 可信+局域网临时 → ✅(当 client 最稳,或当 edge 跑 shell/WASM);**第三方/不可信 → ❌ 现在不安全**;高安全 → 要手动加固 NATS(见 §6) |
| R15 serverless | ⚠️ | **控制环不该 serverless**(常驻有状态实时);**规划/推理该 serverless**(`replan` 当函数、感知用 WASM),Knative 在但 scale-to-zero 还 TODO |

**一句话**:组网/调度/可观测/远程/迁移/包管理这套"边缘编排"它都给到了;**机器人专用 4 块(原生部署 R7、协同运行时 R10、ROS2 R11、真 FL R12)是缺口**,要么自建、要么提需求。

---

## 4. 怎么安装(纯本地局域网,无云)

| 角色 | 装什么 | 命令 |
|---|---|---|
| **控制器节点**(挑一台/云 VM) | k3s server + controller 镜像 + NATS + Postgres + MinIO | `sudo bash infra/scripts/ops/onprem_install.sh` → `kubectl apply -k deployment/onprem`(API `:8080`、NATS `:4222`) |
| **edge 节点**(几台机器 / 机器人机载 PC) | containerd+nerdctl(+可选 NATS)+ Rust **edge-agent** | `sudo bash infra/scripts/recipes/edge_min_stack.sh`;`cargo build --release` 出 agent;`NATS_URL=nats://<控制器>:4222 ./edgecloud-agent`(自注册) |
| **操作端 Mac/Linux** | 啥都不用装 | 浏览器开 `http://<控制器>:8080/ui`;或 `edgectl`(`CONTROLLER_API=http://<控制器>:8080`)/ `kubectl` |

- 机器人机载 PC = 一个 edge;你的 Mac = 操作端(或 CPU 算力节点,见 §6)。
- 远程控制:把控制器 `:8080` 安全暴露或 VPN 进来。

---

## 5. 云-边-端模式 + 三案例

**骨架**:① 拆两镜像(肥/瘦,多架构)② 节点打标签(`compute`/`gpu` vs `robot`/`edge`)③ 重活→compute 出产物到 MinIO → 瘦 deploy→robot 拉产物本地实时跑 → SSE/dashboard 验 → 需要就迁移。命令为**示意**(实际 flag 以 `edgectl`/`/api/v1/runs` 为准)。

**例 1 · 通用 RL**:`rl-trainer`(肥,GPU)训练 → `policy.onnx` 入 MinIO;`rl-runtime`(瘦,arm64)在 robot 拉 onnx 本地推理。
**例 2 · gd**:`gd-planner`(肥,jax/2GO)`replan_from_scene` → `trajectory.json` 入 MinIO;`gd-deploy`(瘦)在 g1 机载跑 `run_real_g1`(governor 50Hz `Move` + arm_sdk;FSM 4→200 起身在 IO 内)。
**例 3 · SPARK**:`spark-sim`(肥,= 现 ubuntu+CUDA+torch 镜像)在 compute 跑 benchmark/合成;`spark-agent`(瘦,arm64,需先按 §6 拆出)在 robot 跑 `run_g1_safe_teleop_real`。

> **延迟红线**:实时控制环(50–500Hz `lowcmd`/`Move` 到机内 `.161`)**必须在机器人 edge 本地**;compute 只送低频 plan/policy/参数,**不是每拍指令**。

---

## 6. 关键设计决策

### 6.1 肥/瘦拆分 + 体积(实测)
- `gd/deploy` **源码才 2.8M**;胖的是依赖:**jaxlib+jax 271M、scipy 120M、numpy 56M、mujoco 53M**,整 venv 691M。**governor 全是纯 numpy**,jax/scipy/mujoco 运行时都不用(import 链顺带拖进来)。
- **瘦边路线**:① 砍 jax(解耦 `deploy`↔jax/envs + 离线把 `body_sdf_scene` 预算成产物)② 砍 mujoco(离线把 actuator spec 预算成静态 JSON)③ 砍 scipy(governor 不用)→ 剩 `numpy+sdk+cyclonedds+源码 ≈ 62M`。
- **"几十 M"能到哪**:Python 库负载可到 ~62M,但**整镜像 ~150–200M**(CPython+numpy 拉不下来)。**要真·几十 M 单二进制 → Rust 原生。**
- **SPARK 同理**:现 Dockerfile = `ubuntu24.04 + CUDA + torch`、**x86_64**,几个 G 且跑不了 Jetson;上机器人**必须拆瘦 agent + slim base + arm64**。

### 6.2 serverless 边界
控制环 ❌ serverless(常驻有状态);规划/推理 ✅(`replan` 当函数、感知 WASM)。

### 6.3 Rust vs C++(edge 控制器)→ **推荐 Rust**
官方 `unitree_sdk2py` 本就**不包 C++ SDK,而是 cyclonedds + IDL + 纯 RPC**;协议我们已摸透(topic/FSM 4-200/CRC/mode_machine/arm_sdk 权重)。所以 **Rust 直接走 DDS**(`cyclonedds-rs` 或 `rustdds`)+ IDL + 重写薄 RPC 即可,**不用碰 C++ SDK**。Rust 优势:内存安全(控制环不崩)+ **和 edgecloud/SPARK 的 Rust edge agent 对齐**(可并入)+ cargo 单二进制。C++ 仅在团队强 C++ / 深度 ROS2 时选。

### 6.4 Mac 的角色
- ✅ **操作端/client**(最干净:提 job、看 dashboard)。
- ✅ **CPU 算力节点**(有 Docker → agent shell 调 `docker run <linux镜像>` 跑 compute 容器,**出产物**)。⚠️ 但 **Docker-on-Mac 到机器人 DDS 过不去 + 无 GPU** → 只接不碰机器人的纯算力任务。
- ❌ **当不了机器人 edge**;控制环永远在机器人本地。

### 6.5 临时/第三方/高安全节点接入
- 可信 + 局域网 + 临时 → ✅(当 client 最稳;或 edge 跑 shell/WASM;心跳超时自动离线)。
- **第三方/不可信 → ❌ 现在不安全**:自注册、无 bootstrap token、NATS 明文且 subject 不隔离。临时隔离 = 单独 NATS account/租户 + NATS TLS + Kata,或只给 client 权限(RBAC scoped)。
- 高安全 → 控制面/沙箱够强(OIDC/RBAC/租户/WORM/Kata/NO_EGRESS),**但 edge↔NATS 要手动开 mTLS+auth + 每节点身份**。

### 6.6 两条 edge 运行时路线(按需选,也可混用)
edgecloud 的边缘有两套形态,各有取舍:

| | **瘦 Rust agent(NATS)** | **K8s-边缘(SuperEdge/KubeEdge 后端)** |
|---|---|---|
| 形态 | 单 agent 二进制 + 容器运行时 | 机器人跑成 K8s 节点(kubelet+pod+addon) |
| 体积/开销 | 轻(几十 M) | 重(k3s/k8s + 组件) |
| **断网自治** | ❌ 心跳超时即离线 | ✅ **断网继续跑 pod 不驱逐 + 节点互检不误杀** |
| **NAT 穿透** | ❌ 要直连 NATS | ✅ **云边隧道够到 NAT 后机器人** |
| 多站点分组 | 手动 | ✅ ServiceGroup 按站点闭环流量 |
| edgecloud 集成 | 原生(agent 自注册) | **后端 provider(浅,放置级 + 含 stub)**,深度功能(自治/隧道/ServiceGroup)要自己起一个真 SuperEdge/KubeEdge 集群 |
| 适合 | 资源紧、要极致小的控制 edge | 要"断网自治 + 隧道运维 + 多机器人站点"的可靠现场 |

> 推荐:**控制环走瘦 agent/Rust 求小求快;要断网容忍 / NAT 运维 → 给机器人套一层 K8s-边缘(优先 KubeEdge,比 SuperEdge 更活跃),把控制/安全 pod 交给它做自治。两者可混用。** 注意:实时控制环本身仍在机器人本地,自治保的是"pod 不掉",不是延迟。

---

## 7. 还能直接拿来用的 edgecloud 功能(你没列但值得)

- **WASM(WasmEdge)+ WASI-NN**:轻量、签名、沙箱化的边缘推理(安全过滤/感知)。
- **云爆发(SkyPilot)**:重训练(3dgs、RL)按需溢出到云 GPU。
- **Knative serverless + `internal/chain` DAG**:`replan`/感知/安全合成 当 serverless(空闲 scale-to-zero 省算力);多阶段感知/规划串 DAG(SLO 感知、含 `collab_planner` 协同,3328 行)。⚠️ 后端 feature-gated、scale-to-zero 还 TODO;`chain` 是最成熟部分。控制环不走它。
- **templates 灰度/蓝绿/回滚**:安全地把新策略/控制器滚动推给机器人。
- **MinIO 产物 + lineage**:给模型权重/`trajectory.json`/checkpoint 做版本管理 + 预签名分发。
- **Prometheus + Grafana**:整队列遥测看板(控制环指标、机器人健康)。
- **WORM 审计 + RBAC + 租户**:多人共享多机器人可追溯。
- **NO_EGRESS 数据治理 / Kata 机密计算**:敏感数据锁本地、不可信代码强隔离。
- **W3 desired-state + reconcile**:声明式"机器人 X 应跑策略 Y",自动收敛。
- **边缘自治 + 云边隧道(SuperEdge/KubeEdge 后端)**:机器人断网时控制/安全 pod 不被驱逐(边缘自治)、节点互检不误杀;隧道让你从外部够到 NAT 后的机器人做运维。代价是 K8s-on-robot,重于瘦 agent —— 详见 §6.6。
- 🔑 **AR + 动捕数字孪生天然合流**:AR transport 既定就是"发 FlatBuffers WorldSnapshot 到 NATS subject",**edgecloud 的 NATS 总线正好是那层**,把 `run_twin_server` 桥上去,部署+AR+遥测统一一张网。**动捕也能接进来**:加一个 `NatNet→NATS` producer(复用 P1/P2 的 `producers/`/`world_state`),把动捕位姿发成一个 subject(如 `world.mocap.g1`)→ 机器人、AR 多端、dashboard 都订阅,**不用每个消费者各自连一遍 NatNet**。
  - ⚠️ **两条边界(直接回答"机器人↔mac↔mocap 还会这么麻烦吗"):** ① **物理网络 edgecloud 不管**(它是应用层编排)—— IP/掩码/多播路由/Motive Local Interface 那套坑(见 [`localization/mocap.md`](localization/mocap.md))**还得配一次**,不会因为上了 NATS 就消失;NATS 省的是"数据分发"的麻烦(发一次大家订),不是"物理联网"的麻烦。② **控制环的定位仍走 Mac 本地直连 `NatNet→shm`**(最低延迟),NATS 只做"扇出"(多消费者 / 跨网段 / 监控 / AR),**绝不进 120Hz 平衡/行走环**(多一跳 NATS = 多一份延迟抖动,和 WiFi 同理,见延迟红线 §5)。
  - 落地形态:控制用定位保持现状(`natnet_shm_writer --natnet → shm → --localization vicon`);**同时**可选发一份到 NATS 给 AR/监控/其它机器人。需 edgecloud 就绪(属 P3,此前刻意推迟到它 ready)。

---

## 8. 要向 edgecloud 团队提的需求

**P0(机器人场景的硬阻塞)**
1. **ROS2/DDS 一等公民**:ROS2 节点部署模板 + **DDS↔NATS 桥**(机器人 ROS2 topic 与 edgecloud 互通)。
2. **原生进程 / "跑这个仓库"工作负载类型**(不止 WASM/容器),或**把 agent 容器执行做完整** → unitree SDK/SPARK/gd 一步部署。
3. **安全节点 enrollment**:bootstrap token + 服务端签发身份 + 撤销(替掉自注册)。
4. **NATS 总线 mTLS + auth + 按租户隔离 subject**(现在明文、共享)。

**P1(重要)**
5. **真·FL 聚合原语**(轮次/权重平均/安全聚合),或 FL coordinator 模板。
6. **完成 collab-inference 运行时(WP8.2–8.6)**(多机器人协同感知/规划)。
7. **overlay/NAT 穿透**(机器人上 WiFi/校园网自动入网;NATS leafnode/WS 或内置 mesh)。**注**:走 **SuperEdge/KubeEdge 后端时其云边隧道已部分解决此问题**(可够到 NAT 后 edge,见 §6.6),此条主要针对"瘦 agent/NATS"路线。
8. **临时节点 lease/TTL**(干净的临时/第三方进出)。
9. **第三方节点 attestation 准入**(基于已有 JWT/Kata-CC,先证明可信再入网)。

**P2(体验)**
10. **机器人设备调度**(robot device class、实时约束)。
11. **Docker-API runner 后端**(Mac+Docker 能原生当 compute 节点)。
12. **打磨 `edgectl`**(机器人工作流的操作 CLI/UX)。

> 每条提需求时附:机器人实验室**场景 + 现状 + 缺口 + 验收标准**,他们能直接 action。

---

## 9. 落地路线

1. **PoC(1 天,先验真)**:一台跑 `onprem_install.sh`(控制器)→ 1–2 台跑 agent → Mac 开 dashboard 看到节点+算力 → 下发一个 hello-world 容器任务,验调度/迁移/SSE。
2. **自架实验室网**:路由器/AP,控制器+edges+Mac 同子网(有线/WiFi 混用,避开校园隔离)。
3. **瘦/肥拆分落地**:解耦 `deploy` 的 jax/envs + 预算 spec/scene 成产物 → `gd-deploy`(瘦)/`gd-planner`(肥)两个 Dockerfile;SPARK 拆 slim arm64 `spark-agent`。
4. **接入 + 验证**:按 §5 三案例跑通(先 sim 标签节点,再 migrate 到真机器人)。
5. **提需求**:把 §8 整理成给 edgecloud 团队的正式 issue。
6. **(可选)Rust 原生 edge 控制器**:把 governor+IO 用 Rust 重写,和 edgecloud Rust agent 对齐 → 单二进制几十 M、实时级。

> 相关文档:实机 bring-up 见 [`REAL_DEPLOY_G1_CN.md`](REAL_DEPLOY_G1_CN.md);AR 孪生架构见 [`ar/ARCHITECTURE_CN.md`](ar/ARCHITECTURE_CN.md)。