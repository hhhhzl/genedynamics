# 通用 AR 架构 —— 多机器人数字孪生（中文版）

引擎无关（Swift/RealityKit **和** Unity/AR Foundation 并存）、多设备（Vision
Pro / 手机 / 平板 / HoloLens / Quest / 网页）、性能极致、且对接边缘/分布式。
在现有 `deploy/ar`（`SceneSource` + `scene_server`）之上**生长**,不推倒重来。

> 英文版见 `ARCHITECTURE.md`。本版额外说明 **P3 分布式层下沉到你的 `edgecloud` 模块**。
>
> **未来兼容生态路线图**(还能接哪些 OpenXR 头显、Unreal、ROS2、传输、非 AR 消费者等,
> 含逐项状态:已做 / 可达 / 红线外)见 `future.md`。

## 0. 唯一的原则

下面一切都来自**把系统拆成四层、每层可独立替换**。兼容性、性能、分布式,本质上都只是"某一层选了哪种实现"。

```
  ┌────────────┐   ┌──────────┐   ┌─────────────┐   ┌──────────────┐
  │  STATE     │ → │  SCHEMA  │ → │  TRANSPORT  │ → │   RENDER     │
  │ 世界模型    │   │ 中立契约  │   │ 发布订阅 +   │   │ 各引擎薄适配  │
  │ 聚合器      │   │          │   │ 边缘网关     │   │              │
  └────────────┘   └──────────┘   └─────────────┘   └──────────────┘
   机器人/障碍/      FlatBuffers     ▼ 下沉到 edgecloud   Swift/RealityKit
   位姿/计划/目标     IDL + ICD       NATS 总线 + 边缘网关  Unity/ARF
   (共享世界系)      (兼容核心)       + controller 编排    WebXR
```

- **两个引擎能互通**:因为它们只共享 **SCHEMA**(一份 IDL → 生成 Swift + C# 代码)+ **约定(ICD)**。别的什么都不共享,各自是独立视图客户端。
- **性能极致**:在 **TRANSPORT + SCHEMA**(零拷贝二进制、增量+关键帧、数据报、客户端插值)。
- **边缘/分布式**:在 **TRANSPORT**——**整层下沉给 `edgecloud`**(见第 6 节)。
- **通用(多机器人)**:在 **STATE**(实体化,而非写死走廊)。

---

## 1. STATE —— 世界模型(把 `SceneSource` 泛化)

现在 `SceneSource` 只持有一个走廊场景。泛化成 **`WorldState`**:一组**实体**,每个 `{id, type, pose, frame, geometry, meta, rev}`。障碍、墙、机器人基座、机械臂连杆、目标、规划路径、**机器人遮挡代理**,统统是实体。

- **生产者(producers)** 往 `WorldState` 喂数据(每个都是小适配器):走廊障碍(`SceneSource`)、机器人位姿(任意 tracker)、关节状态、规划器、ROS2 话题。`SceneSource` 变成**其中一个**生产者,而非全部。
- `WorldState` 记每个实体的 `rev`,让编码器能发**增量**。
- 记录系:**一个可插拔 tracker 提供的共享世界系**(`BaseLocalizationPlugin` 抽象——Vicon / OptiTrack / SLAM / VIO / fiducial 都是可互换后端,见下文"坐标系来源与配准——与跟踪系统无关")。**唯一**的结构性要求是:机器人**和** AR 设备定位进**同一个**世界系。子系(`<robot>/base`、`<arm>/link_k`)引用父系(TF 式)→ 多机器人/机械臂"免费"获得,只要发它们的变换。

> 多机器人/机械臂的扩展是 **STATE** 的事:N 个机器人 = N 个生产者按 id 往同一个 `WorldState` 写。**无需新架构**。

---

## 2. SCHEMA —— 兼容核心(一份 IDL,多语言生成)

这是让 Swift + Unity 互通的关键。契约**只定义一次**,用 **FlatBuffers** IDL;`flatc` 生成 **Swift、C#、Python、TS** 结构体。FlatBuffers **读时零拷贝**(客户端每帧读位姿,无需解析步骤),并支持 schema 演进。

```fbs
// deploy/ar/schema/world.fbs (草图)
namespace twin;
struct Vec3 { x:float; y:float; z:float; }
struct Quat { w:float; x:float; y:float; z:float; }
struct Pose  { p:Vec3; q:Quat; }                  // 在 `frame` 系
enum GeomKind:byte { Box, Sphere, Cylinder, Capsule, Usd, Urdf, Path }
enum EntityType:byte { Obstacle, Wall, Robot, ArmLink, Goal, Path, Occluder }
table Geometry { kind:GeomKind; half_extents:Vec3; radius:float; height:float;
                 asset_uri:string; joints:[float]; polyline:[Vec3]; }
table Entity   { id:string(key); type:EntityType; frame:string; pose:Pose;
                 geom:Geometry; color_rgba:uint; rev:uint; meta:string; }
table WorldSnapshot { schema_version:uint; site:string; stamp_ns:ulong;
                      frame:string;   // 根世界系名,如 "world"
                      tracker:string; // 来源:"vicon"|"optitrack"|"slam"|"vio"|"fiducial"|...
                      units:string; is_keyframe:bool;
                      entities:[Entity]; removed_ids:[string]; }
```

再提供同 schema 的 **JSON 投影**用于调试/浏览器/现有 `scene_server`(保留)。JSON = 人看/调试视图;FlatBuffers = 热路径。

**约定(ICD —— `schema/conventions.md`)** 是兼容性的另一半:
- 坐标系:单一共享世界系(与跟踪系统无关;见下一节),**米/弧度,右手系,z 向上,x 向前**。
- 各引擎基变换(每个客户端各自做;用已知点实测验符号):
  - Unity(左手, y-up):`unity = (-y, z, x)` —— 已在 `FrameRegistration.cs`。
  - RealityKit(右手, y-up, −z 前):`rk ≈ (-y, z, -x)`。
  - three.js/WebXR(右手, y-up, −z 前):同 RealityKit。
- 稳定 `id`(跨帧实体身份)、`rev` 语义、资产格式(**USD/USDZ** 网格、**URDF** 机器人)、颜色/单位。

---

## 3. TRANSPORT —— 下沉到 `edgecloud`(性能 + 分布式)

**这一层整体交给你的 `edgecloud` 模块,不在 enerdynamics 里重造。** 见第 6 节的边界与映射。两个要点照旧:

- **通道拆分**:高频位姿走"丢旧"的尽力而为(NATS core);低频场景/配置走可靠(NATS JetStream)。两种速率解耦。
- **传输载荷**:FlatBuffers(零拷贝);增量 + 周期关键帧,不重发静态场景。

---

## 4. RENDER —— 各引擎薄适配器(两个生态)

每个引擎 = 几百行的适配器:连网关、反序列化 FlatBuffers(零拷贝)、做自己的基变换、按 `id` 实例化/更新实体、做**客户端插值/外推**(用 30Hz 状态渲染出 90Hz)。两端共享的只有:生成的 schema 代码 + ICD。

- **Unity/AR Foundation** —— 已有脚手架(`CorridorRenderer` + `FrameRegistration`);把 JSON→FlatBuffers、泛化成实体、加 `URDF-Importer` 导机器人。一份代码 → iOS/Android/visionOS(PolySpatial)/HoloLens·Quest(OpenXR)。
- **Swift/RealityKit** —— 对应适配器(WebSocket:P2 用 URLSession 连 `scene_server`,P3 起改 NATS-WS;+ FlatBuffers-Swift + RealityKit 实体 + `rk=(-y,z,-x)`)。**复用你现有的 Vision Pro 框架**。
- **WebXR/three.js** —— 零安装的 Android/网页视图。
- **真实机器人遮挡(你的 Q3)**:发一个 `Occluder` 实体到机器人**被跟踪的世界位姿**(URDF/代理网格);各引擎用"只写深度、不画颜色"的隐形材质渲染它 → 真实机器人正确遮挡虚拟墙。

### 4a. Meta / Quest 目标(OpenXR + Meta XR SDK)

**默认头显目标:Meta Quest 3 / 3S / Pro**(passthrough MR)。它们是真正的 MR 设备——6-DoF inside-out 跟踪 + 彩色透视——所以能像 Vision Pro 一样渲染世界锁定的全息体。**Ray-Ban / Ray-Ban Display 不是目标**:单目 HUD,无第三方 6-DoF、无空间渲染 runtime,锁不住全息体。这是硬件/SDK 限制,不是引擎选择问题。

**ARCore 和 Quest 是两个不同的 runtime,不是迁移路径。** ARCore 是 Android **手机**端 runtime;Quest 是 **OpenXR + Meta XR SDK**,不调用 ARCore。这套架构的意义在于:你把每个 runtime 当成**同一棵树的叶子**——**STATE / SCHEMA / TRANSPORT 以及整个 RENDER 适配器(`WorldRenderer` + `WorldClient` + `EntityState` + `(-y,z,x)` 基变换)全不变**。Quest 是 Android,所以 WebSocket / WebTransport 末端也不变。相对手机路径**只有两处不同**:

1. **XR provider 插件** —— 把 *ARCore XR Plugin* 换成 **Meta OpenXR feature / Meta XR SDK**,build target 改 Quest。这是 Unity player settings,**不是代码**——`WorldRenderer.cs` 不 import 任何 XR 包。
2. **配准** —— *谁驱动* `FrameRegistration.WorldOriginAnchor`。手机用 `ARTrackedImageManager` 认 surveyed fiducial;OpenXR 头显用 **`OpenXRAnchorProvider`**(`clients/unity/Scripts/OpenXRAnchorProvider.cs`),即**厂商无关**的"头显贴 Vicon marker"配准器——同一个可替换契约(它只写那一个 anchor Transform)。`QuestAnchorProvider` 是它的薄 Quest 子类;**Pico / Magic Leap 2 / Android XR / Vive XR 直接挂 `OpenXRAnchorProvider`**——配准完全一致,只有 OpenXR feature group 不同(player settings)。

**配准(对应 §4b 表里 Vision Pro 那一行——用你已有的 Vicon):** 头显贴一组刚性 Vicon marker。Vicon 流出头显世界位姿;Quest 自带跟踪给头显在 Unity 里的位姿;anchor 即闭式解 `anchor = headset_unity ∘ L⁻¹`,其中 `L` 是头显 Vicon 位姿过同一个 `(-y,z,x)` 映射(`FrameRegistration.WorldToAnchorLocalRot`,现已成为渲染与配准共用的**单一真源**)。`OpenXRAnchorProvider` 每个 Vicon 样本解一次,并对(准静态的)anchor 做 **EMA 平滑**——既去抖动,又**持续重新钉到 Vicon 真值**,所以一整段会话里全息体都不会漂离真机(一次性 fiducial 会漂)。头部高频运动由 Quest 低延迟本地跟踪渲染,Vicon 流的网络延迟只喂慢速漂移纠正——无害。**不需要 fiducial,也不需要 passthrough 相机权限**,且机器人 + 头显天然在同一个 Vicon 世界系(满足 §1)。

> **安装偏移在上游。** marker→头帧的刚性偏移是生产者/标定的事(在 `producers/tracker_pose.py` 里处理,和机器人 base 一样),所以 Unity 端保持干净:它通过 `SetHeadsetPoseWorld(...)` 消费**已在 Vicon 世界系**的头显位姿,与传输无关(专用 headset-pose WS,或现有流上的一个 `headset/<id>`(`viewer`)实体——见 §1 / `populate_headset`)。Vicon 头显流不稳时把 `LockWhenConverged` 打开。

> 其余生态仍是同一份 Unity 代码:iOS/Android(AR Foundation → ARKit/ARCore)、visionOS(PolySpatial)、**以及所有 OpenXR 头显(Quest、Pico、Magic Leap 2、Android XR、Vive XR…)经 `OpenXRAnchorProvider`**——每个目标只有 XR feature group +(手机上)配准 provider 不同。

---

## 4b. 坐标系来源与配准 —— 与跟踪系统无关

整个设计**只假设一件事:有一个机器人和 AR 设备都能被定位进去的共享世界系。** 谁提供这个系是**可插拔后端**——`deploy/localization/BaseLocalizationPlugin.get_state() -> (qpos, qvel)`,而 #2 的坐标胶水 `Se2Transform.scene_base_from_world(qpos, qvel)` **吃任何后端**。**Vicon 只是一个后端,不是假设。**

随 tracker 变化的只有两件事;其余(STATE / SCHEMA / TRANSPORT / RENDER、坐标胶水、遮挡代理)全不变:
1. **定位后端** —— 每个来源一个 `BaseLocalizationPlugin`。
2. **共同配准** —— 机器人**和** AR 设备怎么进**同一个**系。

| 跟踪系统 | 定位后端 | AR 设备进同一系的方式 | 精度 |
|---|---|---|---|
| 光学动捕(Vicon / OptiTrack / Qualisys) | NatNet / RT / shm 插件 | 设备贴 marker → 动捕直接给世界位姿(最省事) | mm–cm |
| LiDAR SLAM / AMCL | ROS2 odom 插件 | AR 设备在同一张地图重定位,或已知地图位姿放 fiducial | cm–dm,有漂移 |
| VIO / 机载里程计 | 新插件 | 靠共享 fiducial 锚定(VIO 会漂) | 短期 cm / 漂移 |
| 外部相机 + AprilTag/ArUco | fiducial 插件 | 双方都对同一套 fiducial 解算 | 近处 cm |
| AR 设备自带追踪 + 识别机器人 | (无全局)设备算机器人相对位姿 | 世界系=设备系,机器人计划绑上去 | 较脆弱 |
| 共享空间锚(ARCore Cloud / ARKit 协作) | 机器人靠 fiducial 接入 | 多 AR 设备共享一个锚系 | cm–dm |

**精度是参数,不是结构假设。** 把规划 `collision_margin` 和 governor 的 `m_track` 按 tracker 的**精度+延迟**来设:动捕(~cm)可以很薄,SLAM/VIO 就调大或先跑宽松场景。**架构对所有 tracker 完全一致——换插件、设余量即可。**

## 5. 性能极致

1. **零拷贝序列化** —— FlatBuffers;客户端读位姿无需解析。
2. **增量 + 关键帧** —— 不重发静态场景;只发变化实体 + 周期关键帧。
3. **通道拆分** —— 高频位姿走 **数据报(尽力,丢旧)**;低频场景走 **可靠流**。
4. **客户端插值/外推** —— 帧间航位推算 → 30Hz 网络渲染出 90Hz。渲染率 ⟂ 网络率。
5. **同主机共享内存** —— 机器人→网关那一跳免序列化(NATS 同机/边缘网关本地)。
6. **兴趣管理 + LOD** —— 按空间区域/subject 订阅;远处/大量实体降细节。撑多机器人。
7. **QUIC/WebSocket 末端** —— 比裸 TCP 低延迟。
8. **PTP/NTP 时间同步** —— 跨边缘对时;每条消息带 `stamp_ns`,用于插值与多源融合。

---

## 6. 边缘与分布式 —— 下沉到 `edgecloud`(回答你的问题)

**可以,而且应该。** `edgecloud`(`WorkGetBetter/edgeforest/edgecloud`,Go)已经是一套**边缘编排/控制面**,正好提供 P3 需要的分布式底座。映射如下:

| P3 需要的 | enerdynamics 不再自造,改调 edgecloud 的 | edgecloud 现有 |
|---|---|---|
| 发布订阅总线 | **NATS**(替代原方案的 Zenoh) | NATS(`tasks.submit` / `node.score.*` / `wasm.metrics.*` 等已在用) |
| 兴趣管理/路由 | **NATS subject + 通配符** `twin.{site}.{robot}.pose` | NATS 原生 subject 路由 |
| 高频/可靠分流 | NATS **core**(丢旧位姿)+ **JetStream**(可靠场景) | NATS |
| 边缘层级 | NATS **leaf nodes**(每站一个边缘) | edge/gateway(Go,http/mqtt/media + spool 队列) |
| AR 末端扇出 | **NATS over WebSocket**(Unity C# / Swift / 浏览器 nats.ws 直接订阅)或经 edge/gateway | edge/gateway 协议层 |
| 编排/调度/可观测 | **controller**(跨边缘部署/扩缩网关、Prom/Grafana) | controller + edgectl/schedctl + Prometheus/Grafana |

**边界(谁负责什么):**
```
enerdynamics/deploy/ar  ──发布 FlatBuffers WorldSnapshot 到 NATS subject──►  edgecloud
  STATE(WorldState)                                                          NATS(core+JetStream)
  SCHEMA(world.fbs)                                                          edge/gateway(扇出)
  RENDER 适配器  ◄──────── 经 NATS-WS / edge gateway 订阅 ────────────────  controller(编排/可观测)
     │
     └─ 世界原点配准(任意 tracker;见 §4b)
```
- **enerdynamics 只负责** STATE + SCHEMA + RENDER,并**向下调用 edgecloud**(发/订 NATS subject)。
- **edgecloud 负责** TRANSPORT + 分布式 + 边缘编排(NATS、边缘网关、controller、监控)。
- 接口 = **NATS subject + FlatBuffers 载荷**(契约即 `world.fbs`)。
- **ROS2 互通**:加一个小的 **NATS↔ROS2 桥**(或机器人直接发 NATS);可挂在 enerdynamics 现有 ROS2 IO 上。

这样分布式/边缘的脏活全在 edgecloud,enerdynamics 这边的 AR 层保持轻、专注机器人+渲染;edgecloud 的 fleet/调度/WASM/可观测也能复用到多机器人多机械臂场景。

---

## 7. 落到 `deploy/ar` 的模块布局与迁移

```
deploy/ar/
  schema/
    world.fbs            # 兼容核心(IDL)
    conventions.md       # ICD:坐标系/单位/各引擎轴映射/配准/资产
    generated/           # flatc → swift/ csharp/ python/ ts/
  world_state.py         # 通用 WorldState(实体 + 每实体 rev)  [泛化 SceneSource]
  producers/
    corridor.py          # SceneSource 作为生产者
    tracker_pose.py      # 机器人 base+occluder 以及 viewer 头显的世界位姿,来自任意 BaseLocalizationPlugin
    ros2.py              # ROS2 话题 → 实体
  encoders/
    json_codec.py        # 调试/浏览器(今天的契约)
    flatbuffers_codec.py # 热路径;增量 + 关键帧
  transport/
    nats_pub.py          # 发布 WorldSnapshot 到 edgecloud 的 NATS  ← 下沉边界
    ws_json.py           # == 今天的 scene_server(留作调试/浏览器)
  registration/          # 世界原点助手,与跟踪系统无关(Se2Transform 在此)
  clients/
    unity/   # AR Foundation(手机/visionOS) + WorldRenderer/WorldClient;
             #   OpenXRAnchorProvider.cs = 厂商无关 OpenXR 配准器
             #   (Quest/Pico/ML2/Android XR);QuestAnchorProvider.cs = Quest 子类(§4a)
    swift/   # RealityKit(Vision Pro)
    web/     # WebXR / three.js
```

> **Quest/Meta XR 工程配置清单**见 `clients/README.md` 的 "Setup — Meta Quest"
> 一节(包/player settings/场景接线/部署),配准原理见 §4a。

**迁移(每阶段都出价值,旧的不丢):**
- **P0(已完成):** `SceneSource` + `scene_server`(WS+JSON)、`Se2Transform`、Unity 脚手架。
- **P1 泛化:** `world.fbs` + `conventions.md`;`WorldState`(实体);`SceneSource`→生产者;同时出 JSON **和** FlatBuffers。Unity/Swift 适配器读实体。
- **P2 性能(仍跑在现有 `scene_server` 上,不需要 NATS):** WS 载荷从 JSON 换成 **FlatBuffers**(二进制帧);增量+关键帧;客户端插值;(可选)再加 WebTransport 数据报通道。**NATS 属于 P3,这一步不引入。**
- **P3 分布式(下沉 edgecloud):** 接 **edgecloud 的 NATS + edge/gateway + controller**;subject 兴趣管理;多机器人经 ROS2↔NATS 桥接入。

> 今天的 `scene_server`/契约是**向前兼容**的:它变成 `ws_json` 调试网关 + `world.fbs` 的 JSON 投影。永远不重写,只是"长过它"。

---

## 8. 待确认的决策
- 序列化:**FlatBuffers**(零拷贝、多语言)vs Cap'n Proto vs Protobuf。
- 总线:**NATS(edgecloud 已有)** ——故放弃原 Zenoh 提案,与 edgecloud 对齐。
- 末端:**P2 用现有 `scene_server`(WebSocket,无需 NATS);P3 起改 NATS-over-WebSocket**(+ 可选 WebTransport)。
- **Tracker:可插拔**(动捕 / SLAM / VIO / fiducial / 共享锚)——**不假设**任何一种,每个来源一个 `BaseLocalizationPlugin`(见 §4b)。
- 机器人模型源:**URDF**(→ Unity URDF-Importer;→ USD 给 RealityKit)。
- edgecloud 接入点:确认 NATS subject 命名规范、JetStream stream 配置、edge/gateway 是否要加专门的 AR 协议(还是直接用 NATS-WS)。
