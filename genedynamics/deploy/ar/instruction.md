# 走廊 AR 实机实验 · 完整操作手册

把本仓库的 AR 数字孪生 pipeline 跑到**真 Unitree G1 + AR 障碍**的逐步手册。
障碍是**虚拟的**(只在 AR 里),机器人靠 Vicon + 障碍几何"感知"并避让;人戴
Vision Pro / 举 iPad 看到对齐的全息障碍。

> 适用场景默认 **zone_d**(走廊长 4.0m、宽 1.6m、start=(0.5,0)、goal=(3.5,0),
> 2 个虚拟球)。换 zone 时把对应数字替换即可。

---

## 0. 机器角色与网络

| 角色 | 机器 | 跑什么 | 关键软件 |
|---|---|---|---|
| **动捕机** | 动捕主机 | Motive(NatNet 流)+ `natnet_ros2` 驱动 + `natnet_shm_writer`(写 `mocap_state_shm`) | ROS2 + `natnet_ros2`(本仓库 writer) |
| **孪生服务机** | 任意 Linux/Mac(可与机器人机同台) | `run_twin_server.py`(world_ws :8766) | conda `fedguide` + `websockets` |
| **机器人机** | 连 G1 的 Linux | `run_real_g1.py`(governor + 真机 IO) | `unitree_sdk2py` + 本仓库 |
| **规划机** | 任意(可与上同台) | `replan_from_scene.py`(2GO,Docker) | Docker `genedynamics/dev-cpu:torch` |
| **AR 设备** | Vision Pro / iPad | 渲染孪生 | 见 Part B |
| **构建机(苹果端)** | **Mac(Apple Silicon)** | Xcode 出包(Vision Pro/iPad 必需) | Xcode + (Unity) |

**网络**:四台机 + AR 设备接同一局域网(千兆交换机)。记下**孪生服务机 IP**(下称 `TWIN_IP`,例 `192.168.1.50`),AR 客户端要连 `ws://TWIN_IP:8766/`。G1 用网线直连机器人机(默认网口 `eth0`)。

---

## 1. 软件安装(按机器,一次性)

**动捕机(ROS2)** — 装好 ROS2(如 humble)+ `natnet_ros2` 驱动:
```bash
# 在 ROS2 工作空间里
git clone https://github.com/L2S-lab/natnet_ros2 src/natnet_ros2
colcon build --packages-select natnet_ros2 && source install/setup.bash
python -m pip install numpy scipy        # natnet_shm_writer 的 finite-diff 用
```
> 本仓库的 `natnet_shm_writer` 只在**这台 ROS2 机**上跑;它把 NatNet 位姿写进
> `mocap_state_shm`,下游(孪生服务机/机器人机)**不需要 rclpy**,仍用
> `--localization vicon` 读共享内存。详见 §A7。

**孪生服务机(fedguide)**
```bash
/opt/anaconda3/envs/fedguide/bin/python -m pip install "websockets>=12" "flatbuffers>=2.0"
```

**规划机**:确认 Docker 镜像可用:`docker images | grep genedynamics/dev-cpu`。

**机器人机**:按 Unitree 文档装 `unitree_sdk2py`;克隆本仓库;确认 `python -c "import unitree_sdk2py"` 不报错。

**Mac 构建机**:App Store 装 **Xcode**(最新版,含 visionOS SDK);首次打开跑一次 `xcodebuild -runFirstLaunch`。Unity 路线另装 Unity Hub + Editor(见 Part B 选项2/3)。

---

# Part A — 物理与 Vicon 准备(地面标定)

目标:① 在地面建立一个明确的"走廊场景坐标系";② 量出它在 Vicon 世界系里的位姿
`T_world_scene=(x0,y0,yaw0)`;③ 让 Vicon 输出机器人(和可选 AR 设备)的位姿到共享内存。

## A1 硬件清单
卷尺(≥5m)、粉线/激光水平仪、**美纹纸胶带**(彩色,3 种颜色)、记号笔、角尺(画垂线)、
Vicon 标定杆(calibration wand)、G1 基座 marker 簇、打印的 fiducial(A3,见 A6)、龙门吊架/吊带、**物理急停**。

## A2 Vicon 相机标定
1. 按 Vicon 流程做 wand 标定(挥杆采样)直到残差达标。
2. 设地面为 z=0 平面、设一个 Vicon 世界原点(L 形标定块);**Vicon 原点不必和走廊原点重合**——我们用 A4 的变换桥接。
3. 量 Vicon→控制环端到端延迟(可后续);先确保覆盖整条走廊都能稳定看到 marker(无遮挡盲区)。

## A3 地面标出走廊坐标系(**具体怎么贴**)
场景坐标系约定:**x = 走廊前进方向,y = 左为正,z = 向上,右手系,单位米**。
zone_d:走廊长 4.0m、半宽 0.8m、start=(0.5,0)、goal=(3.5,0)。

逐步贴(在空地上选一段直、平、无遮挡的 ≥5m×2.5m 区域):
1. **定原点 O**(场景 (0,0)):在地面选一点,贴一个**红色十字**,中心就是 O。
2. **定 +x 轴(中线)**:从 O 用粉线/激光打一条**直线**,长 4.0m,贴**黄色胶带**当中线。线尾就是场景 (4.0, 0)。沿线每 0.5m 贴一个小刻度并标数字(0.0、0.5、…、4.0)。
3. **定 +y 方向**:在 O 处用角尺画与中线**垂直**的方向(面向 +x 时的**左手边**为 +y),贴一小段标 "+y"。
4. **标 start / goal**:在中线 x=0.5m 处贴**绿色十字**(start),x=3.5m 处贴**蓝色十字**(goal)。
5. **标走廊两壁**(仅作参考,墙是 AR 虚拟的):在中线两侧各 0.8m、平行中线,各贴一条 4.0m 的细胶带(`y=+0.8` 和 `y=-0.8`)。用卷尺逐点量 0.8m 保证平行。
6. **标机器人朝向**:在 start 十字上加一个箭头指向 +x(机器人摆放时脚尖/正面朝它)。

> 贴完检查:O→(4,0) 是直线;start/goal 在线上;两壁到中线恒为 0.8m;+y 在面向 +x 的左侧。

## A4 测量 `T_world_scene`(**具体怎么测**)
`T_world_scene` 把"场景系"放进"Vicon 世界系":`world = R(yaw0)·scene + (x0,y0)`。
只需在 Vicon 里探**两个点**:
1. 用标定杆,杆尖**精确放到 O(场景 (0,0))**,在 Vicon 软件读取杆尖世界坐标 → 记为 `(X0, Y0)`。
2. 杆尖**放到中线 x=1.0m 处(场景 (1,0))** → 记为 `(X1, Y1)`。
3. 计算:
   - `x0 = X0`,`y0 = Y0`
   - `yaw0 = atan2(Y1 − Y0, X1 − X0)`(弧度)
4. 记录测量表:

   | 点 | 场景坐标 | Vicon 世界坐标 |
   |---|---|---|
   | O | (0, 0) | (X0=____, Y0=____) |
   | P | (1, 0) | (X1=____, Y1=____) |
   | → | `x0`=____ `y0`=____ `yaw0`=____ rad | |

   这三个数(`x0 y0 yaw0`)后面所有命令都要用,记为 **`TWS = "x0 y0 yaw0"`**。
   > 例:O 在 Vicon (1.20, 0.40),P 在 (2.20, 0.40) → yaw0=atan2(0,1)=0 → `TWS="1.2 0.4 0.0"`。

## A5 机器人 marker / 刚体
1. 在 G1 **骨盆/基座**刚性贴一组(≥4 个)marker,确保走动时不被手臂遮挡。
2. 在 Vicon 里把这组 marker 定义成一个**刚体(rigid body)**,命名如 `g1_base`,设其本地原点=骨盆中心、x 轴朝机器人正前方。
3. 让 Vicon 实时输出 `g1_base` 的 6DoF。

## A6 打印 fiducial(给 AR 配准用)
1. 打印一张**高对比度图案**(ARKit/ARCore 参考图,纹理丰富、非对称),A3 尺寸,平整裱在硬板上。量出**实际宽度**(米,例 0.40m × 0.28m),记下。
2. 把它**平放或竖立**在走廊旁一个 Vicon 可见、AR 设备方便看到的位置。
3. 用标定杆探它的**中心点**和**+x 边方向**两个点,算出 fiducial 在 Vicon 世界系的位姿 `T_world_fiducial`(同 A4 方法:中心给平移,边方向给 yaw;若竖立还需记法向)。记下。
   > AR 客户端会"看到"这张图算出它在设备系的位姿,再用 `T_world_fiducial` 反推出世界原点 —— 见 Part B 各选项的"配准"。

## A7 动捕接入(NatNet / `natnet_ros2`)+ 共享内存验证

> 本项目动捕走 **NatNet 协议**(OptiTrack/Motive),用 [`natnet_ros2`](https://github.com/L2S-lab/natnet_ros2)
> 把刚体位姿发成 `geometry_msgs/PoseStamped`(话题 `/<刚体名>/pose`,单位**米**)。
> 我们用本仓库的 `natnet_shm_writer` 把它桥接进 `mocap_state_shm`,**ROS2 只活在动捕机这一侧**,
> 下游照旧 `--localization vicon` 读共享内存——无需在 fedguide / 机器人机装 rclpy。
> (若你的动捕是 **Vicon DataStream(非 NatNet)**,改用 `vicon_shm_plugin.py` 里的
> `ViconDemoWriter`(需 `pyvicon_datastream`,注意它是 mm→需 ÷1000);其余完全一致。)

**坐标/单位约定(配错则下游整体被旋转/缩放,务必核对):**
- Motive → Streaming 设置:**Up Axis = Z**、**单位 = 米**(NatNet/ROS 直接发米,writer **不再 ÷1000**)。
- 世界系:右手系、z 朝上、x 朝前、y 朝左(`ar/schema/conventions.md` §1)。
- 在 Motive 里把 G1 骨盆刚体的**枢轴(pivot)设在骨盆中心、+x = 机器人正前方**(同 §A5);
  这样发布的位姿无需再补 mount 偏移。刚体命名 `g1_base` → 话题 `/g1_base/pose`。

1. **起 NatNet → shm 桥接(动捕机,ROS2 env)**:
   ```bash
   # ① 起 natnet_ros2 驱动(serverIP=Motive 主机, clientIP=本机, serverType=multicast/unicast)
   ros2 launch natnet_ros2 natnet_ros2.launch.py    # 或按其 README 配 config/initiate.yaml
   # 验证:话题在发
   ros2 topic echo /g1_base/pose --once
   # ② 起本仓库的桥接,把 /g1_base/pose 写进 mocap_state_shm
   python -m genedynamics.deploy.localization.natnet_shm_writer --rigid-body g1_base
   ```
   看到 `subscribing /g1_base/pose … → shm 'mocap_state_shm'` 且开始周期性打印 `pos=(…)` 即成功。
   写入结构 `q13d` = int64 utime(µs) + 13 float64:pos3 + quat_xyzw4 + vel3 + omega3
   (vel/omega 由位姿有限差分得到;噪声大可加 `--vel-lpf 0.2`;pivot 离地可加 `--z-offset`)。
   **quat 顺序、字节布局已和 `vicon_shm_plugin.py` 对齐**,无需手工核对。
2. 在**机器人机**验证读得到: 
   ```bash
   python -c "from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin; \
   p=ViconShmPlugin({}); import time; time.sleep(0.3); print('state=',p.get_state(),'health=',p.health())"
   ```
   移动机器人,再跑一次,pose 应跟着变、`health()=ok`。

---

# Part B — AR 客户端(四选一,详细 build)

四种都连**同一个** `ws://TWIN_IP:8766/`、渲染同一份 WorldSnapshot;差别只在引擎与打包。
**苹果设备(选项 1/2/3)出包都要 Mac + Xcode;选项 4(WebXR)不用 build、仅 Android Chrome。**

| 选项 | 设备 | 引擎 | 在哪 build |
|---|---|---|---|
| 1 | Vision Pro | RealityKit(原生 Swift) | Mac/Xcode |
| 2 | Vision Pro | Unity + PolySpatial | Unity → Mac/Xcode |
| 3 | iPad/iPhone | Unity + ARKit | Unity → Mac/Xcode |
| 4 | Android 手机/平板 | WebXR(three.js 网页) | 不 build(浏览器打开) |

---

## 选项 1:RealityKit → Vision Pro(原生 Swift)

**在哪 build**:**Mac**(Xcode)。**装到**:Vision Pro。

### 1.1 建工程
1. Mac 上 Xcode → File → New → Project → **visionOS → App**,命名 `CorridorTwinAR`,Interface=SwiftUI。
2. 把本仓库 `genedynamics/deploy/ar/clients/swift/Sources/CorridorTwin/` 里的 4 个 `.swift`
   (`Contract / EntityState / WorldClient / WorldRenderer`)拖进工程 target(或作为本地 SPM 包 `Package.swift` 引入)。

### 1.2 写 app 入口(把 client+renderer 接起来)
新建 `ImmersiveView.swift`:
```swift
import SwiftUI
import RealityKit
import ARKit
import CorridorTwin   // 若作为 SPM 包

struct ImmersiveView: View {
    @State private var renderer: WorldRenderer?
    @State private var client: WorldClient?
    let twinURL = URL(string: "ws://TWIN_IP:8766/")!         // ← 改成你的孪生服务机 IP
    // fiducial 在 Vicon 世界系的位姿(A6 测得),先用平移近似:
    let fiducialWorld = SIMD3<Float>(/*X*/0, /*Y*/0, /*Z*/0) // ← 填 A6

    var body: some View {
        RealityView { content in
            let worldAnchor = AnchorEntity(world: .zero)     // 先放原点,配准后更新其 transform
            content.add(worldAnchor)
            let r = WorldRenderer(worldAnchor: worldAnchor)
            let c = WorldClient(url: twinURL)
            c.onMessage = { [weak r] m in r?.enqueue(m) }
            c.connect()
            renderer = r; client = c
            // 每帧驱动插值
            _ = content.subscribe(to: SceneEvents.Update.self) { _ in
                r.update(now: CACurrentMediaTime())
            }
        }
        .task { await runImageTracking() }                   // 配准:见 1.3
    }

    // 1.3 配准:ARKit 图像追踪 → 把 worldAnchor 放到 Vicon 世界原点
    func runImageTracking() async {
        let session = ARKitSession()
        // 把 A6 的 fiducial 图(实际宽度)放进 app 的 ARReferenceImage 资源
        guard let refs = try? ReferenceImage.loadReferenceImages(inGroupNamed: "Fiducials") else { return }
        let provider = ImageTrackingProvider(referenceImages: refs)
        try? await session.run([provider])
        for await update in provider.anchorUpdates where update.anchor.isTracked {
            let T_device_fiducial = update.anchor.originFromAnchorTransform   // fiducial 在设备系
            // worldOrigin_in_device = T_device_fiducial * inv(T_fiducial_world)
            // 即:已知 fiducial 的 Vicon 世界位姿(A6)→ 反推世界原点在设备系的位姿
            // (此处按你的 T_world_fiducial 组装 4x4;平移近似时仅用位置)
            // renderer?.worldAnchor.transform = Transform(matrix: worldOrigin_in_device)
        }
    }
}
```
> 说明:配准的本质是"app 看到 fiducial → 结合它已知的 Vicon 世界位姿 → 把 `worldAnchor` 摆到 Vicon 世界原点"。最稳的替代:**给 Vision Pro 也贴 Vicon marker**,让 Vicon 直接给头显世界位姿、做一次性 hand-eye(免依赖图像追踪)。

### 1.3 编译上设备
1. Xcode → 选 target `CorridorTwinAR` → Signing & Capabilities → 选你的 Apple ID/Team。
2. 顶部设备选 **Apple Vision Pro**(真机,需开发者模式配对)或 visionOS Simulator(先验逻辑)。
3. **Cmd+R** 编译运行。首次需在 Vision Pro 上信任开发者证书。
4. 戴上 → 进入沉浸空间 → 看向 fiducial 完成配准 → 应看到障碍锁定在地面。

---

## 选项 2:Unity → Vision Pro(PolySpatial)

**在哪 build**:Unity(任意 OS 开发)→ 生成 Xcode 工程 → **Mac/Xcode 出包**。**装到**:Vision Pro。

### 2.1 工程与包
1. Unity Hub → 新建 **3D (URP)** 工程(PolySpatial 推荐 URP),Editor 选装 **visionOS Build Support**。
2. Package Manager 安装:
   - **AR Foundation**、**Apple visionOS XR Plugin**、**PolySpatial**(`com.unity.polyspatial` 全家桶,**需 Unity Pro/Industry 授权**)、
   - **Newtonsoft Json**(`com.unity.nuget.newtonsoft-json`)。
3. Project Settings → XR Plug-in Management → **visionOS** 勾选;PolySpatial 选 **Mixed Reality**(Unbounded volume 适合大场景)。

### 2.2 脚本与场景
1. 把 `clients/unity/Scripts/` 的 `WorldEntities.cs / WorldClient.cs / WorldRenderer.cs / FrameRegistration.cs` 复制进 `Assets/Scripts/`。
2. 新建一个**遮挡材质**:`Assets/CorridorTwin/Occluder.shader`(只写深度不画颜色):
   ```shader
   Shader "CorridorTwin/Occluder" {
     SubShader { Tags{"Queue"="Geometry-1"} ColorMask 0 ZWrite On Pass {} }
   }
   ```
   据它建一个 Material `OccluderMat`。
3. 场景里建空物体 `CorridorTwin`,挂三个组件:
   - `WorldClient`:`Url = ws://TWIN_IP:8766/`
   - `FrameRegistration`:`WorldOriginAnchor` = 拖一个子空物体(配准后被设到 Vicon 原点)
   - `WorldRenderer`:`Client`=本物体的 WorldClient、`Frame`=FrameRegistration、`OccluderMaterial`=`OccluderMat`、`InterpolationDelay`=0.08
4. **配准**:加 `ARTrackedImageManager` + 一个 Reference Image Library(放 A6 的 fiducial,填实际尺寸);写个小脚本在 `trackedImagesChanged` 里把 `FrameRegistration.WorldOriginAnchor` 的 transform 设成 fiducial 的 Vicon 世界位姿换算结果(同选项1思路)。

### 2.3 出包
1. File → Build Settings → 平台切 **visionOS** → Build → 选输出目录 → 生成一个 **Xcode 工程**。
2. 在 **Mac** 上打开该 Xcode 工程 → Signing 选 Team → 设备选 **Apple Vision Pro** → **Cmd+R**。

---

## 选项 3:Unity → iPad /iPhone / 平板(ARKit)

**在哪 build**:Unity(任意 OS 开发)→ 生成 Xcode 工程 → **Mac/Xcode 出包**。**装到**:iPad/iPhone。

### 3.1 工程与包
1. Unity Hub → 新建 **3D (URP)** 工程,Editor 选装 **iOS Build Support**。
2. Package Manager 安装:**AR Foundation**、**ARKit XR Plugin**、**Newtonsoft Json**。
3. Project Settings → XR Plug-in Management → **iOS** → 勾 **ARKit**。

### 3.2 脚本与场景
1. 同选项2 复制 4 个脚本 + 建 `OccluderMat`(同 2.2 的遮挡 shader)。
2. 场景:加 **AR Session** + **XR Origin (AR)**;空物体 `CorridorTwin` 挂 `WorldClient`(URL)、`FrameRegistration`、`WorldRenderer`(同 2.2 赋值)。
3. **配准(iPad 最简单)**:`ARTrackedImageManager` + Reference Image Library 放 A6 fiducial(填实际宽度)。在 `trackedImagesChanged` 回调里:
   ```csharp
   // 伪代码:已知 fiducial 的 Vicon 世界位姿 Twf(A6),AR 检测到它在 AR-session 系的位姿 Taf
   // 则世界原点在 AR-session 系 = Taf * inverse(Twf)
   frameRegistration.WorldOriginAnchor.SetPositionAndRotation(worldOriginPos, worldOriginRot);
   ```

### 3.3 出包
1. File → Build Settings → 平台切 **iOS** → Build → 生成 Xcode 工程。
2. **Mac** 上打开该 Xcode 工程 → Signing 选 Team → 设备选你的 **iPad/iPhone** → **Cmd+R**。
3. iPad 上首次"信任"开发者;举 iPad 对准 fiducial 完成配准 → 看到障碍。

> iPad 必须在 **ARCore 列表对 Android / ARKit 对 iPad/iPhone** 支持范围内(iPad 近年款都支持 ARKit)[[Link](https://developers.google.com/ar/devices#ios)]。

---

## 选项 4:WebXR(Android 浏览器,零安装)

**在哪 build**:不用 build —— 一个静态网页;**用 Chrome 打开网址**即可。**装到**:Android 手机/平板(浏览器)。
最省事的演示路线:不进 App Store、不用 Mac、不装 app。代价:**仅 Android Chrome 支持沉浸式 AR(`immersive-ar`)**;**iOS Safari 不支持**(苹果未开放),visionOS Safari 仅部分支持(visionOS 2+)。配准精度也不如前三种。

现成起步页:`genedynamics/deploy/ar/clients/web/index.html`(three.js + WebXR,自带 WebSocket 订阅 + 插值 + 遮挡 holdout;轴映射同 RealityKit `(-y,z,-x)`)。

### 4.1 前置:必须 HTTPS
WebXR 要求**安全上下文**:除 `localhost` 外,页面**必须经 HTTPS 提供**(否则浏览器不给开 AR 会话)。局域网里给平板访问 → 用自签证书起一个 HTTPS 静态服务(在 `clients/web/` 下):
```bash
cd genedynamics/deploy/ar/clients/web
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 -subj "/CN=twin"
python -c "import http.server,ssl,functools as f; \
s=http.server.HTTPServer(('0.0.0.0',8443), f.partial(http.server.SimpleHTTPRequestHandler)); \
c=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); c.load_cert_chain('cert.pem','key.pem'); \
s.socket=c.wrap_socket(s.socket,server_side=True); s.serve_forever()"
```

### 4.2 WebSocket 连接(注意混合内容)
页面默认连 `ws://<页面host>:8766/`(即 `run_twin_server`)。**坑**:HTTPS 页面连明文 `ws://` 会被"混合内容"拦截。两条路:
- **演示用**:Android Chrome 里临时允许该站点的不安全内容(站点设置 → 不安全内容 → 允许),或
- **正式用**:给 world_ws 套 **TLS**(`wss://`,用反代如 nginx/caddy 终结 TLS),页面里把 `WS_URL` 改成 `wss://...`。

### 4.3 运行
1. 先按 Part C 起好 `run_twin_server`(C2)。
2. 平板 **Chrome** 打开 `https://<TWIN_IP>:8443/` → 接受自签证书警告。
3. 点页面里的 **"Start AR"**(ARButton)→ 授权相机 → 进入 AR。
4. **配准**:起步页用 **tap-to-place**(命中测试点地面 → 放世界原点)——**不是 Vicon 级精度**,够快速演示。要 Vicon 精度:在 `index.html` 里换成基于 fiducial(A6)的图像配准,用 `T_world_fiducial` 设 `worldOrigin`(同前三选项思路);或直接用选项 1/2/3。
5. 应看到障碍 + 机器人 + occluder 锁定地面,随 `run_twin_server` 实时更新。

> 详见 `clients/web/README.md`(浏览器支持矩阵、HTTPS/wss、配准、FlatBuffers 切换)。

---

# Part C — 每次实验的启动顺序(逐步,照着做)

> 约定:`TWS` = A4 测得的 `"x0 y0 yaw0"`;`TWIN_IP` = 孪生服务机 IP。

## C1 起动捕 + 共享内存(动捕机,ROS2)
1. 开 Motive,确认 `g1_base`(和 fiducial 若用刚体)被稳定追踪、无丢帧;Streaming = **Z-up / 米**。
2. 起 `natnet_ros2` 驱动 → `ros2 topic echo /g1_base/pose --once` 能看到位姿。
3. 起桥接 `python -m genedynamics.deploy.localization.natnet_shm_writer --rigid-body g1_base` → `mocap_state_shm` 有数据。
4. **验证**(机器人机):跑 A7 第 2 步那段 `ViconShmPlugin` 检查,`health()=ok`、移动机器人 pose 变化。

## C2 起孪生服务(孪生服务机)
```bash
/opt/anaconda3/envs/fedguide/bin/python scripts/tasks/robot/humanoid/run_twin_server.py \
    --preset zone_d --t-world-scene <TWS> --localization vicon --port 8766
```
- 看到 `[world_ws] serving ws://0.0.0.0:8766/ ... N entities`。
- **验证**:浏览器或 `wscat -c ws://TWIN_IP:8766/` 能收到 JSON;含 `ball_L1/ball_R1/wall_*/goal/start` + `g1/base` + `g1/occluder`,且 `g1/base` 的 pose 随机器人移动而变。
- 没接 Vicon 想先验流程:把 `--localization vicon` 换 `--localization mock`(机器人固定在 start)。

## C3 重规划出 certified 计划(规划机,Docker)
```bash
docker run --rm -v "$PWD:/work" -w /work genedynamics/dev-cpu:torch \
    python scripts/tasks/robot/humanoid/replan_from_scene.py --preset zone_d --out results/ar/run1
```
- 结束打印 `calibrated: best_idx=… m_track=… (CERT)`,产物 `results/ar/run1/level_1/seed_0/trajectory/trajectory.json`。
- 已有现成 certified 计划可跳过本步。

## C4 启动 AR 客户端 + 配准(AR 设备)
1. 打开 Part B 选定的 app。
2. 看向(或举设备对准)**A6 的 fiducial** → app 完成配准(`worldAnchor` 落到 Vicon 原点)。
3. **验证**:在 start 十字处放一个已知小物,AR 里对应位置应有障碍/标记重合(偏差应 < 安全裕度,Vicon 1–2cm 足够)。空场地里应看到 2 个虚拟球 + 走廊墙。

## C5 机器人:干跑 → 落地运行(机器人机)
1. **摆放**:把 G1 基座**踩在 start 绿十字上,正面朝 +x 箭头**;先**吊装/半承重**。
2. **干跑(不发运动,验通信+坐标)**:
   ```bash
   python scripts/tasks/robot/humanoid/run_real_g1.py \
       --plan results/ar/run1/level_1/seed_0/trajectory/trajectory.json \
       --preset zone_d --t-world-scene <TWS> --network-interface eth0 --dry-run
   ```
   看到 `frame glue check: world start → scene pose (0.500, 0.000)` 与 `wiring OK`。
3. **吊装空走**:去掉 `--dry-run`,跑同命令。检查步态启动、不失步、governor 命令平滑。**手握急停**。
4. **落地**:确认安全层 + 急停就位,机器人落地踩在 start,运行:
   ```bash
   python scripts/tasks/robot/humanoid/run_real_g1.py \
       --plan results/ar/run1/level_1/seed_0/trajectory/trajectory.json \
       --preset zone_d --t-world-scene <TWS> --network-interface eth0
   ```
   机器人沿 certified 计划穿行,governor 用注入的 AR 障碍实时避让。

## C6 观察与验收
- **终端**:结束打印 `Exec min body-SDF: +0.xxx → CERTIFIED`、`endpoint dist`、是否摔倒。
- **AR**:人眼看障碍锁定空场地;**occluder 让真机器人正确挡在虚拟墙前**;插值丝滑无跳变。
- **数据**:`results/g1_corridor/real/` 下有逐步 JSONL + npz,用于回放/对比。
- **通过标准**:`certified_safe=True`、无碰撞、实际轨迹 vs 计划误差 < 余量、AR 对齐偏差 < 余量。

---

# D. 安全
- 全程**物理急停在手**;首次落地用**最易场景**(zone_a/zone_d),不要先碰 zone_c 挤压。
- Spark sport-mode 横移钳在 ±0.3 m/s,zone_d 真值余量 ~+0.08m;留足空间。
- 落地前务必挂 `joint_limit + torque_limit + self_collision` 安全层 + 摔倒检测。

# E. 故障排查
| 现象 | 排查 |
|---|---|
| `ViconShmPlugin` 读不到 | `natnet_shm_writer` 没起 / shm 名不是 `mocap_state_shm` / 上游 `natnet_ros2` 没发位姿(先 `ros2 topic echo /g1_base/pose`) |
| `natnet_shm_writer` 收不到位姿 | 话题名 ≠ `/<刚体名>/pose`(用 `ros2 topic list` 核对)/ QoS 不匹配(发布端是 reliable 就加 `--reliable`)/ serverIP·clientIP·multicast-unicast 配错 |
| 姿态被旋转 90° / 上下颠倒 | Motive Streaming 不是 **Z-up**(改成 Z-up);或刚体 pivot 朝向不对(§A5) |
| 位置数值差 1000 倍 | 走了 Vicon DataStream(mm)路径却没 ÷1000;NatNet 路径本就发米,用 `natnet_shm_writer` 即可 |
| AR 收不到流 | `TWIN_IP`/端口、防火墙、`run_twin_server` 是否在跑 |
| AR 障碍漂移/不对齐 | fiducial 实际尺寸填错 / A6 的 `T_world_fiducial` 量不准 |
| 机器人起点不对 | 没摆在 start / `--t-world-scene` 填错(重核 A4) |
| 干跑 pose 不是 (0.5,0) | `T_world_scene` 或 marker 刚体方向错 |
| 计划 `VIOL` 不认证 | 用满配 `replan_from_scene`(非 --fast);必要时调 governor `m_track`(实测 ~0.08 最优) |

# F. 速查
| 项 | 值 |
|---|---|
| 孪生服务 | `scripts/tasks/robot/humanoid/run_twin_server.py` → `ws://TWIN_IP:8766/` |
| 重规划 | `scripts/tasks/robot/humanoid/replan_from_scene.py`(Docker) |
| 实机运行 | `scripts/tasks/robot/humanoid/run_real_g1.py`(`--dry-run` 先验) |
| 动捕桥接 | `python -m genedynamics.deploy.localization.natnet_shm_writer --rigid-body g1_base`(NatNet→shm,ROS2 机) |
| 坐标变换 | `genedynamics/deploy/localization/scene_frame_plugin.py`(world→scene) |
| 契约/约定 | `genedynamics/deploy/ar/schema/{world.fbs, conventions.md}` |
| AR 客户端 | `genedynamics/deploy/ar/clients/{swift, unity, web}/` |
| 仿真复核 | `scripts/tasks/robot/humanoid/run_sport_mode_zones.py`(Docker) |
