# Soft-Robot Co-Design Paper 升级规划（M3BD → 顶会全文）

> 本文档记录把 `_Project__Mode_Robust_Multi_Fidelity_Model_Based_Diffusion_for_Soft_Robot_Co_Design.pdf`（M3BD report）升级成顶会全文（RSS/CoRL/ICRA）的完整决策、实验规划与分步清单。
> 配套：`configs/soft_robot/README.md`（Phase 0–5 runbook，工程级）、记忆 `project_softrobot_codesign_paper.md`。
> 约定：CPU 优先；3D-gen 推理与全量 sweep 走 GPU/RunPod。

---

## Part A — 对话与决策记录

### A.0 项目背景与目标
- **现状**：report（M3BD）讲了三件事——joint model-free/model-based diffusion、risk-sensitive 多 regime、adaptive multi-fidelity——但**实验只做了单个 4×3×4 voxel crawling，且明确没用 model-free prior（uniform prior 起步）**。
- **目标**：升级为顶会全文，**四个贡献全压上**：
  1. **Gradient-free joint co-design**：simulator-reward 的 Monte-Carlo score，不需要可微仿真（对比 DiffuseBot 的梯度 guidance）。
  2. **Multi-fidelity control-variate score estimator**：把"自适应选保真度"升级为有方差/预算分析的估计器。
  3. **Modern-prior robotization 研究**：统一 `{mesh, SDF, point-cloud, Gaussian-splat} → SoftBodySpec` + TripoSG / Hunyuan3D-2 / TRELLIS / CraftsMan3D / DiffGS / MeshFlow 横评。
  4. **Regime DRO / CVaR robustness**：固定温度 risk → 自适应对抗 regime posterior。
- **baseline & 工程参考**：DiffuseBot（位于 `../DiffuseBot`，大写 D；DiffTaichi MPM + PointE）。

### A.1 关键诊断：代码已领先于 report
`genedynamics` 包（分支 `3dgs`）已把 Phase 0–4 做完，Phase 5 = 跑 sweep。比 report 强的地方：

| 组件 | 代码现状 | 文件 |
|---|---|---|
| 3D prior 接口 | lazy 适配器已写（TripoSG），`random_shapes` 可跑 | `genedynamics/morphology/priors/` |
| Robotization | 7-stage mesh→spec 真实可用 | `genedynamics/morphology/mesh_robotize.py` |
| Simulator | **可微 JAX 3D MPM**（Neo-Hookean + 方向肌肉 eigen-stress） | `genedynamics/envs/external/jax_mpm/scene.py` |
| Multi-fidelity | 按 substeps/frames 分级（真·多保真度，非 rollout 数） | `.../mrmfmbd/fidelity_system/` |
| Regime risk | logsumexp + responsibilities | `.../mrmfmbd/mode_system/marginalizer.py` |
| Baselines | CEM / CMA-ES / SHAC + MBD+SHAC | `genedynamics/experiments/framework/baselines/` |

**两个 report 里没记录的关键代码事实**：
- **耦合 joint over `theta=[x_occupancy | phi_ctrl]` 已经实现**（`mrmfmbd_mbd_jax.py` 的 `step()`）：morphology x 与 controller ϕ 一起被 importance-weight 重采样（Eq 4/13/14）。JM2D u-step inner refinement 也已实现。
- **真正的缺口**：model-free prior `p_θ(x)` 现在是 voxel occupancy 上的**平凡对角高斯**（`theta_prior.py`，x_mean=0.6/x_std=0.3），**不是学习到的 3D prior**；`log_prob_batch` 算了但没折进权重。

### A.2 提问与结论逐条记录

**Q1：整体 brainstorm + 基于现有代码怎么搭？**
- 结论：升级工作量主要不在"再发明算法"，而在 (1) 让 joint MF/MB 真在实验里跑起来（modern prior → robotize → co-design）；(2) 把多保真+regime 讲成有原理的算法贡献；(3) 补 DiffuseBot baseline、补任务、补图。

**Q2：joint 形态——真做耦合 vs asymmetric，工作量？**
- 关键发现：**耦合 (x,ϕ) joint 采样器已存在**，不用重写。真正的选择是"morphology prior 住在什么空间"：

| 选项 | 做法 | 工作量 | 风险 |
|---|---|---|---|
| **B｜init-from-prior** | prior→robotize→固定 body，MBD 只 co-design | 1–2 天 | "joint" claim 弱 |
| **A2｜latent-field on fixed grid（推荐）** | shape latent `w` + 小 decoder `g(w)→固定栅格 occupancy+stiffness 场`，喂现有 rollout；prior log_prob 折进权重 | ~1 周 | 保住 JAX 快路径；真·learned-prior joint |
| **A1｜mesh-in-loop** | 每候选 `w→mesh→voxelize→robotize→sim`，拓扑可变 | 2–3 周 | **打破 JAX 快路径**（fast path JIT 编译固定粒子拓扑） |
- 决策：**headline = A2**（同时吃下 joint MF/MB + material-field co-design）；**modern-prior 研究 = Option B 离线 bank**（每 prior 离线 robotize→固定 body→co-design，避开 A1 的墙）。

**Q3：现在是 CPU，哪些先能做？**
- CPU 实测：单 rollout ≈ **110 ms/env-step** → 200-step episode ≈ **22 s**；全量 co-design（K=100×M=32×C=4）≈ **78 h/seed → 跑不动**；但 tiny 配置（K=5×M=4×C=1×50-step ≈ 5 s）≈ **~100 s/轮 → 开发够用**。`random_shapes→robotize` 端到端通（0.01s+2.6s）。torch 无 CUDA → **3D 基础模型 prior 推理这台跑不了**。
- 结论（**后修订，以 B.1 为准**）：CPU 仿真太慢，**凡是要跑 MPM 的（含 tiny smoke）一律放 GPU**；CPU 只做不跑 sim 的纯代码 + kernel/合成 reward 单测。原"tiny 配置 CPU 验证"作废。

**Q4：DiffuseBot 怎么 robotize，跟我们兼容吗？**
- DiffuseBot：point cloud `[1,6,N]` → **神经 SDF**（可微）→ voxel occupancy；design = `{geometry, softness(连续,co-design), actuator[K,voxel](连续场,co-design), coords}`；SoftZoo taichi MPM；**端到端可微**。
- 你的：mesh → 离散 voxelize → `SoftBodySpec{particles_x0, actuator_id(硬1-of-K,几何派生), voxel_id, fiber_dirs(固定), E_per_particle=None}`；JAX MPM；robotize 不可微。
- 兼容性三轴：
  - **① 几何：✅ 兼容**（都归约到栅格 occupancy；DiffuseBot 的 pc→SDF→occupancy 可作为 `pc_robotize` 变体）。
  - **② actuator：⚠️ 部分不兼容**——DiffuseBot 把 actuator placement 当**连续 co-design 场**，你是**几何派生的硬规则**。忠实复刻要把 `scene.py` 的 `A=act·outer(d,d)`（硬索引）改成 `A=Σ_i w_i·act_i·outer(d_i,d_i)`（加权和）。
  - **③ material：⚠️ 需要 A2 stiffness 场**（你现在 `E_per_particle=None`）。
- 关键洞察：**迁移 = SoftBodySpec 的超集扩展，不是重写**；且"扩连续 actuator+stiffness 场"和"做 A2"是同一块工作，**做一次吃两个**。

**Q5：我的方法能做成可微吗？有必要吗（想比 DiffuseBot 强）？**
- 结论：**能（JAX MPM 已可微），但主方法不该做成纯可微**。三条理由：
  1. gradient-free 是护城河——纯可微就是在 DiffuseBot 主场和它打，丢了差异点。
  2. 可微会**打断 multi-fidelity 贡献**——不同保真度的梯度不可比；gradient-free 才是多保真的前提。
  3. contact-rich 梯度病态（符号会错、易 NaN）；MC score 更稳。
- **怎么真正比 DiffuseBot 强**（都不需要可微）：成本（multi-fidelity Pareto）、鲁棒（regime CVaR）、通用（任意 sim 当 oracle）、可靠（contact 上 MC 更稳）；唯一落后的"设计空间丰富度"用 **A2（连续 actuator+stiffness 场）**补平。
- **架构**：zeroth-order 全局（morphology+regime+fidelity）+ first-order 局部 polish（已有 MBD+SHAC）。可选新颖贡献：reliability-weighted `Ŝ=(1−β)·Ŝ_MC+β·∇R`。
- **重要**：把"可微版 ours"（MC score 换成 `∇R`）做出来当 **ablation/baseline**（你要打败的对象）——因为 JAX MPM 已可微，几乎免费，且是"为什么选 gradient-free"的科学实锤。

### A.3 投稿定位：AAAI / ICLR——主题仍是 soft-robot co-design（不改题）
**主题不变**：soft-robot co-design 就是论文的 subject 和标题主词。AAAI/ICLR 完全收 application-grounded 的 co-design / diffusion 工作。下面是"投 AI 会要额外注意什么"，**不是改题**。

会议口味差异（相对机器人会，决定 framing 而非主题）：
- 要一个**清晰的核心方法贡献**（避免"多 trick 拼盘"观感）：组织成"一个 co-design 框架 + 1 个可证明的核心估计器子贡献"。
- 要**强 ablation + 多 seed 统计 + 可复现**；不在意真机部署。
- **theory-lite**：关键命题 + proof sketch（不必大定理）。

**标题保持 co-design 主词**（沿用原题、可微调）：
*"Multi-Regime Multi-Fidelity Joint Model-Free Model-Based Diffusion for Soft Robot Co-Design"*。

**四贡献（仍是 co-design 的贡献，只是用 ML 方法学呈现）**：
1. **Joint MF/MB co-design**：frozen 3D prior（形态）与 reward potential（控制器）通过同一加权 denoising step 耦合（cross-variable shape↔controller）。
2. **Multi-fidelity control-variate score estimator（核心可证明子贡献）**：自适应保真度形式化为固定预算下的低方差/低偏差 score 估计 + budget-dual 分配。
3. **Risk-sensitive CVaR/DRO regime potential**：worst-case 鲁棒的 co-design。
4. **Modern-prior robotization**：统一 {mesh,SDF,PC,GS}→spec + 多 3D-gen prior 横评。

**投 AI 会的加分项（非改题，是增强）**：
- **(T) theory-lite**：估计器无偏/方差缩减、CVaR 性质——命题 + proof sketch。
- **(R) 可复现**：seed 扫 + config 全公开 + 估计器单测。
- **(G, 可选)** generality 主证据 = co-design **内部**广度（locomotion + loco-manip 多任务 × 多 prior × 多 regime）；若想进一步堵 reviewer"是否只对此 sim 有效"，可把估计器**额外**搬到 repo 已有的 trajectory-opt（`double_integrator_box`/`drone_box_3d`）当 **appendix**——可选，不是主线。

---

## Part B — 完整实验规划（CPU 优先）

### B.0 架构约定（所有 Stage 必须遵守：工业级、非侵入、不乱加乱改）
现有代码是 **protocol + registry + typed-config + system-subpackage** 架构。任何扩展都对上一个**现成 pattern**，禁止旁路：

**5 条扩展 recipe（来自实读架构）**：
1. **新增配置旋钮**：YAML `method_params`(dict) → `BaselineConfig.extra`(原样透传) → `MRMFMBDBaseline.run()` 里 `extra.get(key, default)` 读出 → 写进 typed **`MBDConfig`** 字段 → backend 读 `self.config.field`。**新旋钮 = MBDConfig 加字段 + baseline 里 `extra.get`**，不要在中途插逻辑。
2. **新增 baseline**：实现 `BaselineProtocol`（`name` 属性 + `run(config, evaluator, task_spec, *, x_dim, phi_dim, **kwargs) -> BaselineResult`），在 `baselines/__init__.py` `register_baseline(...)`。
3. **新增 system 子包**：`specs.py`（`@dataclass(frozen=True)` + 可变 Config + `default_*_config()` 工厂）+ `__init__.py`（导出）+ 逻辑模块；**baseline 构造预建实例传给 backend**（mirror `fidelity_system` / `mode_system`）。
4. **扩 `SoftBodySpec` 字段**：加 `Optional[np.ndarray] = None` → `build_scene_from_spec` 里 `if not None` 守卫 → `SceneData` NamedTuple 加同名可选字段 → rollout kernel 接线。**`E_per_particle` 就是现成同款模板，照抄即可**。
5. **扩 rollout 输入**：`RolloutRequest` 有 `extra: dict`，新字段走 `extra` 不破坏签名；fast path（`_scene/_mpm_cfg/_mode_friction`）与 slow path（`evaluate_batch`）两条都要兼容。
- 测试：`test/unit/test_*.py`（pytest）；kernel/合成输入单测不跑 sim（CPU）。

**每个 Stage → 架构接缝映射（沿用哪个现成模板，凭什么非侵入）**：

| Stage | 改什么 | 沿用的现成 pattern（模板） | 非侵入保证（退化即旧行为） |
|---|---|---|---|
| **1** spec 扩展 | `SoftBodySpec` 加 `actuator_weight:(N,K)` / `stiffness:(N,)`；`scene._stress_and_J` 加权和 | recipe 4：**`E_per_particle` 同款 Optional 字段** | 缺省 None → 旧 robotizer/rollout 完全不变 |
| **2** A2 decoder | morphology decoder `g(w)`；prior log_prob 入权重；`morph_latent_dim` 旋钮 | recipe 1（MBDConfig 旋钮）+ `_weighted_mean` 已有 `log_prob_batch` | identity decoder + prior 权重=0 → 旧 voxel joint |
| **3** control-variate | 新 `estimator_system/` 子包 + MBDConfig `cv_*` 旋钮 | recipe 3：**mirror `fidelity_system`** | β=0 / 单层 → 旧单保真 score |
| **4** CVaR/DRO | 扩 `mode_system/regime_posterior.py` + `risk_*` 旋钮 | recipe 1 + 已有 `regime_posterior_mode` / `risk_temperature` 字段 | `mode="reward"` → 旧 marginalizer |
| **5** DiffuseBot baseline | 新 `diffusebot_baseline.py` | recipe 2：**`BaselineProtocol` + 注册** | 独立 baseline，0 影响主路径 |
| **6** priors/robotize | `priors/` 加 adapter；`morphology/` 加 `pc_robotize`/`gs_robotize` | **照 `triposg.py` lazy 模板 + 现成 `register()`**；输出同一 `SoftBodySpec` | lazy import；不动现有 `mesh_robotize` |
| **7** loco-manip | `tasks/carry.py` + regime bank | 现有 `push`/`manipuland.py` + `regime_bank_kind` 模式 | 新 `task_id`，不动现有任务 |

> 原则一句话：**每个 Stage 要么"加一个 Optional 字段"、要么"加一个 system 子包"、要么"注册一个 baseline/prior/task"——全是现有架构已有的扩展口，零旁路、零侵入。**

### B.1 CPU / GPU 切分原则（修订：CPU 仿真太慢，凡是要跑 sim 的一律 GPU）
**判定规则：一件事是否 touch MPM rollout？**
- **`[CPU]` 只做"不跑 sim"的工作**：纯代码编写（spec 扩展、A2 decoder、control-variate 估计器、CVaR posterior、统一 robotize、prior adapter）、**kernel/合成输入的单测**（直接调 `_stress_and_J` / 在合成 reward 上验估计器，不跑 rollout）、shape-AE 在静态 occupancy 数据上训练、adapter import 测试、出表出图/聚合、文档。
- **`[GPU]` 跑所有 sim**：任何端到端 co-design loop、所有集成正确性验证（含 tiny-config smoke，放 GPU 上快速迭代）、所有正式 sweep、3D-gen prior 推理、render 的 re-rollout、DiffuseBot 梯度版。
- 一句话：**CPU = 写代码 + 不跑 sim 的单测；只要跑 MPM 就上 GPU/RunPod。**

### B.2 实验矩阵（对应 paper 的表/图）

| 表/图 | 内容 | 主要 config | 依赖贡献 |
|---|---|---|---|
| **Table 1** 主表 | 11 methods × 3 tasks × 5 seeds | `main_v2/*` + baselines | 全部 |
| **Table 2** controller-only | 固定形态、扫控制器优化器 | `controller_only_*`（待建） | 1 |
| **Table 3** prior 横评 | 4–6 priors × 2 tasks × 5 seeds | `crawling_from_mesh.yaml` | 3 |
| **Table 4** fidelity ablation | 7 fidelity 方案 × 5 seeds | `crawling_alm.yaml` 旋钮 | 2 |
| **Fig regime** | 跨 regime 热图 / worst-case | `locomotion_terrain.yaml` | 4 |
| **Fig evolution** | co-design 演化网格（DiffuseBot 风格） | 渲染器 | 1 |
| **Table loco-manip** | push / carry (/grasp) | `push.yaml` + 新建 | 1,3,4 |
| **Generality（可选/appendix）** | 估计器额外搬到 trajectory-opt，堵"只对此 sim 有效" | `double_integrator_box` / `drone_box_3d` | 2（可选强化） |

Table 1 的 11 methods（`configs/soft_robot/README.md` §3.3 已列）：Random prior / CEM / CMA-ES / MBD / MBD+regime / MBD+adaptive iALM / **DiffuseBot-style** / DiffAqua-style / SHAC-only / **Ours(full)** / Ours+SHAC。

### B.3 Loco-Manipulation 实验：怎么选、怎么搭

**为什么 loco-manip 是这篇 paper 的理想 showcase**：manipulation 天生 **contact-rich + 强依赖形态 + 对 regime 敏感（物体质量/摩擦/尺寸）**，**同时压满 4 个贡献**——比纯 locomotion 更能体现 regime-robust 和 multi-fidelity 的价值。

**选择标准（4 条）**：
1. reward 要**依赖形态**（不是只靠控制器就能解 → 体现 co-design）。
2. 有清晰的**latent regimes**（物体质量/摩擦/尺寸 → 喂贡献4）。
3. **contact-rich**（低保真会 mis-rank → 体现贡献2）。
4. 尽量**复用现有 infra**（`push.yaml` / `manipuland.py`）。

**3 个候选任务，难度递增**：

| 任务 | 描述 | infra 现状 | regime | 形态依赖 |
|---|---|---|---|---|
| **Push-to-goal**（先做） | 软体爬行把 box 推到目标 | ✅ 已实现（`push.yaml`，36 regimes，`manipuland.py horizontal_only=True`） | box 质量/摩擦/初始位 | 需要"推面/前肢" |
| **Carry/Transport**（小建，1–2 天） | 把物体托在体上/凹腔里边走边运 | ⚠️ 待建：开 vertical box motion + 接触稳定 reward `G_T` + `tasks/carry.py` | 物体质量/坡度 | 需要"凹腔/托盘"形态 |
| **Grasp-and-move / Scoop**（stretch，appendix） | 形态需形成凹形/夹爪捕获并搬运物体 | ❌ 待建（最强 co-design 故事） | 物体尺寸/摩擦 | 强：必须长出 gripper |

**推荐顺序**：Push（现成，先把 loco-manip 管线跑通）→ Carry（小建，进主表）→ Grasp（有余力当 appendix 亮点）。
**搭建要点**：所有 loco-manip 任务都注册成 `task_id`，复用 `BaselineExperimentPlatform`；regime bank 复用 `regime_bank_kind`（train/test_push 已有，carry/grasp 新增）。

### B.4 DiffuseBot 如何放进你的框架一起跑

**目标**：apples-to-apples——同一 JAX MPM、同一 task/regime/budget 下，DiffuseBot 当 Table 1 的一行。

**两步**：
1. **设计空间对齐（一次性，服务 A2 + DiffuseBot）**：扩 `SoftBodySpec` 加连续 `actuator_weight:(N,K)` + `stiffness/E_per_particle`；改 `scene.py` 的 actuation 为加权和形式。
2. **新增 `DiffuseBotBaseline`**（注册到 `genedynamics/experiments/framework/baselines/`）：
   - prior = PointE（point cloud）或先用 random_shapes/已有 mesh prior 占位；
   - robotize = port DiffuseBot 的 pc→occupancy（CPU 用直接 voxelize 占位，GPU 用神经 SDF 忠实版）；
   - 优化 = **梯度 guidance**（`jax.grad(R)` 穿 JAX MPM，复用 SHAC 的 BPTT 基建）+ classifier-guidance 调度。
   - 这一行就是"first-order 对照"，撑起贡献1 的 gradient-free vs gradient-based 对比。

**不要**直接跑 DiffuseBot 原仓库当主对比（不同引擎不公平）；原仓库只用于交叉验证 / 借数值参考。

---

## Part C — 一步一步完成清单

> 标注：`[CPU]` 现在可做 / `[CPU-dev]` 写+tiny 验、出数等 GPU / `[GPU]` 必须 GPU/RunPod。
> 每步标：依赖 → 产出 → 服务的 paper 部分。

### Stage 0 — 骨架与基线 ✅ 已完成
- [x] **已完成**：`crawling_ground` 端到端 MBD co-design 已跑通，**report 的数值就是这一步的结果**（4×3×4 voxel crawling + 多 regime + fidelity ladder）。无需重做。
- [ ] `[GPU]` 仅剩：确认 RunPod/GPU 通道（`runpod` 分支）——记录"GPU 出数"环境与代码/结果同步方式（后续所有跑 sim 的 Stage 都依赖它）。

### Stage 1 — SoftBodySpec 扩展（服务 A2 + DiffuseBot，一次做完）✅ CPU 部分完成
> **实现发现**：stiffness 其实**早已 wired**——`_E_field_or_default` 已把 `scene.E_per_particle` 喂进 `E_field`→`_stress_and_J` 的 `E`。所以 Stage 1 只新增 `actuator_weight`。全部 **opt-in**：`actuator_weight=None` 时代码路径字节不变。
- [x] `[CPU]` `genedynamics/morphology/protocols.py`：加 `SoftBodySpec.actuator_weight:(N,K)`（`Optional=None`，照 `E_per_particle` 模板；缺省由 one-hot 退化）。
- [x] `[CPU]` `scene.py`：`SceneData.actuator_weight` 字段 + `build_scene_from_spec` 透传 + 新增 `_stress_and_J_weighted`（`A=Σ_i w_i·act_i·outer(d_i,d_i)`）；opt-in 串过 `_p2g_single`/`p2g_3d`/`_env_step`/`_env_step_with_manip` + 全部 4 个单体 rollout（batch 走 vmap 自动覆盖）。stiffness 接 `E_per_particle`（已 wired）。
- [x] `[CPU]` kernel 单测 `test/unit/test_actuator_weight_stress.py`（7 passed）：one-hot == 硬索引、passive 行 == 无 eigen-stress、连续权重 == 手算和；外加 `p2g_3d` 分支接线验证（max diff 2e-11）。默认 `build_scene` 仍正常。
- [ ] `[GPU]` 全 rollout 等价性：one-hot 全 rollout 必须复现旧 crawling 结果（不破坏现有行为）。
- 产出：连续 actuator + stiffness 设计空间。服务：贡献1（设计空间对齐）、A2、DiffuseBot 迁移。

### Stage 2 — A2：shape-latent decoder + joint MF/MB headline（贡献1+6）✅ CPU 核心完成
> 模块 `genedynamics/solvers/single/mrmfmbd/morph_system/`（mirror `fidelity_system`/`mode_system`：`specs.py`+`decoder.py`+`dataset.py`+`__init__`）。decoder 纯 JAX、jittable（hot-loop ready）。当前 decode 只输出 occupancy；stiffness/actuator_weight 头是同一 decoder 的干净扩展（接 Stage 1 的字段），wiring A2 material/actuator co-design 时再加。
- [x] `[CPU]` asset bank：`build_asset_bank.py --prior random_shapes` + `robotize_bank.py --voxel-dims 3,3,3` → `data/asset_banks/loco_cpu`（18 assets，RobotizationSuccess 100%）。
- [x] `[CPU]` tiny β-VAE（纯 JAX + 手写 Adam，env 无 optax）：`train_morph_ae.py` 在 bank 的 occupancy(27) 上学 latent `w`(8) → `data/morph_decoders/loco_cpu`（μ-recon MSE 7e-4）。N(0,I) latent 让 MBD prior 项 = 标准正态 `-½‖w‖²`。
- [x] `[CPU]` decoder `g(w)→occupancy[x_lo,x_hi]` 模块 + 单测 `test/unit/test_morph_decoder.py`（4 passed：形状/范围、jit+vmap、训练降 loss+重建、latent≈N(0,I)）。
- [ ] `[GPU+code]` 把 decoder 插到 `mrmfmbd_mbd_jax.py` rollout 前（`x_dim`→`latent_dim`，rollout 前 `x_morph=g(w)`）——改动 x→occ 映射，**需 rollout 验证，配 GPU smoke 一起做**。
- [ ] `[GPU+code]` 把 `theta_prior.py` 的 `log_prob_batch` 折进 `_weighted_mean`（w 块设 N(0,I)）——同属 hot-loop 改动，配 GPU smoke。
- [ ] `[GPU]` 出数：joint(w,ϕ) vs uniform-prior baseline，证明"learned prior > 平凡高斯"。
- 产出：headline joint MF/MB（CPU 侧 prior+decoder 已就绪）。服务：贡献1、Table 1 Ours、Table 3。

### Stage 3 — 贡献2：multi-fidelity control-variate 估计器 ✅ CPU 核心完成
> 模块 `genedynamics/solvers/single/mrmfmbd/estimator_system/`（mirror `fidelity_system`）。**任务无关**（只吃 reward 数组+costs），第二域可零成本复用。差分（control-variate）实现：`R̃_m = R_hi_m`（m∈S）/ `R_lo_m + Δ̄`（否则），喂 softmax 加权均值——精确端点（K=M→R_hi，K=0→R_lo）。
- [x] `[CPU]` 独立模块：`corrected_rewards`/`cv_score_weighted_mean`/`optimal_subset_size`/`BudgetDual`（dual-ν，Eq 23）+ 诊断。
- [x] `[CPU]` 合成 reward 单测 `test/unit/test_cv_estimator.py`（7 passed）：无偏、等高保真数方差 <0.25×纯高保真、**固定预算下 CV score MSE < 单保真度**、预算可行性、dual-ν 跟踪、诊断（corr>0.9/var_reduction>0.5）。
- [x] `[CPU]` `estimator_system` 任务无关化（设计即满足）。
- [ ] `[GPU+code]` 解开 `mrmfmbd_mbd_jax.py` 的"fidelity 非递减"约束（high-fidelity 可早用）——hot-loop，配 smoke。
- [ ] `[GPU+code]` CV 估计器 + dual-ν 接进核心 score（JAX 镜像 + `nu_max` 联通）——hot-loop，配 smoke。
- [ ] `[GPU]` 出数：Table 4 fidelity ablation（adaptive vs fixed-low/mid/high vs coarse-to-fine）。
- [ ] `[GPU]`（可选/appendix）generality：同一估计器搬到 `double_integrator_box`/`drone_box_3d` trajectory-opt。
- 服务：贡献2（核心子贡献）、Table 4、Fig fidelity。

### Stage 4 — 贡献4：CVaR / DRO regime posterior ✅ CPU 核心完成
> 扩展 `mode_system/regime_posterior.py`，不改现有函数。现有 `risk_sensitive_marginalize_jax` = KL-ball（entropic）对抗 posterior `q∝p·exp(-R/τ_r)`；新增 `cvar_marginalize_jax` = `{q≤p/α}` 歧义集的 **CVaR_α**（最坏 α 比例硬尾平均，精确离散公式）。同签名 drop-in，`REGIME_POSTERIOR_MODES` 加 `"cvar"`。
- [x] `[CPU]` `cvar_marginalize_jax(rewards_mc, log_prior, alpha)`→`(rho_m, q_mc)`：sort+overlap 精确离散 CVaR + 对抗 regime posterior（尾部 p/α）。
- [x] `[CPU]` 合成 reward 单测 `test/unit/test_cvar_regime.py`（7 passed）：α=1→prior-mean、α→0→worst-regime、单调、min≤rho≤mean、q 集中低 reward、CVaR(α→0)≈entropic(τ→0)。
- [ ] `[GPU+code]` 把 `"cvar"` 接进 `mrmfmbd_mbd_jax.py` 的 marginalizer 分派（+ `cvar_alpha` 旋钮，类比 `risk_temperature`）——hot-loop，配 smoke。
- [ ] `[GPU]` 出数：跨 regime 热图 + worst-mode 提升 vs mode-blind MBD。
- 服务：贡献4、Fig regime、Table 1 robustness 列。

### Stage 5 — DiffuseBot baseline 迁移（apples-to-apples）✅ CPU 部分完成
> `baselines/diffusebot_baseline.py`：自包含 Adam 梯度上升 over θ=(x,φ)，`jax.value_and_grad` 穿可微 `rollout_return`（DiffuseBot 梯度 guidance 类比 + first-order ablation）。单 mode + 单（fine）fidelity（DiffuseBot 设定）。pc→occupancy 非神经版 = 已有的 `robotize_point_cloud`（Stage 6），形态 init 走 `morphology` 旋钮。`prior_guidance_weight` = embedding-guidance 的廉价代理。
- [x] `[CPU]` `DiffuseBotBaseline`(BaselineProtocol) + 注册（`list_baselines()` 现含 `diffusebot`）。
- [x] `[CPU]` 梯度 guidance：`jax.value_and_grad(R)` 穿 JAX MPM + 手写 Adam + clip + best-tracking（env 无 optax）。
- [x] `[CPU]` 单测 `test/unit/test_diffusebot_baseline.py`（2：注册可查、无 JAX-MPM evaluator 时 run() loudly raise）。
- [ ] `[GPU]` smoke（tiny config）：梯度版能跑、梯度不 NaN（参考 README §5）。
- [ ] `[GPU]`（可选忠实版）port 神经 SDF-solidify + PointE prior 推理。
- [ ] `[GPU]` 出数：Table 1 "DiffuseBot-style" 行 + "first-order ours" ablation。
- 服务：贡献1（gradient-free vs gradient-based 实锤）、Table 1。

### Stage 6 — 贡献3：modern-prior + 统一 robotize ✅ CPU 部分完成
> 统一 robotize：抽出共享尾巴 `morphology/robotize_common.py:finalize_from_voxelization`（复用 mesh 的 stage helpers，**不动 mesh_robotize 已验证路径**，无循环导入）。pc/gs 各产一个 `VoxelizationResult` → 同一尾巴 → 同一 `SoftBodySpec`。adapters 用共享 `priors/_lazy_mesh.py`（DRY，非 5× 复制 triposg，各 ~15 行）。
- [x] `[CPU]` `pc_robotize.robotize_point_cloud`（点云→occupancy，纯 numpy）+ `gs_robotize.robotize_gaussians`（高斯密度阈值→occupancy）；都走 finalize 共享尾巴。
- [x] `[CPU]` prior adapters：`hunyuan3d`/`trellis`/`craftsman`/`meshflow`/`diffgs`（lazy，照 `_lazy_mesh` 基类）+ 在 `priors/__init__` 注册；`list_priors()` 现 7 个。
- [x] `[CPU]` 单测：`test/unit/test_unified_robotize.py`（4：pc/gs 合成数据→有效 spec、少点软失败、pc≈gs 结构一致）+ `test_prior_adapters.py`（8：全注册、未装则 `MissingDependencyError`+install hint、random_shapes 真采样）。共 12 passed。
- [ ] `[GPU]` 关 Gap A：clone TripoSG + 各 prior 权重，离线生成 asset bank（每 prior）。
- [ ] `[GPU]` robotize_bank 统计每 prior 的 RobotizationSuccess / 多样性 / 下游 reward / 稳定性。
- [ ] `[GPU]` 出数：Table 3 prior 横评。
- 服务：贡献3、Table 3、Fig prior 质性网格。

### Stage 7 — Loco-Manipulation 任务 ✅ CPU 部分完成
> Carry 核心写进 `scene.py`（与 push 同处）：`carry_reward`（forward transport + closing + 接触稳定 `G_T` − drop penalty，纯 jnp）+ `rollout_return_carry`/`_batch`（mirror push，`manip_cfg.horizontal_only=False` → 物体受重力、body 必须托运）。
- [x] `[CPU]` `carry_reward`（jnp 纯函数，`G_T = mean_t 1[obj_y≥carry_y_min]`）+ `rollout_return_carry`/`rollout_return_carry_batch`（mirror push 的 scan）。
- [x] `[CPU]` 单测 `test/unit/test_carry_reward.py`（4：carried>dropped、G_T∈[0,1]、forward 增 reward、drop penalty 生效）。
- [ ] `[GPU+config]` 把 `carry` 接进 evaluator task_id 分派 + `carry.yaml`（mirror push，flip horizontal_only）+ regime bank。
- [ ] `[GPU]` smoke 验证 Carry 形态依赖（无凹腔的体应失败）。
- [ ] `[CPU]`（stretch）建 **Grasp/Scoop**：捕获-搬运 reward + 形态需长出 gripper。
- [ ] `[GPU]` 出数：loco-manip 表（push / carry [/ grasp]），跨 object 质量/摩擦/尺寸 regime。
- 服务：贡献1/3/4 的 loco-manip showcase、Table loco-manip。

### Stage 8 — 渲染器 + 出表出图（关 Gap B）
- [ ] `[CPU]` 写 `scripts/visualizations/render_soft_robot_checkpoints.py`（PyVista 起步，粒子点云着色 by actuator）。注：脚本本身 CPU 写；**出图时的 re-rollout（重跑 (x_k,ϕ_k) 取粒子位置）→ GPU**。
- [ ] `[CPU]` 写 `make_table1.py` / `make_table4.py` / `make_figure3.py` / `make_figure_pipeline.py`（README §3.8 缺的）。
- [ ] `[CPU]` 复用现有 `analyze_*` / `compare_*` 脚本聚合多 seed。
- 服务：所有 Fig/Table 出图。

### Stage 9 — GPU 全量 sweep（RunPod）
- [ ] `[GPU]` 写 `scripts/sweeps/run_sweep.py`（README §3.2 草稿）支持 per-seed override。
- [ ] `[GPU]` Table 1（11×3×5≈165 runs，~250–400 GPU-h）。
- [ ] `[GPU]` Table 2 / Table 3 / Table 4 / regime / loco-manip sweeps。
- [ ] `[GPU]` 汇总到 `results/`，跑 Stage 8 出最终图表。
- 服务：所有正式数值。

### Stage 10 — 写作（AAAI/ICLR；题目仍是 soft-robot co-design）
- [ ] **(T) theory-lite**：propositions + proof sketch——估计器无偏/方差缩减、budget-dual 分配、CVaR/worst-case 性质。
- [ ] 写 Method：joint MF/MB 加权 denoising（coupled vs asymmetric）、multi-fidelity control-variate 估计器（核心子贡献）、risk-sensitive CVaR potential、modern-prior 设计空间。
- [ ] 写 Experiments：4 贡献消融 + DiffuseBot/first-order 对比 + loco-manip + 多 prior/regime 广度 + 多 seed 统计；（可选）appendix 第二域 generality。
- [ ] Related work：model-based/reward-guided diffusion（MBD、DiffuseBot、JM2D、SafeDiffuser）+ multi-fidelity MC + modern 3D-gen priors。
- [ ] 投稿格式（**AAAI / ICLR**，co-design 为题）+ 可复现包（config 全公开 + 估计器单测）+ 附录（grasp、忠实神经-SDF DiffuseBot、证明细节）。

---

## 附：CPU 上先做这些（纯代码 / 不跑 sim）
> 原则：CPU 只写代码 + 跑不 touch sim 的单测；**任何要跑 MPM 的（含 smoke）一律 GPU**。
1. **Stage 1 spec 扩展**（连续 actuator+stiffness 场 + scene 加权 actuation）+ kernel 单测——一次服务 A2 + DiffuseBot。
2. **Stage 2 CPU 部分**（random_shapes bank + 静态数据训 tiny shape-AE + decoder + prior 入权重）——headline 地基。
3. **Stage 3 / Stage 4 独立模块 + 合成 reward 单测**——最 CPU-native，可并行，不依赖 sim。
4. **Stage 6 adapter 代码 + 统一 robotize（pc/gs）**——纯代码，3D-gen 推理留 GPU。
5. **Stage 5 DiffuseBotBaseline 代码骨架 + Stage 7 Carry 任务代码 + Stage 8 脚本**——纯代码。
> **GPU 跑一切 sim**：所有 smoke/集成验证、所有 co-design 出数、3D-gen prior 推理、render re-rollout、DiffuseBot 梯度版、全量 sweep。先把 RunPod 通道（Stage 0 剩项）落实，CPU 写完即可推上去跑。
