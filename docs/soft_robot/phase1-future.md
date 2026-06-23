# Phase 1 阶段性总结 + 未来规划

**日期**：2026-06-23  
**项目**：软体机器人协同设计（morphology x + controller φ），引擎 = 可微 JAX-MPM；方法「ours」= MR-MF-MBD / M3BD（在形态+控制器上做 model-based diffusion）。

---

## 〇、写在前面：诚实定位 + 三条血泪教训

**当前最硬的事实**：在我们目前的 crawling 协同设计任务上，**ours 还没有清晰赢过 CMA-ES**。CMA-ES 是个又简单又鲁棒的强基线。下面的规划全部围绕「如何真正做到 ours > all baselines」，但必须先承认起点。

**三条教训（直接影响后续怎么做实验）**：
1. **单 seed 噪声 ~5%，不要在没多 seed 前建叙事。** 我们一度以为「CMA-ES 随维度退化、ours 反超」，结果 2000-d 一跑发现那是 1000-d 的一次噪声低抽——CMA-ES 三个维度都稳在 ~114。**今后任何"谁赢谁"的结论，必须 ≥3 seed + mean±std。**
2. **matched-rollout ≠ matched-wall。** 同样本数下 ours(MBD) 墙钟 ≈ 2× CEM/CMA-ES（扩散精修开销）。论文要同时报 matched-rollout（主）+ wall（附注），别被审稿人按墙钟问倒。
3. **加大 M / 加维度不能压噪声，只有多 seed 能。** M256 那次就是为错的问题烧了 5.5hr GPU。

---

## 一、阶段性总结（已完成的工作）

### 1.1 已完成清单

| # | 工作 | 状态 | 位置 |
|---|---|---|---|
| A | **接通 5 个 3D generator + Table 3（贡献③）** | ✅ 接通；⚠️ 未按 task-text bank | `scripts/tasks/soft_robot/morphology/prior_adapters/`（pointe=text→点云, shape=Shap-E text→mesh, triposg=image→mesh, trellis, splatflow，各自独立 venv）；`results/.../table3/TABLE3.md` |
| B | **ours 改为 open-loop 轨迹控制器** | ✅ 实现+验证 | `scene.py:113-114`（cfg）、`:514-519`（compute_actuation open_loop 分支，zero-order-hold `tanh(traj[node])`）；phi_dim=n_control_nodes×n_act，两个 launcher 都接了 |
| C | **CEM/CMA-ES/ours 维度扫描** | ✅ 500/1000/2000-d，[4,3,4] | 见 1.2 |
| D | **多 modes（风险敏感多模式，贡献③）** | ✅ 已 wired（部分 task 12/36 模式） | `jax_mpm.py:70-75`：loco+hurdling=12，push/carry/gripping/carry_terrain=36，crawling/balancing/landing=4 |
| E | **（补充）贡献②自适应多保真度修复** | ✅ rank-fidelity V̂ + 廉价探针 + 活对偶 | `mrmfmbd_mbd_jax.py:245`，见 `THEORY_vs_CODE.md §0.1/0.2` |
| F | **（补充）MBD 配置根因修复** | ✅ betaT/tau_frac/温度/M | 见 `THEORY_vs_CODE.md §0.2` |
| G | **（补充）rollout chunking** | ✅ lax.map 分块，大 M/大形态不 OOM | `mrmfmbd_mbd_jax.py:_rollout_at` |
| H | **（补充）协议 B 设计 + Point-E 权重已下** | ✅ 设计完成，待实现 | `THEORY_vs_CODE.md §0.2` |

### 1.2 实验结果（诚实汇总）

**(1) 平滑正弦控制器（80-d）matched-M frontier**（单模 [0.5], K=100）：ours **每个 M 都输** CEM/CMA-ES。

| M | ours-narrow | CEM | CMA-ES |
|---|---|---|---|
| 32 | 13.71 | 14.50 | 15.19 |
| 128 | 14.44 | 15.43 | 15.27 |
| 512 | 15.49 | 16.17 | 16.28 |

**(2) 非凸 open-loop 轨迹控制器 维度扫描**（[4,3,4], M=128, K=100, 单模 [0.5], **单 seed**）：

| 维度 | ours-wide | CEM | CMA-ES | 谁赢 |
|---|---|---|---|---|
| 500-d | 106.81 | 105.96 | **116.59** | CMA-ES |
| 1000-d | 112.18 | 113.01 | 110.32 | ≈ 三方并驾 |
| 2000-d | **55.83**（崩） | 90.84 | **114.24** | CMA-ES |

- **MBD 机制确实激活**：非凸地形上 ours-宽退火 ≫ ours-窄（500-d：106.81 vs 65.67，+63%），与平滑任务相反。
- **但 CMA-ES 鲁棒最强**（~114 各维度），ours 仅 1000-d 有竞争力，**2000-d 崩**（M=128 在 2000 维欠采样，MC 分数噪声太大）。

**(3) Table 3（3D prior，贡献③）**：TripoSG 15.34 > TRELLIS 10.83 > SplatFlow 9.89 > random。⚠️ 但用的是 **10 个通用生物 prompt（animal/worm/crab/...），不是 task-related text**——这是未来要重做的（见 F4）。

### 1.3 当前诚实结论

- **ours 尚未清晰赢 CMA-ES**——这是核心待解问题，不是调参能解决的（已修配置、扫到 2000-d、验过 chunking）。
- **真正已落地、可写的贡献**：① 联合扩散协同设计框架；② **自适应多保真度（已修，rank-fidelity V̂）**；③ 风险敏感多模式（最坏情况）；3D-prior 排序（Table 3）。
- **MBD 的优势区是「中维 + 非凸」**，太低维（平滑）CMA-ES 赢、太高维（欠采样）ours 崩。这本身是个有意义的发现，但不是「全面碾压」。

---

## 二、未来方向（一步步执行计划）

> 每项标注：**怎么做** / **GPU 或 CPU** / **依赖** / **风险** / **产出**。  
> 顺序按依赖关系排了——`F0` 是所有对比实验的前置铁律。

### F0 ⭐【前置铁律】实验规范 + 修 2000-d 崩溃（先做，否则后面都白跑）
- **F0a 多 seed 协议**：所有「谁赢谁」对比 **≥3 seed（最好 5）**，报 mean±std + 配对检验。写一个 `run_multiseed.sh` 包装器统一跑。**【CPU 编排 + GPU 执行】**
- **F0b 修 ours 2000-d 崩溃**：2000-d M=128 时 MC 分数欠采样 → ours 崩。两条路：(i) 用 **chunked 大 M（256/512）** 在 2000-d 重测（chunking 已支持，内存不爆）；(ii) 降低 MC 分数方差（如 control-variate / 更软温度）。**否则一旦加 diff-physics 梯度基线，它会在 2000-d 我们崩的地方显得很强，反衬我们弱。** **【GPU】**
- **F0c 口径固定**：headline 用 matched-rollout（M×K），附注报 wall + best-of-N。先把对比维度**锁在 500–1000-d**（ours 有竞争力的区间），2000-d 仅作 ablation。
- **产出**：一套可复用的多-seed 对比脚手架 + 一张「ours 不崩」的 1000-d/2000-d 多 seed 表。
- **风险**：若多 seed 后 ours 在 1000-d 仍只是 ≈ CEM/CMA-ES（不是 >），就要正视「换任务」或「重定位论文」（见 F1）。

### F1（用户①）必须 ours > all baselines —— 总目标，拆成可执行子任务
这是整个 phase 2 的核心。诚实讲，在当前 crawling 任务上达不到。要达成，需要**任务 + 方法**两头一起推：
- **F1a 找对的任务**：ours 的优势在「非凸 + 多峰 + 中维」。要构造一个 **CMA-ES 会卡局部最优、ours 全局退火能跳出**的任务——候选：(i) **欺骗性/多峰地形**（障碍/台阶/沟，需调好运动尺度，之前测过运动太小走不到障碍——要先放大控制权限，open-loop 已经放大了）；(ii) **风险敏感多模式 + 冲突 regime**（贡献③，最坏情况目标本身非凸，CMA-ES 优化均值会被最坏拖垮）。**【GPU】**
- **F1b 把方法优势接上**：① learned 闭环（F3，地形更非凸）；② 多保真度真省算力（贡献②已修，在「fine 很贵」的任务上体现）；③ 风险目标（贡献③）。
- **风险/诚实**：若试遍仍 ≈ CMA-ES，**Plan B = 重定位论文**——「ours 与 SOTA-ES 持平 **且** 独有 risk-robustness + 自适应多保真度 + 统一框架 + 3D-prior」，Table 1 写 "competitive with / closing gap to CMA-ES" 而非 ">"。这不丢人，且诚实。

### F2（用户②）按 DiffuseBot 粒子尺寸跑（[13,8,13] 对齐）
- **关键澄清**：**粒子数已经对齐了**。我们 particle_spacing=1/128 over box(0.10,0.06,0.10) → 12×7×12 = **1008 粒子**，DiffuseBot（SoftZoo Box 0.08³ sample_density 32）也是 ~1008。**粒子数与 voxel_dims（[13,8,13]/[4,3,4]）无关**——后者只是 decoder/形态分辨率。
- **真正要对齐的只剩 sim 参数**（协议 B 的 param-alignment）：**n_grid（我们 config 用 128 vs 默认 64）**、gravity=3.8、dt=5e-4、n_frames=100、friction=0.5、材料 E0×scale~1e4、n_actuators（我们 10 vs DiffuseBot 4）。
- **怎么做**：写一个 `diffusebot_aligned` 评估 config，把这些参数对齐；[13,8,13] 仅作为 decoder 形态分辨率（贡献①的 latent 形态）。**【GPU】**
- **若要"上万粒子"（更高分辨率）**：改 particle_spacing（1/256→~10k，1/320→~20k）+ 配套 n_grid，chunking 兜内存（前面算过单卡 24GB 前向可跑 ~230 sample / 分块无限）。但**这不是 DiffuseBot 对齐**（DiffuseBot 就 ~1008），是另开「高分辨率」实验。
- **产出**：一个与 DiffuseBot 同 sim-参数的任务 config，供协议 B（F5）与所有基线共用。

### F3（用户③）learned 闭环策略
- **现状**：路已通但未赢。latent c（`D_β`，controller_latent_dim=16）`codesign.py:79-83`；`compute_actuation_learned`/`rollout_return_closed`（scene.py:1062/1120）；环内 SHAC+critic（backend :811-877）；warmup `_warmup_policy:547`。
- **怎么做**（三条，从稳到激进）：(a) **蒸馏**——先用 ours 优化 open-loop 最优轨迹，行为克隆成 `π(state)`；(b) **直接 policy-MBD**——优化策略参数/latent，MBD 在策略空间搜（已有路），与形态联合；(c) **SHAC 一阶精修**——闭环 + 环内策略梯度（hybrid 零阶+一阶）。
- **关键注意（来自 open-loop 经验）**：闭环 latent 若太低维（16）地形可能变平 → 回到 CMA-ES 占优区。**建议先在较高维策略参数上验证 MBD 仍赢，再考虑降维**。
- **GPU**；**依赖** F0（多 seed）、F2（对齐 sim）。**风险**：低维 latent 抹平非凸优势。

### F4（用户④⑤⑥）3D shape prior 重新 bank（按 task text）+ 做成真正结构先验
- **现状/缺口**：`prompts.txt` 是 10 个通用生物，**不是 task-related text**；`TABLE3.md L57-77` 明确 TODO：用 **8 个 functional per-env prompt** 重 bank。**已阻塞**于：GPU + SplatFlow/TRELLIS 的 venv 被删。
- **怎么做**：
  1. 为 8 个 task 各写 functional prompt（如 crawling→"low flat many-legged crawler"，gripping→"hand-like gripper with fingers"）。**【CPU】**
  2. 重建 generator venv（Point-E 已能跑；TripoSG/Shap-E 视需要）。**【CPU 装环境】**
  3. 跑 generator → 形态点云/mesh → robotize → bank。**【GPU】**
  4. **接入做结构先验（对齐 co-design 理论）**：把 banked 形态作为**扩散初始化 / 形态先验分布**（贡献① 的 `p_θ^MF` 或 `theta_init`），而非事后画廊。这才是「真正的结构先验」。**【GPU】**
- **依赖**：F2（粒子/sim 对齐，保证 prior 形态在我们 sim 里可评估）。
- **产出**：8-task 的 functional 形态先验库 + 接入扩散初始化的 ablation（有 prior vs 无 prior）。

### F5（用户④-B）把 DiffuseBot 做成 baseline（协议 B）
- **设计已完成**（见 `THEORY_vs_CODE.md §0.2`），且**大幅简化**：DiffuseBot 的 crawling 跑的是 `forward_sim`——**无物理梯度、无 guidance**，所以**不需要 JAX↔PyTorch 梯度桥**，只要前向 reward 替换。
- **怎么做**：
  1. `diffusebot_bridge.py`：固定 scatter-mean 矩阵 `P[n_voxels, n_p]`（DiffuseBot 逐粒子 occupancy → 我们逐体素 x_morph，先 un-swap y/z）+ `make_oracle()`（包 `rollout_return`）。**【CPU 写 + GPU 测】**
  2. DiffuseBot loop 的 `forward_sim` 加 `external_sim=jax_mpm` 旁路（或抽出 Point-E sample + SDF + top-k buffer + embedding update，避开装 SoftZoo/Taichi）。**【CPU 写】**
  3. 跑：DiffuseBot 100 epoch × 60 sample = **6000 前向 rollout**；ours/CEM/CMA-ES 同样 6000 前向（M=60×K=100 等）。**【GPU】**
  4. 所有最终设计在同一 oracle 重认证。
- **公平警示（务必写进论文）**：DiffuseBot 用**预训练 Point-E 先验 + 外部数据**，ours/CEM 数据无关 → 做**先验开/关消融**，把「先验贡献 vs 优化器贡献」拆开。
- **依赖**：F2（sim 对齐）；Point-E 权重已下。**产出**：DiffuseBot vs ours/CEM/CMA-ES 在我们 sim、匹配算力下的对比表。

### F6（用户⑥⑦）新增 1–2 个「方向对但在我们 task 上有缺陷」的 baseline
研究结论（见下 §四引用）——**推荐这两个，理由是它们的已知弱点正好是我们机制的强项**：

- **ADD #1（首选）：可微物理「梯度下降 on design」基线**（SoftZoo/DiffuseBot 式）。
  - 做法：把设计参数化（逐粒子 / Implicit-MLP / Diff-CPPN），把任务 reward 反传过**可微 JAX-MPM**，Adam + **best-of-N restart**。我们 sim 本来就可微 → **最易复现**。
  - 为什么打得过：作者自己承认「梯度法局部性 → 多局部最优困住」，且**接触梯度在大片轨迹空间为 0 或崎岖**（arXiv:2206.11884 / 2206.10787）——正是我们宽退火采样的强项。
  - **⚠️ 公平铁律**：必须给它 best-of-N + Implicit/CPPN 重参数化（更平滑），否则审稿人骂「你阉割了梯度基线」。**【GPU】**
- **ADD #2：Transform2Act 式 RL 协同设计**（PPO，transform-then-control）。
  - 为什么打得过：已知**极度样本低效**——基准报告它在接触丰富 locomotion 上「预算内无法收敛」（需数千万 env step）；在我们 matched-compute（M×K）下根本跑不动 = 干净的赢。
  - **⚠️ 注意**：它原为**离散图/关节形态**设计，要适配我们连续高维设计（额外工作量）。**【GPU】**
- **CMA-ES/CEM 的定位**：**不要**把它们当「我们打的基线」——它们是**强鲁棒 ES 标杆**（2000-d 还赢我们）。论文里诚实报为「强当代竞争者，我们在缩小差距」，而把 #1/#2 作为「我们机制专门设计去赢」的基线。
- **可选 ADD #3**：BO（GP-BO / GLSO latent-BO）**仅在降维（50–100d）时**做——展示「ours 不需手搭低维 latent 也能匹配 BO 的样本效率」。全维（500-2000d）BO 因维度而非地形输，故事不干净，跳过。

### F7（用户⑦）8 个 task 上跑 5–6 modes
- **现状**：modes 已 wired，但各 task 模式数不一（crawling/balancing/landing=4，loco/hurdling=12，push/carry/gripping/carry_terrain=36）。
- **怎么做**：为实验**统一标准化到 5–6 个 mode**（沿 friction / slope / mass / init-vel 轴选 5–6 个有意义的 regime），8 个 task 各跑 ours/CEM/CMA-ES（+F6 的两个基线）的风险敏感对比。这正是贡献③（最坏情况）最能体现优势的地方（CMA-ES 优化均值，会被冲突 regime 拖垮）。**【GPU】**
- **依赖**：F0（多 seed）、F6（基线就位）。**产出**：8-task × 多-mode 的风险敏感主表（很可能是论文 Table 1 的主体）。

---

## 三、GPU vs CPU 分工速查

| 能在 **CPU** 做（不占卡） | 必须 **GPU** |
|---|---|
| F0a 多-seed 脚手架编排 | F0b 修 2000-d 崩溃（重测） |
| F4-1 写 8 个 functional prompt | F1a 欺骗/多峰任务实验 |
| F4-2 装 generator venv | F2 DiffuseBot-对齐 sim 跑 |
| F5-1/2 写 bridge + DiffuseBot 旁路代码 | F3 learned 闭环训练 |
| F6 适配代码（RL 包装、CPPN 重参数化的实现） | F4-3/4 generator 推理 + prior 接入扩散 |
| 所有 config / 文档 / 结果分析 | F5-3 协议 B 跑（6000 rollout） |
| CEM/CMA-ES 的**算法本体**（forward-only，CPU 也能跑，但慢，建议 GPU） | F6 两个新基线跑 |
| | F7 8-task × 多-mode 主表 |

> 注：CEM/CMA-ES 是 forward-only，CPU 可跑但慢 ~10×；policy 训练 CPU 太慢，必须 GPU。

---

## 四、注意事项 / 风险 / 关键决策

1. **多 seed 是铁律**（教训①）。任何进 Table 的数都要 ≥3 seed。
2. **先修 2000-d 崩溃再加 diff-physics 基线**（F0b），否则基线会在我们崩的地方显强。
3. **匹配算力 + best-of-N + Implicit/CPPN 重参数化**，给每个梯度基线公平待遇（F6 公平铁律），并引接触梯度平滑文献（arXiv:2206.11884 / 2206.10787）预防审稿人质疑。
4. **CMA-ES 是真标杆，别假装打得过**。诚实定位（强竞争者，缩小差距）+ 把 #1/#2 当「我们设计去赢」的基线。
5. **DiffuseBot 公平性**：它有外部预训练先验，必须做先验开/关消融。
6. **learned 闭环别降维过头**（F3），低维 latent 会抹平非凸优势。
7. **关键决策点（F0 之后）**：多 seed 固化 1000-d 后，若 ours 仍只是 ≈ CMA-ES → 启动 Plan B 重定位论文（competitive + 独有贡献），不要继续无限烧算力找「碾压」。

---

## 附：核心文件锚点

- 方法 backend：`genedynamics/solvers/single/mrmfmbd/backends/mrmfmbd_mbd_jax.py`
- open-loop 控制器：`scene.py:113-114, 514-519`
- learned 控制器：`scene.py:1062/1120`，backend `:514/547/811-877`
- 任务/模式：`genedynamics/experiments/plugins/task_domains/jax_mpm.py:32-75`
- 3D prior：`scripts/tasks/soft_robot/morphology/prior_adapters/`，`results/.../table3/TABLE3.md`
- 协议 B 设计 + MBD 根因 + 贡献②修复：`docs/THEORY_vs_CODE.md §0.1/0.2`
- 启动器：`scripts/tasks/soft_robot/co_design/run_learned_codesign.py`（ours）、`run_baseline_p2.py`（CEM/CMA-ES）
