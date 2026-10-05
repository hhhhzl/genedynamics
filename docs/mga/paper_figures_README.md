# MGA：三页实验图表与叙事安排

审计日期：2026-09-19。本文整理已经讨论确定的展示方案，并以当前论文、配置和落盘结果限定结论。三页指实验部分的总版面预算，包含图、表、caption 和解释文字；不是每个环境各占一页。

配套文档：[H1 四任务报告缺口与修复顺序](humanoid_report_README.md)。完整工程历史见 [experiment_run_plan.md](experiment_run_plan.md)。当前论文实际位置是 [`latex/latex_mga`](../../latex/latex_mga)，不是仓库根目录下的 `latex_mga`。

## 当前论文更新：Scanning seeds 0–9

2026-09-22：Scanning 主表、正文引用数字和 Appendix G 的聚合图/统计已切换到全部 seeds 0–9。数据仅来自 `results/arm/surface_scan`；主表八个方法 × 13 suites × 10 seeds 的 1,040 条结果均齐全。方法顺序为 ISSA、ATACOM、MPPI、DIAL、PegasusFlow、MGA、w/o RL prior、w/o LRC；`no_retraction` 对应 w/o LRC。

分组仍为 Hard 4、Soft 4、Hybrid 3、Unseen 2，不把 unseen 再算入 Hard/Soft。每个 seed 内对组内 suites 等权平均，再对十个 seed 均值求均值与总体标准差；SSR 阈值、归一化力阈值和失败保留规则不变。主图及 Visualizations 的固定 seed 0 示例不随统计扩容重新挑选。下文 seed 0–1 的版式开发记录保留为历史，不是当前论文统计或最终方法列表。

| 分组 | MGA：SSR / nCVaR95 | w/o RL prior：SSR / nCVaR95 | w/o LRC：SSR / nCVaR95 |
|---|---:|---:|---:|
| Hard | 100.0% / .498 | 100.0% / .486 | 67.5% / .521 |
| Soft | 100.0% / .466 | 100.0% / .488 | 70.0% / .498 |
| Hybrid | 100.0% / .658 | 96.7% / .738 | 56.7% / .702 |
| Unseen | 50.0% / .429 | 55.0% / .433 | 75.0% / .502 |

跨 13 suites，MGA、w/o RL prior、w/o LRC 的 operational success 分别为 120/130、120/130、87/130；zero-violation success 为 119/130、117/130、86/130。接触有效覆盖率分别为 96.03±1.38%、97.10±1.26%、84.75±6.19%。w/o LRC 的终点进度仍达 99.88%，但 43 次失败中 42 次仅覆盖不足、1 次仅接触保持不足，因此“推进到终点不等于忠实覆盖”仍是 Surface 的主要证据。Unseen 上 MGA 未取得最高成功率，这一边界保留在正文。

对应入口仍是 `scripts/paper/mga/plot_surface_mechanism.py`（默认 seeds 0–9，`--appendix` 生成聚合附图）；输出仍在 `reports/mga/paper_figures`，论文副本在 `latex/latex_mga/figures/exp`。主表位于 `tex/peg_insert_results.tex`，统计定义位于 `sections/appendix_environments.tex`，连续量与 strict-success 敏感性检查位于 `sections/appendix.tex`。

Peg 主图 (f) 保留同一固定示例、全部未平滑真实 wrench 样本及首次完成标记；峰值为 MGA .791、w/o prior .870、DIAL .901、ISSA .974。ISSA 是现有 nominal execution reference，不能作为匹配的 Sensing-OOD 对比；图内脚注删除，来源限定保留在 caption 和图的 JSON 中。该示例支持较低峰值，不意味着 MGA 在所有 seeds 上 force tail 最低或完成最快。

(g) 展示同一次执行的计划选择日志：prior proposal 被采用、refined plan 未通过模型复核、task-owned emergency 被选择。它不是三种互斥接触模式，也不是实际违规或 jam 计数。该示例 emergency 在首次完成后发生，且由 task override 触发，不能用来证明 jamming recovery；它与 (f) 的实际执行响应应分开解释。

## 1. 三页怎么放

| 页 | 内容与建议占比（含 caption） | 读者应得到的结论 |
|---|---|---|
| 第 1 页 | 通用表约 30%；Scanning 机制图约 50%；setup/RQ/解释约 20% | 在同一安全成功口径下比较方法；展示 geometry、可实现响应和 stiffness 如何共同维持扫描进度 |
| 第 2 页 | PegInsert 时间图约 40%；H1 机制图约 40%；解释约 20% | PegInsert 说明 prior 与 model-based 的互补和局限；H1 说明接触几何如何扩展到物体与全身运动 |
| 第 3 页 | Scanning + insertion 实机综合图约 65%；平台、重复试验、解释约 35% | 用真实执行轨迹说明部署可行性，与仿真的受控机制比较形成互补 |

最终是一个通用表、三个仿真图组、一个实机图组。H1 主图优先 Unjamming，Walk-and-Push 是通过严格 walking 验证后的扩展面板。Force Regulation/Fixed-Stance Push 的完整图、全部曲线和消融放 Appendix。

比例是排版起点。缩版时先减少重复 panel 和装饰图，保留坐标、单位、阈值与清晰的字，不把所有信息缩成不可读的整页拼图。

## 2. 每个实验回答什么 RQ

沿用目前 [Introduction 的三项 contribution](../../latex/latex_mga/sections/intro.tex)：C1 是 RL policy 到结构化短时域 proposal；C2 是共同 closed-loop rollout 下的 model-based generative annealing；C3 是 motion–impedance 的 realization-aware manifold refinement。

| 实验 | 主 RQ | 主要对应的 contribution | 证据边界 |
|---|---|---|---|
| Scanning | 几何和接触响应变化时，manifold refinement 能否兼顾已执行路径覆盖、接触力和形变？ | C3，辅以 C2 | hybrid 对 stiffness/geometry 有开发证据；不把扫描优势全部归因于 RL prior |
| PegInsert | policy proposal 经 model-based refinement/revalidation 后，是否比直接 RL 或去掉 prior 更容易安全完成插入？ | C1 + C2，C3 提供 refinement 机制 | 分 ID、Pose-OOD、Sensing-OOD 报告；prior 的好处是条件性的 |
| H1 push | 同一框架能否处理解卡/偏航纠正，并进一步协调支撑、手部接触和物体运动？ | C3 向 whole-body 扩展，结合 C2；C1 待 full-MGA 验证 | Unjamming 是主机制任务，Walk-and-Push 是扩展；当前开发结果尚不能证明 full-MGA 全身优势 |
| 实机 | model-based 执行组件能否在真实机器人上完成曲面接触扫描和入孔对齐？ | 部署可行性 | 已记录实机没有 RL prior；入管案例未激活 contact-force refinement，不能作为 full-MGA 或 jam recovery 的实机证明 |

可靠性与 gate 用来解释“何时相信修正、何时拒绝候选”，不单凭 gate 接受率给出可靠性正确或绝对 OOD 安全的结论。需要对应的错误预测、执行风险和消融证据。

## 3. 通用表：主文只放 SSR 和 nCVaR95

沿用已经指定的列顺序：

```text
Task | Setting | Metric | RL | ISSA | ATACOM | MPPI | DIAL | PegasusFlow | w/o RL prior | MGA
```

这里有一个必须写清楚的区别：正式运行的八个算法包含 `Model-based Only`；上述主表为了突出 prior 消融，展示的是 `w/o RL prior`。二者是不同配置，不能互相改名或替换数值。MBO 的完整结果放 Appendix；如果主文必须同时展示八个算法和 prior 消融，则增加第九个方法列。

建议表格层级：

- Surface Scan → Hard / Soft / Hybrid → SSR、nCVaR95。Hard/Soft 各覆盖 plane、cylinder、convex、bumpy、unseen，Hybrid 覆盖三种 stiffness map；每个 suite 等权，并说明分组规则。Appendix 逐一展开全部 13 suites。
- PegInsert → ID / Pose-OOD / Sensing-OOD → SSR、nCVaR95，保留三组，不混成总均值。
- H1 → Unjamming / Walk-and-Push → SSR、nCVaR95；Walk-and-Push 只有通过新协议验证后才能填入。Force Regulation/Fixed-Stance Push 完整指标放 Appendix。

指标解释：

1. SSR 是“完成该任务且满足该任务规定的安全条件”的跨 seed 比例。PegInsert 要求插入成功、全规定评估窗口无 F/T 违规且无 jam；H1 还要检查非手接触、姿态/跌倒，Walk-and-Push 要检查真实步行。Surface 必须明确覆盖/接触/安全门槛，不直接借用 PegInsert 定义。
2. nCVaR95 是每条轨迹归一化接触载荷中最高 5% 样本的均值，再跨 seed 汇总。PegInsert 使用四项 wrench utilization 的最大值：横向力/20 N、轴向力/30 N、弯矩/1.5 N·m、扭矩/1.0 N·m。Surface 可由实际法向力除以对应 `f_max` 计算；H1 当前字段为 `physics_force_normalized_cvar95`，使用真实物理子步手力。
3. 这列主要度量接触载荷尾部及其距限制的余量，不独立度量时间振荡。低 nCVaR95 也可能来自停滞或不接触，所以必须和 SSR 一起读。Surface 的形变、H1 的平衡不包含在这个“接触力尾部”标量里，要在任务图与完整指标中保留。
4. 同一任务内方法必须用相同采样频率、评估窗口和终止规则。不同任务可以有不同物理采样率，但不能把三任务的 nCVaR95 再平均成一个“总体安全排名”。短暂超限可能仍对应 nCVaR95 < 1，是否违规由完整轨迹的阈值判断。
5. 展示全部 seeds 0–9；注明已接受的 seed overlap 与被开发阶段查看过的评估条件。SSR 给成功数/10，Appendix 给二项比例区间；连续量给跨 seed 均值、离散度及配对差值。10 个 seed 的差异不能自动写成显著优势。

## 4. Scanning 主图：空间位置上的机制，而不只画 cylinder

建议标题方向：**Maintaining contact progress across geometry and compliance changes**。

Panel A 是全部八算法的 spatial mechanism strip，可排成 2×4 小图。固定同一 suite、同一 seed、同一视角与色标，每格叠加目标路径和实际 EE 路径。优先使用确实跨过 hard–soft 边界的 hybrid 案例，背景表示真实 stiffness map；轨迹颜色只表示一个量，例如归一化法向力。标出未覆盖区和失去接触的位置。八算法包括 MBO，不用不同方法各自最好看的 seed。

Panel B 在相同弧长/材料坐标下，对齐背景材料 stiffness 与执行的法向 stiffness `nᵀ K n`。再用紧凑曲线展示法向力或形变及阈值。重点比较 MGA、no-controllability-geometry 和 no-retraction；这部分回答“为什么需要可实现响应与流形部署”。`no_rl_prior` 与 `no_learned_reliability` 在全部 13 个 suites 的配对结果放入主表/Appendix，用来回答 learned components 是否跨 rigid、soft、unseen 和 hybrid 条件提供收益。实际测得的响应相关性不能替代对应的单因素消融。

Panel C 可用小型散点/配对差值图概括 hard、soft、hybrid 的 coverage–force/deformation tradeoff。空间不够时移到 Appendix，保留 A+B。不同 geometry 的结果仍在完整表和媒体索引里，不把一个 hybrid 示例称为所有曲面验证。

coverage 必须由整条已执行 EE 轨迹计算；“访问过的最远弧长”和“容差内实际访问过的路径比例”分开记录。只看最终点或 command 进度不能证明覆盖。

### 4.1 Surface seed 0--1 版式开发结果

`results/arm/surface_scan` 现已包含全部八个算法与四个消融在 13 个 suites、seeds 0--1 上的完整结果，共 312 个 task runs。续跑容器正常退出；每个方法均有 26 个 `results.json`、26 条轨迹、13 个 suite summary、`overall_summary.json` 和 `protocol_manifest.json`。这些数值只用于冻结版式和统计实现；正式论文数字在相同定义下用 seeds 0--9 替换，不再改变图的编码、阈值或布局。

Surface 主文采用容忍一个离散接触瞬态的 operational SSR：

```text
coverage >= 0.90
AND settled_contact_loss_rate <= 0.10
AND force_violation_rate <= 0.01
```

其中 coverage 使用整条 executed EE trajectory、实际接触 mask 和 5 mm 路径容差；settled contact loss 排除前 10 个接触建立步。100-step 协议下，1% force-violation tolerance 至多容忍约一个离散控制采样的瞬态。Appendix 同时报告 `force_violation_rate == 0` 的 strict SSR，避免 operational tolerance 隐藏真实越界。该口径在查看 seeds 2--9 之前冻结。

当前 operational SSR（成功数/运行数）为：

| 方法 | Hard | Soft | Hybrid | Overall |
|---|---:|---:|---:|---:|
| Standalone RL | 0/10 | 0/10 | 0/6 | 0.0% |
| ISSA | 0/10 | 1/10 | 0/6 | 3.8% |
| ATACOM | 2/10 | 6/10 | 4/6 | 46.2% |
| MPPI | 3/10 | 2/10 | 2/6 | 26.9% |
| DIAL | 6/10 | 7/10 | 3/6 | 61.5% |
| PegasusFlow | 1/10 | 2/10 | 1/6 | 15.4% |
| Model-based Only | 7/10 | 9/10 | 6/6 | 84.6% |
| **MGA** | **8/10** | **10/10** | **6/6** | **92.3%** |

主表仍将 SSR 与 nCVaR95 成对阅读。当前分组均值如下；每格为 `coverage / nCVaR95`，nCVaR95 是实际法向力 CVaR95 除以该 suite 的 `f_max`：

| 方法 | Hard | Soft | Hybrid |
|---|---:|---:|---:|
| Standalone RL | .349 / .607 | .388 / .519 | .399 / .600 |
| ISSA | .413 / .596 | .525 / .534 | .441 / .714 |
| ATACOM | .786 / .534 | .919 / .425 | .884 / .606 |
| MPPI | .994 / 1.056 | .975 / .960 | .779 / .874 |
| DIAL | .966 / .955 | .995 / .818 | .838 / .766 |
| PegasusFlow | .878 / 1.184 | .956 / 1.089 | .690 / .844 |
| Model-based Only | .891 / .581 | .974 / .435 | .922 / .642 |
| **MGA** | **.908 / .551** | **.995 / .415** | **1.000 / .659** |

这两组开发数据支持的 Surface 结论是：MGA 不必在每个单项力指标上最小，而是在 geometry/compliance shift 下取得最高的安全完成率；Soft 达到 10/10，Hybrid 达到 6/6。MBO 是有竞争力的强基线，DIAL/ATACOM 也有非零且可观的成功率，因此叙事不是“其他算法完全不能工作”，而是 MGA 改善 progress--safety tradeoff。Hard 的 `rigid_unseen` 仍是 MGA 的主要失败来源，不隐藏在总体均值中。

消融的 operational SSR 为 MGA 24/26、w/o RL prior 24/26、w/o controllability geometry 24/26、w/o retraction 17/26。SSR 饱和时，prior 和 geometry 的贡献用 paired nCVaR95/coverage 解释：RL prior 在 Hybrid 将 nCVaR95 从 .728 降至 .659；controllability geometry 将整体 nCVaR95 从 .551 降至 .524；retraction 则直接把 SSR 从 65.4% 提升至 92.3%。`no_learned_reliability` 与 MGA 的全部物理任务指标相同，因为当前 learned reliability 对所有这些运行 abstain 且不拥有 authoritative promotion。PegInsert 的独立 P4 study 同样没有形成稳定控制增益，因此正式 PegInsert MGA 冻结为 core-only；论文不再用三个任务中的任一个声称 learned reliability 带来性能提升。

### 4.2 Scanning 在仿真综合图中的 60% 版面

Scanning 占 Scanning/PegInsert 综合页左侧 60%，不重复主表已经给出的 SSR/nCVaR95。按两列内部网格组织：上部约 58% 高度用于执行结果，下部约 42% 高度用于机制时间/空间对齐。

**上部：八算法 spatial execution strip。** 使用固定的 `hybrid_stripes, seed=0`，排成 2×4 小图，方法顺序与主表一致：RL、ISSA、ATACOM、MPPI、DIAL、PegasusFlow、MBO、MGA。所有小图使用同一相机、同一路径范围和同一 `[0, 1]` force-utilization 色标；背景画真实 hard/soft stiffness bands，黑色细线为 reference path，彩色线为 executed、contact-valid EE path，灰色虚段只表示未覆盖参考区。标题只给方法缩写，角标给 coverage，不在图内重复 SSR/nCVaR95。seed 0 对所有方法固定，不能逐方法挑最好 seed。当前该 suite 上 MGA coverage 为 1.000；ATACOM/MBO 为 .901/.931，而 MPPI、DIAL、PegasusFlow 为 .822/.782/.327，能够直观看到同一材料切换下的覆盖差异。

**下部：一条 MGA hard--soft transition 的 realization trace。** 仍使用 `hybrid_stripes, seed=0`，共享横轴为 executed material/path coordinate `scan_xi`，而不是把不同接触时刻直接做跨 seed 平均。第一行用浅色背景画 `k_surf` 区域，并叠加执行的法向 stiffness `n^T K n`；第二行画实际 `F_n/f_max` 与安全线 1，同时用短标记注明 contact loss、retraction/emergency 或 gate rejection 的真实事件；第三行只在空间允许时画累计 contact-valid coverage，否则移至 Appendix。主文不再加入第二个 SSR 柱状图，也不画没有候选级日志支持的虚构 proposal cloud。

这两个 panel 可以直接由现有轨迹生成：`task_signals` 已保存 `positions`、`reference_path`、`coverage_valid_mask`、`scan_xi`、`k_surf`、`normal_stiffness`、`force`、`f_max`、`in_contact` 与 gate signals；每步 `infos` 保存 prior acceptance、revalidation、emergency 和 reliability 状态。版式阶段读取 seeds 0--1，最终仅替换聚合表/置信区间和必要的代表性标注，spatial strip 的预先固定 seed 与视觉编码保持不变。

P8 的旧 hybrid 记录只作为机制开发历史；论文统计现在统一从 `results/arm/surface_scan` 读取。当前 learned reliability 在材料切换上 abstain，支持的是 task-owned model-based certificate，而不是“learned reliability 准确识别所有 hybrid 风险”。

## 5. PegInsert 主图：保留一个随时间变化的指标

主曲线选 **normalized wrench utilization `rho(t)`**，主图横轴用真实时间（秒），安全边界统一画 `rho=1`。这样保留对力控动态的直观判断，也把横向/轴向力与力矩放到同一个可读的刻度上。

推荐三个并列 panel：ID、Pose-OOD、Sensing-OOD。每个 panel 主线展示 MGA、w/o RL prior、Standalone RL 和 PegasusFlow；所有算法的统计已由主表覆盖，八算法完整时间曲线放 Appendix。细线/小标记保留逐 seed 失败与越界，不能让均值曲线遮掉单次冲击。接触切换时刻不同，曲线跨 seed 平均的含义必须说明；代表性实轨迹和逐 seed 轨迹优先于把错位冲击平均成平滑带。

在时间轴底部仅标记接触、retraction、完成/失败等真实事件，不再加第二根 depth 纵轴。是否完成由终点标记与主表 SSR 回答，depth 曲线放 Appendix。若需要接触对齐视图，作为补充图保留 `t=0` 事件定义，主图仍显示实际执行时长。

`free-space → impact → alignment → insertion` 是任务过程的解释，不是每条轨迹一定完整经历的四个离散标签。只有日志支持时才标注实际模式；顺序可能重复、跳过，失败也可能停在入口。impact 尖峰、alignment 的局部调整和反复 recovery 不人为补画。

主文不再堆叠 `e_xy`、`e_theta`、depth–error corridor 图。它们连同四项原始 wrench 曲线、jam/recovery、完整 Pareto 放 Appendix。`rho(t)` 能让振荡可见，但“力控更稳定”的数值结论还需力跟踪误差、接触丢失/抖振或峰值等支持。六维原始 F/T 图适合 Appendix，统一坐标系并标注真实/有偏观测来源。

### 5.1 当前 PegInsert 论文数据源

> 2026-09-24 clean-room reset：下述论文结果已整体归档到
> `results/_archive/peg_insert_protocol_reset_20260924/local/paper_composite`。
> 活动目录 `results/arm/peg_insert` 不再包含这些旧数字；在 P5 正式补跑前，
> 不得从归档自动恢复或把旧数据与新协议混合聚合。

论文此前采用的 OOD 结果曾整合在 `results/arm/peg_insert`。从归档根目录读取 MGA 与 w/o RL prior，可复现当时论文表中的同一组数字：

| 方法 | ID SSR / nCVaR95 | Pose-OOD SSR / nCVaR95 | Sensing-OOD SSR / nCVaR95 |
|---|---|---|---|
| MGA | 90% / 0.722 | 50% / 0.903 | 90% / 0.780 |
| w/o RL prior | 90% / 0.729 | 60% / 0.817 | 70% / 0.726 |

ID 与全部 baseline 保留原来的 canonical 记录；MGA 与 no-prior 的 Pose/Sensing 两组来自历史升级执行。四个 OOD suite 的 seeds 0–9、`results.json`、执行轨迹、PNG 和 GIF 已整套复制，旧目标 OOD 则备份在 `results/_development/peg_insert_pre_paper_ood_merge_20260913`。来源、suite tree hash、主表数值和备份路径记录在 `results/arm/peg_insert/paper_result_manifest.json`。`reports/mga/peg_insert` 也已经仅从这个统一根目录重新生成，共聚合 300 runs。

这些文件的内部 `config_snapshot` 保留原始执行路径和 protocol metadata，用来维持 provenance。当前论文结果仍是“原 ID/baseline + 升级后的 MGA/no-prior OOD”的组合，而不是一次新执行的统一 final 矩阵。已有叙事“ID 打平 no-prior、Sensing 改善、Pose 仍输 PegasusFlow”适用于这份明确锁定的组合。

新主表、时间曲线和 GIF 仍统一以 `results/arm/peg_insert` 为入口，但只能在
clean-room 240-run 矩阵完成后生成。旧 `paper_result_manifest.json` 位于归档，
只描述历史 composite 来源，不能作为新结果的 provenance。

## 6. H1 主图：Unjamming 讲几何，Walk-and-Push 讲真实全身扩展

Panel A：Unjamming 俯视图，画箱体轮廓的时间序列、yaw、手部接触位置与允许的目标区。用 MGA 与 DIAL/PegasusFlow，以及 no-controllability-geometry/no-retraction 的失败轨迹解释 realized contact geometry。只有任务实际有障碍时才画墙，不能给 Force Regulation/Fixed-Stance Push/Walk-and-Push 加同一堵墙来制造视觉效果。

Panel B：Walk-and-Push 侧视 motion strip，标记箱体、pelvis 和左右脚落脚位置；紧凑地附上 foot-contact 时间条与手力/限制。需要同时看见身体前移、支撑前移和左右脚 lift–land，不能用箱体到线或滑脚冒充 walking。

Force Regulation 的 15/30 N 阶跃响应、Fixed-Stance Push 的 10 cm nominal/mass–friction OOD 曲线，以及 Unjamming/Walk-and-Push 的全部力、支撑、yaw 与消融放 Appendix。当前 Walk-and-Push 仍未通过，主图可以预留版面，不能填“理想成功数据”作为实测结果。具体缺口和修复验收见 [H1 README](humanoid_report_README.md)。

## 7. 实机一页：轨迹覆盖与真实执行

沿用已经讨论的实机/仿真分工：仿真突出受控比较和 mechanism，实机突出实际到过哪里、如何保持接触以及如何完成对齐。

- 左侧约 1/4：Nova 5、六轴 F/T 传感器、末端和两种任务装置的实机照片，尺寸与坐标简洁标注。
- 中部约 2/5：table、foam、phantom 三行实际扫描轨迹，叠加目标路径；颜色表示补偿后的法向力/目标或接触状态。同一行给完成路径长度和全部重复试验的覆盖/力误差摘要。
- 右侧其余空间：实机 insertion 的关键帧与已记录的横向偏移–高度轨迹，标出入口和完成位置。明确 robot-reported axis offset、tip offset 与机械间隙的区别。

现有 [`real_results.tex`](../../latex/latex_mga/tex/real_results.tex) 与 [实机 Appendix](../../latex/latex_mga/sections/appendix.tex) 已记录 table/foam/phantom 各 5 次扫描；完整 session 完成数分别为 5/5、4/5、5/5，不能把 foam 扫描完成改写成所有 session 成功。insertion 目前只有一个展示案例，且没有启用 contact-force refinement，不报成统计成功率或主动解卡验证。

实机不画一整页六维力曲线。六轴原始/补偿后数据、显示滤波、真实时间、force holds/recovery、标定限制和重复试验表放 Appendix。显示的平滑不用于计算峰值或误差；轨迹覆盖若尚未从日志算出，先标数据待提取，不能把“有末端位置日志”当作已经验证了覆盖。

## 8. Reward/cost 曲线、GIF 和导出

RL learning curves 建议放 Appendix：横轴训练 environment steps，展示同预算下 reward、评估 SSR 和 safety cost，标明 checkpoint selection。MGA 与 standalone 使用同一个 prior checkpoint；MGA 的在线 refinement 不应画成独立 PPO 训练曲线。DIAL/MPPI/PegasusFlow 无 RL 训练，不强加训练曲线。跨 reward 定义的任务不共享 reward 排名；正文用任务完成和实际安全指标回答 RQ。

每个环境都配真实 rollout GIF/视频，PDF 用关键帧和补充材料链接。全部八算法及已规定消融保留 per-seed 媒体；主文使用预先指定的同一 seed 或接近中位表现的代表例，并说明选取规则。最好 seed 可以作为定性示例单独标注，不能替代代表样本；失败帧不能删掉。

保持已有输出架构：

```text
results/<domain>/<task>/<group>/<algorithm>/level_<suite>/seed_<seed>/
  results.json
  trajectory/trajectory.json
  trajectory/trajectory_best.png
  trajectory/trajectory_best.gif
reports/mga/<task>/                         # 跨算法汇总、源数据与导出图
latex/latex_mga/figures/exp/                 # 论文采用的最终图
```

共享指标、统计和绘图扩展放在 `genedynamics/experiments/plugins/metrics/`、`plugins/visualizations/`、`utils/` 的对应现有模块；`scripts/paper/mga/` 调用它们。现有 `plot_peg_insert_draft.py` 只是旧 guided 数据的视觉草稿，不能当作 final additive 全图生成器，也不能把已执行轨迹画成“候选 proposal 分布”。

建议下一次正式出图顺序：先锁定数据来源和比较口径 → 核对必需 task signals → 输出通用表 → 画 Scanning/PegInsert → H1 验收后填 Unjamming/Walk-and-Push → 提取实机轨迹 → 汇总 GIF 索引 → 编译 PDF 检查三页布局。没有单候选轨迹日志时，不制作声称展示 RL 候选优于 Gaussian 的机制图。
