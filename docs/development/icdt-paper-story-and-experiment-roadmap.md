# ICDT 论文叙事与实验完成路线

更新：2026-11-05（America/New_York）  
目标：RSS 2027  
论文入口：[`latex/latex_icdt`](../../latex/latex_icdt/)  
结构参考：[`latex/latex_mga`](../../latex/latex_mga/)  

本文档记录本轮对话形成的论文逻辑、当前证据边界和后续实施顺序。论文正文只
保留方法、理论、实验协议和可引用结果；诊断过程、阶段状态、无效运行、代码迁移
和待办事项留在开发文档。已有冻结结果按其原协议保留，不因叙事更新而重命名或
重判。

## 1. 本轮确定的研究中心

论文不再围绕“把 source action 直接搬到 target”展开，而围绕以下问题：

> 源技能包含有用的接近、接触、模式选择和恢复结构；不同身体和控制器会改变
> 这些指令的实际响应。有限目标交互下，如何识别仍有用的源响应，将其变成目标
> 身体可执行的行为，并在源结构与目标任务冲突时有条件地偏离？

这个中心把已有工作分成三条互补路线：

- Human2Sim2Robot、Play2Perfect、TactAlign、KITE 等说明物体运动、交互意图、
  触觉和精密操作先验可以跨 embodiment 提供结构。
- ASID、AdaptSim、FADA 等说明部署差异需要目标侧辨识、仿真适应或执行模块
  适应。
- DIAL-MPC 和普通 response MPC 说明可以在线利用模型重新优化，但会引入额外
  在线计算。

ICDT 的位置不是否定这些路线，而是连接“源行为结构”和“目标执行响应”：目标
干预用于校准响应，source-conditioned map 产生目标指令，选择性响应约束决定何时
保留源行为。任务目标始终存在，源对应关系不能定义任务成功。

## 2. Introduction 的六段逻辑

1. **技能复用需求。** 已有技能包含路径之外的交互结构；重新收集完整目标演示
   和完整目标训练成本高。
2. **近期方向已经解决的部分。** 共享意图、物体运动先验、目标辨识、任务驱动
   适应和在线规划都已显示价值。
3. **剩余缺口。** 共享坐标不说明 target 能安全实现 source response；模型更准
   也不自动决定是否保留某个恢复模式；纯 task optimization 在有限数据下又可能
   丢失有用结构。
4. **贯穿全文的例子。** 插入时相同横向修正可能在 source 解卡、在 target 产生
   力峰值。正确迁移可能需要改幅值、速度、时序，甚至放弃该模式。
5. **方法。** 有限干预校准 task-relevant response；bounded residual transport
   改变实际请求；task band、历史兼容、模型支持和执行阶段共同控制 source
   anchoring。
6. **贡献与证据。** 每项贡献对应一个 RQ、一组主表比较和最多三项主要消融。
   不要求方法在每个 cell 都获胜；要求预先指定的现象可重复、可归因，并完整
   保留反例。

## 3. 三项贡献及其成立条件

### C1：通过执行机制定义的行为迁移

**主张。** Cross-embodiment transfer 应区分 source command prior、target skill
map、target execution kernel 和最终行为分布。内部 command 或 embedding 更接近
并不等于任务事件被保留。

**需要的证据。**

- transport 必须改变经过 decoder、bounds、recovery 和 selector 后的实际请求；
- 同一语义任务在不同身体上产生正确的事件顺序，而不只到达终点；
- source mode 在 target 可行时可被实现，不可行时的偏离单独报告；
- Panda→xArm7、Panda→UR5e 和固定臂→Go2+Z1 中至少形成两类不同形态证据。

### C2：干预校准的选择性响应约束

**主张。** 有限目标交互提供 task-relevant execution information；source response
只在任务兼容、历史匹配、模型有支持且执行阶段相关时形成额外偏好。

**需要的证据。**

- 相同总目标访问预算下，full 与无校准对照在预声明条件上比较；
- full 与“保留 task band、去 response”的对照隔离 source response 的增量；
- full 与 ungated response 对照展示 selective anchoring 在 source-compatible 与
  source-incompatible 条件中的不同作用；
- 改善必须出现在受约束任务成功或预声明任务事件上，不能只依赖 response MSE。

### C3：条件性事件分析与递增复杂度验证

**主张。** 任务事件能否迁移取决于 source margin、calibration error、anchoring
error 和 unsupported mass。这个条件性结论应在从受限几何到身体支撑耦合的复杂度
阶梯中验证。

**需要的证据。**

- 八个语义子任务覆盖几何时序、持续接触、混合恢复、动态环境和身体稳定；
- 九行 general table 使用共同的四个指标；
- 事件图解释成功或失败发生在哪个 contact phase；
- matched-budget 性能、端到端 latency 和最差组边界同时报告。

如果 clean ablation 不能显示 response anchoring 的独立价值，C2 必须收缩；如果
有效指令没有变化，C1 不能由内部 loss 下降支持；如果只在 stance system 中出现
差异，论文只能写系统级 morphology case，不能写普遍 transport 收益。

## 4. Research Questions

| RQ | 问题 | 主要贡献 | 主要任务 | 决定性比较 |
|---|---|---|---|---|
| **RQ1** | 有限目标交互能否识别足以改善迁移的任务相关执行响应？ | C1、C2 | `dual_route_carry`、`aperture_brake`，并以 `dual_channel_contact` 加入首个接触条件 | full vs no-calibration；ASID、AdaptSim、FADA；预算曲线 |
| **RQ2** | 持续接触、阶段切换和环境运动出现时，何时应保留 source response，何时应偏离？ | C2、C3 | `corner_guided_seating`、`jam_recovery`、`compliant_fixture_insertion` | full vs no-response vs ungated；最强 task baseline；事件与力峰值 |
| **RQ3** | 相同语义能否跨臂自由度和支撑形态实现，并保持合理在线成本？ | C1、C3 | Panda→xArm7／UR5e；`stance_insertion`、`reposition_then_dock` | direct source、ASID、AdaptSim、DIAL-MPC、response MPC；端到端 latency |

RQ 级结论来自其指定任务和 matched mechanism test 的共同模式，不要求 full 在
八个子任务的每一个 seed、mechanism 或 baseline 上都占优。正式运行前仍需根据
pilot 方差冻结最小有意义效应和统计规则，不能看到最终结果后再定义“多数”。

## 5. 四个环境家族与八个核心子任务

| 家族 | 核心子任务一 | 核心子任务二 | 可选扩展 |
|---|---|---|---|
| 受限空间持件转运 | `dual_route_carry`：双路线持件绕障和定姿保持 | `aperture_brake`：完整持件穿过实体开口、短距制动、低速保持 | `sequential_aperture_reorientation` |
| 导面接触定位 | `dual_channel_contact`：双物理通道、受载接触、模式出口 | `corner_guided_seating`：L 形导槽转向、持续载荷、末端止挡 | `edge_slide_pivot` |
| 连接器装配恢复 | `jam_recovery`：受载卡滞、卸载撤回、重对齐、插入保持 | `compliant_fixture_insertion`：被动单轴夹具在接触下运动 | `insert_then_lock` |
| 身体支撑耦合装配 | `stance_insertion`：固定足支撑下执行同一恢复语义 | `reposition_then_dock`：支撑域内目标重定位、settle、对接 | `disturbed_docking_recovery` |

当前八份合同已放入五个既有 config family 的 `protocol/subtasks/`，共享 loader 为
[`subtask_contracts.py`](../../genedynamics/tasks/transfer/subtask_contracts.py)。状态为：

- 已实现合同：`dual_route_carry`、`aperture_brake`、`dual_channel_contact`、
  `corner_guided_seating`、`jam_recovery`、`compliant_fixture_insertion`、
  `stance_insertion`。其中 `aperture_brake`、`corner_guided_seating` 与
  `compliant_fixture_insertion` 已各完成 4/4 fresh CPU source qualification、物理子步
  evaluator 与 anti-bypass／行程稳定检查；其余既有子任务仍需按新协议做 fresh
  source qualification 和 evaluator 复核。
- 合同冻结、实现待完成：`reposition_then_dock`。
- pending suite 在 `require_runnable=True` 时必须于 physics 启动前拒绝，不能用旧
  环境冒充新子任务。

机制轴仍是每个子任务内部的 train/development/heldout 变化，不能把两个摩擦系数
命名成两个任务。第二核心子任务必须具有新的事件关系、实体几何或身体协调要求。

## 6. General table 与 targeted studies

### 九行 general table

| 行 | 角色 |
|---|---|
| ICDT | 完整方法 |
| Task-only | 同容量内部系统参照；不是外部 SOTA |
| FADA† | planner 固定、执行适应的近期对照 |
| DIAL-MPC† | 在线扩散式 annealing / replanning 对照 |
| Response MPC | 普通模型优化对照 |
| Public geometry/feedback | 共同几何和反馈是否已经足够 |
| Public source prior | 直接源迁移 |
| AdaptSim† | task-driven simulation adaptation |
| ASID† | active identification 后的任务学习 |

† 表示本仓库 task adaptation。每种 adaptation 必须说明保留的作者机制、改动的
训练链、可访问数据和原生系统差异。当前 FADA 名义能力不足，因此其低分只能描述
当前适配，不能用来宣布原论文方法失败。

ACP 与 Human2Sim2Robot 按已确认决策转到表外。它们只有在任务适配策略完成
source/nominal qualification 后，才进入对应 RQ 的资源分离专项比较；缺失策略不记
为 0。Play2Perfect、KITE、TactAlign 等只在任务接口和公开实现允许忠实复现时增加，
不能因为论文新就删除其核心机制后占一行。

### General table 的四个指标

1. **Constrained success ↑**：完成任务且所有碰撞、接触、力、支撑和身体条件通过。
2. **Physical-constraint violation rate ↓**：任一物理约束失败的 episode 比例。
3. **Failure-penalized normalized completion time ↓**：成功用完成时间／时限，失败取 1。
4. **End-to-end p95 control latency ↓**：状态处理、网络或 planner、模型查询、
   selector、projection、decoder 和 WBC 的完整同步控制路径。

峰值力、模式完成率、恢复率、最差组成功率、模型误差和目标交互预算是 RQ 图或
附表指标，不再继续扩充 general table。家族内两个子任务等权，四个家族是否再聚合
须在正式协议中冻结。支撑形态展示不能掩盖前三类操作任务的退步。

## 7. 当前结果能够支持的故事

当前 GPU0 独立正式批次 v2 已完成 36 个方法×任务单元、232 个 actor 作业和
11,600 条物理结果。下表是目标身体上 source-like、nominal、delay/gain shift 和
payload shift 的合并受约束成功率；不是未来八子任务、等预算 general table。

| 方法 | Dual-route | Dual-channel | Jam recovery | Stance insertion | 前三类均值 |
|---|---:|---:|---:|---:|---:|
| **ICDT** | 100.00 | 98.75 | 42.19 | 86.88 | 80.31 |
| Task-only | 100.00 | 97.50 | 47.19 | 60.16 | 81.56 |
| FADA† | 0.00 | 0.31 | 0.00 | 0.00 | 0.10 |
| DIAL-MPC† | 100.00 | 0.00 | 36.25 | 83.75 | 45.42 |
| Response MPC | 100.00 | 99.69 | 48.44 | 70.78 | 82.71 |
| Public geometry/feedback | 100.00 | 97.50 | 50.00 | 87.50 | 82.50 |
| Public source prior | 0.00 | 0.00 | 62.50 | 87.50 | 20.83 |
| AdaptSim† | 98.44 | 99.06 | 62.81 | 86.72 | 86.77 |
| ASID† | 96.88 | 98.44 | 59.69 | 87.50 | 85.00 |

### 当前可以讲的现象

1. **简单任务饱和，不能证明 transport。** Dual-route 中多种方法达到 100%；
   dual-channel 中 ICDT 比 task-only 高 1.25 pp，95% 区间为
   $[-2.19,5.00]$ pp。这支持增加 aperture braking 和 corner contact，而不是把
   小差异写成稳定优势。
2. **静态恢复暴露当前方法缺陷。** Jam recovery 中 ICDT 为 42.19%，AdaptSim
   为 62.81%，差值为 -20.63 pp，区间 $[-33.13,-7.19]$ pp；ICDT 的 185/320
   失败均为 overforce。CPU 配对审计进一步发现 158 次是 ICDT 与 task-only 共同
   超力，27 次是 ICDT 独有退化，11 次是 ICDT 独有收益；160/185 涉及 lateral
   force limit，185/185 终止时 `retreating=false`。27 次独有退化全部属于 mode 0，
   之前都发生过 transport admission，但只有 8 次在终止控制当下仍 admission，
   说明主要问题包含共享 early-contact/reference-unloading recovery 和延迟力后果，
   不能只调最后一 tick 的 transport gate。这使“事件充分的 response 和
   contact-phase anchoring”成为方法问题，而不是需要隐藏的负结果。
3. **支撑形态上完整系统有效，但尚未完成 attribution。** Stance insertion 中
   ICDT 比 task-only 高 26.72 pp，区间 $[21.56,31.41]$ pp；它与 AdaptSim/ASID/
   public references 接近。现有 full 与 task-only 使用的 completion、selector 和
   projection 不完全匹配，因此只能写 system-level gain，不能单独归因 transport。
4. **三个复杂子任务已完成可运行性闭环，但还不是方法结果。**
   `aperture_brake` 已实现四边实体开口、完整 L 形物体包络、frame-contact 物理子步
   证据和短距制动保持，fresh CPU source qualification 为 4/4；
   `corner_guided_seating` 的 fresh CPU source qualification 为 4/4，四次峰值力
   10.57--10.61 N、侧向峰值 5.79--5.81 N，均按加载接触、第一段、拐角、第二段、
   实体止挡的顺序完成；`compliant_fixture_insertion` 也是 4/4，夹具最大位移
   0.227--0.273 mm、峰值速度 1.92--2.56 mm/s、峰值加速度 0.637--0.666 m/s$^2$，
   无行程端支撑并完成相对插入和共同稳定。它们证明环境、公共 source 与 evaluator
   可执行，尚不能证明 ICDT 优于 baseline；下一步必须在同一冻结协议上做 paired
   method pilot。
5. **关键接触的现有正式结果只支持局部机制故事。** 对 jam recovery 的 320 条／方法
   结果按 actor instance、request 和 reset hash 重新配对后，ICDT 的总成功率为
   42.19%，低于 AdaptSim 20.63 pp、ASID 17.50 pp、response MPC 6.25 pp 和
   task-only 5.00 pp，只高于 DIAL-MPC 5.94 pp。在 `latency_gain` mode 1 cell 中，
   ICDT 为 40%，分别高于 AdaptSim 10 pp、ASID 22.5 pp、response MPC 15 pp，并与
   task-only／DIAL-MPC 持平；这可以动机化“延迟接触响应”研究，但不能替代 overall
   ranking。新一轮 clean actor paired pilot 没有复现该性能差异，见第 7 点。
6. **初步预算证据是负向模型诊断。** 目标响应模型在 30/60/120 秒预算下的 held-out
   log MSE 分别为 0.007443、0.007861、0.008412，source/nominal held-out MSE 也不单调；
   当前没有“更多目标数据稳定降低误差”的证据。fresh physical pilot 中 full（60 秒）
   与 no-calibration（0 秒）均为 16/16 成功、零违规；full 的归一化完成时间为 0.7185，
   no-calibration 为 0.7572，但均值差 -0.0387 主要由 `payload_shift/mode_0` 的
   -0.1783 驱动，四个 cell 中有两个方向相反。这是局部时间信号，不是稳定预算收益。
7. **fresh 关键接触 pilot 证明当前两类 cell 区分力不足。** full、AdaptSim、ASID、
   response MPC、task-only 和三个 clean ablation 在相同 16 个 fresh reset 上全部
   16/16 成功、零违规；128 份物理证据均独立复审通过。full 的归一化完成时间 0.7185，
   ASID 为 0.6984、AdaptSim 为 0.7400、task-only 为 0.7213；full 并未稳定领先。
   no-response-anchoring 和 unconditional-response-anchoring 分别为 0.7169 和 0.7161，
   没有 response term 或 selective gate 的正增量。response MPC 的 episode-level actor
   p95 中位数为 0.282 s，明显高于 full 的 0.079 s，但这仍是 actor-only development
   指标，不能替代正式端到端 latency。

### 目前不能写的结论

- 不能写 ICDT 总体优于 ASID 或 AdaptSim；前三类均值仍落后。
- 不能写 response matching 已稳定改善 task performance。
- 不能用 FADA 当前适配低分推断原方法机制无效。
- 不能写 Go2+Z1 结果证明导航或 learned locomotion；当前是 fixed-foot stance。
- 不能写 Panda→UR5e 已完成；当前正式固定臂 target 是 xArm7。
- 不能写等预算 sample efficiency 或 latency 优势；现有主表预算不相等且尚无
  共同口径的九方法端到端 p95。

## 8. 三个主要消融

只保留以下三项作为正文主要消融：

| 名称 | 目标函数或模型变化 | 必须保持相同的部分 | 回答 |
|---|---|---|---|
| No calibration | 使用 nominal response model/context | 数据权限以外的网络、warm state、训练步、selector、执行接口 | 目标干预是否提供有效信息 |
| No response anchoring | $C+\beta h+\Omega$ | task band、模型、数据流、网络、optimizer、sample stream、execution | source response 是否有独立增量 |
| Unconditional anchoring | $C+\beta h+\lambda R+\Omega$ | 与 full 完全相同，只去掉 gate 的选择作用 | 选择性是否避免 source bias |

现有 task-only 只有 $C$，同时删除 response 和 task band，仍作为 system reference，
不能代替 no-response ablation。三项消融必须共享实际 command authority、mode
policy、checkpoint budget、online admission 和 recovery wrapper。若这些条件不同，
结果改为系统比较，不能写单因素归因。

## 9. 理论目前的缺口与修改方向

### 9.1 保留的理论核心

- command law 与 target execution kernel 共同决定 behavior law；
- reweighting 不能产生 source support 外的 command，transport 可以移动 support；
- task-relevant identifiability 只要求 probe null space 不影响 task quantity；
- 条件性事件界把 source failure、small margin、calibration/anchoring error 和
  unsupported mass 分开。

### 9.2 必须补强的部分

1. **Response descriptor 尚不 event-sufficient。** 当前短时增量和控制频率峰值
   不足以编码恢复顺序、持续接触、物理子步冲击、fixture settling 或 body support。
   下一版定义任务共享的 event response：事件发生时间／顺序、阶段内接触覆盖、
   子步力风险、相对 fixture motion 和 body feasibility。训练可用可微 proxy，正式
   success 仍由独立 evaluator 判定。
2. **短时界不能直接推出整段任务成功。** 需要按 phase 写有限时域组合：每阶段
   的 response error、margin 和 support risk 单独估计，再明确组合损失。先采用
   有限阶段 union bound 或 stopping-event 分解，不宣称无限时域统一保证。
3. **Ensemble spread 不是校准误差上界。** 当前动态夹具开发结果显示 spread 与
   实际误差相关性不足。它只能作为经验 gate 特征；若要理论使用，必须用独立
   holdout 做 coverage calibration，或把未校准部分放入 $p_{\rm bad}$。
4. **Self-collected logs 不保证新 contact branch 可辨识。** 理论和实验同时报告
   task-relevant excitation、事件覆盖和未访问分支；不能用总体 MSE 代替覆盖。
5. **条件性事件界不证明优于 task-only。** ICDT 与 matched task-only 具有相同
   表达能力。论文将优势定位为有限数据和优化下的 source-informed bias，由消融和
   budget curve 验证；除非新增明确假设和证明，不写 sample-complexity theorem。
6. **Training gate、checkpoint fallback 和 online admission 必须分开。** $G=0$
   只删除该样本的 response term，不等于运行时回退 task-only。理论符号和方法图要
   分别表示三种选择。
7. **执行接口 $\eta$ 是 attribution 边界。** completion、projection、selector、
   recovery 和 WBC 的变化会改变实际 trajectory law。所有 component claim 都要
   固定 $\eta$，并用 issued request 证明处理确实进入物理执行。

理论修改完成后，RQ2 的 positive/negative contact cells 应直接对应事件界中的
large-margin supported、small-margin 和 unsupported 三类条件，而不是事后按胜负
解释 gate。

## 10. 从当前结果到论文级大表还缺什么

### 任务闭环

- `aperture_brake` 已完成四边实体开口、完整 L 形 object-envelope evaluator、物理
  子步 frame-contact 证据、公共 braking/hold source skill 和 4/4 CPU 资格；下一步是
  learned method／baseline 小规模 paired pilot；
- `corner_guided_seating` 已完成实体折角导槽、公共 source、受载转向、实体止挡和
  4/4 CPU 资格；下一步是 learned method／baseline 小规模 paired pilot；
- `compliant_fixture_insertion` 已完成接触后公共激励、物理子步证据、行程端拒绝、
  相对插入与共同稳定判据，并以 4/4 CPU source qualification 闭环；下一步是
  learned method／baseline 小规模 paired pilot；
- 实现 `reposition_then_dock` 的目标重定位、稳定窗口和 docking 闭环；
- 四个既有任务用新语义合同重新做 source qualification，不重贴旧结果标签。

### 方法与 baseline 闭环

- 先放行共享 early-contact/reference-unloading recovery，再验证覆盖延迟力后果的
  force-aware admission；CPU 诊断已经排除“185 次均由 transport 单独造成”的解释；
- 三项 clean ablation 已冻结为 no-calibration、no-response-anchoring 和
  unconditional-response-anchoring；full 与三项消融共八个 mode-specific CPU
  preflight、GPU 训练和 selector 已通过；同 mode 的 batch-index stream 与 task-only
  起点完全一致。128-episode paired task pilot 已完成但全部饱和，不能扩成正式规模；
- 对 FADA、ASID、AdaptSim、DIAL-MPC 逐项完成 source/nominal qualification，记录
  原生机制和 task adaptation 边界；
- 九行方法铺满八个核心子任务。未执行保留 missing，不能填 0；资格失败和真实
  目标执行失败分开；
- ACP/H2S2R 只做有资格的 targeted study，不阻塞 general table。

### 统计与资源闭环

- pilot 后冻结 interaction-time grid、独立训练实例数、每实例条件数和最小效应；
- 目标数据、主动 probe、完整 task rollout、选模和失败全部计费；
- 预先冻结 train/development/heldout mechanism split；
- 保持 paired initial conditions，按训练实例和物理初态层次重采样；
- 统一测量九方法的 end-to-end p50/p95、deadline miss、GPU time 和 wall time；
- 按 family 和 subtask 报告，不让聚合均值隐藏某个 contact 或 support group 为零。

### 形态与可视化闭环

- 完成 Panda source、xArm7 target、UR5e target 的同任务可达性和 source skill
  qualification；
- 生成相同初态、相同语义和相同相机的 source→target episode GIF；
- `stance_insertion` 作为不同形态可行性和支撑约束展示，headline performance 仍由前三类
  操作任务和 matched morphology comparison 决定；
- 如果加入实机，至少提供一类接触任务的 source/target transfer、力和失败全分母；
  没有实机时标题和主张保持 cross-embodiment simulation，不写 sim-to-real 结论。

## 11. 命名与代码迁移规则

新论文、配置、runner 输出和分析只使用语义 ID：

| 历史别名 | 新语义 ID |
|---|---|
| T1 | `dual_route_carry` |
| T2 | `dual_channel_contact` |
| T3 | `jam_recovery` |
| T4 | `stance_insertion` |

迁移规则：

1. 旧结果、旧 SHA、旧报告和历史命令不可改名；它们是不可变证据。
2. 旧 `T1`--`T4` 只允许在一个 legacy reader 边界映射为语义 ID；新写出的 manifest、
   result、figure、CLI help 和 LaTeX 不再产生数字别名。
3. `configs` 继续按现有 robot/task family 组织，subtask 语义放在
   `protocol/subtasks/`；不创建 `results/<project>/<run_id>/`。
4. `scripts/paper/cross_ebdm` 保持论文入口，逐步把 `run_t2_*`、`train_t3_*` 等
   主动脚本迁移为由 config 和 semantic ID 驱动的通用入口。算法、task physics、
   evaluator 和统计逻辑分别留在既有 package 层，paper script 只做薄编排。
5. `genedynamics/solvers/single/<method>` 保存各算法内部实现；task-specific controller
   与 evaluator 留在 `genedynamics/tasks/transfer`；高层 `learning` 不出现算法名。
6. 每次迁移先用 `rg` 分类 active code、compatibility reader 和 immutable evidence。
   只有 active code 需要重命名；禁止为“清理”而修改历史结果内容。

## 12. 实施顺序与放行门槛

### A. 语义和证据冻结

- 八份合同、共享 loader、source snapshot 和 semantic entry 完成；
- active runner 不再把数字别名当 canonical identity；
- 唯一仍 pending 的 suite `reposition_then_dock` 在 physics 前被拒绝；aperture、
  corner 与 compliant 已通过 runnable gate 和 fresh source qualification；
- 当前 v2 结果只作为已有四子任务 evidence preview。

**出口：** 合同测试、legacy read 测试、result path 测试通过，历史 artifact hash
不变。

### B. 静态接触归因

- CPU 配对诊断已完成；下一步配对放行共享 early-contact/reference-unloading
  recovery，并在 heldout 请求验证 delayed-force-aware admission；
- 训练同架构 nominal response artifact；
- 在 matched execution 下完成 no-calibration、no-response-anchoring、
  unconditional-response-anchoring 三项消融；full 与三项消融的八个 mode-specific
  CPU preflight、单实例 GPU 训练和 128-episode paired pilot 已完成；三个消融均未在
  当前两类 cell 上表现出成功率或时间收益，下一版必须先建立有区分力的事件条件；
- 用 contact phase、首个冲击、issued request 和子步峰值回答原因。

**出口：** full 与每个消融只差声明因素；source/nominal qualification 通过；失败
全保留。若 response 没有独立增量，先收缩贡献再扩展规模。

### C. 新复杂度闭环

- aperture、corner、compliant fixture 已完成物理、evaluator、source skill 与 fresh
  qualification；下一项是 reposition/dock，并为前三项补 learned method／baseline
  paired pilot；
- 每个任务先物理／evaluator、再 source skill、再公共 feedback、最后 learned method；
- 先跑小规模 paired pilot 验证任务有区分力且没有 evaluator bypass。

**出口：** 四个新任务均由 pending 变为 runnable，并有 fresh source qualification、
anti-bypass 测试和至少一个非 oracle 公共参照成功。

### D. Baseline 资格和八任务 pilot

- 九行方法逐任务做 source/nominal 资格；
- 给合理原生训练预算，记录 learning curve 和失效原因；
- targeted ACP/H2S2R 独立决定，不拖延主表。

**出口：** 每个 general-table cell 有 qualified executable policy 或明确的 pre-run
missing reason；正式 freeze 前不能根据 target 排名删 baseline。

### E. 协议冻结和正式 GPU cohort

- 根据 pilot 方差冻结实例数、seeds、预算、机制组合、指标和统计脚本；
- 新鲜 train/development/heldout split；
- GPU0-only 要求如未被用户更新则继续执行；正式运行不在中途调参或替换失败。

**出口：** 8 subtasks × 9 rows 的四指标 general table，三项 clean ablation，三组
RQ 图，分层区间、预算和 latency 审计齐全。

### F. 论文证据闭环

- Introduction 的三项贡献逐条链接到主表、ablation 或 event figure；
- 结论按最终 evidence 缩放，保留失败边界；
- 统一 GIF、主图、表格和 appendix provenance；
- 运行 LaTeX、引用、版面、数字来源和 artifact hash 检查。

## 13. 预期论文图表

1. **方法总图：** source prior、target intervention、response calibration、selective
   anchoring、target execution kernel 和行为事件。
2. **任务复杂度图：** 四个 family 每个展示两个 core subtasks，箭头表示增加的事件
   关系，不表示简单参数变难。
3. **General table：** 九方法 × 四 family／八 subtask 的四个指标。
4. **RQ1 图：** constrained success 与 heldout response error 对总目标 interaction time。
5. **RQ2 图：** full/no-calibration/no-response-anchoring/
   unconditional-response-anchoring 在 compatible 与 incompatible contact cells 的
   事件成功、首次力峰值和 recovery timeline。
6. **RQ3 图：** Panda→xArm7/UR5e 和 Go2+Z1 的同语义 GIF／轨迹，附 p95 latency 与
   support failure boundary。
7. **失败图：** jam overforce、corner contact loss、fixture travel/settling 和 support
   infeasibility 的首个失败事件，不只展示成功样例。

## 14. 下一项具体工作

论文 draft 已使用三个贡献、三个 RQ、八个语义子任务、九行 baseline 和四个主表
指标。aperture、corner 和 compliant fixture 已完成 source 可运行性闭环；静态 jam
recovery 的离线配对强 baseline 对照、三项 clean ablation 合同和 0/30/60/120 秒模型
预算诊断也已完成。full 与三项消融的八个 mode-specific 单实例训练和 128-episode
fresh paired pilot 已闭环，但八方法全部成功，拒绝了“直接扩大这两个 cell 即可回答
RQ1/RQ2”的设计。下一项工作是在不观察方法排名的条件下，按物理事件预先定义更有
区分力的 contact cells：优先使用 `corner_guided_seating` 的转角持续载荷和
`compliant_fixture_insertion` 的夹具运动／settling，再为 `aperture_brake` 加入 heldout
短制动距离。每个新 cell 先要求公共 source 可行且 task-only 不饱和，再以同一套
full／强 baseline／三消融做小规模 paired pilot。只有出现跨 seed 和至少两个 cell
一致的机制增量，才冻结多实例正式设计；与此同时实现 `reposition_then_dock`，不能把
本次单实例饱和结果写成正式大表排名。
