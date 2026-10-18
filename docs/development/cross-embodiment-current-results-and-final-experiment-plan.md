# Cross-EBDM 当前实验结论与论文大实验缺口

> **2026-10-17 P5 完成更新：** 本文其余内容记录 P5 冻结前的中间状态，其中“31/36”“至少 5 个实例”“五格消融”和“P6 尚未放行”等表述已失效。当前权威状态、八实例六格消融、21,200 条完整演练、精确确认包与理论修订见 [P5 预访问冻结与 P6 执行入口](cross-embodiment-p5-preaccess-closeout.md)。P6 已通过执行 admission，但 confirmation cohort 仍未访问。

2026-10-17。本文档是开发记录，不进入 LaTeX 论文正文。它以 P4 联合验收、P5 归因勘误、九行四任务主表合同、P6-readiness 五格消融、P6 协议有效性验收和 task-compatible gated transport v2 方法身份 closeout v3 为准；旧版 T4 训练语义混杂的 8/8/16/16 不再用于归因。

## 当前结论

当前证据不支持“ours 在四任务上整体最强”或“response matching/transport 稳定提高任务成功”。可以保留的开发结论是：

1. 共享的 safe-settled guard 修复了 T4 中多个方法共同遇到的有限状态活性瓶颈；它是公共控制工程，不是 transport 增量。
2. 在统一训练/执行语义的 T4 单实例 B60 消融中，matching 与 task band 只在组合时解决 mode 1，Full 为 16/16，交互为 +7/16。
3. 同样的 B60 五格消融在 T3 上给出相反结果：band-only 为 15/16，Full 为 10/16，交互为 -9/16。T1/T2 五格全部 16/16，无法从成功率识别任一因素。
4. v2 在冻结 B60 warm/RNG/sample stream 上把任务项保留在 band 内，并只在 task-compatible、source-supported、低 ensemble disagreement 的样本上施加 anchoring；最终 checkpoint 相对 matched task-only 选择。T3/T4 的模型代理差分别为 `+0.000563` 和 `+0.000025`，均在 `+0.05` 门内。
5. v2 的配对目标开发中，T1 为 `16/16` 对 task-only `16/16`，T3 为 `7/8` 对 `6/8`，T4 为 `4/8` 对 `4/8`。只有 T3 出现一个有利 discordant pair；T1 饱和，T4 无 discordance。T2 在 strict-contact 下 Panda source 为 `4/4`，但 v2 与 task-only 在 known-nominal xArm 均为 `0/4`，八次都在出口前未建立导面接触，因而未打开目标。
6. 因而 matching 与 task band 的作用是任务、模式、支持区域和预算相关的条件行为；不存在已验证的跨任务稳定主效应。
7. 当前所有结果都是开发集。v2 只有一个训练/校准实例，T4 目标开发仍只有一个 mechanism/body/support cell，不能用于最终统计推断。
8. P6 协议有效性门已经关闭：T2 strict-contact 判据在 4/4 源轨迹上可行，并把旧合同下成功的 10/10 抬起绕行轨迹全部判为失败；T4 两个机制组乘两个 support cells 均有 10/10 hash-unique 初态，16/16 任务轨迹通过独立 body/WBC/support 审计。
9. DIAL-MPC 的真实 sampling/annealing/receding-horizon 路线已移植：T1/T2 在 known-nominal 资格门被拒，T3 为 `9/16`，T4 为 `15/16`。T3 的 mode 0/1 为 `1/8` 与 `8/8`，与旧 Full 的 `8/8` 与 `2/8` 呈相反模式偏置，说明在线规划也没有形成跨模式稳定优势。
10. H2S2R source-trajectory 的 T4 固定训练机会已完成：两个 mode 各 4096 epochs，最终 known-model monitor 合计 `0/8`，目标为 `0/16`，其中 8 次 overforce、8 次 joint-limit。该失败已在目标前出现，首先指向当前 reward/训练/任务适配，而不是 target mechanism 本身。

机器 closeout 位于：

```text
results/_development/p6_readiness_20261016/results/cross_embodiment/
  four_task_benchmark/protocol/p6_readiness_ablation_step1_v1/
  level_B_log_60/seed_0/report.json
```

报告 SHA256 为 `8d58ac71142034ea50f33a5275ab7c4e4403046146ce7fc7486318e529633952`，四任务 target audit、80 个 episode/任务、收费回放、方法顺序、成功数和交互全部通过哈希复核。Step 1 已完成，P6 尚未放行。

协议有效性验收位于：

```text
results/_development/p6_protocol_freeze_20261016/results/cross_embodiment/
  four_task_benchmark/protocol/p6_protocol_validity_gate_v3/
  level_read_only/seed_0/report.json
```

报告 SHA256 为 `3685c113a40ea45e7359b13097cb1bd1dee5928858a6994eb93c962e00d96c2c`，43 项检查全部通过，`p6_admitted=false`；该报告冻结时剩余主表实现缺口为 9。v1 的失败审计被保留：它因 5 个姿态误差在 JSON 往返后出现约 `1e-17` 的浮点差而误报，v2 只把序列化数值比较改为 `1e-12` 容差；v3 只登记两个已完成的 source-prior 资格失败单元。后续 DIAL-MPC 四格和 H2S2R-T4 已完成，当前缺口降为 5；这些更新没有改变实验、任务判据或 P6 非访问状态。

## 方法身份 closeout：冻结 v2

旧 Full v1 的目标函数在 band 内会把 task loss 替换成纯违例惩罚，且最终选择相对 warm checkpoint，而不是 matched task-only。新版本 `task-compatible-gated-transport-v2` 在打开任何 fresh target seed 前完成了三项结构修正：task loss 始终保留；response anchoring 只在 task-only-relative compatibility、公开 source-condition support、ensemble disagreement 阈值和任务执行 mask 同时通过时启用；最终候选相对 matched task-only 的 held-out 模型任务代理选择。T3/T4 的支持和 disagreement 阈值分别由 held-out source/known-nominal 状态的 higher empirical q95 冻结，没有读取目标成功、reward 或 evaluator 标签。

训练、资格和目标开发均通过独立审计：4000 个主更新和 4 个恢复证明完全匹配历史 sample-stream SHA；T3/T4 的 source 与 known-nominal 中，v2/task-only 均为 `4/4 + 4/4`；fresh target 的成对结果为 T3 `7/8 vs 6/8`、T4 `4/8 vs 4/8`。方法身份 closeout 位于：

```text
results/_development/p5p6_method_completion_20261017/_reports/
  p5p6_method_identity_closeout_v3.json
```

该 closeout 冻结 v2，禁止再按开发或 P6 结果选择 v1/v2。任何 objective、gate、threshold rule、action semantics 或 checkpoint-selection 修改都必须创建新方法版本并使当前确认 ancestry 失效。该 closeout 当时记录 10 个开发缺口；后续完成 DIAL-MPC 四格和 H2S2R-T4 后，当前还缺 4 个独立实例、5 个开发主表单元和全部 36 个最终确认单元。T1 v2 已完成，T2 因 strict known-nominal 资格拒绝仍未形成目标开发格；`p6_admitted=false` 保持不变。

## 四任务 B60 五格可比消融

五格依次为不继续训练的 `frozen-warm`、同容量 `task-only`、`matching-no-band`、`band-only` 和 `Full`。每个任务内部共享 warm、网络容量、训练器语义、有效动作、数据、采样流、优化器、更新数、decoder、evaluator 和目标初态；四任务之间允许任务规定的执行 mask 不同。

|任务|frozen-warm|task-only|matching-no-band|band-only|Full|交互|主要现象|
|---|---:|---:|---:|---:|---:|---:|---|
|T1 ObstacleTransfer|16/16, 8.363 s|16/16, 8.314 s|16/16, 8.316 s|16/16, 8.359 s|16/16, 8.345 s|0|成功率完全饱和，Full 也不最快|
|T2 GuidedContactAlignment|16/16, 15.341 s|16/16, 15.110 s|16/16, 15.199 s|16/16, 15.409 s|16/16, 15.525 s|0|旧合同成功率完全饱和，Full 在五格中最慢；10 条抽查成功轨迹在 strict-contact 下均失败|
|T3 KeyedInsertionRecovery|9/16, 13.520 s|8/16, 13.895 s|12/16, 12.969 s|15/16, 12.326 s|10/16, 13.280 s|-9|band-only 最好；matching 在 band 内减少 5 个成功|
|T4 MobileKeyedInsertionRecovery|8/16, 12.905 s|8/16, 13.899 s|9/16, 14.139 s|8/16, 13.429 s|16/16, 12.583 s|+7|正效应全部来自 mode 1；仅单训练实例、单物理初态和单 support cell|

T3 的 mode 0/mode 1 成功数分别为：frozen `4/5`、task-only `4/4`、matching-no-band `7/5`、band-only `8/7`、Full `8/2`。T4 为 frozen `8/0`、task-only `8/0`、matching-no-band `8/1`、band-only `8/0`、Full `8/8`。相反符号集中于困难模式，说明必须围绕 hybrid recovery 分支解释，不能用整体响应 RMSE 代替。

## P4/P5 任务与强参照现象

P4 的四预算开发结果仍用于观察预算曲线和公共参照。下表的 `Full` 是旧 v1，每格 16 个开发请求；T4 的 Full/公共参照采用 safe-settled guard 开发比较，FADA 仍来自旧 P4 公共恢复栈，最终主表必须统一重跑。冻结的 v2 只完成了上节所列 T3/T4 fresh-seed 小规模比较，不能与本表混成一个方法版本。

|任务|Full|task-only|ordinary MPC|public geometry|source prior|DIAL-MPC|FADA adapted|H2S2R source-trajectory|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|T1|16|16|16|16|资格拒绝|资格拒绝|0|0|
|T2|16|16|16|16|strict 资格拒绝|strict 资格拒绝|0|0|
|T3|10|8|15|16|16|9|0|0|
|T4|16|8|16|16|16|15|旧公共栈 0|0|

各任务现象：

- **T1**：Full、task-only、MPC、geometry 和全部内部消融均满成功。P4 四预算也全部满成功，Full 平均时间一直不优于 ordinary MPC。该任务现在只能支持“方法可执行”；最终需要未见机制、约束裕度和 clearance 才能产生信息。
- **T2**：P4 四预算的 Full/task-only/MPC/geometry 在旧合同下全部满成功。Full 在正预算下快于 MPC/geometry，但慢于 task-only；最新五格中 Full 最慢。strict-contact v2 已冻结：接触必须先于出口，遍历段接触率至少 0.5，最长连续失联不超过 0.15 s。源端 4/4 可行；五个消融各抽取两个旧成功轨迹，共 10/10 因“出导面前未建立接触”被新合同拒绝。所有 T2 最终分数必须按新合同重跑。
- **T3**：Full 随 B0/B30/B60/B120 为 16/15/10/16，成功随日志预算明显非单调。B60 ordinary MPC 为 15，geometry/prior 为 16，Full 只有 10。响应 matching 在无 band 时为 +4，在有 band 时为 -5；局部响应误差更小不能推出任务改善。
- **T3** 的 DIAL-MPC 为 `9/16`，mode 0/1 为 `1/8` 与 `8/8`，普通 response MPC 为 `15/16`。DIAL 使用约 4122 个 candidate rollouts/control，actor-only p95 为 11.865 ms；最终仍需在隔离资源下测端到端时延。
- **T4**：旧 guard 下 Full/task/MPC/geometry/prior 的 B60 成功为 9/0/0/16/12；共享 safe-settled guard 后为 16/8/16/16/16。该变化证明旧 latch 是公共瓶颈。统一 recovery trainer 后 Full 才在五格中出现 +7 交互，但 geometry/prior 同为 16 且更快；Full 不是整体最强。DIAL-MPC 为 `15/16`，mode 0/1 为 `7/8` 与 `8/8`，约 4126 个 candidate rollouts/control，actor-only p95 为 13.215 ms。新协议的四个机制/support cells 各得到 10 个不同 post-preparation 物理状态，最小两两积分状态距离为 0.00878--0.02129。公共参照在 nominal support 为 8/8，在 0.90 support 为 0/8；后八次均在 114--170 个策略控制步后触发 arm joint-limit，同时 body height、tilt、WBC residual 和四足支撑 16/16 通过，因此困难 cell 和失败全部保留。

## Baseline 当前完成度

必须同时报告两个完成度。**开发实现矩阵**有 9 行、4 个任务，共 36 个单元；按最终冻结的 v2 方法行计，当前 **31 个完成、5 个缺失**。**最终确认矩阵**仍是 **0/36**：开发结果不能复用为 P6 论文数字，旧 T2 和单-cell T4 尤其不能迁移。新增的 `task-only` 四格来自可比五格开发实验；它是同容量、同训练器的必要主表参照，不再只放在消融表。`public source prior` 的 T1/T2 bounded opportunity 也已关闭：T1 在 Panda 为 4/4、known-nominal xArm 为 0/4 timeout；T2 strict-contact 在 Panda 为 3/4、known-nominal xArm 为 0/4 timeout。两格均在 gate 前停止，不打开未知目标，也不把未运行目标伪记为 0。剩余缺口为：

- Ours v2：T1 已完成并为 `16/16 vs 16/16`；T2 的 calibration/训练完成，但 strict-contact known-nominal 中 v2/task-only 均 `0/4`，目标按门控未执行，因此 T2 仍缺。
- ACP：T1/T2/T4 未跑，T3 只是未配对的部分证据；四格都视为未完成。

当前九行解析协议为 `configs/cross_embodiment/four_task_benchmark/protocol/main_table_v5.json`（SHA256 `3911df26d379c046301ed98ac4298668e6132a60c34d01053bd3de5baad058a9`）；只读合同审计 SHA256 为 `4941dcbb0252991b0f4ea2ba98e7eab3c12011ac99b05a94403f0651ec208c20`，验证 9 行、4 任务、恰好 5 个缺口且 `admit_final_run=false`。

FADA、H2S2R、DIAL-MPC、ordinary MPC、public geometry 和 public source prior 的四任务开发格均已关闭；ours 仅缺 T2，ACP 缺四格。FADA 四任务目标均 0/16，但其 source actor 已资格通过；这只说明当前 task-adapted Planner/IDM 路线失败，不能推广成原 humanoid 方法无效。H2S2R 四任务均 0/16，known-nominal monitor 也失败，当前结果更接近 reward/训练适配不足。ACP 的学习 policy 尚未完成 matched C-supervised 路线；teacher 不能代替学习 policy。DIAL-MPC 已保留 node-spline sampling、diffusion/annealing update 与 receding horizon；普通 Adam MPC 仍作为不同路线单列。

|主表行|权限|当前开发证据|进入最终主表前还缺什么|
|---|---|---|---|
|Ours gated transport v2|B-log|T1 16/16、T3 7/8、T4 4/8；相对 task-only 分别 0、+1、0；T2 strict nominal 资格拒绝|解决或冻结 T2 资格拒绝计分；至少 5 个独立实例；四任务新合同重跑|
|Task-only|B-log|16/16、16/16、8/16、8/16；与 Full 同容量|至少 5 个独立实例；作为 Full 的首要同权限主对比|
|FADA task-adapted|B-log|四任务开发目标均 0/16；T1--T3 原生 task-adapted source 均 4/4|按最终 T2/T4 合同和新 cohort 重跑；保留“task-adapted、非原论文定量复现”标签|
|ACP task-adapted|C-supervised|T3 源 14/16、旧目标 0/16，但未完成 matched C-supervised；其余任务没有学习 policy|完成四任务示范定义、学习 policy、配对资格和开发评估；teacher 不作为 ACP 成绩|
|H2S2R source-trajectory|C-supervised-model|T1--T4 各 0/16；T4 两个 mode 各完成 4096 epochs，known-model monitor 合计 0/8，目标失败为 8 overforce + 8 joint-limit|按最终 T2/T4 合同与新 cohort 重跑；保留固定训练机会和“task-adapted、非原 human-assets/FABRICS 定量复现”标签|
|DIAL-MPC task-adapted|B-log-model|T1/T2 资格拒绝；T3 9/16；T4 15/16；T3 mode0 只有 1/8；已完成 sampling/annealing、模型调用与独立回放审计|按最终多机制/多 cell cohort 重跑；隔离资源测端到端 p50/p95/deadline miss；保留资格拒绝|
|Ordinary response MPC|B-log-model|T1/T2/T3/T4 为 16/16、16/16、15/16、16/16|strict T2、两个 T4 support cells、独立校准实例和端到端时延|
|Public geometry|no-target-learning|四任务开发均 16/16；T2 旧合同、T4 单 cell|新合同全量重跑；它是当前最强公共参照之一|
|Public source prior|no-target-learning|T3/T4 为 16/16；T1/T2 在 known-nominal 0/4 后正确停门|冻结最终资格拒绝的计分规则；不能把未执行目标描述成物理目标 0 分|

最终主表建议保留当前四条近期路线：FADA、ACP、Human2Sim2Robot-source-trajectory、DIAL-MPC，以及 ordinary MPC、public geometry、public source prior；将 **task-only 同时放进主表**，因为它是检验 source anchoring 增量的同容量必要参照。frozen-warm、matching-no-band 和 band-only 留在消融表。不同权限的方法继续显示在同一资源--性能表中，但只在相同 permission class 内做“优于”推断。

## 主表指标

每个任务单元使用四个通用性能量，所有失败进入分母：

1. `constrained success rate`：完成、请求模式匹配、deadline 和全部 force/collision/joint/body/WBC/support 判据同时满足。
2. `constraint violation rate`：任一子步违反冻结约束。
3. `capped normalized completion time`：`min(time, deadline)/deadline`，失败记 1。
4. `worst-group constrained success`：两个机制组及 T4 body/support cells 中的最差组成功率。

此外把 `qualification / pre-deployment rejection rate` 作为强制状态列。它不替代四个性能主指标；它用于区分真实目标执行失败、资格拒绝和实现未完成，防止未执行方法因 violation 为零而显得更安全。

资源列至少包含 permission class、目标交互秒数/transitions、demonstrations、外部预训练、训练更新和 GPU 小时、每控制步模型调用、端到端 p50/p95 时延与 deadline miss rate。现有 actor-only timing 不能代替最终端到端时延。

任务专属诊断放在补表/图：T1 clearance 与终端 SE(3)；T2 continuous guide contact、峰值总力/侧向力、stall；T3/T4 jam recovery、retry、keyway margin；T4 body tilt/height、support、WBC residual 和 arm/base motion。源--目标响应距离只作机制诊断，不作性能主指标。

## 最终采样与统计

当前冻结下限为：每个需要学习或校准的方法至少 5 个独立训练/校准实例；至少 2 个 target mechanism groups；每任务每实例至少 40 个配对 target 请求；每任务至少 10 个真实不同物理初态；T4 至少 2 个 body/support cells。一个 checkpoint 下的 policy seeds 不是训练重复，一个 deterministic reset 下的多个策略种子也不是初态重复。

T1--T3 冻结为 `10 初态 × 2 请求模式 × 2 机制组 = 40` 个请求/实例。T4 冻结为 `10 初态 × 2 请求模式 × 2 机制组 × 2 body/support cells = 80` 个请求/实例。全部方法共享同一 request package，不替换失败。

主表是 9 行；其中 Full 和 task-only 已同时属于五格消融，所以只需额外运行 frozen-warm、matching-no-band、band-only 三行，总共仍是 12 个唯一方法变体。若保守地把 12 行都按 5 个实例交叉，T1--T3 需要 `12 × 3 × 5 × 40 = 7200` 个 originals，T4 需要 `12 × 5 × 80 = 4800` 个，共 12000。若 public geometry/source prior 两个确定性方法只执行一次唯一 request package，而不伪造训练重复，则物理下限为 `(10 × 5 + 2) × (3 × 40 + 80) = 10400` 个 originals；统计分析不得把对确定性结果的别名复制当独立样本。收费精确回放会再增加同量执行，尚不含 source/nominal qualification、训练、预算曲线和针对性 RQ。先实测一个完整实例的训练/评估吞吐，再冻结总 GPU/CPU 小时和存储。

推断以训练/校准实例为最高重复层，在 mechanism/initial-state group 内做配对。使用 hierarchical paired bootstrap；每个 RQ 预声明一个主对比，RQ 内多比较用 Holm。报告效应量和区间，不把控制步当独立样本。最终访问前要用开发数据做 power simulation，明确最小有意义效应；5 个实例只是下限，若 10 个百分点成功差仍无法得到可用区间，应增加到 8--10 个实例。

## 针对性 Research Questions

主表回答总体任务表现；下列子矩阵只选能直接回答问题的方法，不再跑全笛卡尔积：

|RQ|问题|方法|任务/指标|
|---|---|---|---|
|RQ1|同权限下 ours 是否提高受约束任务表现？|全部主表行；主推断按 permission class 分层|T1--T4，四个通用指标|
|RQ2|matching、task band 与 continued training 如何作用？|五格消融|T1--T4；成功、时间、mode、response/event margin|
|RQ3|目标数据效率如何？|ours、task-only、FADA、H2S2R、ordinary MPC|预冻结少数预算点；成功--交互/计算前沿|
|RQ4|接触保持与卡滞恢复是否改善？|ours、task-only、ACP、ordinary MPC、geometry|严格 T2、T3、T4；contact ratio、force、jam recovery|
|RQ5|摊销执行相对在线规划是否有价值？|ours、task-only、DIAL-MPC、ordinary MPC、geometry|相当任务表现下的端到端 p95、miss、model calls|
|RQ6|身体/支撑耦合下是否保持收益？|ours、task-only、FADA、DIAL-MPC、geometry|T3→T4、至少两 body/support cells；最差组和 body/WBC margin|
|RQ7|“distribution transport” 是否真的改善条件对应，而非只改变动作？|五格消融、task-only、source prior；重点 T3/T4|请求模式混淆、条件 mode coverage、hybrid-event 对应、response 距离与任务结果的联合图|

ACP 使用 C-supervised，H2S2R 使用 source trajectory/model supervision，DIAL-MPC 使用模型 rollout；它们可以同表展示，但若要做因果式优劣比较，需要给 ours 一个相同信息权限版本，或只报告资源--性能前沿。

## 理论与论文需要修改的地方

1. **中心主张要从稳定 transport 优势收缩为选择性 anchoring 的条件命题。** 旧 v1 的 T4 matching×band 交互为正、T3 为负；v2 的 fresh-seed 结果变成 T3 一个有利 discordant pair、T4 无 discordance。理论必须允许 gate 关闭时退回 task-first 行为，并明确不预测跨任务正号。
2. **论文方法必须等于实际实现。** 当前 draft 仍包含尚未形成主证据的 stochastic flow、adaptive probe design、ensemble CVaR、UR5e/移动平台和真实部署。实现并消融它们，或从主方法删除；不能把 blueprint 当完成算法。
3. **task-band 命题已与 v2 训练规则对齐一半，仍缺模型到真实任务的桥。** v2 已把 task loss 保留在 band 内，按样本使用 detached support/uncertainty gate，并相对 matched task-only 做 checkpoint fallback；这消除了 warm-relative selection 和 band 内 task loss 消失的问题。仍需界定随机优化误差、soft penalty 的瞬时超限、有限验证集误差，以及 model task proxy 与真实 constrained success 的偏差；`+0.05` 只能称模型侧选择容差。
4. **identifiability 到任务的常数未验证。** `kappa, L_Theta, epsilon_K`、访问支持外概率和 uniform kernel error 都没有实验估计。若不能验证，应把桥接命题降为适用条件，保留一个与实验直接相连的有限策略/事件 margin 结果。
5. **hybrid event metric 需要包含控制器记忆。** T4 证明的是公开 predicate 下 retreat latch 最迟在 K 次连续满足后释放，不是安全或成功定理。事件距离必须包含 phase、guard counter、integral 和 recovery mask；TCP RMSE/平均力不能替代。
6. **相同控制类没有表达优势。** v2 与 task-only 容量相同，source anchoring 改变选择准则，不扩大可执行集合。论文贡献只能来自样本效率、条件对应或在相同任务近优集合内更好的选择。T3 的 `1` 个有利 discordance 是线索，T4 的 `0` 个 discordance 说明这项增量尚未跨任务成立。
7. **“distribution transport” 缺少分布证据。** 需要报告请求模式条件对应、mode coverage、source-conditional response discrepancy 和必要支持移动；若最终只有 deterministic residual 与任务成功，应缩小名称和贡献表述。
8. **统计理论需覆盖多校准层级。** 当前 Hoeffding 命题条件于一次校准；论文结论需要把训练/校准实例作为最高随机层。可保留有限策略界作辅助，主要推断使用预声明的层级配对设计。
9. **T2/T4 的任务有效性合同已补强，但 v2 最终方法结果仍不完整。** T2 strict-contact 和 T4 两个 body/support cells 已冻结并通过物理有效性验收；v2 在 T2 已暴露 source→known-nominal 接触保持失败，目标因资格拒绝未执行，T4 仍只有一个 development mechanism/body/support cell。理论和结果必须使用新合同产生的多实例重复实验。
10. **sim-to-real 标题需要真实系统证据。** 若最终没有真实机器人目标实验，应从标题、摘要和结论删除 sim-to-real，定位为 cross-embodiment sim-to-sim benchmark/method。保留 sim-to-real 时，至少需要一个真实固定臂接触任务及相同冻结协议；T4 的真实移动平台证据更强但成本更高。
11. **资格拒绝与主表全覆盖的定义冲突。** 当前合同要求所有九行覆盖四任务，又禁止把未运行目标记为 0。最终应把预部署拒绝定义为算法级结果：`constrained success=0`、capped time 为 1、worst-group success 为 0，并另报 rejection rate；同时将“无目标物理执行”明确标注，不能把算法拒绝写成目标机制执行失败。开发阶段的 bounded reproduction failure 继续保持非目标分数。
12. **方法身份已冻结，但证据强度尚不足以进入昂贵确认。** v2 已作为最终 ours 版本冻结；后续不再按性能选择 v1/v2。进入 P6 前仍需处理 T2 strict 资格拒绝、至少 4 个额外独立实例、实例级 power simulation 和两个 T4 support cells。若这些开发重复显示 v2 只与 task-only 持平，论文必须把贡献写成可审计的 selective anchoring/fallback，而不能宣称普遍性能提升。
13. **在线规划结果要求把计算与模式鲁棒性写进理论。** DIAL-MPC 在 T4 接近满成功，在 T3 却集中失败于 mode 0；旧 Full 的 T3 困难模式恰好相反。仅用“规划方法更强”或“摊销策略更稳”都解释不了结果。理论应把 mode-conditioned feasible basin、有限 rollout 预算和闭环事件转换列为条件，并把 model calls/control 与 deadline miss 纳入性能定义。
14. **baseline 失败需要分解成方法失败与适配失败。** H2S2R-T4 在 known model 已经 0/8，FADA/ACP 也存在任务适配层限制。论文应报告 source、known-nominal、target 三段漏斗；只有通过前两段后在目标退化，才能作为 cross-mechanism/embodiment 泛化证据。

理论最有希望的主线是：task-first 目标定义模型任务近优区域；公开 support 与 ensemble disagreement gate 决定哪些样本允许 source-conditioned response anchoring；在 gate 关闭时目标退化为 task-only，在 gate 打开且模型误差小于 hybrid-event margin 时，anchoring 才可能改善事件对应。checkpoint fallback 只给 held-out 模型代理上的近似非退化，不推出真实任务成功或优于 task-only。T3 的一个正向翻转和 T4 的完全持平用于检验 gate 的充分性与覆盖率，不能被平均 response RMSE 替代。

## 后续执行顺序

1. **已完成：** T1--T4 的 B60 五格可比开发消融与四任务机器 closeout。
2. **已完成：** T2 strict-contact、两个机制组、每 cell 十个物理初态、T4 两个 body/support cells、四个通用指标、failure taxonomy、最小效应、统计和 stop rule 的协议有效性冻结。
3. **已完成：冻结方法身份 v2。** source/known-nominal q95 gate calibration、4000 次可比训练更新、32 个资格 originals 和 32 个 fresh target originals 均已通过独立审计；v2 不再与 v1 做性能选择。T3 为 `7/8 vs 6/8`，T4 为 `4/8 vs 4/8`，只作为方法 sanity gate。
4. **完成剩余 5 个开发实现缺口：** 先冻结“资格拒绝如何进入全覆盖主表”的算法级计分规则，并据此关闭 ours v2 的 T2；然后完成 ACP T1--T4 的 matched C-supervised demonstrations、学习 policy、source/known-nominal 资格和配对开发评估。teacher 不能代替 ACP policy，合理训练机会后的失败必须保留。所有 T2 方法按 strict-contact v2。
5. **生成 4 个额外独立实例并做 power freeze。** 最终 ours v2 及每个学习/校准 baseline 至少 5 个独立训练/校准实例；先用当前完整实例实测的训练、资格和评估吞吐建立资源表，再以配对 discordance和实例间方差决定 5 或 8--10。确定性 baseline 在相同 request package 上评估，不伪造训练重复。
6. **完成 P6 任务矩阵的开发预演。** T1--T3 每实例覆盖两个机制组和 40 个请求；T4 每实例覆盖两个机制组、两个 body/support cells 和 80 个请求。这里仍使用开发 cohort，只验证执行、存储、统计和失败记账，不复用为最终结果。
7. **封存 confirmation package。** 冻结请求、初态、机制、身体、checkpoint、预算、依赖和代码 SHA，并复核第 4 步冻结的资格拒绝计分与 rejection rate；在此之前最终 cohort 不可访问。
8. **一次性运行 P6 主表与五格消融。** 全部 9 行参加四任务主表；ours v2/task-only 复用主表执行，额外三格完成消融；失败不替换，方法变化必须创建新版本和新确认集。
9. **运行 RQ 子矩阵：** 数据效率、接触恢复、在线规划时延、身体耦合和条件分布对应；只使用预声明的方法、任务和预算。
10. **重写论文：** 方法段只保留实际实现；理论对齐条件性主张；结果主文只放主表、关键消融、OOD/body、latency 和必要的对应证据，开发流水继续留在 `docs/development` 与复现包。

在第 4--7 项完成前，不应访问最终确认集。现有开发证据足以冻结方法版本和缩小主张，但不足以决定论文结论。
