# ICDT 论文叙事、当前证据与下一轮实验规划

更新：2026-11-05（America/New_York）。11 月 4 日版本确立论文叙事；11 月 5 日新增第六节，将其细化为子任务搭建、比较设计和投稿排期。本节是当前研究叙事、证据解释与后续工作顺序的依据；下方历史记录保留当时的事实、决策和 SHA，其中的“当前”“尚未启动”和 GPU1 队列状态不再代表现状。历史结果是否有效，以各自冻结协议与有效性记录为准，不因研究叙事更新而重判。

当前论文入口为 [`latex/latex_icdt`](../../latex/latex_icdt/)，组织方式参考 [`latex/latex_mga` 的 Introduction](../../latex/latex_mga/sections/intro.tex) 和 [Experimental Analysis](../../latex/latex_mga/sections/exp.tex)。[`latex/cross_ebdm`](../../latex/cross_ebdm/) 是既有设计背景；运行日志、试验决策、阶段命名和任务清单放在本开发文档，论文保持学术 draft。本轮只修改论文和文档，不启动新训练、仿真或 GPU 作业；后续算力要求继续为 GPU0-only。

本版 [论文 PDF](../../latex/latex_icdt/main.pdf) 已同步摘要、Introduction、相关工作、方法与条件性理论、实验设计、真实结果表和结论。正文在第 8 页结束，含参考文献共 9 页；[排版检查](../../latex/latex_icdt/notes/layout_check.json)通过，未发现溢出、未解析引用或书目错误。新增子任务与未完成消融明确标为计划；当前稿件不把预期贡献写成已完成实验结论。编译使用临时目录，仓库中仅保留 PDF 和检查报告等交付文件。

## 当前完成状态与证据边界

GPU0 独立正式批次 v2 已完成 **36 个方法×任务单元、232 个 actor 作业、11,600 条物理结果**，执行状态为 `FORMAL_TABLE_ASSEMBLED`，主表标记 `formal_performance_evidence=true`、`all_cells_cpu_reaudited=true`。逐 actor 的独立审计和 CPU 组表均已完成，描述性不确定性采用 10,000 次重采样。该 v2 是保留的有效冻结实验，不重新标为 invalid，也不与旧 GPU1 v1 混合。

- [正式主表](../../results/_analysis/four_task_transfer/independent_main_table_formal_v2/main_table.json)、[不确定性](../../results/_analysis/four_task_transfer/independent_main_table_formal_v2/uncertainty.json)、[统计协议](../../results/_analysis/four_task_transfer/independent_main_table_formal_v2/protocol.json)及[执行状态](../../results/_analysis/four_task_transfer/independent_main_table_execution_v2/state.json)记录这轮终态。
- [v2 启动回执](../../results/_analysis/four_task_transfer/independent_main_table_execution_v2/launch_receipt.json)冻结 1,703 份源码／任务配置哈希。旧 GPU1 v1 的[无效记录](../../results/_analysis/four_task_transfer/independent_main_table_execution_v1/invalidation.json)保持独立。
- 每个固定臂 actor 的请求为 5 个物理初态槽位×2 个 mode×4 个机制组（`source`、`nominal`、`latency_gain_shift`、`payload_shift`）＝40；移动身体再乘 2 个支撑条件＝80。可训练方法有 8 个实例，两种公共固定参照各 1 个身份。全部固定臂请求的身体均为 xArm7，移动请求均为 Go2+Z1；这里 `source` 指目标身体上的 source-like 机制，不是 Panda source qualification。表中结果是这套目标执行 suite 的合并结果，不能重新命名为“只在未知目标偏移上的成功率”。
- 后续修改训练、actor、任务、身体、指标或测试分布，均属于新的开发版本与独立冻结 cohort。已经查看的 v2 结果可以用于提出假设，不能重新当成这些新假设的未访问测试集。

## 一、论文中心与 Introduction 的推导

> 已有源技能包含有用的交互结构，但不同身体未必能用相同指令实现它。ICDT 利用有限目标干预校准执行响应，迁移其中仍然有效的结构，并在源行为与目标任务发生冲突时允许偏离。

继承 MGA 的组织方式：先提出连续的技术缺口，再让方法、贡献、RQ 和实验逐项回应。主表回答完整系统表现，少量消融与事件图回答原因。不要求每个单元都获胜；需要在预先指定的重要条件下形成可重复、可解释且资源代价清楚的收益。出现反例时缩小主张或修改机制，不能通过事后挑任务、删失败或改指标制造优势。

Introduction 按以下六段组织：

1. **技能复用需求。** 源机器人已学会绕行、贴靠、寻位和恢复；迁移到另一机械臂或带臂移动身体时，希望复用技能，而非重新收集整套目标演示。身体、控制延迟、负载和接触顺应性会改变相同指令的执行结果。
2. **近期路线及其价值。** Human2Sim2Robot、Play2Perfect 强调源行为、物体语义与精密装配先验；ASID、AdaptSim、FADA 强调利用目标信息校准机制或执行模块；TactAlign、KITE 强调接触反馈与跨身体交互表示。它们共同说明源结构有价值、目标执行差异必须处理。
3. **具体缺口。** 不声称“现有方法都只迁移动作、没有接触”。问题收敛为：有限目标数据下，如何判断哪些源交互响应值得保留，并将其转化为目标身体可执行、可完成任务的行为？源响应有帮助的区域和源策略不兼容的区域需要区别处理。
4. **贯穿全文的装配例子。** 两个身体都能到达入口，但相同横向调整可能分别解除卡滞或引发力峰值。模型预测改善还不足以决定是否保留源恢复方向、速度和时序；目标工作空间或支撑约束不允许源方式时，需要有原则地偏离。
5. **ICDT 方法。** 短时干预提供任务相关执行信息；校准模型支持 source-conditioned transport；目标任务损失持续存在，源响应约束在任务兼容且模型有支持的区域选择性施加；已有身体控制接口负责闭环执行。
6. **三个贡献及证据安排。** 两项方法贡献加一项系统验证贡献，不把标准辨识、普通残差网络、公共 decoder/WBC 或未证明有效的迭代网络另列为原创贡献。

拟用题目：**ICDT: Intervention-Calibrated Distribution Transport for Cross-Embodiment Contact Manipulation**。最终是否使用 “Sim-to-Real”，取决于真实机器人证据；当前仿真不能替代实机结果。

| 贡献 | 拟提出的主张 | 必须对应的证据 |
|---|---|---|
| **C1：面向实际执行响应的技能迁移** | 连接源条件技能分布、目标指令和目标闭环响应，通过目标可执行映射迁移任务相关交互结构 | 实际有效指令与响应变化、模式语义保持、相同语义在不同身体上可执行 |
| **C2：干预校准的选择性响应约束** | 用有限目标交互校准任务相关响应，在有效区域保留源结构，在任务冲突或支持不足时放松约束 | 校准、响应、选择性三项受控消融；数据预算与事件条件下的机制证据 |
| **C3：递增交互复杂度下的性能与资源验证** | 在受限几何、持续接触、混合接触恢复和身体支撑耦合中验证完整方法的价值与边界 | 强 baseline 主表、目标访问预算、端到端执行成本、真实跨自由度／移动身体结果 |

以上是待证贡献设计。当前证据尚未证明总体优于 ASID/AdaptSim，也未证明单独 transport 在关键接触任务上带来稳定增量；不能把研究目标写成已取得的结果。

### 相关方向与 baseline 定位

| 路线／工作 | 叙事用途和比较角色 | 当前处理 |
|---|---|---|
| [Human2Sim2Robot](https://human2sim2robot.github.io/)、[Play2Perfect](https://play2perfect.github.io/)（Karen Liu 相关工作） | 物体语义、源行为先验与精密装配 | H2S2R 保持表外；Play2Perfect 的多指手 play prior 与固定持件场景有差异，适合作专项比较候选，不能直接填主表 |
| [FADA](https://github.com/LeCAR-Lab/FADA)、[DIAL-MPC](https://github.com/LeCAR-Lab/dial-mpc)（Guanya Shi 相关工作） | 执行适应与在线模型优化 | 核心主表保留，明确本仓库适配身份与原系统差异 |
| [ASID](https://weirdlabuw.github.io/asid/)、[AdaptSim](https://proceedings.mlr.press/v229/ren23b.html) | 主动辨识后学习、任务驱动仿真适应 | 已产生有竞争力的当前结果，继续作为强参照；分别是 2024／2023 工作，不统称“近两年方法” |
| [TactAlign](https://www.roboticsproceedings.org/rss22/p006.html)、[KITE](https://arxiv.org/abs/2606.22113) | 接触语义和跨身体意图／执行分离 | 用于精确定位缺口；在代码、接口与数据可复现性核实前不承诺新增正式主表行 |

论文定位不能依靠“只与旧方法比较”。近期工作用于建立已有能力和剩余问题；主表方法同时需要问题匹配、可执行性和清楚的数据权限。ACP／Human2Sim2Robot 按用户已确定的范围保留表外，不因源学习尚未建立就持续无上限调参，也不把其缺失执行记成主表零分。

## 二、三个 Research Questions 与递增复杂度实验

| RQ | 所回答贡献 | 重点环境／证据 | 关键比较 |
|---|---|---|---|
| **RQ1：有限目标交互能否提供足以迁移源技能的执行信息？** | C1、C2 | 受限转运与基础导面定位；响应预测与受约束成功率随目标预算变化 | ASID、AdaptSim、FADA、无校准消融；完整采集／选模／适应账本 |
| **RQ2：接触持续性、切换和环境动态增加时，何时应保留源响应，何时应偏离？** | C2 | 曲线导引、卡滞恢复、柔顺夹具；接触事件、峰值力、恢复过程 | 无响应约束、无选择性约束，以及最强任务 baseline |
| **RQ3：相同任务语义能否跨自由度和支撑形态实现，并保持合理执行成本？** | C1、C3 | Panda→UR5e；固定臂→Go2+Z1；身体与机制变化交叉设计 | 直接源迁移、ASID、AdaptSim、DIAL、公共反馈；端到端延迟和身体约束 |

### 四个环境家族：每类先完成两个核心子任务

复杂度通过新的任务关系增加，而不是把两种摩擦系数命名成两个任务。下表第二核心子任务均须单独冻结可行性、成功判据与源技能；“拟新增”表示尚无完整实验结果。先形成 **8 个核心子任务**，第三列只有机制证据确有缺口时再扩展。

| 环境家族／现有配置语义 | 核心子任务一 | 核心子任务二 | 可选第三子任务 |
|---|---|---|---|
| **受限空间持件转运** `obstacle_transfer` | 双路线持件绕障、定姿到位（复用现有基础） | 窄开口穿越后短距制动定位（拟新增；姿态与速度时序） | 连续不同取向开口，中途重定向 |
| **导面接触定位** `guided_contact_alignment` | 双通道接触引导定位（复用现有严格合同） | 沿折角或曲线导槽进给至止挡（拟新增；持续接触与方向切换） | 贴边滑移后绕支点旋转定位 |
| **带恢复的连接器装配** `keyed_insertion_recovery`／`compliant_keyed_insertion_recovery` | 偏置插入、卡滞撤回及重对齐（复用现有任务） | 单轴柔顺夹具插入（已有开发物理与预测数据，策略实验待补） | 插入后小角度锁止，增加接触阶段 |
| **身体支撑耦合装配** `mobile_keyed_insertion_recovery` | 固定站姿执行同一插入（复用现有基础） | 支撑范围内调整身体位置、稳定后对接（拟新增；显式身体协调） | 对接保持受到扰动后恢复定位 |

冻结时统一处理以下原则：

- 延迟、增益、负载、摩擦、夹具刚度是子任务内部的机制泛化轴，训练／开发／留出组合明确分开；身体变化与机制变化交叉设计，避免混淆二者。
- 新任务需要相应源技能，例如锁止不能假设由旧静态插入 prior 自动覆盖。跨身体比较使用同一个任务的源 prior，目标执行接口与可访问状态清楚声明。
- 主比较选择双方共同可达且满足物理约束的目标范围；源模式在目标不可行的条件作为独立诊断，解释选择性偏离，不能让主表主要测不可达初态。
- 当前导面合同在 1 ms 物理子步检查接触：覆盖率至少 50%，最大连续脱离不超过 150 ms，持续到请求模式出口。它允许有限短暂脱离；新增曲线任务若要求连续接触，应另立合同，不能追改旧成功判据。
- 当前固定臂主表目标为 xArm7，尚无正式 Panda→UR5e 结果。UR5e 是 RQ3 的新增跨自由度证据；当前 Go2+Z1 是站姿操作／支撑证据，尚不构成步态、导航或实机移动操作证据。
- v2 的目标身体 source-like、nominal 与偏移组已合并计分。未来协议需显式记录每个请求的身体与机制身份，分别报告目标 nominal／shifted 条件并保留目标 source-like 诊断；真实源机器人的 source qualification 单独进行。完整分层表保留，不重新计算旧表后追称其为原预注册主指标。

### 九行主表及三个主要消融

| 主表行 | 作用 |
|---|---|
| ICDT | 完整方法 |
| Task-only | 同容量内部参照，可保留在 general table，但不作为外部 SOTA |
| FADA | 冻结行为规划、适应执行模块的近期参照 |
| DIAL-MPC | 在线迭代模型优化参照 |
| Response MPC | 普通模型优化参照 |
| Public geometry/feedback | 公共反馈是否已足够解决任务 |
| Public source prior | 直接源迁移参照 |
| AdaptSim | 任务驱动仿真适应 |
| ASID | 主动系统辨识与后续任务学习 |

每个主表方法参与全部核心子任务，使用相同任务成功／约束合同。算法可以保留各自合理的原生控制结构；系统比较不要求都套成同一算法。实现缺失记“未执行”，真实执行失败保留，不能混为零成功。FADA、DIAL、ASID、AdaptSim 的任务适配身份、作者算法核心、执行改动和训练权限逐项说明。

写成统一目标形式：`J = E_i[C_i + β h_i + λ sg(G_i) R_i + Ω_i]`，其中 `C_i` 是任务损失，`h_i` 是相对冻结 task-only 的任务带 hinge 惩罚，`R_i` 是源响应约束，`G_i` 是每次更新重算并停止梯度的选择性门控，`Ω_i` 表示共同正则／控制约束。下表省略样本下标与期望，`h` 表示这一相同 hinge 项。正文突出三个消融：

| 消融 | 相对完整方法的主要改动 | 所回答问题 |
|---|---|---|
| **无目标校准** | 用 nominal context／model，其他训练和执行条件匹配 | 有限目标干预是否提供有用执行信息 |
| **无源响应约束** | `C + β h + Ω`，保留 task-band | 源响应项是否有独立任务增量 |
| **无选择性约束** | `C + β h + λ R + Ω`，令源响应项不再被 gate 选择 | 强制保留源响应在哪些条件有偏置 |

当前纯 task-only 的 `C` 与完整方法同时改变 response 与 band 项，不能单独识别 response 的作用。三组主要消融匹配 warm start、数据流、优化步数、动作权限、执行器、selector、模式策略和选模预算；若某项必须变化，应显式作为系统比较而非单因素归因。纯 task-only 继续保留，旧六格目标分解放附录；MLP／四步迭代参数化也放附录，不能因四格接近饱和而把 diffusion 作为主贡献。

### General table 的四个指标

| 指标 | 定义与目的 |
|---|---|
| **受约束成功率 ↑** | 完成任务且满足碰撞、接触、力和身体支撑合同 |
| **物理约束违规率 ↓** | 发生任一合同约束违规的 episode 比例，区分安全失败和普通未完成 |
| **失败惩罚的归一化完成时间 ↓** | 成功取完成时间／任务时限，失败取 1；全体请求均进入分母 |
| **端到端 p95 控制延迟 ↓** | 包含状态处理、策略／模型、在线候选选择、decoder／WBC 等实际控制链；固定硬件、同步与计时边界 |

按家族展示；每个家族内子任务等权，再按预冻结规则做家族级聚合。操作性能的主聚合与移动身体扩展分开报告，保留当前“前三类操作／第四类形态扩展”的科学角色，不让移动身体掩盖操作退步。未来是否采用不同聚合需要提前定义，不能改写 v2 的原统计合同。

完整子任务结果放附表，预算曲线、最差组成功率、模式对应、恢复率、峰值力和模型误差放 RQ 图。当前移动任务最差组成功率为零，必须继续展示；未来主表第四项换为延迟不意味着删除这一失败。当前 v2 的第四项是最差组成功率，**尚无统一口径的九方法端到端 p95 主表**，不能从单次网络前向或不同 actor 的局部计时拼接伪造该列。

候选目标交互时间网格为 0／30／60／120 秒，具体数值须在试验前冻结；允许 ASID／AdaptSim 等主动选择如何使用预算，不限定所有方法只能执行同一种两秒探针。预算比较覆盖 source 数据、目标校准、候选选择、模拟目标、部署目标和选模访问；目标物理交互按完整流程累计，不能只报最后部署阶段。不同权限方法保留各自可用输入与成本，并设计共同预算曲线；不因预算不同就自动认定原实验无效，也不能把当前表解释成等预算排名。

## 三、现有结果：能够讲什么、缺少什么

### 已完成 v2 的受约束成功率

单位为百分比；下表是目标身体上四种冻结机制的合并结果，不是目标偏移组单独统计。`source` 机制标签及历史任务编号只用于精确指向既有证据。固定参照每任务只有一个 actor，不伪造八个训练实例。† 表示当前仓库任务适配版本，不能与原论文完整系统无条件等同。

| 方法 | 受限转运 | 导面定位 | 静态插入恢复 | 移动支撑插入 | 前三类均值 |
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

描述性区间按独立训练 actor 和冻结机制／支撑组内共同物理初态块重采样，mode 保持配对，固定 actor 不重复伪造。以下均为 **ICDT 减比较方法的成功率百分点差**，95% 区间仅条件于已冻结机制、支撑和目标仿真器，不是所有未知机器人上的保证；没有多重比较家族校正。

| 比较 | 差值（pp） | 描述性 95% 区间（pp） | 当前解释 |
|---|---:|---:|---|
| 导面定位 − task-only | +1.2500 | [−2.1875, +5.0000] | 高成功率，尚无稳定增量 |
| 静态插入 − task-only | −5.0000 | [−18.1250, +8.4375] | 点估计退步，区间跨零 |
| 移动插入 − task-only | +26.7188 | [+21.5625, +31.4063] | 完整系统有清晰差异；不能独立归因 transport |
| 静态插入 − AdaptSim | −20.6250 | [−33.1250, −7.1875] | 现有强 baseline 在该任务领先 |
| 静态插入 − ASID | −17.5000 | [−30.9375, −3.7500] | 同样存在操作性能缺口 |
| 前三类均值 − task-only | −1.2500 | [−5.7292, +3.5417] | 不支持总体 transport 增量 |
| 前三类均值 − AdaptSim | −6.4583 | [−10.7292, −1.7708] | 不支持完整方法总体领先 |
| 前三类均值 − ASID | −4.6875 | [−9.3750, +0.1042] | 没有建立优越性 |

| 已观察现象 | 可写的内容 | 仍需完成 |
|---|---|---|
| 受限转运多个方法 100% | 基本跨接口迁移可执行 | 用有意义的几何／时序任务关系提高区分度，而非只加随机噪声 |
| 导面定位接近天花板 | 严格合同下高成功率 | 在相同动作权限下识别 response 的独立效应；当前差值区间跨零 |
| 静态插入 ICDT 135/320 成功、185/320 超力 | 暴露混合接触与模式恢复短板 | 优先早期力事件、模式策略、动作投影和模型有效时域，不能只增加种子 |
| 移动插入 ICDT 556/640、task-only 385/640 | 完整系统可在该身体运行，并优于此内部参照 | 对 ASID／公共反馈无优势；80 次 joint-limit、4 次 timeout 及最差组 0 均保留 |
| 柔顺夹具 holdout 的短时预测改善 | 校准模型在新初态上具有局部预测价值 | 任务闭环收益、长时域预测和事件安全证据仍缺失 |
| 参数化四格 63/64、62/64、64/64、64/64 | 两类参数化均可执行 | 不能支持 diffusion 或 anchoring 的稳健优势 |

特别需要保留以下归因限制：

1. 当前插入／移动的 fidelity 记录标记 `single_factor_transport_attribution:false`。完整 actor 含模式选择、投影或选模逻辑，不能把其全部差值写成源响应项的因果贡献。
2. FADA 的当前低分伴随新 source／nominal qualification 中尚未建立 learned 闭环能力，且本地采集／训练链不等同作者完整 oracle PPO＋DAgger。冻结 planner／更新 IDM 的证据有效，但不足以把零分解释为原方法的固有机制失败；需明确 source teacher 与 learned source 的区别。
3. 当前 AdaptSim／ASID 等探索和选模成本不相同。现有表提供完整系统表现，尚未给出统一完整目标预算下的效率结论。
4. 固定臂结果针对 xArm7；不能展示这些轨迹却标注为 UR。Go2+Z1 结果支持站姿操作与支撑扩展，不能写成动态步行或实机成功。
5. [独立柔顺夹具预测报告](../../results/arm/compliant_keyed_insertion_recovery/validation/fixture_response_model_holdout_20261031/report.json)的 8 条新公共 episode、3,623 个窗口中，v2→v3 的 3／6／10 tick 整体位置 MAE 为 7.72→4.47、9.16→9.02、13.79→13.81 μm；事件窗口为 24.30→23.64、40.26→37.46、50.63→49.38 μm。改善主要在短时，不能从平均误差推出接触安全或任务收益。优先研究接触建立、峰值力、撤回、再推进和完成等事件。
6. 现有四步确定性 DDIM-style 分支直接优化 ICDT 目标，没有 score-matching 预训练。它是迭代 transport 参数化，不能称作已学得概率 score 的 diffusion policy，也不能由 64/64 天花板开发结果宣称分布建模优势。

期望论文形成的证据结构是：简单任务保持竞争力；预先指定的重要接触条件出现可重复、可解释的收益；新的身体能够实现相同技能语义；数据或执行成本提供额外价值。当前已有完整有效主表，但中间两项核心机制与性能证据仍需研究，不能先写成全面胜过强 baseline。

## 四、理论调整：让数学结论对应三个贡献

本轮 [ICDT 方法稿](../../latex/latex_icdt/sections/method.tex)已写入两条基础 lemma 与一条带证明的条件事件保持命题；工程审计、历史诊断和完整技术边界放附录或本文档。旧 [finite feasibility 理论草稿](../../latex/cross_ebdm/sections/finite_feasibility_theory.tex)作为背景，不把大量诊断边界并列为论文贡献。数学推导已落稿，不等于当前响应模型已经满足命题的事件充分性、覆盖和误差假设。

1. **执行映射与支持。** 说明 transport 可改变纯 reweighting 无法创造的指令支持，并把结果通过目标 decoder／executor 推到执行响应空间。同时承认同容量 task-only 具有相同表达类，不能由支持论证推出 transport 相对 task-only 的天然表达优势；实际候选收益来自有限数据和有限优化下的归纳偏置。
2. **任务相关校准。** 给出任务响应可辨识所需的局部激励与覆盖条件；不要求识别全部物理参数，但必须覆盖决定任务事件的方向和时间尺度。系统辨识本身是标准工具，ICDT 的作用是连接辨识信息、响应迁移和目标任务。
3. **响应对应到任务事件。** 明确足以定义任务事件的响应状态、源成功事件裕度、校准／transport 误差、模型支持范围及身体失败事件，再建立条件性的任务失败界。不能把“响应改善会改善任务”直接假设后当作中心证明。

当前稿件 `eq:event-transfer-bound` 的条件结论为：

\[
P(\text{目标失败}) \le P(a(Y_s)\le0) + P(0<a(Y_s)\le r)
+ \frac{L}{r}(\epsilon_{\rm cal}+\epsilon_{\rm anchor}) + p_{\rm bad}, \qquad r>0.
\]

这里在固定匹配条件和有限时域上耦合源／目标真实及模型响应；支持、任务兼容且身体约束满足的事件为 `A`，`P(A^c) ≤ p_bad`。任务 margin `a` 在共同响应度量下为 `L`-Lipschitz，并在 `A` 上充分定义成功。校准误差和 anchor 误差是在 `A` 上带指标的期望误差。草稿利用三角不等式、Markov 不等式与失败事件并集界给出证明；不需要把“响应改善直接使任务更好”作为结论本身的假设。

**尚待完成的是理论与实际模型之间的联系。** 当前增量／力描述子并不编码所有恢复 guard、累积接触约束和闭环历史，因此现有 anchoring loss 不是上述命题的证书。需要构造事件充分的响应、测量或控制访问分布上的校准误差、支持外概率与身体失败，并处理多步闭环累积；不能从短时预测 MAE 直接填入整段 episode 的误差项。接触阶段、延迟历史、峰值力／积分量应按真实成功合同选择。该界说明条件性事件保持，不比较 task-only，不保证样本复杂度或完整系统任务优势。论文保留这些边界，实验负责检验条件及实际增量。

必须同步修正的边界包括：soft task-band 不是物理任务非劣保证；ensemble disagreement 未经校准不是误差上界；训练 gate 为零不保证共享网络输出等于 task-only；公共模型、selector、decoder 和 WBC 都可能决定实际行为，归因必须覆盖完整执行链。当前导面 transport 主要改变切向和姿态，法向保持与部分刚度由公共控制器处理，不能据此宣称 transport 学会了完整柔顺控制。理论陈述应对应可实现的有限预算 learner，而不是利用不可实现的 population oracle 要求每个状态都支配 task-only。

## 五、后续落地顺序与架构整改

### 工作顺序和交付条件

| 顺序 | 具体交付 | 完成条件 |
|---|---|---|
| **架构与命名收口** | 冻结证据所对应源代码／配置快照；语义任务标识；薄 paper 入口；配置化设备与环境；旧格式兼容读取 | 当前运行时代码不再依赖 task-number 身份，代表性配置与 actor 行为等价，历史 SHA 证据可追溯 |
| **贡献与协议冻结** | 三个贡献／RQ、八个核心子任务、四指标、baseline 权限和完整预算、目标机制分层与真实源资格分离 | 每个实验明确回答哪个问题，成功／约束判据、统计单位及主要聚合在新测试前确定 |
| **插入机制诊断与 baseline 质量** | 定位早期超力；事件／动态响应；相同执行器下 matched actor；baseline 训练链与名义能力说明 | 能区分实现差异、公共控制边界和算法机制；不以低质量适配支撑优越性 |
| **小规模 RQ 开发实验** | 三个主要消融、预算曲线、模式与事件分析；先做有区分度的核心子任务 | 形成清晰机制证据或可解释适用边界，再决定哪些修改值得正式冻结 |
| **跨身体与正式主表** | 真正 UR5e 目标；移动支撑扩展；全部主表 baseline；独立新 cohort | 请求、方法、预算、输出路径和指标冻结；完整执行并保留真实失败 |
| **论文与展示** | Introduction、方法、精简理论、general table、三组 RQ 图和代表性视频 | 每个正文结果主张有对应证据；展示标明真实身体、条件和成功／失败，不挑片段掩盖限制 |

下一项具体研究交付是：完成语义命名与入口整改；冻结八个核心子任务合同；随后集中处理静态插入的超力，在同一执行器下比较完整方法、无响应约束、无选择性约束。该结果决定柔顺夹具能否成为主要展示任务。不要未经机制检查立即把旧大表扩大重跑。

### 当前架构问题与迁移方式

2026-11-04 只读检查确认：`scripts/paper/cross_ebdm` 有 **49 个文件名**含 `t1`–`t4`；`qualify_method_behavior.py` 为 **5,601 行**，`train_t4_completion.py` 为 **2,018 行**。当前 [main_table_execution.py](../../genedynamics/tasks/transfer/main_table_execution.py)反向导入 `scripts.paper.cross_ebdm`，并硬编码 `/tmp` 环境、设备和特定运行路径。问题包含依赖方向与职责，不能只重命名文件。

| 当前结构／符号 | 计划迁移 |
|---|---|
| `T1`／`T2`／`T3`／`T4` 作为运行时任务身份 | `obstacle_transfer`、`guided_contact_alignment`、`keyed_insertion_recovery`、`mobile_keyed_insertion_recovery`；旧结果读取层保留显式映射 |
| 大型 task-number trainer／qualification script | 算法实现进入各自既有 solver；任务语义进入 task adapter；paper 只解析 config 并调用公共入口 |
| ASID 等按任务编号重复训练入口 | 单一方法入口，通过已有任务 adapter 和配置表达差异 |
| `T2_TRANSPORT_COORDINATES` 一类内部字段 | 使用导面接触语义命名；旧 checkpoint 通过版本化兼容读取，不原地改二进制 |
| runtime 导入 paper script | admission／assembly 等通用实现置于既有运行时模块；paper 单向调用 package |
| 写死设备、环境和批次目录 | 由既有实验 recipe、环境配置与启动参数提供；算法源码不承担资源调度 |

沿用 MGA 已有 [suite 入口方式](../../scripts/paper/mga/run_peg_insert.sh)：物理任务配置定义具名 `suites`，继续使用 `_base.yaml` 和 `main`／`baseline`／`ablation`／`training`／`validation`／`protocol` 分类。子任务放在所属任务家族配置中，不再新增一套顶层 `solvers`、`integrations`、`cross_embodiment` 或混杂算法的高层 learning 树。每个算法在其 solver 下维护实现；通用 learning 层维持高层职责。具体改动先对照项目原有架构规则，不能机械地把所有文件移动到新目录。

**先保存与冻结证据一致的源码／配置快照，再迁移当前实现。** 1,703 份清单是来源索引，不等同完整内容快照；需要核对现有训练／评估快照是否足以恢复，并仅对缺失部分保存精确版本。若当前文件已变化，应从原冻结快照恢复对应内容，不把后来文件错误盖章为原版本。v2 的结果、配置证据、checkpoint 和有效性记录不原地改写；兼容读取解释旧标识，新运行用语义名与新 cohort。历史文字中的 T1–T4 允许为溯源保留，不能用全库字符串替换破坏证据。

正式数据规模需按科学问题节制。维持现有实例／请求规模，四家族各两个子任务约对应 **23,200 条正式评估 outcomes**，还不含消融、预算曲线和新增身体。因此先完成有限开发实验与方法资格，再决定正式样本量；不得把这个数量当成本轮自动启动授权。后续保持 GPU0-only，论文／文档修改不伴随后台计算。

## 六、2026-11-05 执行规划：八个核心子任务如何搭建

本节是后续实施方案，不表示新增环境、重构或训练已经执行。11 月 5 日只读复核确认 v2 仍为 `FORMAL_TABLE_ASSEMBLED`，36 个单元和 11,600 条 outcome 完整；49 个含任务编号的 paper 文件、运行时反向导入 paper 及硬编码资源也仍存在。下一轮目标保留四家族、每类两个核心子任务、全部九方法主表与三个主要消融；先集中完成能回答问题的开发证据，再冻结完整大表。

### 投稿安排先更新

11 月 5 日查阅 [RSS 2027 官方征稿](https://roboticsconference.org/information/cfp/)与[日期](https://roboticsconference.org/)：Stage 1 截止 **2026-12-04 23:59 AoE**，最多 **5 页正文＋1 页参考文献**，允许实验处于初步或未完成状态，但要求方法与假设完整。Stage 2 完整稿截止 **2027-04-16**，需要执行评审确定的 Extension Charter，正文最多 8 页。当前 8 页正文的 ICDT draft 应保留作为完整稿，同时在同一 LaTeX 工程准备符合 Stage 1 要求的精简版本；已有排版检查通过不代表符合 Stage 1 篇幅。

因此分成两个交付时间尺度：12 月 4 日前交付清楚的方法、可检验的三个 RQ、可信初步比较和一组干净机制实验；完整八子任务、预算曲线、UR5e 与可能的实机证据继续推进，不把所有内容强塞进第一阶段。具体审稿扩展按届时正式要求调整，不能假设一定获邀。

### 共用结构与搭建验收

沿用 [MGA 入口](../../scripts/paper/mga/run_peg_insert.sh)、[_base.yaml 的 suites](../../configs/arm/peg_insert/_base.yaml)及统一 experiments runner。下列名称是拟新增的语义 suite，不是另建八套算法目录：

| 家族 | 核心子任务一 | 核心子任务二 | 可选第三项 |
|---|---|---|---|
| 受限持件转运 | `dual_route_carry` | `aperture_brake` | `sequential_aperture_reorientation` |
| 导面接触定位 | `dual_channel_contact` | `corner_guided_seating` | `edge_slide_pivot` |
| 连接器装配恢复 | `jam_recovery` | `compliant_fixture_insertion` | `insert_then_lock` |
| 身体支撑耦合装配 | `stance_insertion` | `reposition_then_dock` | `disturbed_docking_recovery` |

物理几何与动力学进入既有 `envs/domains/manipulation/transfer_workcell`；任务身份、观测、公共接口、事件判据进入既有 `tasks/transfer`；算法训练和推断进入各自 solver；paper 入口负责配置展开、调用和汇总。静态插入与柔顺夹具已有不同配置目录，继续保持，论文把两者归为一个家族。高层 learning 只维护真正共享的数据／checkpoint 能力，不混入各算法训练。

配置沿用 `configs/<embodiment>/<task>/{_base.yaml,main,baseline,ablation,training,validation,protocol}`。结果继承当前框架的 `results/<embodiment>/<task>/<category>/<method>/level_<suite>/seed_<seed>`，正式批次通过已有 recipe/protocol 和已声明输出根区分，禁止覆盖旧证据，不改为 project/run_id 层。suite 表达子任务／条件；设备、种子、总预算按实验配置统一管理，不允许某方法在 suite 中偷偷扩大预算。

每个子任务达到“搭好”的标准包括：实体场景与复位；公开观测／动作权限；独立事件与成功／违规判据；合格源技能及公共反馈；训练／开发／留出分布；九方法适配入口；原始物理证据和一条成功／典型失败可视化。判据读取物理状态，不信任 policy 自报 phase。先在 CPU 检查正常、错序、终点绕过、超力和支撑失败，再进行小规模开发；导面／峰值力等需要物理子步证据。

### 受限持件转运：路径语义到姿态与速度协调

**`dual_route_carry`：继承现有双路线任务。** 保留刚性附着的 L 形持件、上方／侧向绕障、目标位姿保持和碰撞／关节限制；它不包含重新抓取。补源轨迹全程可达性与新接口迁移等价检查，源 prior 对同一任务跨目标身体保持冻结。既有转运成功容差为 5 mm／3°、保持 0.3 s、时限 10 s；新协议若改变它们须另行记录。延迟、负载属于同一子任务的机制测试，不另计为子任务。

**`aperture_brake`：新增实体开口与短距停靠区。** 在[既有 workcell](../../genedynamics/envs/domains/manipulation/transfer_workcell/workcell.py)增加有边框的开口和出口后的终端区域。源技能为接近→调整持件姿态→整体穿越→减速→到位保持。开口净空根据整个持件包络定义，不能只让 TCP 穿过；evaluator 检查持件完整穿越事件、正确方向／顺序、无边框碰撞、终端位姿和低速度保持。空中绕过边框、仅 TCP 越线、碰撞后到达均不能算成功。开口几何作为已知任务信息同等公开，真实延迟／负载隐藏。

先用较大净空与足够制动距离验证可行性，再缩小几何和时序裕度，最后加入预先固定的机制偏移。几何参数根据双方可行性与源技能确定，不能看到 ICDT 排名后筛选任务。其 RQ1 价值是区分“轨迹看起来相同”与“真正执行时能否及时制动”；成功率饱和时同时看完成时间与违规率，不预设 transport 必赢。

可选第三项新增两个取向不同的开口和中间转动空间，需要新的多阶段源技能与两个依次穿越事件；八核心任务完成前不开展。

### 导面接触定位：载荷保持到转角接触切换

**`dual_channel_contact`：保留当前严格合同。** 复用现有双通道实体导面、接触材料和 [1 ms 子步 evaluator](../../genedynamics/tasks/transfer/guide_evaluation.py)。首次受载接触在模式出口前发生，规定区间接触覆盖至少 50%，连续脱离不超过 150 ms，出口后允许已有声明的抬起转运；不能将它改写成全程零脱离任务，也不追改 v2。

**`corner_guided_seating`：新增水平面内 L 形导槽。** 第一版采用底面、两段侧壁、圆角过渡和实体末端止挡；先避免空间曲面造成的法向变换与网格接缝混淆。源技能为建立接触→第一段滑动→保持载荷转向→第二段滑动→止挡稳定。公开状态增加由已知几何得到的局部切向、路径进度和下一段目标；接触 geom 身份可供独立 evaluator 使用，不能无声明地变成策略特权输入。

必须同时改实体几何、任务 spec、源技能和 evaluator；现有 [ContactConfig](../../genedynamics/tasks/transfer/contact.py)对场景／profile 有约束，仅加 YAML 不够。当前接触法向定义为 world +z，正 `command[12]` 表示向下的压紧力，水平底面转角可复用这一约定；后续若改为空间曲面，须显式扩展局部 wrench 坐标映射。

完成要求为按顺序经过两段及转角、在规定导引区间内维持声明的受载接触、抵达止挡并稳定，且全程不超力。新合同单独冻结接触比例、允许脱离和载荷范围；预设中间事件阻止抬起越墙、跳过转角或直接飞到终点。难度从浅转向到直角／更小圆角，再交叉摩擦和延迟。主要回答 RQ2：何时保留接触结构、何时调整时序或方向。公共反馈的权限、投影和选择器需要对方法一视同仁，否则无法识别 transport 的作用。

可选 `edge_slide_pivot` 增加实际支点和末端旋转，需要新的扭矩与接触模式判据，不能只更换终端姿态。

### 连接器装配恢复：静态混合事件到柔顺环境

**`jam_recovery`：优先解决现有静态任务。** 当前几何已有 keyway 与防转肩部；[恢复判据](../../genedynamics/tasks/transfer/contact_reference.py)已经要求真实受载接触后撤回：首次载荷超过 0.3 N 后至少 12 mm 上撤，并从实际轨迹区分模式。保留加载→卸载／撤回→重对齐→插入→受力保持事件，不把无恢复的直插替换为同一任务。当前 ICDT 185/320 失败均为 overforce，因此先定位首次冲击前后的有效动作、投影、接触阶段与物理子步力峰值，确认失败来自执行、预测时域还是源约束施加的阶段。

**`compliant_fixture_insertion`：在已有被动夹具上补齐闭环。** [实体实现](../../genedynamics/envs/domains/manipulation/transfer_workcell/contact_workcell.py)已有 world x 单轴 passive slide、弹簧阻尼和机械行程，无夹具驱动器；现有开发配置是 0.5 kg、800 N/m、12 Ns/m，行程 ±12 mm。这些是起始开发参数，不是最终难度冻结值。[任务 wrapper](../../genedynamics/tasks/transfer/compliant_socket.py)已提供测得 fixture pose/twist 和相对坐标评价，隐藏质量／刚度／阻尼。

仍需建立动态任务源技能、九方法训练／部署适配、物理子步 fixture 位移／速度记录和稳定性合同。当前控制端点的最大位移／速度只作诊断，不能冒充完整夹具稳定判据。成功在当前夹具坐标系判断相对深度、姿态、恢复及受力保持；额外约束夹具末态速度／稳定窗口及机械行程。推动整个 socket 到世界目标、或顶住机械限位把柔顺夹具变成固定夹具，不能绕过任务。

先做固定／较硬夹具的对照，确认操作激发了可测运动，再增加较软或较低阻尼条件，并留出刚度—阻尼组合。响应模型从全窗口平均误差扩展到接触建立、卸载、峰值力、夹具运动和重新插入事件；新任务是否支持贡献最终由闭环比较判断。该子任务是 RQ2 最重要的候选展示，不把已有微米级短时预测改善称为操作收益。

可选 `insert_then_lock` 需要可实际转动锁止的 bayonet／L 槽几何和锁止事件。当前防转 keyway 不能通过追加旋转指令变成锁止实验。

### 身体支撑耦合装配：已有站姿执行到目标驱动协调

**`stance_insertion`：保留现有四脚站姿操作。** 复用 Go2+Z1 浮动动力学、足接触、stance WBC，以及身体高度／倾角／支撑／关节／QP失败判据。需纠正“身体完全静止”的说法：[当前配置](../../configs/quadruped/mobile_keyed_insertion_recovery/_base.yaml)已有 `[0.005,0.005,0] m/s` 持续 6 s 的公共身体运动请求。这里 stance 指固定足支撑而非基座不动；当前没有步态／导航。

**`reposition_then_dock`：新增有目标的重定位和稳态事件。** 复用机器人与连接器资产，增加经站姿／手臂共同可达性核查的对接位点。技能过程为观测位点→在当前足支撑域内调整身体→满足低速度和姿态稳定窗口→开始对接恢复→插入保持。它区别于旧版固定时长的身体漂移，不能仅改名字或 drift 速度算新子任务。

共同接口需允许有界 body target/request，并对所有方法提供同等可观测量与控制权限；WBC 执行请求并维持支撑，不暗中替某种策略完成高层重定位。若新增动作通道，全部相应 actor/adapter 都要重新适配，不能只让 ICDT 使用。evaluator 从实际 base、tool、fixture 与足接触记录确认顺序；禁止滑脚或挪脚绕过支撑域。按目标相关重定位要求和共同可行区域布置场景，不能靠大量不可达初态制造失败。

难度从小 body shift 到更明显的臂—身体协调，再加入支撑增益偏移或插入反力。这类主要回答 RQ3 的形态可行性，投入优先级低于静态／柔顺插入，不要求它承担“全面超过 baseline”的主结论。可选第三项才加入可复现扰动与恢复时限；真正行走另立任务，当前八任务不隐含导航开发。

### 方法、baseline、消融和理论一起推进

算法先做两件事：一是记录并匹配完整执行接口 η，分清请求、投影后动作、恢复覆盖和在线 selector；二是在关键事件上校准响应，加入任务需要的接触阶段、因果延迟历史、夹具相对运动及物理力峰值。导面既有 warm/残差饱和问题先检查有效动作差异，必要时对完整方法和参照同时调整输出尺度／权限，不能仅增大 ICDT 权限。

九方法继续覆盖全部八子任务。ASID／AdaptSim 已是强任务适配参照；优先维持其辨识／适应闭环与完整预算。DIAL 保留公共响应模型上的任务适配身份。FADA 安排一次有界源闭环诊断：先核查源 teacher→采集分布→动作映射／IDM→learned rollout，再决定是否需要补源训练；当前 learned source／nominal 的三个固定臂结果均为 0/64，`admission passed` 仅表示身份／执行审计通过。真实失败保留在适配行，但不能把这些零分当作胜过作者完整系统的主要证据。同步更新过期 SOURCE/fidelity 状态；ACP 与 Human2Sim2Robot 继续表外。

主消融保持三项：无目标校准；去 response 保留 task-band；保留 response 去 gate。数据、warm start、动作权限、恢复、选择器、训练／选模预算在对应比较中固定。主表 task-only 仍保留，但不替代纯 response 消融。MLP／迭代参数化仅作为附录诊断。

理论继续围绕现有两条 lemma 和条件事件界：明确每个子任务的事件充分响应、裕度和支持范围；用独立轨迹检查预测与事件是否对应，区分模型误差和公共执行器改变。ensemble spread 只能作经验筛选，不能当误差证书。数学结论说明何时可以保持事件；实际优于强 baseline、有限数据受益仍由预指定实验回答。可允许部分子任务持平或负收益，要求主要收益可重复且机制解释一致，不要求逐格全胜。

### 比较规模、统计与视频

主表保留四指标：受约束成功率、物理违规率、失败惩罚的归一化完成时间、端到端 p95 延迟；家族内子任务等权，正文按四家族聚合，附表展示全部八项。最差机制组、模式恢复、峰值力／接触曲线放 RQ 图，不删除现有失败边界。图的重点是：RQ1 的预算—成功／预测曲线；RQ2 的源响应保留／偏离、事件时序及三个消融；RQ3 的身体变化与延迟／可行性。

基本主表延续 Panda→xArm7 与 Go2+Z1。UR5e 在三个固定臂家族各预先指定一个代表子任务，建议窄口制动、折角导引、柔顺夹具，并与原身体在相同任务条件下交叉比较；纳入这些比较的九方法都要执行，不能仅展示 ICDT。当前 UR 资产／IK 路径存在不等于资格与结果已完成。每个任务家族至少有一组同条件 source/target 视频，明确机器人、机制、是否成功，展示完整关键事件；移动对照标为 Go2+Z1，不标成 UR。

开发先用少量独立 actor、共享初态的配对请求诊断；正式实例数由开发方差与最小有意义差异决定。训练、开发和正式名单分开；同一身体内方法共享物理初态，跨身体匹配任务几何／初态语义而非要求不同机器人全状态哈希相同。正式统计保留 actor 和初态块层级，mode 配对、机制／支撑分层；固定参照不伪造八个训练 seed。主要比较提前指定，其余区间作为描述性分析。

0／30／60／120 s 仍是待冻结预算候选，计入目标探索、校准、选择和失败尝试，允许主动方法自行使用交互；源数据和模拟训练成本另外公开。当前一次静态插入实例中 AdaptSim 为 10 个目标 episode／1,616 控制步，ASID 为 3／137，不能直接解释为等预算效率比较。部署观测／模型／选择器／投影／WBC 统一计时，不能把 actor 前向延迟填作端到端延迟。

按当前密度作容量规划：主表为 `9×8=72` 个方法—子任务单元、23,200 个评估 episode；三消融全部覆盖另需 9,600，总计 **32,800**，尚不含训练、资格、预算曲线和 UR。三个 UR 代表子任务以相同密度执行九方法另需约 6,960。该数量是预算估算，不是已经冻结的样本量或启动任务。

v2 汇总没有可靠总起止时间，本轮抽取 36 个方法—家族首 actor 的 report，按实际 episode `wall_seconds` 与实例数外推旧四任务约 34.84 串行小时；同复杂度八子任务主表约 69.68 小时。这只含 episode 执行，不含训练、JIT、启动、CPU审计、排队；修复早期失败会延长 episode，新场景也会改变耗时。正式排程前各方法／新子任务实测少量 rollout 吞吐，再估 GPU0-only 总时间，不给无依据的“一晚全跑完”承诺。

### 实施顺序与近期时间目标

以下是工作目标，不是研究效果或完成时间保证；发现新问题时更新依赖和证据，不靠反复试正式集推进。

| 工作窗口 | 应完成的内容 | 交付与后继条件 |
|---|---|---|
| 11 月 5–10 日 | 保全冻结来源；语义身份和薄入口迁移设计／实施；八子任务合同；子步记录与全链计时 | 新配置可展开、旧 actor 代表性行为等价、来源可恢复；不重写 v2 |
| 11 月 10–20 日 | 两条并行线：静态插入超力与匹配消融；折角导槽和柔顺夹具闭环搭建。同步有限 FADA 源诊断 | 一组完整、可归因接触对照；新任务源／公共控制可执行，评价防绕过 |
| 11 月 20–28 日 | 冻结一组关键接触开发假设与新鲜评价；近期强 baseline、三个消融和初步预算证据；窄口任务接通 | RQ1／RQ2 有可解释证据及负例，既有四任务大表作为系统背景 |
| 11 月 28–12 月 4 日 | Stage 1 方法、假设、初步证据与6页版本；核对引用和结果 | 达到初稿投稿要求；未完成实验明确列为后续工作 |
| Stage 1 后至下一轮冻结 | 八子任务全部搭建、九方法适配、目标驱动身体协调、UR 代表任务、三消融完整覆盖 | 正式协议、预算和名单冻结；GPU0执行、CPU审计、完整主表与RQ图 |
| 获邀后至 4 月 16 日 | 按 Extension Charter 补实验与分析、完整稿／视频；有实机资源再执行预定硬件分支 | 每个贡献有对应证据、全部失败保留、结论范围与结果一致 |

实机分支需要确认可用机械臂、力／位姿观测和夹具资源；优先在导面／柔顺插入中选择一个与贡献直接对应的任务。资源尚未确认前，仿真主线照常推进，但不把 “Sim-to-Real” 写成已验证结论。第三可选子任务、复杂曲面、锁止和动态行走均后置，完整八核心目标不因此减少。

最先实施的具体工作包是：**来源保全＋语义任务入口＋八份子任务合同；随后并行静态插入机制修正与两项接触环境建设。** 新正式大表等待接口、baseline身份与评价合同稳定后启动。

## 历史记录（2026-11-02 及以前；仅供溯源）

以下保留原始开发记录与当时的 lead。其 GPU1 授权、待启动状态、旧 contribution 设想和历史任务编号均按记录发生时理解；当前状态、论文范围和下一轮顺序以上方 2026-11-04 版本为准。历史性能、无效标记及 SHA 不作追溯性修改。

更新：2026-11-02（America/New_York）。本文件是开发与决策记录；LaTeX 保持论文 draft。以下“论文贡献与下一轮实验”取代历史段落中以四任务逐请求战胜 task-only 为先决条件的规划。历史冻结实验、判定与 SHA 保留；新研究问题须在下一批测试前登记，不能追认成旧实验的预注册主张。下面“当前实施”的 GPU0-only 描述记录当时授权；之后用户授权本轮在物理 GPU1 上执行有界诊断、动态响应训练与四格对照。2026-11-02 用户将未完成 goal 的设备要求改为 GPU0；GPU1 正式主表旧批次的超时与 GPU0 新版本见文件末尾。

## 当前推进：GPU1 归因对照、动态响应与 baseline 名义诊断

T2 的 `configs/arm/guided_contact_alignment/protocol/backbone_attribution_v1.json` 已在新训练前冻结 MLP／四步确定性 DDIM-style transport × task-only／selective 四格。每个 backbone 内 task-only 和 selective 从同一 warm checkpoint 出发；四格在每个实例共用同一 branch 样本流、response、source bank、输出通道、残差半径和公共执行器。训练已在物理 GPU1 上完成八个实例、合计 96,000 次更新，耗时 959.5 秒；没有新增物理调用或打开确认集。32 个 checkpoint 的两模式构造、有限输出与推理延迟预检已通过：MLP／迭代模型的参数约 291k／308k，单次前向中位延迟约 0.30／0.72 ms。随后冻结的两种机制偏移×两个 mode×两个 reset slot 的八个新请求，已在八实例四格上一次性配对完成，共 256 个开发 episode；它不是四任务正式主表。Diffusion 直接用原 ICDT 目标反传，没有 score-matching 预训练，因此目前只可称为迭代 transport 参数化，不能宣称学到概率 score 或额外的采样不确定性。

在物理评估仍运行时，只用冻结的 1,600 个公开 validation states 做了 CPU 动作支持诊断；均匀抽取 256 个状态，并逐一核验八实例四格 checkpoint SHA。T2 五个可改通道的归一化请求残差接近共同半径 0.00625 的比例，MLP task-only/selective 各实例中位数为 96.7%/97.1%，迭代模型为 99.2%/99.0%。selective 与 matched task-only 的全 13 通道平均绝对请求差，各实例中位数仅约 1.03e-5（MLP）与 4.71e-5（迭代模型）；逐实例仍有少数最大差达到半径两端的 0.0125。证据与精确定义保存在 `results/_analysis/four_task_transfer/backbone_attribution_saturation.json`。同一批公开状态上的八实例 warm checkpoint 已先达到 92.9%--98.5%（MLP）和 92.8%--100%（迭代模型）的五通道边界占用，逐 checkpoint 哈希与统计见同目录 `backbone_attribution_warm_saturation.json`；饱和发生在 task-only/selective 分支分开之前。这是已访问 validation 数据上的表达支持诊断，不是独立任务效果；它提示当前公共投影和有限残差半径可能压缩 source anchoring 对实际请求的可辨识作用，下一版须在相同执行权限下匹配地调整两分支，而不能只调 selective 的损失权重。

四格物理开发集的完成结果为：MLP task-only 63/64、MLP selective 62/64、迭代 task-only 64/64、迭代 selective 64/64；四格约束违反均为 0。平均失败罚时归一化完成时间依次为 0.94873、0.95013、0.94839、0.94627。按八个训练实例和每个机制×mode 的两个共同请求块分层配对重采样，MLP selective 减 task-only 的成功率差为 -1.56 个百分点，描述性 95% 区间 [-9.375,+6.25] 个百分点；迭代模型两格成功率均为 100%，该批内差值与区间均为零，但不能推断所有机制条件等效。迭代 selective 减 task-only 的完成时间差为 -0.002125，区间 [-0.00664,+0.00266]，没有稳定时间优势。原运行器把请求 0--3 与 4--7 分别重采样，只固定机制而没有固定 mode 配比；原 `report.json` 完整保留，其中旧 paired interval 已由 `results/arm/guided_contact_alignment/validation/backbone_attribution_development_v1/independent_cpu_audit.json` 的修正值取代，后续代码也已修正并通过回归测试。

独立 CPU 重审核验 256/256 个实例×方法×请求组合、结果文件与证据 SHA；所有离散成功／失败、模式、约束和 evaluator 来源一致。186/256 份证据逐位复算完全一致，另 70 份只有至多 8.67e-19 的浮点标量差，低于本次重审明确记录的 1e-12 绝对容差；因此审计报告分别保存严格复算计数与容差复算结论。失败为 3 个 timeout，均保留，没有补跑。端到端 actor 动作调用的 episode 内 p50 中位数，MLP 两格约 1.04 ms、迭代两格约 1.48 ms；这项迭代参数化增加了在线调用成本。T2 当前开发集接近成功率天花板，且 warm 已饱和、分支实际请求高度重合；这批对照不能支持 source-response anchoring 的任务性能增益，也不能将迭代参数化的 64/64 宣称为对 MLP 的稳健优势。

动态 T3 的单轴夹具分段响应网络已在物理 GPU1 完成 600 次更新，输入仅含因果公共机器人／夹具历史与已发请求，输出为 3/6/10 control tick 响应和控制频率力峰值。首版训练集 910、验证集 914 个窗口来自四条轨迹；127 个夹具运动事件窗口中，3/6 tick 的位置 MAE 为 13.30/18.45 μm，高于零位移参照的 8.48/16.50 μm；10 tick 为 18.57 μm，低于零参照的 26.78 μm。更强的 >100 μm 事件中三个时域均略优于零参照。这个模型是已观测小位移条件下的开发预测器，不是未知接触机制的可靠外推或物理子步安全证书。先前自动审批对特定并发作业的拒绝已由本轮用户明确授权并发使用 GPU1 取代；当前训练遵守本轮 GPU1 资源界限。

在同一数据、网络与 600-update 预算下，另训练一版以部署时的“事件概率×条件位移”为目标的 v3，并以独立脚本重算先前已经访问过的 5,480 个验证窗口。`results/_analysis/four_task_transfer/fixture_event_objective_comparison_20261031/report.json` 记录 v2→v3 的全窗口 3/6/10 tick x 位移 MAE 为 7.65→4.33、9.17→8.71、13.15→12.30 μm；按各段真实事件分组为 24.38→23.95、41.24→35.83、48.89→42.79 μm。静止组 6 tick 由 3.29 增至 3.75 μm；不乘事件概率的条件幅度误差在 6/10 tick 也变差，故改善限于当前部署输出与已访问数据。分析核对 v3 四份源码快照、训练 recipe、数据及 checkpoint SHA；两版均未做新的物理调用。这是开发性预测诊断，不能推出任务收益、未知机制泛化或子步安全。

为避免只报告已访问验证数据，冻结两个 checkpoint 后使用任务本地 `fixture_response_holdout_*` 配置新采集 8 条 CPU 公共轨迹：软夹具 50 N/m 与较硬夹具 800 N/m 各四条，均为新 reset seed，原始证据 8/8 审计通过，且与两个模型的训练 bundle 无 episode seed 或轨迹 SHA 重叠。数据 recipe 保留 3,623 个窗口；数据格式中的 fit/validation 标签只用于打包，八条全部是两模型的外部 holdout。冻结评估报告 `results/arm/compliant_keyed_insertion_recovery/validation/fixture_response_model_holdout_20261031/report.json` SHA 为 `68b281ff688cb45b07f8f92867ed83e1fad545e70dcea5ef5b011954375cf3b7`，核对原始 evidence、公共轨迹、模型与评估源码 SHA。v2→v3 的总体 3/6/10 tick x MAE 分别为 7.72→4.47、9.16→9.02、13.79→13.81 μm；真实事件窗口分别为 24.30→23.64、40.26→37.46、50.63→49.38 μm；近静止窗口为 5.40→1.78、3.38→3.73、5.55→5.85 μm。两种夹具各自也有 3 tick 改善；6/10 tick 的静止误差增加，较硬夹具的总体误差略增。因而 v3 有新初态上的短时预测增益，但长时域的权衡仍未解决。这批新采集的 8 条 episode 是预测 holdout，不能计为任务成功率、模型更新或物理子步安全证明。

ACP T1 固定单示范 512-update GPU1 诊断已完成：0/128/256/512 各一次相同训练起点闭环均 timeout；teacher-forced 平移误差均值由 3.93 mm 到 5.93 mm，末端误差由 82.9 mm 到 102.5 mm，未建立 learner 名义能力。H2S2R T2 配对诊断在相同初始模型下每分支 131,072 transitions、四个共同 monitor 点全部无名义任务成功；旧训练终止分支最终 timeout，新 reference-end 分支最终 overforce，后者跟踪误差 76.9 mm 高于旧分支 60.3 mm。它没有证明 lifecycle 修复带来学习收益，也没有证明原生 human-video 方法失败。FADA 额外完成固定上游训练 pass 的 22 个方法 AST 逐项相等和源码 SHA 核验，结果在 `results/_analysis/four_task_transfer/baseline_pilots/fada_native_pass_identity.json`；T1--T3 三个目标适配 checkpoint 各有 42/42 planner 张量逐位未变、82/82 IDM 张量更新，结果在同目录的 `fada_target_planner_freeze_identity.json`。这些是方法部分忠实性证据，脚本教师、DAgger 采集、名义能力和独立训练实例仍须分别确认。

为区分 ACP 原生 100 ms 稀疏路标本身与学习误差，在既有八条 native19 teacher 轨迹的实际访问状态上，只用完美未来标签重建同一 60-control chunk 的插值参考。八条的平均参考位置误差为 0.160 mm，最大瞬时误差为 3.081 mm；逐轨迹哈希和计算定义保存在 `results/_analysis/four_task_transfer/baseline_pilots/acp_sparse_oracle_offline.json`。这说明稀疏时间表示在已记录 teacher 轨迹上通常能近似其位置参考；它不是将完美标签闭环重放，更不能排除分布偏移、姿态／刚度误差或阶段切换导致的 ACP learner 失败。

ACP 作者仓库 `baselines/ACP/README.md` 的公开下载数据包名为 `flip_up_230.zip` 与 `vase_wiping_200.zip`；本项目当前任务适配只给 T1 提供六条 fit、两条 validation 原生动作示范，数量级显著更小，而且将原生视觉输入替换成了公共低维历史。现有 0/8 目标 pilot 因此只说明这条有限示范适配失败，不能作为强数据条件下 ACP 原方法性能的结论。主表若要把它称为强外部方法，下一步必须先声明并提供足量、任务多样的适配示范，保留与 ours 不同的监督权限和采集成本，再用两个独立训练实例及不相交 nominal/shifted pilot 检验学得策略。

CPU 只读质量审计 `results/_analysis/four_task_transfer/baseline_pilots/acp_teacher_diversity_audit.json` 按实际训练来源核对四任务 teacher manifest 的 SHA 和八条轨迹：T1--T3 使用后续 `scalar_2175` 合格集合，T4 使用其现有合格集合；每个任务均为两个 mode 各三条 fit、一条 validation，八条全合格且无 manifest 标记的 canonical duplicate。首版 `demonstrations/teacher_collection` 曾有 mode 失败，不能误认为当前训练来源。四任务各自八条示范的 goal 均相同；同 mode 成对轨迹在相同控制索引的 TCP 平均距离最大值依次仅为 1.83、1.11、5.88、0.059 mm。这些数值不等于全状态覆盖测量，但足以说明仅改变 seed 扩大现有配方不能证明示范覆盖新的目标、接触构型或机制。GPU1 完整 ACP 训练应等到新采集 recipe 在两模式、fit/validation、目标与机制维度上建立真实多样性，并保留每次失败和监督成本；八条 teacher 正对照仍不等于 learned ACP 名义能力。

作为开发试验，T1 后续在 `configs/arm/obstacle_transfer/validation/acp_native_impedance_teacher_mechanism_diversity.json` 预先指定三种单独机制变化与一种留出的组合变化；物理 GPU 未参与采集。两个 mode 的八条 teacher 示范均成功、精确回放与证据审计通过、没有 canonical duplicate。相对原名义示范，同 mode 的配对 TCP 平均路径距离中位数为 4.67/4.11 mm（mode 0/1），但目标几何仍固定。合并后 16 条示范有 12 条 fit、4 条 validation，3,260/1,104 个稀疏 native19 窗口，轨迹无重复且两 split 不相交；数据资格报告在 `results/arm/obstacle_transfer/baseline/acp/demonstrations/mechanism_diversity_sparse_dt100ms/report.json`。完整来源与路径统计追加于上述 diversity audit。

按 `configs/arm/obstacle_transfer/training/acp_mechanism_diversity_pilot.json` 在物理 GPU1 完成一次 8,000-update ACP 开发训练，耗时 337.47 s、四个固定检查点全部保存；held-out 去噪损失最低的是 6,000-update checkpoint，值 0.01235。256 个 validation 窗口的 native reference 平移重建平均误差约 7.88 mm，这不是闭环任务误差。该 checkpoint 在原 fit mode 0 与 held-out mode 1 示范 reset 上的 CPU 闭环均 timeout，最终位置误差 28.28/117.11 mm，两份证据独立审计通过；报告在 `results/arm/obstacle_transfer/baseline/acp/diagnostics/mechanism_diversity_nominal_learning_cpu/report.json`。增加这组机制示范没有建立 learner 名义能力，不能据训练损失下降放行 ACP 主表。

为定位 100 ms 稀疏路标及 12 路标执行节奏本身是否足以失败，同两条已访问 teacher 轨迹又以**完美未来标签**作了四次 CPU 闭环 oracle 对照：逐控制标签与稀疏 chunk 在两个 reset 上分别均为 2/2 成功，四份证据审计通过，保存在 `results/_analysis/four_task_transfer/baseline_pilots/acp_sparse_oracle_closed_loop/report.json`。这只排除了上述两条轨迹上“即使标签完美也因稀疏执行必然失败”的解释；未来 teacher 标签不可在线获得，不能把 oracle 计为 baseline 或泛化性能。下一步须诊断 learned 轨迹的首个偏离点、采样噪声与闭环反馈条件，再决定是否扩大更多目标几何的监督数据。

同一报告又从原始逐控制证据重算首次偏离：两条 learned 轨迹在控制步 0 已发出与 teacher 不同的 xyz 请求，且均在第 5 个控制步超过 1 mm 的 TCP 路径差；对应的完美标签稀疏 oracle 沿已记录 teacher 前缀最大偏离分别仅 4.80/2.38 mm。该诊断把失败起点定位在 learned request，而非仅在若干次 1.2 s 重规划之后才发生；不过它不区分模型训练误差、采样方差和输入条件设计各自的因果份额。主表准入前应在固定 checkpoint 上做同 reset 多采样与首个 chunk 的 teacher-forced 对照，不要仅增加示范数后重报旧试验为正式性能。

首个 chunk 的固定 checkpoint 多采样已在同一报告补做：两个原 reset 各保持公开初态不变，仅以八个预列采样种子生成第一条 native19 request，不推进任务。解码后、公共执行界限前的 xyz 速度请求与 teacher 首条请求的 L2 差，mode 0 为 0.132--0.643 m/s（中位 0.204），mode 1 为 0.098--0.726 m/s（中位 0.254）；共两次 reset、16 次网络调用、零 task step。ACP 适配器当前把预测的绝对 reference pose 与上一公共 reference 的位姿差除以 20 ms 形成速度请求，因此毫米级位置误差可能被放大为较大的初始速度，再遭公共边界截断；这是由代码和已观测首请求作出的**候选机制推断**，尚未隔离训练误差与控制器语义。修复前不能把当前失败称作 ACP 原方法的接触迁移极限。

另对 ACP 固定作者源码完成只读来源审计：主仓库与外部力编码器共 20/20 个 manifest 文件哈希吻合，90 个可按同名方法比较的 AST 中 82 个完全相同；八处差异逐项列在 `results/_analysis/four_task_transfer/baseline_pilots/acp_source_identity.json`。ACP 作者源码数值对齐的 12 项 CPU 测试通过，FADA／H2S2R 的源码与训练链合同共 21 项 CPU 测试通过。这些结果只增强移植机制可信度，不证明学习后的 ACP、FADA 或 H2S2R 具有名义闭环能力。

任务侧四格开发评估沿用公共 episode/evidence 流程，并记录完整 actor 单次动作延迟。完整训练后先在真实公共验证状态上预检全部 32 个 checkpoint 的两模式 actor 构造、有限输出与 GPU1 p50/p95/p99 延迟；通过后才冻结八个新请求并执行配对开发 cohort。独立四任务主表仍要等外部 learned baseline 的机制、名义能力与 pilot 资格闭合后再冻结。

为诊断首版响应模型的覆盖不足，CPU 又保留五组各四条完整公共轨迹：软夹具、极软夹具、普通接触 x 脉冲、越过名义横向间隙的接触脉冲及极软夹具加接触脉冲。六组共 24 条轨迹全部通过证据重审，且没有按成功筛选；后四组依据前一组诊断逐步设计，属于开发数据，不能作为预注册的独立测试。单独降低弹簧到 50 N/m 或施加受限横向脉冲都没有把最大夹具位移显著推过约 0.23--0.35 mm，公开日志显示实际 TCP/夹具相对 x 位置在脉冲时几乎不变。混合数据包已固定为 5,471 fit、5,480 validation 窗口，>100 μm 事件分别有 352／389 个；同架构、同种子、同 600-update 的 GPU1 对照顺序执行并保留两版模型。

该混合模型随后顺序完成 GPU1 训练。按完整 episode 聚合其 >10 μm 事件窗口相对零位移参照，10 tick 在 12 条验证轨迹中有 10 条改进，6 tick 为 6/12，3 tick 为 0/12；六组逐条件的同窗对照保存在 `results/arm/compliant_keyed_insertion_recovery/training/segmented_response_mixed_gpu1/coverage_contrast.json`。三成员 ensemble spread 对实际绝对误差的 Spearman 相关在各时域接近零或为负，见同目录 `uncertainty_diagnostic.json`，因此不能把 spread 当成已校准的误差上界。24 条轨迹中的后续激励是观察前一组结果后设计的开发数据；这些结果只帮助选择下一版接触激励与模型，不构成独立机制泛化估计。理论草稿据此明确区分逐时域／逐接触模式误差、事件 margin 与未经校准的 ensemble spread。

对 `validation/contact_x_probe_engaged` 四条完整公共日志再按预声明正向脉冲的 15 个控制周期重算：实际发出的 world x 请求均值为 4.75--5.31 mm/s，但 TCP 相对夹具的 x 净位移仅约 -0.8--4.0 μm，夹具 x 净位移为 0.192--0.256 mm。两者几乎同步位移是从公开观测作出的推断，说明该脉冲没有形成足够的相对接触激励；继续只增加同类脉冲幅值不能替代新的、独立声明的动态机制设计。这是对已有开发日志的只读重算，没有新物理调用或新的验证样本。

## 当前实施：方法定义对齐与动态任务／名义能力诊断

用户已授权落实 brainstorm 评审后的第 1 项，并行第 2 项。本轮范围是准确的方法定义、有限预算理论、最小被动柔顺 T3，以及已有 baseline 的名义学习诊断。Diffusion 底座、正式大表和新的外部 baseline 属于后续工作，不因本轮修改自动放行。

### 方法身份与理论对齐

通过实际 YAML 解析、SHA 绑定 checkpoint 加载和 CPU actor 构造，核验了四任务的 ours、task-only、ungated、band-only 共 16 个单元、144 个实例／mode actor。证据：`results/_analysis/four_task_transfer/method_identity/report.json`。没有 reset 或 physics，也没有把构造通过解释为行为等价或性能通过。特别是 T2 的当前 YAML 引用 `actor_bindings_mode_conditioned_contact_establishment_v2.json`，不能仅查看同目录的旧 `actor_bindings.json` 判断实际方法。

| 任务 | 实际完整方法身份 | 对部件归因的影响 |
|---|---|---|
| T1 | `ours_gated_v2`；`PublicPlannedObstacleDenseActor` | 与 task-only 同 actor 类，仍需核对训练与选择预算 |
| T2 | `ours_mode0_contact_established_transport_v5`；共享 terminal wrapper 包含接触建立 admission | transport 只改切向／姿态五通道；normalized residual radius 为 0.00625，法向、刚度、目标力由公共任务控制器持有 |
| T3 | `ours_task_first_mode_specific_dual_physical_selected_v21`；`PublicModeSpecificRecordedCompletionDenseTransportActor` | 使用 mode-specific refinement、物理 checkpoint 选择以及在线 response admission；旧 ungated 与 band-only 不是仅差一个损失项的对应版本 |
| T4 | `ours_gated_v2_completion_projected`；`PublicCompletionDominantRecoveryExecution` | task-only 仍为 liveness wrapper，不能把主表全部差异归因于 anchoring；同起点 projection 因子实验另行解释 |

当前候选的正确形式是 `Bound(base_proto + residual)`：源 chunk、源／目标历史与控制器记忆是条件输入；公共 task controller 生成残差中心。之后公共 decoder、边界与具体执行绑定决定实际请求。候选默认 H=10，控制周期 20 ms，候选时域 0.2 s；含在线 selector 的 actor 应将所有候选和模型查询计入延迟。训练 gate、离线 checkpoint 选择和在线 admission 是三个不同对象。

初始化来源也按实际工件区分。T1 的完整方法与 task-only 分别来自两次历史训练目录，当前构造审计不认证它们共享 warm/优化预算。T2 的 `policy_instances_v3/report.json` 与 `train_t2_policies.py` 记录六分支从同一个 warm checkpoint／optimizer 状态复制，并使用共同 branch sample stream。T3 的 mode-specific training identity 明确绑定已训练 task-only 的 shared-start report，随后进行重定中心、fresh optimizer、按 mode 续训和候选选择。T4 的 `completion_tail_aligned_transport/checkpoints/report.json` 明确从前一版 completion-aligned continuation 续训，而主表 task-only 是历史冻结 checkpoint。后三者各自的训练报告与样本流保留；不能把四任务笼统称为同 warm、同预算的单因素实验。

论文方法和理论已同步：population oracle 与有限数据／优化预算的 task-only learner 分开定义；响应 bridge 是额外条件，未宣称从损失形式自动得到样本复杂度优势；两 checkpoint 界限定同一分布、短时时域、proxy、执行接口和采样协议，不覆盖多 checkpoint 搜索、在线切换或 episode success。训练 `G=0` 只去掉该样本的响应项，不等于在线回退到 task-only，也不保证条件效果为零。

后续 backbone 四格须同时重建 MLP 与新底座的匹配实验；共同 warm 可以来自预训练 task-only，但两个后续分支必须从同一 checkpoint 出发且获得相同续训预算。完整 objective 与 task-only 同时相差 band 和 response，必须以 band-only、同 band 的 ungated 对照分别识别作用。旧结果按原身份保留。

### 动态 T3 与数据边界

新增任务限定为 `compliant_keyed_insertion_recovery`：单轴被动 spring-damper socket，独立 task-local config／合同，原静态 T3 默认行为不变。公共 world 观测增加 fixture pose/twist；机器人关节数组排除夹具自由度。公共几何控制器在测得的夹具相对坐标中工作，当前相对速度请求转换回 world 执行，历史请求保持原 world 语义。夹具质量、刚度、阻尼及仿真内部状态不进入策略输入。

公共轨迹采用显式 `public-target-task-data-v3`／`public-fixture-world-v1`，旧 v1/v2 不接受新字段。响应数据采用 `public-fixture-segmented-response-v1`：过去四帧 world 机器人与夹具观测，已发出的 world 请求，3/6/10 tick 的 TCP pose/twist/wrench 与夹具 position/twist 增量，控制频率力峰值单独保存。未来夹具状态只能是标签。执行、evaluator、轨迹入口和响应窗口共用单轴夹具验证，拒绝横向位移／速度与非 identity 旋转；不能用一个合法旋转矩阵绕过当前 `passive_x_v1` 合同。新 schema 不被旧 response checkpoint 静默接受，也不表示动态 response model 已训练完成。数据按完整 episode 拆分 fit/validation，所有失败保留。

CPU 首轮已完成，证据为 `results/arm/compliant_keyed_insertion_recovery/validation/cpu_qualification/report.json`。预声明的 150-control 无接触回弹检查显示弹簧阻尼能量衰减；xArm7 两种 mode × fit/validation 共四条完整公共几何轨迹为 4/4 成功、零约束违反、零 backend/evaluator 状态不一致，完整 evidence 独立审计通过。接触引起的夹具最大位移约 0.229–0.272 mm。这证明任务与数据链可执行，仍是温和开发条件，不是 learned transport 优势或三类源适用条件的性能实验。

审计进一步修复通用 IK 的机器人／夹具自由度边界：只优化和裁剪机器人关节，保留夹具初始坐标，并按机器人实际六自由度识别 UR 多起点求解。相同四条预声明请求已在 `validation/cpu_qualification_robot_ik/report.json` 完整复跑，仍为 4/4 成功、零违规、零状态不一致，控制步数和动作均不变。一个公共日志 digest 完全相同，另外三个存在至多约 4.55e-13 的观测浮点差异；`recheck_comparison.json` 保留对照，不冒称新旧 SHA 相同。

修复后四条日志经 `scripts/paper/cross_ebdm/prepare_fixture_response.py --config configs/arm/compliant_keyed_insertion_recovery/training/segmented_response_data.json` 转成 `results/arm/compliant_keyed_insertion_recovery/training/segmented_response_data/robot_ik`：910 fit、914 validation 窗口，feature 943D、request 10×13、分段响应 3×27。窗口不跨 episode，公开日志 digest 与数据包 roundtrip 已核验；无新增物理调用、无成功筛选、未训练模型。原资格报告和原数据包保留。当前接触引起的位移小于 0.8 mm clearance 与 2 mm success tolerance；下一轮须先固定有区别力的柔顺条件，不能以这四次成功宣称动态适配有效。

兼容性检查还修复五个已有静态 T3 训练／校准入口：读取后明确限定 v1/v2 和静态环境，两条 completion 数组入口也拒绝 fixture 字段，避免忽略夹具观测后静默训练旧模型；相关 22 项测试通过。动态数据及旧 public data/response 的 20 项测试通过。动态任务／接触／evidence 的 53 项测试通过，资格执行体迁回 task 层后其中 29 项复查通过，数量不重复相加。统一 runner 曾完成一条成功资格轨迹，该工件来自 IK 修复前，保留原来源，不冒称当前源码已重新运行。新 paper 命令保持薄入口；旧 completion 厚训练脚本的迁移欠项已在架构文档列明。

### Baseline 名义学习诊断

名义能力必须由学习后的策略闭环执行证明。ACP 的 32/32 是 teacher 正对照；每任务六条 fit 示范和有限 action reconstruction 不能代替 learner 的名义能力。H2S2R 的四任务 1M-transition 训练已完成，但名义 monitor 无成功；这既不能证明收敛，也不能排除训练数据／目标语义问题。FADA 继续保留已有效的 source 正对照，并区分完整原生采集流程与 task-adapted source acquisition。

本轮按预声明少量请求，对 ACP 原 fit／held-out 示范起点和 H2S2R selected／final checkpoint 进行 CPU 名义执行，结合既有学习曲线诊断；新增失败不替换。发现可证实的移植语义差异时，修复使用独立 recipe、保留旧默认和历史结果，再做有界学习对照。目标失配失败不要求修到成功；名义学习尚未成立时，不把这些失败作为优于原 SOTA 的证据。汇总证据为 `results/_analysis/four_task_transfer/baseline_pilots/nominal_learning_diagnosis_closeout.json`，其中绑定各项原始报告与 SHA。

第一批 16 条 CPU 名义执行已完成，全部失败。ACP 每任务各测试一个原 fit 和一个 held-out 示范起点：T1 两条 timeout，T2/T3/T4 两条均 overforce，因此当前问题不只发生在 held-out 泛化。H2S2R 每任务在同一新 nominal seed 比较 selected/final：T1 为 timeout/collision，T2、T3 均 timeout/timeout，T4 为 timeout/joint_limit。各任务报告位于 `results/<embodiment>/<task>/baseline/<method>/diagnostics/nominal_learning_cpu/report.json`。这些小样本定位学习链问题，不估计论文性能。

H2S2R 已核实的移植差异是 reference 生命周期：原生 force-tracking 在 reference 结束时终止且关闭 success 早停，当前适配在 reference 结束后保持最后一帧到共享任务终止。原生 tracking 本来没有 success bonus，不能将缺少 bonus 当作 bug。本轮只引入默认关闭的 reference-end termination 训练选项；共享任务成功／失败合同不变，因此只是部分生命周期对齐。新旧终止的前缀／回报比较和有限 native PPO smoke 单独记账，不据此宣称训练已修好。ACP 则使用固定单示范、固定更新预算的拟合诊断，同时报告去噪 loss、采样动作重建与真实闭环，避免把 loss 下降等同于名义任务学会。

| 诊断 | 实际运行与结果 | 可据此采取的下一步 |
|---|---|---|
| ACP T1 原 fit 示范单轨迹续训 | 固定 batch8、LR5e-5、128 updates，在 0/32/64/128 各执行一次同起点闭环，4/4 timeout。固定采样的 teacher-forced 平移误差均值约 2.86/4.21/4.34/5.01 mm，闭环终点误差 64.8/97.1/84.0/76.4 mm；耗时 127 s | 问题已在训练起点出现，应检查执行时间尺度／状态分布与局部预测的闭环偏移；此预算未证明模型容量不足或方法失效 |
| H2S2R T2 reference 终止对照 | 同 checkpoint 的前 503 步动作和奖励完全一致；旧环境继续到 1000 步，额外累计约 254.77 奖励；新设置在 reference 结束时 done，但不标成功或 task timeout | 已证实训练生命周期不等价，保留显式 opt-in 修正；不能把这项修复直接解释为性能改进 |
| H2S2R T2 同初始化配对续训 | 每分支 8 actors×16 horizon×64 epochs＝8192 transitions，0/16/32/64 epoch 各一次完整任务 monitor。epoch0/16 的模型参数与归一化逐 tensor 相同，epoch32 开始分化；两分支前三级均 timeout，末级均 axial support loss，成功 0/8；耗时 264.7 s | 最后 tracking error 降至 28.5/24.6 mm 的同时，episode 提前在约 55% reference 处支撑失败，不能称为任务收益；需要继续核对任务约束与原生 tracking 目标的关系 |
| FADA source 证据复核 | T1 保留合格旧 source，T2/T3 使用已修复 source，分别 8/8 成功；T1 曾出现的 repair 4/8 回退未采用 | 名义 source 能力已有依据，可继续方法忠实性与目标迁移比较；仍需区分 task-adapted acquisition 和上游完整采集流程 |

本轮 baseline 共 30 次新增名义诊断 episode、ACP 128 updates、H2S2R 16384 training transitions，另有 128 transitions 的 native learner 接口 smoke；没有新 shifted-target 测试，未使用 GPU。ACP 更大单示范诊断和 H2S2R 配对续训的 GPU0 recipe 已写入各自 task-local training 配置，但未启动；单次上限分别为 512 updates／1800 s 与每分支 131072 transitions／总 3600 s，均限制 4 GiB。不能因为 CPU 诊断未成功就自动扩为四任务重训。

最后审阅将物理 GPU UUID 从算法源码移回各自 recipe，运行时仍核对物理设备、`cuda:0` 和 `CUDA_VISIBLE_DEVICES=0`；两份 recipe 的 plan-only 检查通过。新增配置边界后相关测试为 25 passed、2 skipped（有意隐藏 CUDA），没有重跑训练。原 closeout 中的 9 passed／2 skipped 与 `source_files` 保留为当时收口记录，不能要求其源码哈希等于此后配置校验清理后的文件。独立复核 closeout 的 36 个工件 path/SHA 引用全部一致。

### 本轮交付后的顺序

1. 保留现有冻结结果及完整 actor 身份，先把下一轮内部消融的初始化、执行 wrapper、在线候选选择和训练／验证预算冻结为可比较的一组；论文系统表仍允许外部方法采用自己的原生控制结构。
2. 基于当前动态 T3 接口，先建立独立的柔顺条件与 whole-episode 数据划分，训练并验证分段 response model，尤其检查接触／回弹附近误差；当前四条几何轨迹仅用于链路资格。
3. 并行解决 ACP／H2S2R 名义学习问题，固定诊断预算与假设后再扩大训练；FADA 保留已有有效 source，推进尚缺的方法忠实性部分。GPU0 有空闲后才执行已经准备的有界 recipe。
4. 实现少步 diffusion transport 时，使用实际 `base_proto + residual` 接口，先验证完整采样链的近零初始化、输入梯度、有效请求及 20 ms 控制截止时间，再做 MLP／diffusion × task-only／selective 四格。Diffusion 的收益和 anchoring 的收益分别解释。
5. 四格与外部 baseline pilot 合格后再冻结独立主表；T1 保留为简单机制诊断，动态 T3 承担接触适配，T4 保持 morphology extension。推转或擦拭扩展依据缺失的机制证据选择，不同时扩建所有候选任务。

## 论文贡献与下一轮实验（2026-10-30）

### 科学目标与评判范围

完整方法相对成熟外部方法的任务表现、数据需求与执行成本是主比较；task-only 是同容量内部消融。论文不要求 transport 在每个任务、每个实例或每个请求上都超过 task-only，也不要求每项指标都最好。个别退步要保留并解释，但不采用“任何 adverse pair 即否决整篇论文”的规则。T1--T3 承担操作性能比较，T4 承担同一技能接口向移动身体扩展的证据，不承担 transport 必须额外提高成功率的义务。

系统比较和部件归因使用不同证据：完整方法可以比较各自合理训练后的最终系统；声称收益来自 response anchoring、gate 或校准时，仍需控制相应消融中的其他因素。目标成功奖励驱动的任务级 admission 会改变数据权限与算法，不因旧门失败就默认添加。下一步优先完成 baseline 和可视化，暂停为了满足旧严格门而继续增加 ours wrapper/selector。

本轮将论文故事收紧为三层。第一层是系统结果：完整 Cross-EBDM 在 T1--T3 相对成熟 learned baseline 形成一致、可解释的总体优势；允许某个方法在某个任务或某项代价指标领先，论文看预先冻结的宏平均、失败罚时、约束违反和数据/计算成本，而不是要求 ours 填满每个最优单元。T1 主要检查饱和能力是否保持，T2/T3 承担接触与恢复条件下的主要差异。第二层是机制解释：task-only 只回答 transport 的条件性增量；简单、同分布条件可以持平，响应失配、接触和恢复条件出现多数或有实际意义的收益即可，不再要求逐任务或逐请求全面超过。第三层是适用范围：xArm7、UR5e 与 Go2+Z1 的可复核成功轨迹证明同一语义接口能跨自由度和支撑形态运行；T4 只承担 morphology feasibility，不承担主性能优势。

下一版不再用“transport 必须在每个任务、每个指标都超过 task-only”作为论文放行条件。RQ1 的主结论看 T1--T3 等权宏平均 constrained success，并同时报告失败罚时、最差组、约束违反与资源；能够支持论文的结果是 ours 在预先冻结的主要聚合指标上形成领先或更好的性能--资源 Pareto 点，并在多数任务或预先说明的困难任务子集优于各强 baseline。逐任务退步和代价必须展示，但单个 adverse cell 不否定总体贡献。RQ2 用 task-only 和机制消融回答增量作用，允许 transport 在 T1 这种饱和任务上没有收益；只要 T2/T3 的预期困难条件出现可重复、实际有意义的收益、总体没有明显系统性伤害，就能支持“条件性 transport”而不是“逐状态支配”。RQ3 只要求同一语义接口在 xArm7、UR5e 和 Go2+Z1 上形成可复核成功轨迹，T4 不承担性能领先义务。

正式实验采用下面三条放行规则，避免再次退回逐格全胜的旧判据。第一，RQ1 以 T1--T3 等权宏平均为主：完整方法达到最佳或统计上不可区分的最佳组，并在至少两个任务或预先冻结的 T2/T3 困难子集形成实际有意义的优势，即可支持总体系统贡献；某个 baseline 在某项任务、时间或资源指标领先不否定结论。第二，RQ2 只要求 T2/T3 聚合或困难子集相对 task-only 呈稳定正增量、T1 保持且安全代价不恶化；它不要求每个 seed、请求或任务都获益。第三，RQ3 只要求冻结协议下的成功轨迹、约束审计和可复现视频；T4 不进入 T1--T3 的主排名。若最终只满足其中一部分，论文按证据缩小主张，不用事后改变测试集合或指标来追求“全胜”。

### 当前结果与可讲的内容

当前完整四任务表是 `results/_protocol/four_task_transfer/confirmation_v14/statistics_contract_complete/main_table.json`。它记录在线 evaluator 的结果，输入完整性检查通过；正式流水账未保存完整物理轨迹，因此不能逐条离线复算全部物理判据。这一限制不把已发生的执行自动变成无效或未运行。

| 方法/身份 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|
| Ours 完整系统 | 320/320 | 320/320 | 320/320 | 638/640 |
| DIAL-MPC task-adapted | 320/320 | 0/320 | 284/320 | 640/640 |
| Ordinary response MPC | 320/320 | 313/320 | 308/320 | 488/640 |
| Public geometry/feedback | 40/40 | 38/40 | 40/40 | 80/80 |
| FADA v14 旧 task-adapted（最终比较无效） | 0/320 | 1/320 | 0/320 | 0/640 |
| ACP v14 占位拒绝（未执行目标） | 0/320 | 0/320 | 0/320 | 0/640 |
| Human2Sim2Robot v14 占位拒绝（未执行目标） | 0/320 | 0/320 | 0/320 | 0/640 |
| Task-only（消融） | 320/320 | 317/320 | 320/320 | 447/640 |

Ours 四任务已执行轨迹均未记录物理约束违反。T1 大量方法达到成功率天花板，应作为基本能力与执行成本参照；T2 更适合检验接触稳健性；T3 适合检验恢复、完成效率及困难机制；T4 已具有形态扩展展示价值。公共几何参照很强，不能从高绝对成功率直接推出 learned method 的独特贡献。

按下一版论文的主范围只聚合 T1--T3，ours 当前相对 ordinary response MPC 的成功率宏平均高 1.98 个百分点，失败罚时归一化完成时间低 0.0291，最差组成功率高 6.25 个百分点；逐任务表现为 T1 成功率持平、T2 高 2.19 个百分点且少一次约束违反、T3 高 3.75 个百分点并将归一化完成时间降低 0.0990。相对 DIAL-MPC，T1 成功率持平，T2 高 100 个百分点，T3 高 11.25 个百分点。相对 public geometry，T1/T3 成功率持平，T2 高 5 个百分点并消除该参照的 5% 约束违反；该参照只有一个确定性身份和每任务40请求，因此这里只作描述性比较。这个模式已经支持“简单任务保持能力，接触与恢复任务受益”的候选故事。它还不能支持“SOTA 优越性”，因为 v14 的 FADA 行属于旧的 under-qualified 路线，ACP/H2S2R 是未执行目标的占位拒绝，不能把这些零分当作算法表现。

新的 baseline foundation 已经产生真实开发证据，但三者均未达到论文放行标准。FADA 的 source-preserving T1 0/30/120秒曲线均为0/8 timeout，T2/T3 当前曲线也均为0/8，T4 identity-matched pilot 为0/8 overforce；所有执行无基础设施错误，source 正对照与 evidence audit 已通过。ACP 已完成四任务 native-impedance teacher 32/32 正对照、四任务 diffusion policy 训练及 held-out native-action 数值核验；最新 native sparse target pilot 为 T1 0/8、T2 1/8、T3 0/8、T4 0/8，失败以 timeout、overforce和支撑丢失为主，执行和 evidence audit 无错误。Human2Sim2Robot 的 robot-source task adaptation 已完成 T1/T2/T3 各1M transitions：T1 target pilot 为0/8 timeout、零违规且8/8 mode match；T2为0/8 timeout、零违规但0/8 mode match；T3为0/8、2 overforce＋6 timeout且0/8 mode match。T4 的1M-transition训练已完成，所选 epoch128 的独立目标 pilot 为0/8（7 overforce、1 timeout），零执行错误。它没有作者的人体视频资产，因此不能称为原生 human-to-sim-to-robot 全流程复现。以上只能证明适配链可运行并暴露了闭环失败，仍需方法忠实性审查、独立训练实例和冻结的正式请求，才能进入论文主表。

task-only 只用于回答部件是否有增量作用。v14 的 T1--T3 宏平均成功率差为 +0.3125 个百分点，归一化完成时间差为 −0.00185；逐任务为 T1 成功率持平且稍慢，T2 多3次成功且稍快，T3 成功率持平且稍快。T3 专项困难集另有 64/64 对 62/64 和共同成功轨迹平均快 2.30% 的结果。它们足以说明 transport 没有在 T1--T3 造成系统性退化，并给出 T2/T3 条件收益线索；由于最终 actor 与旧六格消融身份尚未完全对齐，不能把这些数字写成单一 response/gate 因素的最终因果结论。T4 的 638/640 对 447/640 是很强的系统差异，但下一版只把它用于形态覆盖和补充消融，不让它主导 RQ1 的 T1--T3 性能结论。

更细的证据：

- T3 独立困难机制对照：`results/arm/keyed_insertion_recovery/validation/finite_time_mode_specific_dual_physical_selected_transport_confirmation/gpu1/report.json`，8 实例、64 对请求，完整方法 64/64、task-only 62/64，均零约束违反；62 对共同成功平均减少 13.34 controls（2.30%），7/8 实例时间方向非负。成功率 McNemar p=0.5，不能声称已显著提高成功率。时间的逐请求符号检验没有处理共享训练实例，下一步补实例/请求块层级区间，并报告含失败罚时的时间。它是 mode-specific refinement、物理 checkpoint 选择与运行时 response admission 的完整管线结果，不是 response 权重的单因素实验。该 actor 已进入 v14，因此不能把两个报告当完全独立的两次确认。
- v14 T3 六格表 gated-full 为 320/320，ungated-full 为 284/320。冻结绑定实际为 `ours_task_first_mode_specific_dual_physical_selected_v21` 对旧 `ungated_full`，因此 +11.25pp 是两条完整管线的差异；尚不能单独归因 gate。需要重新对齐后再回答 gate RQ。
- T4 同起点 2×2：`results/quadruped/mobile_keyed_insertion_recovery/validation/transport_completion_factorial/same_start_response_anchoring_gpu1/report.json`，task-only/anchoring × projection off/on 四格各 64/64。可以展示两者都能在这些支撑条件完成任务，不能说 anchoring 在这组条件带来额外成功率。这个 null 结果不阻止 T4 作为 morphology extension。
- v14 的 `transport_claim.json` 当前输出 `core_claim_supported=false`。旧自动门把 T1--T4 等权宏平均，并因 T4 相对 task-only 的两个 adverse pair 否决 stable-transport；同时 ACP/H2S2R 仍是 coverage-only 占位行，外部强方法比较也未闭合。这个 `false` 是旧严格合同的结果，不等于下一版论文贡献已经被否定。下一版把 T4 单列为形态证据，RQ1 的主聚合只覆盖 T1--T3；完整方法只需在可信外部 learned baseline 中取得总体领先或统计并列领先，并在多数任务或预先声明的困难子集显示收益。最终判定仍必须等待 FADA、ACP、H2S2R 的可信目标评估。

### 三个 Research Questions 与候选贡献

| 问题 | 所回答的贡献 | 重点任务与对比 | 还需补充 |
|---|---|---|---|
| RQ1：完整方法相对成熟方法，在任务表现、目标数据与计算成本上有什么优势？ | 干预校准与可执行分布迁移组成的完整方法 | T1--T3，FADA、ACP、H2S2R、DIAL/MPC、公共反馈；四任务都展示 | 可信 baseline 训练、目标评估、资源表；目前不能宣称已经胜过 ACP/H2S2R |
| RQ2：校准、响应匹配及选择性约束在什么条件下有用，代价是什么？ | 有条件的源响应正则化与任务/行为权衡 | 重点 T2/T3，task-only、无校准、无 gate、对应关系打乱等受控对照 | 对齐最终 actor、训练与选择预算；校准预算曲线、条件事件/模式对应、层级不确定性 |
| RQ3：同一任务语义与技能接口能否迁移到不同运动学和支撑形态？ | 跨 embodiment 的可执行性及范围 | Panda→xArm7；新增 Panda→UR5e；Panda(T3 skill)→Go2+Z1 | 真实 UR 目标实验；现有身体的 GIF/视频与可复现轨迹；T4 无需优于 task-only |

当前可写的是高完成率的系统能力、困难 T3 的条件性完整方法收益、T4 的移动形态扩展，以及 source anchoring 有适用边界。低预算优势、响应项的独立任务收益、比 ACP/H2S2R 更好以及实机 sim-to-real 尚需证据。理论保留可行支持、响应/模型误差与事件裕度的条件关系；公共执行器归因恒等式是解释工具，不单列为原创理论贡献。有限训练下的泛化/正则化收益须实测，不能从精确 task-optimal 解的性质推出或否定。

因此当前最合理的 contribution 不是“transport 在所有条件统一支配 task-only”，而是“当源任务语义仍有效、目标执行响应发生可辨识偏移时，经过校准和选择性接纳的 transport 保持简单条件的任务能力，并在接触／恢复困难条件改善成功或完成效率”。现有 T1--T3 数据支持这个故事的方向，但 task-only 因果对照和三条外部 learned baseline 尚未闭合，所以现在可以写候选叙事，不能写最终优越性结论。

### 理论应如何服务新的贡献叙事

现有理论中“真实任务成本的精确 task-only 最优解不会被额外 response 正则自动支配”这一命题应保留，因为它排除了从训练目标直接宣称普遍任务优势。但它不是论文实验要击败的对象：实际 task-only 是有限目标数据、有限更新、有限模型和冻结选择预算下得到的估计器。下一版理论应把 population optimum、有限预算 task-only learner 与 selective transport learner 明确分开，避免用一个不可实现的 oracle comparator 否定有限数据归纳偏置的可能收益。

理论主线改成三个可由实验检验的问题。第一，校准后的 source-response 在声明的任务／mode 条件下提供有用归纳偏置，是有限预算收益的候选解释；正方向 bridge 只能在误差小于裕度时给出相应的条件成本比较，不能单独推出估计误差下降或样本复杂度优势。目标数据需求必须通过 T2/T3 预算曲线、held-out response 误差和 constrained success 验证。第二，训练 selective gate 控制响应梯度，不能将 `G=0` 当成执行回退或零效果；若研究在线选择器，必须用真实执行选择事件和同一增广状态定义归因。对应实验是 mode／事件分层的 full、task-only、ungated 与 shuffled-correspondence 对照，以及错误接纳与覆盖分析。第三，可执行支持与公共 decoder/WBC 给出部署范围和约束接口，不给出 learned transport 的任务优势；T4 和 UR5e 视频验证这一范围，性能归因由 T1--T3 的受控比较完成。

需要补出的数学结果优先是有限预算 oracle inequality 或局部强凸近似下的样本复杂度比较：把 task-only 的估计误差、可迁移 response signal、false-admission 质量、校准误差与 source-bias 分成可观察或可上界的项。若数据不支持估计这些项，理论就保持定性分解，不把无法验证的全局 Lipschitz、安全或 task dominance 写成主要 theorem。最终实验若只在 T2/T3 困难条件显示收益，理论陈述也限定在相同条件集；T1 持平和 T4 不增益是预期边界，不是反例。

### 外部 baseline 的实际缺口与执行顺序

三者已有 `genedynamics/solvers/single/<method>` 原生 PyTorch 核心、task controller 接口和开发 pilot，因此工程接线并非从零开始；但论文层面的“接入”尚未完成。v14 中 ACP/H2S2R 的行是占位失败，FADA 行来自旧的 under-qualified 路线，全部不得当作 SOTA 主表证据。每条路线还要通过方法忠实性、source/known-nominal 正对照、合理训练机会、冻结 checkpoint 选择、独立目标请求和资源记账六项 admission；通过后即使目标任务失败，也把失败作为有效结果保留，不继续修到成功为止。

1. **FADA：冻结方法忠实的最终适配路线。** planner/IDM数值诊断、source正对照和T1 source-preserving 0/30/120秒曲线已经完成。下一步对照上游训练链审查oracle、DAgger、planner冻结和仅IDM目标适应是否完整保留，然后在至少两个独立训练实例上重复小规模pilot。若数值、source与机制门通过而目标仍失败，就把失败保留为方法在当前跨 embodiment 合同下的结果，不再按目标成功率继续修。
2. **ACP：先补 learner 名义闭环与数据规模诊断。** 四任务native-impedance teacher、diffusion policy、held-out native-action核验和target pilot均已完成；开发失败已不是“没有训练policy”或旧dense/sparse action-time混用。下一步审查reference/virtual pose、时变刚度标签和100ms action horizon与上游的一致性，先检查当前 learner 在 fit／held-out 示范起点是否能闭环完成，再决定有限数据规模对照；固定刚度对照用于解释adaptive compliance的作用，不替代外部主表结果。
3. **Human2Sim2Robot：完成四任务并限定复现名称。** T1/T2/T3各1M transitions和target pilot已完成，T4训练已完成；四任务名义学习能力仍待诊断。四任务结束后检查object trajectory reward、初始化、动作尺度、domain randomization和checkpoint选择，再做独立实例pilot。现有路线必须称为 `Human2Sim2Robot task-adapted (robot-source trajectory)`；原生human-video全流程依赖作者资产，获得前不能把该适配路线写成完整原生复现。

首轮 foundation 审计进一步缩小了缺口。FADA 的 checkpoint、密封 rollout 与 current actor 在 CPU 上对 T1/T2/T3 可逐步重放，最大动作误差分别为 5.96e-7、4.77e-7、9.54e-7。T4 历史 rollout 的 recipe 确实绑定同一 checkpoint SHA，但 current actor 从第0步即相差0.557；归档的训练 source snapshot 又没有保存当时的 task actor，因而旧 rollout 不能证明当前接口身份。该结论已写入 `results/_analysis/four_task_transfer/baseline_pilots/fada_t4_actor_identity_diagnosis.json`；当前 checkpoint 的35D observation、13D action、199D command、joint chart和body-relative chart均与现有runner匹配，新的8请求 identity-matched pilot 随后已执行，结果见下方当前状态。64-window 归因中，T2 的 planner/teacher-forced IDM/end-to-end normalized MSE 为 3.253/0.229/1.783，是当前最明确的模型链失败；T1 为0.422/0.006/0.015，T3为1.329/0.030/0.159，不能把三者合并成同一种失败。GPU 重放的细小数值漂移会超过原2e-6 identity阈值，因此身份判断沿用 CPU，GPU只用于训练/数值评估。

ACP 的示范管线现已检查 trajectory、executed episode、teacher labels、canonical model-data 与逐文件 SHA，并固定 fit/validation 拆分。早期 scalar-800 公共教师路线曾在 T1--T3 出现mode覆盖不足、终端误差和phase停滞；这些失败发生在公共教师通过 ACP native-19 chart重新解码之后，且全部保留。后续 native-impedance teacher 已在 T1--T4 各执行8条全新轨迹：独立重审32/32严格成功、零约束违反、两mode均为3 fit＋1 validation、零canonical duplicate、零替换，逐条exact replay和evidence audit全通过，共18,121 controls。统一审计位于`results/_analysis/four_task_transfer/baseline_pilots/acp_four_task_native_impedance_positive_controls.json`。四任务policy也已按修复后的官方100 ms action-time语义训练，GPU1 held-out native-action重建统计全部有限且严格加载checkpoint；这关闭了有限动作重建检查，但尚未证明 learner 名义闭环能力；下一步优先检查 fit／held-out 起点执行和示范规模，再决定独立实例复核。evaluator、任务阈值和既有失败保持不变。

Human2Sim2Robot 的 CPU learning-chain 审计排除了首帧不可达与参考时钟停滞：四任务初始对象跟踪误差为0.812--4.77mm，远低于0.2m停钟边界，neutral reference 均继续推进。旧 checkpoint 全部0/4，但失败不同：T1 后期退化为动作饱和与 joint limit，连续指标反而选择 epoch512；T2/T3主要 overforce；T4为 joint limit。新的1M-transition robot-source sanity 使用64 actors，在64/128/256/512/1024 epochs各保留checkpoint，并用成功、违约、参考完成度、末端跟踪和动作饱和共同选择。T1已完成全部1,048,576 transitions，wall time 3,527.65秒；epoch64/128/256/512均为0/8 success、零违规，参考完成度中位数为22.3%/38.6%/97.8%/36.7%，动作饱和为7.55%/31.9%/21.5%/25.0%。epoch1024退化为8/8 collision、33.3%参考完成度和33.9%动作饱和。冻结排序选择epoch256；所选checkpoint的独立8请求pilot仍为0/8 success、8 timeout、零约束违反、8/8 mode match、零执行错误，4,000个control/network calls全部保留且evidence audit通过。因此增加至1M transitions没有产生T1任务成功，但这只说明增加至该预算仍未成功，不能排除训练量不足或训练目标／生命周期不匹配；97.8%参考完成度仍不能写成任务成功。T3同预算训练及独立target pilot已完成，结果为0/8 success、2 overforce＋6 timeout、0/8 mode match、零执行错误。T2也已完成1,048,576 transitions，wall time 8,262.97秒；五个monitor均为0/8 success，冻结规则选择零违规且连续指标最好的epoch512。独立target pilot为0/8 timeout、零约束违反、0/8 mode match和零执行错误。T4配置通过CPU audit和plan-only后已在GPU1启动，保持原1M-transition预算、monitor点和冻结选择规则；epoch64/128/256/512均为0/8，依次表现为8次joint-limit、8次timeout、8次joint-limit和8次overforce。参考完成度中位数为89.24%/100%/96.20%/100%；epoch512末端跟踪误差降至61.0mm、动作饱和降至3.69%，但全部轨迹发生overforce。按冻结排序，当前仍由零违规的epoch128领先，训练继续到epoch1024后才最终选择。T4按历史速度估计约23,917秒，资源上限为27,000秒。H2S2R在目标 nominal simulator 中训练和监控，资源表必须记为 target-model transitions 与 monitor episodes；“零 target call”只可指零真实机器人/密封确认调用，不能写成零目标仿真数据。

截至本轮，三条路线的实际状态如下，均不能直接抄入最终主表：

- **FADA** 的T2/T3旧目标数据曲线、T4 identity-matched pilot和T1新的source-preserving 0/30/120秒曲线均已完成。T1三个预算均0/8 timeout；T2各预算均0/8且适应后主要早期overforce；T3各预算均0/8，失败在overforce与timeout之间变化；T4为0/8 overforce。全部target execution无基础设施错误并通过evidence audit。T2/T3 repaired source正对照为8/8；T1 repaired checkpoint退化为4/8，故已否决并封存其派生曲线，新的source-preserving路线保持原source checkpoint且source正对照8/8。当前缺口是上游机制逐项签字和独立训练实例，不能继续根据target成功率改路线。
- **ACP** 已修复官方100ms action-time语义，完成四任务native-impedance teacher、diffusion policy训练、validation选择和held-out native-action GPU核验。最新native sparse固定8请求结果为：T1 0/8 timeout；T2 1/8 success、6 overforce＋1 axial-support loss；T3 0/8、8 overforce；T4 0/8、7 overforce＋1 timeout。全部无执行错误且evidence audit通过，约42.5--42.9ms的单次采样统计已记录。旧的dense-label/sparse-execution混用已排除；当前缺口是上游机制逐项签字和独立训练实例。
- **Human2Sim2Robot task-adapted** 的T1/T2/T3 1M-transition sanity和独立目标pilot均已完成。T1冻结选择epoch256，独立pilot为0/8 timeout、零约束违反、8/8 mode match；T2选择epoch512，独立pilot为0/8 timeout、零约束违反、0/8 mode match；T3选择epoch256，独立pilot为0/8、2 overforce＋6 timeout、0/8 mode match。三者均零执行错误并通过evidence audit。T4已完成1,048,576 transitions，训练耗时12,816.24秒；冻结选择epoch128，独立8请求为7 overforce、1 timeout，零执行错误。原生human-video资产仍不可得，最终名称必须保留 `task-adapted (robot-source trajectory)`。

原生方法核对来源（2026-10-29）：[FADA 官方代码](https://github.com/LeCAR-Lab/FADA)、[ACP 官方代码/公开数据](https://github.com/yifan-hou/adaptive_compliance_policy)、[Human2Sim2Robot 官方流程及资产申请](https://github.com/tylerlum/human2sim2robot)。FADA 的原生训练链为 oracle→Planner-IDM DAgger→冻结 planner/适应 IDM；ACP 的核心是参考/虚拟位姿与时变刚度；H2S2R 的核心是对象轨迹与操作前初始化驱动模拟 RL。任务改编应保留这些机制并明确替换的观测/执行层。

质量检查关注机制忠实、动作/数据正确、合理训练机会、学习曲线、source/known-nominal 正对照与有效目标执行。工程代码已经接入不等于论文 baseline 已有效接入；按论文标准，三条路线目前都尚未放行。FADA 的 source-preserving T1 0/30/120秒开发曲线已在GPU1完成，三个预算均0/8 timeout；它与T2/T3/T4的既有开发失败只说明当前适配闭环没有完成任务。ACP 的四任务native-impedance正对照已32/32通过，四任务policy、validation loss 与 held-out `native_actions` GPU数值核验均已完成，最新四任务target pilot只有T2成功1/8，其余任务均0/8。H2S2R 已完成T1/T2/T3的1M-transition曲线和独立target pilot，三者均0/8；T4同预算训练已完成，独立pilot为0/8（7 overforce、1 timeout）。若方法忠实审查、正对照和合理冻结预算通过后仍然失败，就把失败作为该方法在当前跨 embodiment 合同下的真实结果，无需修到它必然成功；在这些门关闭之前，不把开发零分写入论文主表。公共评测已为ACP、FADA和H2S2R统一记录新网络调用次数及在线推理时延；FADA/H2S2R旧pilot的空时延字段保持原样，后续正式pilot使用新统计，不改写旧工件。

### 主表、消融与统计的下一版组织

- **主表 8×4**：ours、FADA、ACP、H2S2R task-adapted、DIAL-MPC、response MPC、public geometry、source prior；每个外部 baseline 参加四任务。task-only 进入单独消融表，可在主表附一行供读者定位，但不承担外部 SOTA 的角色。T4 是扩展列，不与 T1--T3 强制合成一个“全胜”门。
- **通用指标**：constrained success↑、实际物理 constraint violation↓、失败按时限封顶的 normalized completion time↓、worst-group success↑。拒绝率/执行数单列，不把拒绝当物理超限。资源副表列目标日志秒数/示范数、训练 transitions/GPU hours、在线 p50/p95 延迟/模型调用/超时率。episode wall time 不代替在线推理延迟。
- **资源公平性**：按真实 source/log/demo/model 来源比较。当前 ours、DIAL、response MPC 绑定共同 response models，主要差异是部署时在线优化；不能用硬编码权限标签判不公平。ACP 的目标示范与 H2S2R 的源轨迹来源须逐项列明。允许方法使用原生所需资源做完整系统性能—成本比较，同时对可匹配的路线另做共享日志实验；只有后者支持同信息量因果归因。
- **内部消融**：保留六格框架，先修复最终 actor 与六格身份的一致性；same warm start、训练流、训练/验证预算、公共执行器和 checkpoint 规则匹配。物理 checkpoint 选择用掉的目标开发访问也进入预算。T2/T3 做重点机制诊断，T1/T4 的零增量如实保留。
- **研究结论**：新测试前冻结主要任务、主指标和比较家族；默认 T1--T3 等权成功率宏平均为系统主比较，同时展示所有逐任务结果及风险/时间权衡。可由多数任务获益或明确机制子集获益支撑限定贡献，不要求逐请求支配。报告效应量和实例×请求块层级区间，选定比较做多重检验；不靠重复 seed、改变统计门或看过测试后挑胜出子集获得结论。T4 单列覆盖范围与完成率。

### GIF 与跨自由度扩展

当前四个成品 GIF 只渲染目标侧：T1--T3 为 xArm7，T4 为 Go2+Z1。它们对应从 Panda source skill 出发的四条成功迁移结果，但画面中没有 source Panda。还没有 Panda/Franka→UR5e 的 Cross-EBDM rollout，也没有 source/target 成对动图。不能将 xArm 轨迹换外观标为 Franka→UR。

先制作现有身体四组 GIF/MP4：T2/T3/T4 已有完整开发 `evidence.json.gz`，可按 qpos/qvel 恢复同一物理场景后离线渲染；T1 选择有完整记录的合格开发轨迹，或用冻结 actor 单独采集展示 episode。v14 正式表没有逐步全轨迹，不把这些 clips 冒称该批正式 episode 的直接录像。借用已有 `scripts/paper/mga/render_mechanism_scenes.py` 的 visual-only mesh 恢复模式，保持碰撞几何、状态、工具与任务判据一致；按真实时间同步展示 source/target、阶段及接触/支撑信息，包含成功和代表性失败。

现有目标身体的首批四组 GIF 已于 2026-10-29 完成，统一索引为 `results/_analysis/four_task_transfer/visualization/report.json`。T1 是单独采集的 development-only ours episode；T2 来自 ours predeployment qualification；T3 来自 selected-transport 专项 confirmation；T4 来自 completion-admission validation。四条所选轨迹均成功且无记录到的约束违反。渲染逐帧恢复 evidence 中的 qpos/qvel，不插值、不推进物理；官方 xArm7、Go2 和 Z1 网格仅作为 visual-only geometry 加回，每条轨迹在最多16个保存状态上验证所有 body FK，最大误差均为0。它们证明现有目标身体可制作可复核视频，不是 v14 全账本的录像，也不是 Franka→UR5e 结果。

真正 Franka Panda→UR5e 按 T1→T3→T2 推进：先冻结目标身体的公共任务可执行性，再收集 UR 响应、训练/校准 Cross-EBDM actor，执行独立目标 rollout 后渲染 source/target 成对 GIF。论文可视化的四个面板应为 T1--T3 的 Franka Panda→UR5e，以及 T4 的 Franka Panda任务语义→Go2+Z1；T4 保持移动身体与支撑形态变化，无须为凑四张 Franka→UR 动图而改变它的科学目的。

T1 的公共 UR5e 能力已经从失败探针推进到开发资格。枚举全部8个起点 IK 分支后，分支3/7能连续覆盖两条路线且无碰撞，最大 Jacobian 条件数约13.0/14.4；当前任务 workcell 绑定分支7。UR 专用的弧长参考仍使用原10秒 deadline、原 evaluator 和标准动作界。无扰动的 mode×速度×旋转配置16例为16/16成功、零违规；随后按v14的 latency/gain 与 payload 两个机制 cell、4个reset slot×2 modes运行16例，仍为16/16成功、零违规，388--398 controls完成。该结果只证明公共任务与 UR 动力学接口可执行，尚未形成仓库内正式 evidence，也不证明 Cross-EBDM transport 成功。response schema 已包含 `ur5e` body one-hot，单轨迹 renderer 也支持 UR5e；剩余关键步骤是用现有任务证据入口保存资格轨迹，建立 Panda source bank 与 UR target response/calibration，运行真正的 ours actor，再制作成对动图。T3/T2 的 UR 公共能力和 actor 都尚未资格。

### 接下来交付的工作包

| 顺序 | 工作 | 完成证据 |
|---|---|---|
| A，冻结 Research Questions | RQ1比较完整系统与强外部baseline；RQ2用task-only及最小消融解释transport何时有用；RQ3检查跨自由度和支撑形态。RQ1主聚合固定为T1--T3，T4只作morphology extension | 新协议在任何正式独立测试请求前冻结；v14 与所有现有pilot只标作development evidence；不设置逐格战胜task-only的否决门 |
| B，关闭 baseline 方法忠实性 | H2S2R T4队列已完成；逐条审查FADA的oracle→planner/IDM DAgger→冻结planner/适应IDM、ACP的reference/virtual pose＋时变刚度diffusion policy、H2S2R的object-trajectory RL和初始化。原生资产缺失时将H2S2R明确标为robot-source task adaptation | 三者各有上游commit、替换接口清单、source/known-nominal正对照、训练曲线、冻结checkpoint规则和资源账。target成功不是admission条件，基础设施错误、source失效或机制被删才是不合格 |
| C，完成四任务外部 baseline pilot | 每条baseline先用至少两个独立训练实例和不相交请求做T1--T4小规模pilot；只允许修复能由数值对齐或接口测试证明的实现错误，不按目标成功率继续挑参 | 每条路线四任务都有真实checkpoint、学习曲线、独立target execution、推理延迟和失败分解；没有placeholder、predeployment拒绝或source失效冒充算法结果。合格失败会原样进入后续正式测试 |
| D，冻结论文大表 | pilot关闭后冻结8个训练实例、机制cell、请求包、各方法原生训练／目标数据预算、主指标和统计层级；正式请求与开发请求完全不相交 | 主表方法为ours、FADA、ACP、H2S2R task-adapted、DIAL-MPC、response MPC、public geometry/source prior；task-only移到消融表；所有baseline均参加四任务，主排名只聚合T1--T3 |
| E，执行大表和资源表 | 学习方法使用同一独立实例数和公共请求包，记录所有失败 | T1--T3 报 constrained success、失败罚时 normalized completion time、constraint violation、worst-group success 和宏平均；T4 单列成功／覆盖；资源表报告目标数据、训练 transitions/GPU hours、在线 p50/p95 延迟和模型调用 |
| F，回答 RQ2 | 用最终actor对齐task-only、无response calibration、无selective gate、打乱correspondence等最小受控对照；T2/T3为主，T1作饱和检查，T4只作补充。结论看多数困难条件和实际效应，不要求每格获胜 | same warm start、数据、训练／选择预算、公共执行器和请求；报告条件收益、gate覆盖、效应量和实例×请求块区间，避免把完整管线差异归因给单一部件 |
| G，完成 RQ3 可视化 | UR5e按T1→T3→T2完成公共资格、response/calibration、ours独立rollout；同时采集对应Franka Panda source evidence；T4配对Panda keyed-insertion source skill与Go2+Z1 target | 三条Franka Panda→UR5e成对evidence GIF/MP4，加一条Panda skill→Go2+Z1；现有xArm7/Go2+Z1 target-only GIF保留为补充材料，不改外观冒充新机器人 |
| H，论文收口 | 用 RQ1--RQ3 的成立范围改理论、实验叙事和 supplementary | 主张总体或多数困难任务优势、T2/T3条件性机制收益和跨形态可执行范围；逐格失利照实展示；仿真不冒称实机 sim-to-real |

B、C、D 是主表的硬依赖。G 在 baseline admission 关闭后可与 E、F 并行，不必等待整张大表；T4 不再占用修算法的关键路径，只需保留一个可信成功轨迹和四任务统一表中的对应单元。

实现仍沿用现有 solver/frontend/backend、task controller、task-local config 与 `scripts/paper` 薄入口。通用 learning 层保持无算法专属目录；本规划不新增顶层 baselines/integrations/solvers，也不按 P 编号建立运行源码。

## 历史记录：confirmation v1 与后续开发

架构迁移和 GPU0-only confirmation 执行已经完成；“架构迁移未通过、确认集未访问、只有五个实例”等旧状态不再适用。旧多 GPU attempt 仍保存在 invalid_runs，未被重新纳入。当前 `confirmation_v1` 共 21,200 条计分记录、14,720 次物理执行、6,480 次预部署拒绝；主表 9×4、消融 6×4、学习/校准方法 8 个独立实例。确定性参照只有一份 40/80 请求；拒绝 slot 不是独立训练重复。

**执行完整性通过，不等于论文合同完整通过。** 本次发现并区分了三项问题：

1. 原 `aggregates.json` 的 worst group 只按机制聚合；冻结设计实际要求 **mechanism × requested mode × body/support**。派生统计独立重算，修正 12 个标量单元；原件和 SHA 不变。
2. 四任务冻结合同都要求实际模式等于请求模式，但当时执行的 evaluator 没有独立比较二者。正式工件缺少 observed mode 和逐步轨迹，无法离线补判。现有成功率只能称 **recorded evaluator success**，不能标为完整合同认证后的 constrained success。加入模式条件可能降低已有成功数，不能直接猜测哪些 episode 应改判失败。
3. T4 正式执行遗漏冻结的 `public_folded_reset_v1` 和 `.002` joint noise，实际使用默认零扰动。收据哈希混入 sample ID 和 requested mode；每个机制×支撑×模式组的10个公共参照请求只有一个动作哈希和一个终点签名。代码快照与正式执行源码 SHA 对齐，足以确认初态合同没有落实；缺失完整 integration 状态，不能补造原始物理收据。**T4 区间只能条件于实际重复初态，不能支持原声明的随机初态分布。**

CPU 反例已经保存：一条已有 T3 开发成功轨迹实际 mode=0，把 requested mode 改为1，冻结 evaluator 仍返回 success=True。该验证没有运行物理。当前 run 不删除、不覆盖，也不擅自更改其 invalid 标签；保留为存在明确评估限制的历史执行证据。修复后需要新版本和独立确认 cohort，不能把本轮种子调参后再当新测试。

## 产物与复现

所有下列文件位于 `results/_protocol/four_task_transfer/confirmation_v1/`：

- `statistics/report.json`：独立完整性审计、输入/输出 SHA、冻结协议来源、统计实现与限制。
- `statistics/main_table.csv` / `.json`：完整 36 格主表，四项指标、区间、请求数、实际执行数、拒绝率和实际执行条件下的约束违反率。
- `statistics/ablation_table.csv` / `.json`：完整 24 格消融。
- `statistics/contrasts.csv` / `.json`：主对比、外部参照描述性差值、2×2 因素效应与交互、gate 对比。
- `statistics/failure_groups.csv` / `.json`：机制×支撑×请求模式分组及错误、力、步数统计。
- `statistics/aggregate_corrections.json`：12 个旧 worst-group 标量的修正清单。
- `diagnostics/contact_latency.json`：T2/T3 原始证据、模式判据反例、代码/数据 SHA、未运行的开发矩阵。
- `diagnostics/support_gate.json`：T4 关节/支撑证据、四种 gate 的区别、训练/执行计数及未运行矩阵。
- `diagnostics/audit.json`：136 个独立路径/SHA 引用及12个统计输出校验通过，模式漏洞反例由主审计再次在 CPU 重现。
- `diagnostics/reset_contract.json`：新增只读审计，逐任务核对冻结 reset 与执行绑定、身份混入哈希的反例、T4 重复动作/终点旁证。原统计报告及旧审计不覆盖重写。
- `source_snapshot.zip` / `.json`：本轮修改前的实现与配置备份，保存原运行评估逻辑；当前代码不再冒用该版本的放行状态。

沿用现有入口；指定一个尚不存在的输出目录即可重现，不写回原件：

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
.runtime/baseline_tasks_torch/venv/bin/python scripts/paper/cross_ebdm/summarize_results.py \
  --confirmation results/_protocol/four_task_transfer/confirmation_v1 \
  --output /tmp/cross_ebdm_statistics_reproduction
```

报告逻辑在既有 `genedynamics/experiments/utils/metrics.py`；paper 脚本只提供参数和声明式对比。没有新增 solver、learning、integrations 或阶段命名源码。13 项合成数据 CPU 回归测试通过，覆盖交叉配对、拒绝分母、确定性身份、最差组、Holm，以及哈希/身份损坏。

## 统计定义与边界

冻结 v1→v2→v3 设计已从架构快照 `source-overlay.zip` 逐个成员 SHA 恢复。主科学对比是 B-log 下 gated full−task-only，10,000 次 hierarchical paired bootstrap。实现以训练/校准实例为最高层，在固定机制/支撑层内抽取初态块；同一初态的两个模式一起抽，所有被抽实例共享请求块抽样，避免把一份请求包当八份独立请求。seed=267100 沿用开发 rehearsal；共享请求轴及具体 Holm 家族划分是本次透明的实现澄清，不冒称冻结时已逐项枚举。

区间是 pointwise 95% empirical percentile，条件于当前机制/支撑组合。补充 exact instance sign-flip 使用完整 2^8 枚举，在每个主表/机制问题家族内对三个均值指标×任务×对比做保守 Holm。它条件于已评估请求和实例差的对称性，不是随机化试验；非线性 worst-group 只给 bootstrap 区间。不同权限的方法只报告描述性差值和区间，不做优越性检验。

新增 reset 审计进一步限制 T4：十个 nominal initial slots 并非十个随机物理初态。已保存区间仍是对原记录的可复现描述，不能解释为对冻结初态分布的泛化误差估计；不通过删重复行、改分母或补随机种子把旧运行“修成”正式证据。

全拒绝行保留完整计分分母和冻结罚分（success=0、violation=1、capped time=1），不把八个 slot 当训练实例、不报告虚构训练 CI。另列实际执行中的 violation；无执行时该项为空。全成功/全失败可能产生零宽经验区间，不是概率等于0/1的证明，也不是等效性认证。

main 的 ours/task-only 与 ablation 的 gated_full/task-only 共3,200条计分记录在动作哈希、评分和步数上一致，不能作为第二份独立证据。执行 wall_seconds 含12-worker争用，不能当控制调用 p95。

## 主表结果（已有 evaluator 评分）

T1--T3 学习/校准行每格320请求，T4为640；确定性行分别40/80。R表示全体预部署拒绝，协议得分0，实际目标执行0。

| 方法 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|
| Ours gated v2（工件名 ours_frozen_v1） | 320/320 | R | 133/320 | 285/640 |
| Task-only | 320/320 | R | 133/320 | 284/640 |
| FADA task-adapted | 0/320 | 0/320 | 0/320 | 0/640 |
| ACP task-adapted | R | R | R | R |
| H2S2R source-trajectory | R | R | R | R |
| DIAL-MPC task-adapted | R | R | 107/320 | 320/640 |
| Ordinary response MPC | 320/320 | 0/320 | 149/320 | 317/640 |
| Public geometry | 40/40 | 0/40 | 20/40 | 40/80 |
| Public source prior | R | R | 20/40 | 40/80 |

Ours 不是当前最强方法。成熟论文的 task-adapted 路线与原生系统要分清：资格拒绝不能写成目标物理失败，FADA 的失败不能推广为原论文无效。

| 配对成功率差（百分点） | T3：估计 [95% CI] | T4：估计 [95% CI] |
|---|---:|---:|
| Ours−task-only（主对比） | 0.00 [−7.19, 6.25] | +0.16 [−3.44, 3.44] |
| Ours−ordinary MPC（跨权限描述） | −5.00 [−13.75, 3.75] | −5.00 [−10.47, −0.31] |
| Gated−ungated | −1.56 [−7.81, 3.75] | −0.47 [−3.59, 2.34] |
| Matching×band 交互 | −1.25 [−4.38, 1.25] | +0.31 [−0.47, 1.41] |

主对比有利/不利配对：T3为14/14，T4为8/7，补充 Holm p均为1。没有 transport 稳定增益证据，也不能把未显著解释为完全等效。T4 相对 ordinary MPC 的描述性差值偏负，更不能继续宣称最强。

| 消融 | T3成功 | T4成功 |
|---|---:|---:|
| frozen_warm | 123/320 | 292/640 |
| task_only | 133/320 | 284/640 |
| matching_no_band | 134/320 | 286/640 |
| band_only | 141/320 | 284/640 |
| ungated_full | 138/320 | 288/640 |
| gated_full | 133/320 | 285/640 |

六格T1全部成功，T2全部拒绝。T3的latency组、T4的derated组形成共同零分层；修正后这些任务的主表/消融 worst-group 均为0。

## 三类失败：已确认什么、下一版改什么

### T2：严格接触合同与公共路线冲突，必须改绑定

目标 xArm 的公共路线仍使用 lift→横移→末端下压，源接触运动和目标空中横移却被映射到同一 phase。已有16份资格记录中，ours/task-only 各source 4/4、nominal 0/4；nominal先在130--247 ticks穿过出口，424--468 ticks才建立接触。正式 ordinary MPC 的payload组160/160未在出口前接触，latency组160/160 timeout；geometry对应各20/20，FADA320/320 overforce。

下一版在现有 task controllers 内提供严格接触公共 reference：建立接触→保持法向力→导面切向运动→出口→终点保持。所有依赖该参考/phase语义的方法共享，训练和资格合同同步。不能靠加epoch、放宽contact ratio或恢复抬起绕行来解决。

先做 source/nominal 两模式的公共能力验证。预定开发矩阵为旧/新controller×2domain×2mode×4新初态，共32 episodes；新reference需16/16满足完整模式、接触、力和时限合同。失败保留，未达标不进入目标比较。

### T3：共同位置进展失败，因果因素尚未隔离

正式 latency组是 delay3(60ms)、gain=.7、payload=.5，另组同时变成delay1、gain=.85、payload=.8，不能把组差异单独归因于delay。ours/task/MPC/DIAL各0/160 timeout；geometry/source-prior各0/20 timeout。geometry末端位置误差12.62--23.56mm，明显大于2mm阈值，而姿态已接近目标。

保留开发记录中，另一个gain/payload组合的delay3曾4/4成功，因此并无“delay3必不可行”的证据。正式工件没有phase/guard时序，当前不能确认卡在withdraw、alignment、guard latch或插入进展。

下一版先补可观测量，不先加gain或延长deadline。固定public_geometry，delay0/3×gain1/.7×payload.5/.8×2mode×4新初态，共64个开发episode。记录首次phase分歧、guard计数、reference lead、补偿量、发出/实际施加动作与轴向进展，再只选一个共享执行因素做修复。修复配对矩阵在因果诊断后另行预定义；本轮没有预先决定“增大gain有效”。

### T4：共享执行层缺少关节预测约束，优先保住余量

正式derated组中 ours/task/MPC/DIAL 各320/320 joint_limit，geometry/prior各40/40；FADA该组也320/320 joint_limit。8条保留开发轨迹进一步定位到Z1 joint4上限：derated首步裕量约.047--.063rad，nominal约.106--.118rad。轨迹仍保持四脚支撑，body height/tilt通过，不能笼统写成“支撑丢失”。具体joint4定位属于开发证据，正式工件没有足够关节时序直接逐条确认。

公共WBC仅约束torque、接触力和acceleration，没有预测joint-position/velocity约束；关节上限在执行后才报错。body gain在名义QP之后缩放力矩，小QP residual不能证明实际动力学残差小。候选arm residual在withdrawal prefix之后才可接管，部分开发derated轨迹接管前已经耗尽裕量。

下一版在现有 `envs/.../floating_contact.py` 的WBC/执行层增加基于可测q、qd的关节制动/预测裕量约束，在 `tasks/transfer/controllers` 添加共享参考governor与anti-windup。所有baseline同享；不泄露真实hidden gain，不移除derated cell，不放宽关节容差。优先验证共享prefix留下可执行的撤回空间，不能只把joint_limit换成timeout就称修复成功。

预定矩阵为原/新共享执行器×2mechanism×2support×2mode×4新初态，共64个开发episode。新执行器须保留nominal任务能力、满足全部硬约束，并在derated每种机制/模式完成完整任务；具体裕量阈值依据模型/测量误差在开发前冻结，不能从本轮confirmation挑阈值。

## Gate、模型和理论的修改边界

要严格区分训练anchoring gate、模型验证checkpoint selector、在线source-distance fallback、T4公共恢复prefix gate。

- 16个T3/T4训练实例的最终candidate均通过selector，checkpoint fallback=0/16。
- 保存的训练minibatch anchoring均值：T3为78.15%，T4为46.69%；task-compatible部分约99.98%/99.72%。这不是正式轨迹gate覆盖率。
- 正式ours的network calls等于控制步数，可排除该实现中的source-distance fallback；T4 wrapper每步先算candidate再决定是否执行，因此不能由network calls推断learned command生效比例。正式未保存common/learned计数，无法事后补出。
- 当前响应输出/任务proxy主要围绕TCP、wrench和force peak，不含joint margin、支撑状态和完整hybrid event。因此checkpoint proxy容差δ=.05不能推出真实安全、模式匹配或任务非退化。

方法下一步应在公共执行能力修好后再决定。若修改响应schema，应加入phase、guard memory、issued/applied action history及任务相关event/joint margin，先在独立开发数据验证误差和事件裕量，再训练新版本；不能仅因成功率低就扩大matching权重或增加训练预算。相同策略类中的anchoring不增加表达能力，其贡献须由同权限task-only对比的数据效率、条件事件对应或任务增益证明。

论文理论收缩到条件性模型选择：模型误差和事件裕量满足条件时可讨论响应对应；训练gate不是在线安全shield，gate关闭的局部损失不等于最终优化策略等同task-only。未知的uniform error、Lipschitz/identifiability常数只能作为假设。没有实机证据时，不把本轮仿真写成已完成sim-to-real。

新诊断要求把理论的适用对象进一步写清楚，暂不把未验证命题写进论文为已证结论：

1. **定义完整事件成功。** 成功必须包含物理终止、实际模式匹配和任务接触条件；连续TCP响应误差小并不推出离散事件顺序正确。相应命题须假设事件边界有正裕量，或显式给出事件错误项。
2. **包含控制记忆。** 延迟、issued/applied reference、phase、guard latch和积分器会使同一TCP/wrench对应不同后续行为。若仍用当前响应schema，应说明它不是充分Markov状态；不能凭一步拟合误差推出整个恢复过程对应。
3. **区分有界与可完成。** T4无关节越限但timeout表明关节约束可行不等于任务可达，更不等于混合控制器有限时间完成；需要独立的可达/进展假设，名义QP残差也不能代表隐藏gain下的动力学误差界。
4. **实验分布必须真实执行。** 风险和非退化结论所依赖的初态分布，需要物理多样性和跨方法配对证据；T4重复初态的v1结果不满足原声明的分布，不能只调整统计记号。
5. **贡献只来自共同条件下的增量。** 公共guard、接触reference和WBC修复必须给全部适用baseline；这些是共享执行能力，不能作为transport独占收益。接着用同权限task-only与新消融证明transport改善，否则保留“尚未证明”的结论。

## 本轮实际修改与开发验证

所有新行为均为显式版本或 profile；旧结果和旧 SHA 保留。

- `tasks/transfer/evaluation.py`、`guide_evaluation.py`：完整模式条件进入 constrained success，物理成功单列，unknown/ambiguous 不视为匹配；T2 同时要求严格接触。版本为 `cross-embodiment-mode-correct-v2` / `t2-strict-guide-contact-mode-correct-v3`。
- `tasks/transfer/episode.py` 与既有 env adapter：保存真实 reset qpos/qvel、排除身份字段的物理哈希、请求/施加动作、私有物理轨迹和公共 controller memory。诊断字段没有加入 actor observation。复位模式在准备前登记；执行后记录失败仍收费并保留 raw transition。
- 既有 experiment framework：通用 gzip evidence 和 SHA、真实 flat trajectory 及字段偏移；aborted 结果仍进入分母，复位异常证据持久化，禁止覆盖失败记录。`verify_results.py --evidence` 为纯 CPU 重判薄入口，不能触发物理重放。
- FADA 自身 `learning/task_training.py` 修复 T2 误用插入任务模式分类；Cross-EBDM `learning/contact_data.py` 原有 T2 分类正确，仅补已支持环境白名单。没有训练新模型或改旧 checkpoint。
- 四任务 `_base.yaml` 标记 `development_revision_pending`、confirmation 已访问。原主表配置不会把新版代码与旧 cohort 当作新正式实验运行；开发使用既有 task/validation 结构。

T2 新 reference 仅位于 `tasks/transfer/controllers/guide_contact.py`，以 `reference_version=public_guide_contact_v1` 选择。4次 CPU smoke：Panda 两模式成功（7.42/8.02秒），xArm 两模式 overforce（3.80/5.02秒，侧向峰值11.67/10.61N）。完整证据归档在 `results/arm/guided_contact_alignment/qualification/contact_reference_smoke/`。**未通过资格门，32例矩阵未启动。** 下一版候选是移动/保持阶段目标2N、终点仍5N；这是待检验方案，未偷换严格判据。

T4 在现有 `floating_contact.py` 加入可选 `joint_margin` profile（预测/制动、参考限幅、anti-windup）及显式 `folded_stance` reset。8个 smoke 请求完整保留：2构造失败、4 legacy joint_limit、2新执行器 timeout。新执行器两次750步无物理违约，但停在phase0/1、未完成撤回。证据在 `results/quadruped/mobile_keyed_insertion_recovery/qualification/joint_margin_execution_smoke/`。**未通过资格门，64例资格矩阵未启动。**

T4 smoke 的两个 old/new 配对也不能当严格同状态因果证据：新执行器影响准备阶段，同seed初始关节观测相差7.28/6.77mrad；公开 issued reference 与实际 governed reference 最大分离58.7/10.5mm。下一版必须固定准备过程，再区分 constraints-only、公开参考 governor 和身体姿态协调。公开参考不能直接替换为隐藏 delayed reference。固定底座 IK 未解出撤回、抬高底座10mm后有解仅是数值线索，不是不可行/可行的证明。

T3 通过统一 runner 完成1次独立证据链核验（seed930300、372步、完整模式成功、离线重判一致），未算入因素矩阵。64例因素矩阵采用 `validation/mechanism_factorial.yaml`，source/config SHA 在 `diagnosis_plan.json` 预先冻结：delay0/3×gain1/.7×payload.5/.8×mode0/1×新seed940300--940303；固定公共反馈、原15秒时限，CPU-only，无训练、重放、失败替换或中途修改。最终诊断见该结果目录的 `diagnosis.json`。

矩阵已完成，47/64完整模式成功、17/64 timeout、0物理约束违反，17个失败的实际模式也均正确；64条评分和账本可独立离线重判，107份冻结源码 SHA 一致。

| T3 gain / payload | 两种delay、两模式、四初态合计 | 结果 |
|---|---:|---|
| .7 / .5 | 0/16成功 | 全部timeout，phase3缓慢下插 |
| .7 / .8 | 16/16成功 | 重载在该控制结构下反而改善进展 |
| 1 / .5 | 16/16成功 | 两种delay均成功 |
| 1 / .8 | 15/16成功 | 另1例retreat latch阻塞 |

16例共同失败没有任何retreat激活；末端轴向reference lead接近−10mm上限，仍在缓慢下降，未在15秒内达到就座接触。取消delay仍为0/8，说明delay不是这个共同失败的必要条件。**优先修复对象是共享轴向控制余量/进展，不能用safe-release或延长deadline来掩盖。** gain×payload的交互不允许把“更重”直接定义为更难。

额外1例单列：delay3/gain1/payload.8/mode0/seed940302，retreat共240步，其中183步已满足safe且clear_count≥4，末尾仍retreat。它没有负向lead饱和，属于另一种guard liveness线索；新safe-settled release是否改善任务仍需单独对照，不能凭离线可释放条件宣称有效。

T3随后完成了单因素修复验证。实现只增加 `insertion_lead_z_m`，phase0--2仍为10mm，只有插入阶段比较10/15mm；可微decoder同步修改，默认值保持旧行为。源/名义门控在两个mode、两个新reset上两版本均4/4成功、零违规，逐seed完成时间完全相同。门控通过后，在共同失败的delay3/gain.7/payload.5条件上用四个新reset做配对：10mm为0/8成功、全timeout且零违规；15mm为8/8成功、零违规，mode0完成时间9.14--9.30秒，mode1为13.06--13.20秒。结果分别归档于 `validation/insertion_lead_qualification` 与 `validation/insertion_lead_response`，16份response证据均通过独立CPU重判。该证据支持“轴向进展余量解决了这一个共享失败层”，不支持transport收益，也不是实测力安全保证。另1例retreat-latch失败仍是独立问题，未与本修改捆绑。

T2也按预定只测试移动阶段2N、终点5N的版本，使用xArm两mode×两个新reset共4例。mode0为2/2严格成功，但mode1为0/2且均overforce，侧向峰值13.85--14.26N；比旧smoke的mode1峰值更差。证据归档于 `validation/contact_load_qualification` 并通过独立CPU重判。**该候选未通过资格门，不继续扫载荷，也不启动32例矩阵。** 离线分解显示，两次失败都发生在已通过mode1出口之后：第二切向轴达到0.02m/s上限，同时法向卸载请求已达到0.01m/s上限，峰值仍增长到29.67--33.89N。这否定了“共同失败只由法向目标过高造成”的简单解释，指向切向推进、法向响应与执行滞后的耦合。下一版恢复5N目标，只增加移动阶段的**公共wrench切向governor**：在预先冻结的硬限值分数带内连续缩放切向速度至零，保留法向卸载；判据、mode顺序、deadline和终点均不变。先做xArm两mode新reset小门控，全部成功且零违规后才考虑32例矩阵。具体输入SHA、post-outlet统计和决策见同目录 `diagnosis.json`。

数值配对边界也保留：每个factor cell有4个不同物理初态；delay/gain/mode配对哈希一致，payload配对的qpos/TCP最大差分别为8.882e−16rad/2.220e−16m，因此原始精确哈希不一致。全局8个hash来自4seed×2payload，不能当8个独立初态。延迟命令在FP32统一表示后逐元素相等，但原先绝对1e−7的严格检查仍标false。`diagnosis.json` 明确保存这些限制，事后精度分析没有改任务判据或原始结果，不声明所有审计项全真。

前一轮相关测试共115项通过（含已有浮动WBC物理单测）；本轮新增的接触载荷、插入lead及可微decoder对齐测试12项通过，完整method/controller套件修复版本合同后167/167通过。T3新增 `public_v4` 合同并保留旧 `public_v3`，显式纳入execution/reset profile，未覆盖旧合同。另64例因素矩阵、4例T2候选、8例T3名义门控与16例T3慢响应对照均由统一runner完整记录，所有物理任务失败均保留。依赖MJX的transfer backend测试在当前CPU运行环境中有20项因`mujoco.mjx`未安装而未能执行，与本轮控制器修改无关。原confirmation的15个输入引用与12个统计输出SHA保持不变。没有运行GPU、训练或commit。

新 reset 的 qpos/qvel 等物理哈希用来验证状态多样性，不冒称完整 integration/controller continuation receipt；下一次严格同状态 replay 实验仍须单独冻结并保存完整 carry。

## 接下来的执行顺序与放行条件

1. **已完成：CPU统计及诊断。** 原工件SHA保持，36格主表/24格消融、95%区间、分组、拒绝记账和三类诊断已落盘。论文合同仍因mode缺口不能放行。
2. **已实现模式判据与统一证据记录。** 已有开发轨迹正反例与新统一入口证据链通过；旧 confirmation 执行器不接新大表，它的旧协议现已关闭。T4 另需把 folded reset、非零扰动和独立物理哈希写入下一版冻结合同，并通过真实初态多样性及跨方法配对检查。
3. **共享能力部分放行。** T3 的15mm插入阶段lead已通过名义门与已诊断慢响应层，可进入更宽的开发资格矩阵，但仍需单独处理retreat-latch组。T2的2N候选和T4 joint-margin候选均未通过，不启动各自32/64例资格矩阵。任何后续候选保留全部失败，不复用旧 confirmation 做选择。
4. **通过公共能力门后，重新资格验证相关baseline并更新方法训练。** 不能只让ours享受新执行器。只修有证据的适配/合同bug；算法机制本身失败可以保留，不能为了“每个baseline都修成功”无界调参。源码继续属于各solver包，task绑定属于controllers。
5. **确认transport增量再冻结下一轮大实验。** 同预算/同样本/同容量比较task-only、band-only、ungated、gated；分别记录四种gate和完整模式事件。选择规则只看开发集；旧confirmation已访问，不能再用于模型选择或新独立检验。
6. **新版本的一次性确认与论文。** 冻结新代码/模型/合同/独立请求包，保留九行主表和六格消融；先完成主要成功证据，再有针对性地做数据预算、隔离资源端到端时延、body coupling和conditional transport RQ。当前结果不支持“ours最强”；RSS 2027的科学增量要由新证据决定。

下一步是修好并验证公共能力，而不是直接开始新一轮大表或训练。模式正确、物理可行、reset多样性和因果诊断必须先成立，再讨论transport的独立增量。

## CPU离线统计收尾与下一版冻结方向（2026-10-18）

本轮明确禁用CUDA，只重算统计和读取既有证据。`summarize_results.py` 在 `/tmp/cross_ebdm_statistics_closure_20261018` 重算成功：21,200条计分记录、14,720次物理执行和6,480次预部署拒绝均通过输入SHA复核；12个数值输出与当前权威统计逐字节一致。只有派生 `report.json` 不同，因为当前脚本新增了两条已由 `reset_contract.json` 确认的T4 reset限制，数值没有变化。为保持原审计链，旧报告及其SHA没有覆盖。完整收尾记录位于 `results/_protocol/four_task_transfer/confirmation_v1/diagnostics/common_failure_closure.json`。

T2的九个顺序开发候选共36例，仅8例成功。这个合计只描述搜索历史，不是一个可泛化成功率估计。28个失败正好分为14个首段移动冲击和14个outlet后冲击；28/28在违规前一控制周期仍低于25 N总力和10 N横向硬限，22/28甚至低于各自60%阈值。典型outlet样本在一个20 ms周期内由5.98 N总力、2.81 N横向力跳至25.09 N和10.82 N。现有执行在一个控制周期内运行10个2 ms物理子步，因此纯反应式wrench阈值没有足够预警。当时据此冻结了actor级公共预测接触路线；该路线及后续变体的验证已经完成且未过门，不能继续作为下一版方案。完整否证与新的2 ms共享执行层决定见下文“后续 CPU 机制闭合与修改决定”。

T3的64例因素矩阵为47/64成功、17个timeout、零物理违规。16个共同失败全部集中在gain=.7、payload=.5，与delay和mode无关；另1个是独立retreat latch。15 mm插入阶段lead在名义配对中与10 mm同为4/4，在慢响应配对中由10 mm的0/8变为8/8，八个discordant pair全部有利于15 mm；开发性双侧exact McNemar为0.0078125。该检验是在已诊断条件上做的，不能当正式确认性p值。这里冻结的全64例矩阵和独立retreat-release对照现已完成；两个正交门及组合结果见下文，并继续保留`public_v4`。

T4的legacy为0/4且全是joint limit；当前joint-margin profile有2个构造失败和2个执行timeout。两条执行轨迹把最小关节余量保在33.4/56.7 mrad、四脚支撑不丢失，却只访问phase 0/1，目标距离从约19.5 mm恶化到29.3/37.1 mm，issued/governed reference最大分离10.5/58.7 mm。这个profile同时改变准备、预测关节约束、隐藏执行器reference clamp和anti-windup，无法归因。下一版先固定并复用同一post-preparation状态，把执行器拆成legacy、predictive-bounds-only和bounds+public-reference-governor；物理约束留在`floating_contact.py`，公开进展与anti-windup放在现有task controller。若仍安全但不可完成，再通过已有`ContactCommands.body_twist`接入所有方法共享的公开body posture intent。最终门仍要求完整任务成功和零硬违规，不能把joint_limit换成timeout视为修复。

理论相应增加三条边界：采样系统的事件裕量必须覆盖控制周期内的接触增长；前向不变性与有限时间进展是两个独立前提；T4还需要在关节/支撑约束和可用body-intent通道下的受限可达性。公共reference、guard和WBC修复属于共享benchmark能力，只有公共门通过后的同权限对比才能归因于transport。

### 后续 CPU 机制闭合与修改决定

上述 T2/T3 计划已经继续按 CPU-only 开发协议执行，完整的机器可读结论位于 `results/_protocol/four_task_transfer/confirmation_v1/diagnostics/shared_failure_revision.json`。这些顺序开发结果用于选机制，不能混入正式主表，也不能把自适应搜索汇总当成功率估计。

T2 共执行了15组 actor-rate 候选、每组两个mode各4个新reset，共120例。只有 post-outlet lift 的mode 1取得4/4，其余没有严格成功；合计60个overforce、48个timeout、4个axial-support loss、3个guide-contact loss和1个coverage failure。降低法向目标、actor级一步切向投影、低刚度/各向异性刚度、接触建立后的wrench ramp、法向滤波、有界卸载和现有2 ms法向force channel均未同时得到任务完成与严格接触安全。这个结果把失败边界定位到接口时标：NumPy/JAX后端每2 ms都已经计算完整TCP wrench，但公共actor每20 ms才看到一次；当前`hybrid_wrench`在子步内只接收法向标量。20 ms actor无法对同一周期内已经发生的切向/总力冲击提供硬保证。

因此下一版不再增加`guide_contact.py`的actor子类或标量配置。T2改为一个所有方法共享的 **substep wrench governor**：在`tasks/transfer/contact.py`中放置唯一的backend-neutral纯数值函数，NumPy与JAX后端在每个2 ms子步以已经存在的完整TCP wrench和名义desired wrench调用它。它维护当前/上一子步wrench的增长包络，把切向努力投影到预先声明的横向/总力软预算；包络不可行时，在本20 ms周期剩余子步锁存耗散式切向停止和法向卸载。子步wrench、触发与干预只作为evaluator诊断，不能加入actor observation，也不能读取`axial_support`、几何接触身份或未知mechanism。该执行器要对主表全部方法一致启用，因此属于benchmark公共能力，不能算Cross-EBDM贡献。

T2的验证顺序也冻结：先给十个子步增加只读诊断并在已知mode-0冲击种子上确认guard在首次硬限越界前触发；再做纯函数的NumPy/JAX对齐、有限值、slew和预算性质测试；随后做xArm两mode×4新reset，必须8/8严格成功、零硬违规、零接触合同失败；只有通过后才扩大到Panda/xArm及所有主表actor的共同执行器资格。若无法为周期内wrench增长给出经过验证的上界，这个governor只能称经验执行门，不能在理论中称force-safety certificate。

T3已经完成两个正交门。15 mm insertion-only lead的完整64例新reset矩阵为63/64、零违规，唯一失败是预先单列的retreat latch；safe-settled release在新鲜legacy/new对照中由legacy 7/8变为new 8/8，归档条件回放也在430步（8.6秒）完成。最终将15 mm lead和safe-settled release组合后，delay×gain×payload×mode的16格、每格4个新reset全部成功，即64/64、零违规，最大总力10.3117 N、最大横向力3.3508 N，终态retreating为0。下一步把这个组合冻结为新的T3合同版本，保留`public_v4`；当前证据仍标记development，只证明公共任务机制可行，不证明transport增益。

T4结论没有被T3结果改变。下一版仍先复用完全相同的post-preparation积分状态，隔离legacy、predictive-bounds-only和bounds+public-reference-governor；只有安全但不可达时，才把已有公开`body_twist`通道作为所有方法共享能力接入。资格门要求真正完成任务、保持支撑和关节约束、显式记录public/applied reference；把joint-limit改成timeout不算修复。

理论应据此拆成四个独立命题：控制周期内wrench增长受界时的采样安全；共享执行器下的前向不变性；T3的有限时间进展与safe-settled混合liveness；T4在关节、支撑和body-intent约束下的可达性。transport只进入这些公共前提成立后的任务策略比较，不能吸收公共governor、reference或WBC修复带来的收益。

### T2 控制周期边界诊断与下一版决定

在统一runner的真实`seed=sample_id`映射下，CPU重新执行mode-0的930270与930272，并只开启evaluator-only的2 ms子步记录。930270在第175个控制周期的substep 0由零wrench直接跳到31.7367 N（横向10.8847 N），此时governor未激活；但前一个控制周期的substep 9仍处于激活状态且以零wrench结束。930272则在危险峰值发生时保持激活，最大总力19.8317 N、横向5.6546 N，最终严格成功。完整事件记录位于`diagnostics/t2_substep_boundary_pair.json`。

这说明当前7/8不是需要继续找静态阈值，而是实现把`previous_substep_wrench`、guard latch和contact dwell都局部初始化在每次20 ms `_step`内。它能覆盖同一控制周期内的分离/重碰，却会在控制周期边界丢失已经触发的事件状态。下一版把这三个量纳入可保存、可批处理、NumPy/JAX一致的`ContactState`，触发后持有固定一个完整控制周期的10个子步预算，跨`_step`递减；稳定接触dwell可以提前释放。非激活状态必须逐位返回名义wrench，诊断状态不能进入actor observation。首先增加“substep 9触发、下一tick substep 0仍激活”的边界测试，再重跑xArm两mode各四个seed；只有8/8严格成功、零硬违规、零接触失败才合并现有mode-conditioned公共binding并扩大资格。

T3不再继续调参：15 mm insertion-only lead与safe-settled release已经组合达到64/64，应冻结成保留`public_v4`的新合同。T4先做严格同一post-preparation continuation state的48例隔离矩阵：legacy、predictive-bounds-only、bounds+public-reference governor，覆盖2 mechanism×2 support×2 mode×2 fresh state。只有安全但仍timeout时才单独冻结所有方法共享的`body_twist`分支。机器可读的完整决策与源码/evidence SHA位于`diagnostics/next_revision_decision.json`。

### T2 跨周期 governor 后续诊断与回退

跨 tick 的10子步事件预算和公开方向门控已经实现并通过CPU纯函数测试，但完整任务门没有通过。本轮共执行108个CPU自适应开发episode，全部失败均保留；这些计数只用于机制选择，不是成功率估计。开发集930270--930273已被反复查看：阻尼比2时mode 0可达4/4、mode 1仅2/4；恢复原始阻尼比1并以公开issued tangent判定+x风险后，mode 1恢复4/4、mode 0仅2/4。全局中间阻尼1.5仍是6/8。材料参数只能在两个mode之间移动失败，不能作为公共任务修复。

后续只在已触发事件内修改法向阻尼也没有形成稳定工作点：120 N s/m达到开发集最好7/8，但剩余失败是0.30秒断触；90 N s/m降为4/8；按相对速度符号分段为5/8。把无支撑后的切向命令直接清零同样为4/8，并产生3个timeout和1个overforce。原因不是缺少另一个标量，而是现有`boundary_remaining`把制动、回接和释放压成同一状态；重复无支撑事件会重置预算，无法同时保证有限进展、低冲击回接和稳定释放。

因此材料已恢复原始阻尼比1，未通过的`qualified v16`类和binding已删除，没有新增validation配置。保留的实现只有跨tick有限预算、NumPy/JAX共同函数、evaluator-only诊断，以及以公开issued tangent而非合成阻抗wrench判断+x路线风险。下一版必须在现有shared contact executor内加入可保存的`nominal/brake/recontact/release`离散状态，分别持久化phase age、上一子步wrench、support dwell和release progress；先用现有子步轨迹离线冻结转换和回接能量包络，再用未使用的新cohort跑8/8开发门。930270--930289均已用于选择，不能再称独立确认。完整尝试矩阵、回退内容和下一版状态机合同位于`diagnostics/t2_governor_followup.json`。

T2当前仍未放行；T4的48例严格同continuation-state隔离矩阵也尚未开始。T3保持64/64冻结结论。GPU训练、baseline资格和论文主表继续等待T2与T4公共能力门。

### T2 有限状态 release 的 CPU 否证与联合修订

在把phase、phase age、上一子步wrench、support dwell和release progress纳入`ContactState`后，按公开issued tangent分开处理两条路线：+x使用有限brake，+y只在导面前缀内失去测得支撑时回接。耗尽开发集930270--930273上的最好结果为7/8：mode 1为4/4，mode 0为3/4，零硬力违规；唯一失败是seed 930273的一段8控制步（0.16秒）轴向支撑缺口，比冻结上限多一个20 ms采样。2 ms子步记录显示这段时间并非连续自由飞行：每个控制周期内部多次重建接触，但有限brake在同一周期内反复释放和重触发，周期末采样恰好仍可能无支撑。

随后用41个额外开发episode和2次单例子步重放检验release机制，所有失败均保留。持续release并恢复完整切向wrench时，mode 0四个样本全部overforce；把法向改成只在失联时回接仍然全部overforce，说明切向推进和法向冲击不能分开处理。把release切向wrench限幅后出现2个overforce和2个contact-loss，因为上层内部reference继续积分，被暂时抑制的位移转化为储存的阻抗能量。把reference直接贴回TCP则4/4零硬违规但全部在入口timeout；固定2 mm lead同样4/4 timeout，6 mm单例恢复可达性却把接触缺口扩大到0.48秒。把support dwell从3增加到5也只得到2/4。固定dwell、固定lead和单独wrench release均被否证，源码已回退到上述7/8开发状态；它没有qualified binding，也不能用于主表。

下一版不再把release理解成一个独立wrench缩放器，而把**允许的切向努力与内部reference增量作为同一个约束决策**。离线先从已保存的2 ms轨迹分别估计brake、free-gap、reimpact和stable-support的wrench增长及相对法向速度包络；随后构造每个子步的可行区间：一侧包含总力/横向软预算、command slew和储存阻抗能量上界，另一侧包含非零推进量与deadline可达性下界。anti-windup只累计实际放行的reference增量。若任一历史pre-impact或长gap状态上的安全上界低于进展下界，则设计在运行新物理样本前直接判不可行；这只是合同一致性检查，不是反事实成功声明。

只有该联合区间、NumPy/JAX对齐、checkpoint恢复、非激活逐位透明、reference-energy记账、有限phase时长和有限进展测试全部通过，才使用尚未访问的930290--930293做xArm两mode各4例开发门。门槛仍是8/8严格成功、零硬违规、零contact-gap失败，且不修改评估器、deadline、mode顺序或目标。机器可读的尝试矩阵、151个累计CPU开发episode记账和当前源码SHA已更新到`diagnostics/t2_governor_followup.json`。

三类共同失败的下一版处置因此明确分开：T2进入上述reference/wrench联合可行性设计；T3的15 mm insertion-only lead与safe-settled release已经组合达到64/64，下一步只冻结新合同，不再调参；T4先完成相同post-preparation continuation state上的legacy、bounds-only、bounds+public-reference隔离矩阵，再根据“安全但不可达”证据决定是否启用所有方法共享的`body_twist`。理论继续分别陈述采样事件裕量、前向不变性、有限时间进展、混合liveness和受限可达性，不能用一个transport误差界覆盖这些公共执行前提。

### 当前 CPU 离线收尾与下一版冻结决定

本节覆盖上面“尚未开始”或“仍待隔离”的旧状态，但保留旧文字作为开发过程记录。2026-10-18再次以`CUDA_VISIBLE_DEVICES=''`重跑统一统计入口到`/tmp/cross_ebdm_statistics_closeout_20261018`：21,200条计分记录、14,720次物理执行、6,480次预部署拒绝保持不变，主表、消融、对比、失败分组和汇总的12个JSON/CSV数值文件与权威工件逐字节一致。没有物理重放、训练或GPU调用。机器可读收尾位于`results/_protocol/four_task_transfer/confirmation_v1/diagnostics/shared_failure_closeout.json`。

T2已经完成最后一轮已访问开发集回归。把稳定支撑dwell从3个2 ms子步改为4个后，930270--930273两个mode共8例为8/8严格成功、零硬违规；最大总力21.6602 N、最大横向力8.0023 N、最大采样接触缺口0.14秒，终态governor全部回到nominal。这个cohort累计经历了自适应机制选择，不能当资格估计。下一版不再改机制或扫标量：只把临时runner中的mode-0路线迁入现有`guide_contact.py`，与既有mode-1 post-outlet路线在现有`bindings.py`形成一个公共binding；补齐checkpoint、非激活逐位透明、4子步release、reference-energy和有限进展测试后，再一次性运行尚未落盘的930290--930293两mode共8例。只有8/8严格成功、零力/接触失败、有限phase退出和独立证据重判全部通过，T2才放行。

T3无需继续修改控制律。15 mm insertion-only lead解决共享有限进展失败，safe-settled release解决独立retreat-latch；组合矩阵已达到64/64模式正确成功、零违规。下一版只冻结一个描述性合同、binding、源码SHA和64格证据manifest，同时保留`public_v4`及全部旧开发证据。它证明的是共享任务机制可行，仍不构成transport优势。

T4的严格同continuation-state隔离已完成48例：legacy为8/16成功和8次joint limit；bounds-only及bounds+public-reference均为0/16成功、16次timeout、零违规。这证明关节前向约束有效，但固定底座下任务不可达，公开arm reference也不能单独恢复可达性。按预定条件分支加入所有方法可见的世界平面`body_twist=[0.005,0.005,0]`、仅前6秒启用后，同16份收据为16/16成功、零违规、四脚支撑保持，最大总力19.3497 N、最大横向力4.4866 N；16份证据的独立CPU重判全部通过。T4开发能力门因此通过。最小实测关节余量只有0.000226 rad，所以理论只能陈述这些收据上的经验安全与受限可达性，不能宣称对未测失配的鲁棒不变性。

下一版的改动边界由此冻结：T2保留当前有限状态substep governor，只完成公共接口、测试和新鲜门；T3冻结15 mm lead与safe-settled release；T4冻结`joint_margin_public_reference`和公开平面body-intent，并对所有适用baseline同享。理论分别写采样事件裕量、reference能量记账、前向不变性、有限时间进展、混合liveness与受限可达性。GPU0训练、baseline资格和新大表必须继续等待T2新鲜8例门通过；公共执行器的收益不能计入Cross-EBDM transport贡献。

### T2 新鲜门失败后的 CPU 子步诊断与修订冻结

上节计划中的930290--930293新鲜门已经实际执行并永久封存，不能再次作为独立资格集。两种mode共8例只有5例严格成功：mode 1为4/4，mode 0为1/4；两个mode-0 episode因0.24秒和0.16秒严格接触缺口失败，另一个因横向峰值11.0457 N过力失败。最大总力为24.8761 N。8份证据均通过独立CPU重判，审计没有再次运行物理，因此这不是记录或评估器误差。

三个失败episode随后各进行了三次带2 ms子步记录的收费CPU诊断回放；其中一次诊断聚合因过力episode没有outlet而失败，失败本身及全部60,390个物理子步均已记账。最终回放逐项复现原outcome、步数和力峰值。930291的12采样/0.24秒缺口期间，内部x reference lead从约4.956 mm增至5.679 mm，等效x弹簧努力从10.78 N增至12.35 N。930292的8采样/0.16秒缺口只比冻结上限多一个20 ms样本，但附近无支撑运动的x lead达到约8 mm，governor在同一控制周期内反复释放和重触发。930293最终过力时公共actor已经发出零x/y速度，内部仍保留5.381 mm的x lead、约11.70 N的x弹簧努力和0.0326 J切向弹簧能量。证据位于`results/arm/guided_contact_alignment/validation/finite_state_contact_qualification/diagnosis.json`；错误的第一版gap聚合及失败诊断尝试也保留，没有覆盖或删除。

因此上一节“保留机制、只迁移接口”的判断已被新鲜门否证。根因是现有2 ms状态机只投影`desired wrench`，而50 Hz reference在进入十个物理子步前已经完整积分；phase按固定age回到nominal时也不要求当前存在稳定支撑。被wrench guard暂时拒绝的运动仍作为阻抗reference储能，随后在重接触时释放。继续增加dwell、固定毫米lead、降低actor速度或增加actor阈值都不能修复这个状态不一致。

下一版冻结为**切向reference接纳与wrench联合governor**，仍只修改现有共享执行路径。`substep_wrench_governor`下的切向reference改为按2 ms积分；每个子步把弹簧加阻尼切向努力投影到由现有0.8总力/横向软预算导出的球内，只把实际接纳的reference写回状态。brake/recontact期间不累计前向reference；没有测得支撑时不再仅因phase age而回到nominal；连续4个低风险支撑子步后进入已有`RELEASE` phase，逐步恢复切向努力。每个20 ms公共边界通过现有`reference_position`字段反馈实际接纳的reference，使所有方法看到共享执行器真实接受的运动。不会新建solver、learning、integration目录，也不会加入按阶段命名的源码树。

运行新物理前必须通过NumPy/JAX纯函数对齐、brake/recontact/release中的checkpoint恢复、关闭该profile时的逐位透明、持续失联下reference能量有界、恢复阶段不累积前向reference、连续支撑下的条件phase liveness、可接纳支撑运动的有限进展，以及公共/内部reference记账相等。930270--930273和930290--930293只用于访问集机制回归。全部访问集通过后，重新检查930294--930297未落盘，再一次性执行两个mode共8例；门槛仍是8/8严格成功、零硬力/接触违规、终态nominal且phase age为零、8/8独立证据审计通过。

理论也随证据收紧：把接纳后的reference作为控制器状态，联合约束切向储能和子步wrench；phase liveness只能在支撑重现假设下陈述。没有经过验证的plant wrench增长界和闭环储能界之前，T2仍是经验公共执行门，不能写成force-safety certificate。T3保持15 mm insertion-only lead加safe-settled release的64/64冻结结果；T4保持公开body-intent的16/16冻结结果。GPU0训练、baseline资格和四环境大表继续等待修订后的T2新鲜门通过。

### 固定低能量上限的 CPU 否证与下一版实现决定

上节提出的reference接纳方案已经在全部已访问seed上完成12轮、192个CPU episode的自适应回归，共1,586,810个物理子步。所有失败均保留，这些结果只用于机制选择。持续release、支撑条件接纳、闭环接纳和恢复记忆等路线在“安全但timeout”与“推进但overforce”之间切换，没有一条达到16/16。最后的`finite_release_soft_budget`候选保留50 Hz reference积分、有限brake、公共anti-windup反馈和温和的失联法向恢复，只在BRAKE内把存储弹簧努力压到固定2.5 N。它的mode 1为8/8成功；mode 0为0/8、全部timeout；零力/接触违规，最大总力6.8058 N、最大横向力2.7226 N。因此它证明了低能量候选可以安全，却不能满足冻结deadline。

保存轨迹给出了可量化的进度缺口。mode 0进入第一轴阶段时x约为0.38036 m，20秒结束时只有0.40231--0.40278 m，实际第一轴平均速度为1.289--1.316 mm/s。以同一候选mode 1从outlet到完成所需的4.86秒作为后半程保留时间，mode 0必须在前半程保持至少6.115--6.127 mm/s；平均需求是实际值的4.71倍。最大x reference lead只有3.65--3.75 mm。这个timeout不是偶然seed或终态phase问题，而是固定安全cap删除了任务所需的控制 authority。

T2的三种outcome现在有统一但不能互相替代的解释。接触缺口来自被wrench guard拒绝的运动仍留在reference状态；过力来自保留切向能量与无支撑法向回接在一个2 ms步内共同释放；timeout来自固定低cap破坏有限时间可达性。继续扫dwell、固定lead、固定弹簧cap或actor采样阈值只会在三者之间移动失败。

下一版冻结为**双侧切向viability governor**，只修改现有shared contact路径。保留50 Hz actor reference积分、有限brake、公共accepted-reference反馈和温和法向回接。删除固定2.5 N stored-spring cap；每个BRAKE子步把完整切向努力（弹簧加阻尼）投影到现有0.8横向软预算与扣除法向努力后的0.8总力软预算交集，再从投影后的完整努力反解实际接纳reference，同时单独限制存储弹簧能量。记录动态允许努力、阻尼努力、接纳/拒绝reference增量、存储能量、支撑和phase，不新增solver、learning、integrations或阶段命名源码目录。

安全上界之外新增公开路线/deadline进度下界。离线记录态必须保留至少6.13 mm/s的第一轴authority；若任何必需状态的安全允许区间与进度区间没有交集，直接判定当前task/controller合同不可行并重设计公开路线，不再消耗新seed。该下界是开发期admission test，不改变20秒evaluator，也不是反事实成功声明。

执行顺序固定为：先通过NumPy/JAX与checkpoint对齐、非激活逐位透明、完整切向努力软预算、存储能量、有限phase、条件support liveness和6.13 mm/s进度测试；再在已访问930270--930293上要求16/16严格成功且零违规；仅在此前提满足时确认930294--930297仍未使用，一次性运行两mode共8例，并要求8/8严格成功、零硬力/接触违规、终态nominal和8/8无物理重放证据审计。T3的64/64合同和T4的16/16公共body-intent合同保持冻结。T2通过前继续禁止GPU训练、baseline资格和四任务主表。

机器可读的完整尝试表、逐seed进度量和下一版合同位于`results/arm/guided_contact_alignment/validation/coupled_reference_admission/finite_release_soft_budget/offline_diagnosis.json`；跨T2/T3/T4决定位于`results/_protocol/four_task_transfer/confirmation_v1/diagnostics/next_revision_decision.json`。统一离线统计仍为21,200条记录、14,720次物理执行和6,480次预部署拒绝，12个数值输出逐字节一致；旧确认run依旧整体无效，不能因统计可复现而恢复资格。

### 2026-10-19 T2 物理子步证据与公共能力门收尾

本节取代上面所有“下一版待实现”的T2计划，但不删除历史失败。统一离线统计仍为21,200条计分记录、14,720次物理执行和6,480次预部署拒绝，12个数值输出逐字节一致；它们继续只是旧invalid confirmation run的可复现描述，不能恢复为论文确认结果。

T2首先暴露了评估证据的采样别名。旧严格评估器用每个20 ms控制窗口末端的`axial_support`和`normal_force`代表整窗接触；把物理步从2 ms缩至1 ms后，16例中有10例因窗口末端恰好落在接触抖动的零力相位而被判为失联。记录器现保存每个物理子步的轴向接触、法向力和TCP位置，`t2-strict-guide-contact-mode-correct-v4`在物理时间尺度上计算接触覆盖率与最长连续失联，冻结的0.5覆盖率和0.15秒上限没有修改。精确证据把该轮结果修正为15/16；唯一真实失败是mode 0 / seed 930271的303 ms飞行段。

对真实飞行段加入仅作用于向外运动的速度制动后，已访问16例达到16/16，但第一次一次性资格集930294--930297只有7/8：seed 930297在1 ms子步内产生36.5961 N约束冲击。间隙渐启虽然修复该种子，却在已访问seed 930293产生37.7193 N冲击；近表面软落地包络又让三个机制端点全部恶化到39.1420--40.8513 N。这些失败均永久保留。子步诊断显示高层恢复力只能改变冲击相位，无法用上一子步零wrench预测下一子步的接触约束冲量。

最终修订把MuJoCo导面接触时间常数从20 ms增至40 ms，阻尼比保持2，物理步保持1 ms，公共控制周期保持20 ms；该construction-only参数属于所有方法共享的T2物理单元。930270--930293的16例访问集为16/16，峰值总力9.1442 N、横向力3.1731 N、最长真实失联0.028秒；失败资格集降级后的930294--930297扩展回归为8/8，峰值总力8.3772 N、最长失联0.030秒。两组共24/24、零违规，全部证据通过零物理独立重判。

随后确认930302--930305在任何结果目录中均未出现，冻结v2资格配置并一次性执行两个mode共8例。结果为8/8严格成功、零约束违规、零mode mismatch、8/8 governor以nominal且phase age为零退出；峰值总力7.7389 N、横向力2.6593 N、最长真实失联0.021秒。8份压缩证据的离线审计全部通过。T2公共开发能力门因此通过。它证明冻结共享执行器和接触单元上的经验可行性，不是论文任务性能证据，也不是统一force-safety certificate。

T3继续冻结15 mm insertion-only lead加safe-settled release的64/64、零违规开发能力门；T4继续冻结`joint_margin_public_reference`加公开平面body intent的16/16、零违规开发能力门。三个公共能力门现均通过。机器可读总表位于`results/_protocol/four_task_transfer/confirmation_v1/diagnostics/t2_t3_t4_capability_closeout.json`。

下一步不能直接沿用旧P6主表。先把1 ms物理步、40 ms接触材料、物理子步接触证据、T3组合合同和T4公开body intent写回所有适用方法的共享协议；重跑零物理协议有效性与架构等价检查；再让每个baseline和ablation在完全相同的共享合同下通过已冻结的predeployment qualification。完成后才可在GPU0上训练，并生成全新的独立四任务实验roster。公共执行器、评估器、接触材料、路线和body-intent修复的收益不得归入Cross-EBDM transport贡献。

### 2026-10-21 CPU 离线统计最终收尾与下一版协议决定

本轮只做CPU离线计算，没有调用GPU、训练或物理仿真。统一统计入口重新生成了`statistics_v2`：21,200条记录、14,720次物理执行、6,480个预部署拒绝和60个聚合cell。第二次在`/tmp`独立推导后，13个输出文件（含报告）逐字节一致；输入SHA在分析前后均通过。统计、benchmark合同、理论、mode评估和episode证据共70项测试全部通过。`statistics_v2`补入旧报告漏写的两项限制：执行器没有持久化`observed_mode`，以及T4没有真正使用声明的folded reset与0.002 joint perturbation。因此旧confirmation仍整体无效，统计只能描述已记录分数。

旧表不能支持方法优势。T1中ours、task-only、ordinary response MPC和public geometry都为1.0成功，任务饱和。T2中ours与task-only均在部署前被拒绝，执行的公共/response参照成功率为0且违规率为0.5，FADA成功率为0且违规率为1，没有可识别的transport比较。T3中public参照为0.5、ordinary response MPC为0.465625、ours与task-only完全相同为0.415625、DIAL为0.334375。T4中public参照为0.5、ordinary response MPC为0.495312、ours为0.445312、task-only为0.44375，同时所有执行的learned row最差组成功率为0。全部冻结消融的success contrast经Holm校正后均为1.0；main ours与`gated_full`、main task-only与ablation `task_only`又是相同测量别名，不能当独立重复证据。

下一版只允许三类共享修订。T2为所有方法共同冻结1 ms physics、20 ms control、40 ms contact time constant、damping ratio 2、substep wrench governor和物理子步evaluator v4；开发门为新鲜8/8、零违规、零mode mismatch，访问回归24/24。T3为所有方法共同冻结15 mm insertion-only lead与safe-settled release；已有64/64、零违规开发门。T4为所有方法共同冻结`joint_margin_public_reference`与公开平面body intent，同时修复reset为canonical folded preparation、0.002 joint perturbation及收费50 preparation controls；已有16/16、零违规能力门只证明这些收据上的经验可行性。

协议还必须补齐三个横向缺口：所有任务持久化`observed_mode`并把`observed_mode == requested_mode`纳入成功；收据分别保存post-reset physical state、controller state与episode record哈希；FADA的T2 source loader使用导面x/y crossing语义，只有T3/T4使用insertion/load-withdrawal语义。所有main、baseline和ablation只能继承同一任务合同，不得覆盖上述共享单元。资产缺失或known-nominal资格失败仍作为预声明的predeployment rejection保留在主表，不能删行或换方法。

执行顺序冻结为：先新建而非覆盖T2/T3/T4合同版本并解除四个`_base.yaml`对旧confirmation包的绑定；随后重跑零物理协议有效性和架构等价；再对每个baseline、ablation与ours执行相同predeployment qualification并据新证据更新actor binding；全部合格后才冻结GPU设备策略并训练/校准；最后生成完全独立、未使用reset cluster的新四任务roster并只启动一次。旧seeds、schedule、outcomes和在旧confirmation上选择的checkpoint均不得复用。机器可读决定集中在`results/_protocol/four_task_transfer/confirmation_v1/diagnostics/next_revision_decision.json`，统计报告位于`results/_protocol/four_task_transfer/confirmation_v1/statistics_v2/report.json`。

理论对应收紧为五点：把共享executor/task cell与transport policy显式分解；要求采样安全允许集与deadline可达进展集交集非空；把hybrid guard liveness与有限时间进展分开；在关节、支撑、力与公开body-intent约束下陈述可达性；以物理reset cluster定义独立统计单位。共享执行器、evaluator、接触材料、路线、reset或body-intent带来的性能变化都不能计入Cross-EBDM贡献。

### 2026-10-21 共享协议写回后的 CPU 放行复核

本轮继续保持`CUDA_VISIBLE_DEVICES=''`，没有物理仿真、训练或GPU调用。统一统计入口独立重算到`/tmp/cross_ebdm_cpu_closeout_20261021`，`statistics_v2`的13个文件全部逐字节一致；完整复核与三类失败的判别方案位于`results/_protocol/four_task_transfer/diagnostics/cpu_offline_closeout_20261021.json`。旧`confirmation_v1`仍只具有诊断价值，不能恢复为论文性能证据。

T2的1 ms physics、20 ms control、40 ms contact time constant、damping ratio 2、substep governor与物理子步评估，T3的15 mm insertion-only lead与safe-settled release，T4的joint-margin执行、folded reset、0.002 rad扰动与公开平面body intent，作为下一版**共同任务/执行合同**是正确的。52个leaf的零物理协议与架构等价检查已经通过。三类后续诊断分别以子步接触/力/accepted reference、phase/guard/轴向进展、joint margin/support/body intent为最小观测量；共享接口错误只允许在共同adapter修复，经过共同合同后仍失败才归为方法机制失败或保留的predeployment rejection。

复核发现当前放行链仍少一层。`method_packages_20261021`的60行检查只验证配置继承、工件SHA、actor构造和CPU/GPU设备放置，物理步为零；它不是source/known-nominal行为资格。尤其T2的15行中仍有12行沿用旧接触单元下的拒绝，其中包括ours、task-only和全部六个消融。旧共享单元已经被判为共同根因后，这些拒绝不能未经新物理门直接继承，也不能把“包检查通过”写成“方法资格通过”。T3/T4合同也发生变化，同样需要共同的行为资格，只是旧资产缺失等与共享执行无关的拒绝可以保留并重新核验来源。

因此GPU放行顺序修正为：先在`scripts/paper/cross_ebdm`实现一个共同的行为predeployment入口；让每个可执行actor instance在同一v2 source/known-nominal门上运行并保留全部失败；据结果区分共享integration bug、方法机制失败和资源拒绝；随后只用开发数据冻结训练计划，在物理GPU1（进程内逻辑`cuda:0`）训练/校准并更新actor manifest；manifest变化后再做一次零物理架构和包检查。当前`confirmation_v2`包在这些步骤前已经生成，虽然仍是`sealed_unaccessed`，但不得启动或修改，应作为过早封存的审计工件保留；最终工件冻结后生成使用新seed的后继包。

正式执行器也尚未适配该版本：`genedynamics/tasks/transfer/confirmation.py`仍声明`four-task-transfer-confirmation-v1`，要求`frozen_after_architecture_equivalence_v1`和`zero_physics_preflight`。下一版必须改为包版本感知的admission，验证`protocol_validity`、architecture、行为predeployment、训练、GPU1 calibration和最终manifest，强制`CUDA_VISIBLE_DEVICES=1`且进程内仅见`cuda:0`，并审计分离的physical/controller/episode reset哈希。完成这些条件前保持GPU训练和正式roster关闭。

### 2026-10-21 CPU 行为资格收尾与 GPU1 放行

第一版行为资格错误地要求所有目标actor在Panda源场景上执行。FADA按其声明语义拒绝了这种source/target混用，因此该运行的720条记录已整体封存为invalid，复用数为零。修订后的v3把两个问题分开：source/training数据只通过actor manifest及其全部工件SHA验证；共同物理行为只在各任务声明的目标机器人known-nominal场景上比较。v3包含60个逻辑行，其中40行可执行、20行保留明确的工件或资源拒绝，共1112个CPU episode；每个任务使用2个新reset槽和两个mode，没有打开confirmation状态。

v3完成1112/1112。35行通过行为门，5行被行为拒绝，8行等待T2新训练工件，8行当前协议工件不可用，4行保留Human2Sim2Robot外部资源拒绝。四个`public_geometry`任务sentinel共16/16通过且没有执行异常，所以共同T1--T4任务单元允许进入GPU训练。全部1112份压缩episode证据已由第二条CPU路径逐一重算SHA并离线重判，问题数为零。新零物理审计覆盖52个leaf，protocol validity与architecture equivalence均通过、物理步数为零；T2 manifest已绑定通过资格的`public_finite_state_guide_contact_v2`公共参照。

可执行方法只出现三类失败。第一类是74个有限时间进展失败：T1 FADA 32个、T2 FADA 2个、T2 response MPC 11个、T3 FADA 29个。诊断读取phase占用、accepted task-space进展、issued action和终态距离；保持共同deadline不变。第二类是75个约束控制失败：T2 FADA 30个、T2 response MPC 10个、T3 FADA 3个、T4 FADA 32个。诊断比较最后安全子步、首次违规子步、切向/法向命令、accepted reference和governor phase；公共参照在相同cell通过，因此不能按方法放宽共同力限。第三类是T2 response MPC的11个导面接触合同失败，全部发生在mode 1且力很低；下一步只检查该方法的x/y crossing source语义、响应预测和outlet前支撑覆盖。只有故障能在公共sentinel复现时才允许修改共同任务单元。

下一版保持v3资格协议，不再修改T2/T3/T4共同合同。GPU1只训练T2缺失的八实例工件：main Cross-EBDM、main task-only及六个消融alias；已经行为拒绝的baseline保留拒绝，缺资产路线继续记入主表。训练后更新T2 actor manifest，再重跑零物理manifest/package/device检查和新训练行的v3行为门。最终manifest冻结后才生成使用全新reset cluster的后继confirmation包；过早生成但未访问的`confirmation_v2`继续封存，旧invalid run与本轮开发资格都不得进入论文性能统计。

机器可读收尾位于`results/_protocol/four_task_transfer/diagnostics/cpu_behavioral_closeout_20261021.json`；行为报告与独立审计位于`results/_qualification/four_task_transfer/behavioral_predeployment_20261021_v3/`；新零物理审计位于`results/_architecture/cross_ebdm_shared_protocol_20261021_v3/`。
### 2026-10-21 T2 task-owned transport and fresh-seed qualification

The first GPU1 policy run was not admissible.  Its learned actors still used the obsolete lift/corridor base: that base reproduced the exact mode split of the failure on the frozen four-request development set (mode 0: 2/2 overforce; mode 1: 2/2 contact established only after the outlet), while the frozen finite-state public route was 4/4.  The v1 transport also modified all 13 request coordinates, including the normal contact command, impedance log-scales, and force target.  The machine-readable diagnosis is `results/_protocol/four_task_transfer/diagnostics/t2_policy_interface_failure_20261021.json`.

The learned and response-MPC wrappers now use the same mode-specific finite-state base as the public sentinel.  T2 transport owns only coordinates 0, 1, 3, 4, and 5; the task cell owns normal velocity, six impedance coordinates, and normal force.  A zero-residual actor is bitwise equal to the public base and passed 4/4.  Eight v2 policy instances were trained on physical GPU1, but full-radius policies achieved only 23--28/32 across ablations and ours achieved 24/32.  Evidence showed that the normalized 0.025 radius produced a 0.002 m/s tangent change, large relative to the public route's 0.004--0.0075 m/s commands.

A fixed development sweep selected the largest tested common radius with 32/32 success: execution scale 0.25, giving an effective normalized radius 0.00625.  These development resets are not qualification evidence.  Eight v3 instances were retrained for 96,000 total GPU1 updates; all 48 selected checkpoints passed an independent hash/version/radius audit.  On a disjoint seed anchor (2921000), fresh T2 qualification produced: ours 32/32, task-only 32/32, gated-full 32/32, matching-no-band 32/32, public geometry 4/4, frozen-warm 30/32, band-only 31/32, ungated-full 31/32, response MPC 21/32, and FADA 0/32.  All 324 evidence files passed independent re-audit.  The post-v3 zero-physics audit passed 52 configurations with zero physics steps.

This establishes T2 predeployment viability for the main method, but not a transport performance advantage: ours, task-only, gated-full, and matching-no-band are indistinguishable on the small nominal qualification set.  Formal evaluation remains closed until a complete fresh-seed four-task qualification, package/placement recheck, and a new independent confirmation roster are sealed.

### 2026-10-21 全四任务资格重跑与离线失败收尾

最终manifest下的四任务行为资格已从冻结plan完整执行，共60个逻辑行和1368个CPU物理episode；运行没有打开confirmation状态，也没有读取旧invalid outcome。40行通过、8行按实际行为拒绝、8行因当前协议工件不可用而未执行、4行保留Human2Sim2Robot外部资源拒绝。全部1368份压缩证据由第二条CPU路径逐份重算SHA和evaluator，1368/1368通过、零问题。报告位于`results/_qualification/four_task_transfer/behavioral_predeployment_20261021_v7/`，机器可读离线收尾位于`results/_protocol/four_task_transfer/diagnostics/fresh_predeployment_offline_closeout_20261021.json`。

公共合同没有出现跨方法回归。T1中ours、task-only、response MPC、public geometry和六个消融全部通过；T2中ours、task-only、gated-full、matching-no-band均32/32，public geometry为4/4；T3和T4除FADA外的所有可执行main方法、公共参照和消融均通过。T2的旧接触合同失败在当前版本为0，说明有限状态公共路线和task-owned坐标修复已消除旧接口错误。共同T2/T3/T4合同保持冻结，不再修改力限、deadline、mode、route、reset或body intent。

当前143个行为失败分成两类，第三类作为回归门为零。有限时间进展失败64个：T1 FADA 32、T2 FADA 2、T3 FADA 30。约束控制失败79个：T2 FADA 30、T2 response MPC 11、T2 frozen-warm 2、band-only 1、ungated-full 1、T3 FADA 2、T4 FADA 32。接触合同失败0个。FADA与response MPC下一步先补方法侧诊断：FADA记录teacher-forced误差到闭环首次分叉，response MPC记录预测力裕量、优化前后代价、投影前后动作和首次违规子步。只有source fixture、归一化、坐标语义或adapter能复现错误时才修实现；若这些检查通过且公共sentinel通过同一cell，则保留行为拒绝，不为baseline放宽共同任务。

该资格集只用于admission，不能支持transport性能优势：T1饱和；T2中ours与强task-only同为32/32；T3/T4在名义资格cell中同样持平。下一步先将零物理审计与最终资格SHA绑定，再升级正式runner的版本化admission。ACP、T1/T2 DIAL和T1/T2 public-source-prior的`current_protocol_artifact_unavailable`是实现/可用性缺口，不能伪装成方法失败计零；它们必须在正式roster前成为真实实现，或作为明确的非性能可用性条目保留。之后才能生成与所有开发/资格seed不相交的新roster并运行主表。

理论相应增加一个明确条件：沿可达轨迹，安全允许动作集合必须与在deadline内产生正进展的动作集合有非空交集。公开body intent属于共同executor能力；采样接触安全、hybrid guard liveness、前向不变性、受限可达性和有限时间完成继续分别陈述。Cross-EBDM的贡献只能来自共同executor通过后、相同权限和配对reset下ours相对task-only及冻结消融的增量证据。

### 2026-10-21 最终资格的 CPU 统计收尾与方法侧诊断冻结

本轮只读取已经封存的`behavioral_predeployment_20261021_v10`，未打开confirmation状态、未运行新物理、未训练，也未读取正在执行的部分结果作性能比较。独立审计已覆盖全部1,440份资格证据。CPU入口`scripts/paper/cross_ebdm/analyze_predeployment.py`生成60行统计、逐模式Wilson描述区间、失败cell和机器可读诊断计划；输出位于`results/_analysis/four_task_transfer/predeployment_20261021_v10/`。这些区间只描述admission样本，不能作为论文性能置信区间。

最终资格共有1,257次成功、104次timeout、79次约束违规，基础设施错误为零。41行通过行为门，11行按行为拒绝，ACP四行保留算法级预部署拒绝，Human2Sim2Robot四行保留外部资源拒绝。四任务`public_geometry`共同sentinel为16/16，当前T2导面接触合同失败为0，因此没有证据支持再次修改共享力限、deadline、mode、route、reset、T3 lead或T4 body intent。

三类失败采用互斥的主结果口径并保留一个零事件回归门：

- **有限时间进展失败：104次。** T1 FADA 32、T1 public-source-prior 4、T2 DIAL 32、T2 FADA 2、T2 public-source-prior 4、T3 FADA 30。下一版读取phase/stage占用、issued/applied action、accepted reference进展、终态误差和hold/mode状态；不延长共同deadline。
- **约束控制失败：79次。** T2 FADA 30、ordinary response MPC 11、frozen-warm 2、band-only 1、ungated-full 1、T3 FADA 2、T4 FADA 32。下一版记录最后安全/首次违规物理子步、预测与真实总力/横向力裕量、投影前后请求、accepted reference、joint margin和support；不按方法放宽共同约束。
- **导面接触合同失败回归门：0次。** 继续记录首次接触/出口子步、出口前支撑覆盖、mode crossing语义和task-owned坐标相等性。任何回归先定位adapter；只有公共sentinel在相同cell复现时才允许重审共享单元。

静态CPU接口审计确认两个实现缺口。T2 learned dense与ordinary response MPC都保留冻结的task-owned坐标，但新接入的DIAL和public-source-prior直接发出完整13维候选，绕过了同一投影，可改写法向速度、六个阻抗坐标和法向力。这违反T2 actor manifest声明的共同接口，属于baseline adapter缺陷，不能解释为DIAL或直接source prior机制本身失败。下一版先让两者复用现有task-owned projector；DIAL/source-prior仍只拥有切向平移和旋转提议，任务单元继续拥有法向、阻抗和力。

FADA现有资格证据只保存网络调用计数，尚不能区分源码复现错误与闭环机制失败。下一版复用`solvers/single/fada/learning/source_training.py`的离线诊断，对四任务全部冻结实例保存逐phase teacher-forced action误差、Box越界、归一化和decoder检查；目标开发回放再保存teacher/predicted action及首次闭环分叉。只有source fixture、normalization、decoder或adapter错误可复现时才修实现；若source检查通过而autoregressive闭环仍分叉，保留FADA预部署拒绝作为方法限制。

修改后的CPU验证顺序固定如下。第一步增加DIAL/source-prior两模式的零物理task-owned逐位相等检查。第二步使用四个全新开发请求，对两种方法分别比较旧/投影实现，共16个CPU episode；修订实现每种方法必须4/4成功且零违规。第三步完成全部FADA离线source fixture与首次分叉审计。第四步对ordinary response MPC和DIAL补目标函数前后值、预测力裕量、候选/reference差值与首次违规子步，并在四个全新T2请求上与公共base配对。第五步只检查T1 direct source prior的单位、坐标系、mode和Box解码；接口正确后的直接迁移失败保留为no-target-learning baseline结果，不允许用目标数据拟合它。

上述审计是在`confirmation_v5`执行期间发现的，因此该package不能成为最终论文roster。专属进程组已停止，GPU1已经释放；部分run保留8,508/21,200条记录，其中5,929条目标物理执行、2,579条预部署拒绝。run和package均写入invalid seal，全部记录与200个seed复用数为零，不做选择性回放。下一版只有在adapter修复、完整零物理检查、全部方法predeployment重新通过后，才生成完全不相交的新package和roster。

该结论不要求修改ours，也不允许修改共同任务合同。论文理论继续把共享executor的可行性与transport增量分开；当前新发现进一步要求方法比较中的action interface一致。最终transport贡献仍只能由相同task-owned权限、相同公共executor和配对reset下，ours相对task-only及冻结消融的正式新cohort增量支持。

### 2026-10-21 v11 CPU 离线收尾、三类失败与风险模型决定

修复后的最终资格版本为`behavioral_predeployment_20261021_v11`。1,440个CPU资格episode由独立路径全部重判，得到1,241次成功、102次timeout、97次约束违规、0次导面接触合同失败和0次基础设施错误；36行行为合格、16行行为拒绝，另有4行ACP算法级拒绝和4行Human2Sim2Robot外部资源拒绝。四任务public-geometry sentinel为16/16，learned dense、ordinary response MPC、DIAL和public source prior的T2 task-owned接口静态检查全部通过。因此不再修改共同T2/T3/T4合同，也不把剩余baseline失败重新解释为共享执行器问题。CPU派生统计位于`results/_analysis/four_task_transfer/predeployment_20261021_v11_v2/`；误带v10修复前决策文字的第一份v11派生目录已显式封存为invalid，原始v11资格结果没有变化。

三类失败的处理已经冻结。102次有限时间进展失败继续做方法侧phase、issued/applied action、accepted-reference和终态误差诊断；只有source fixture、归一化、decoder、objective或adapter错误可以修实现，接口正确后的闭环停滞保留为方法限制。97次约束控制失败使用最后安全/首次违规1 ms子步、预测/实际力裕量和reference记账诊断；其中20次来自T2的ours、task-only和六个消融，说明它不是单一checkpoint偶然失败。导面接触合同当前为0，只保留回归门，不改变route、force limit、deadline或evaluator。

针对learned-family的T2过力，CPU已否证三条简单路线。固定执行半径在384个learned episode中每个尺度都失败，最小半径仍有10次overforce；四种固定坐标scope也没有在保留任务平移能力的同时消除失败；现有50 Hz response head对6次真实overforce提前检出为0，实际峰值28.6--33.4 N时预测最大值约4.5--5.9 N。因此下一版不再缩半径、删平移坐标或复用旧response force head，而增加独立的public substep-risk ensemble：输入公共历史和十步候选请求，标签为每个1 ms物理子步上的总力/横向力最大值，按episode block做单侧校准，并在`[1, .5, .25, 0]`中选择最大安全插值；零插值必须逐位等于已资格化公共base。

新采集的588个开发episode全部审计通过，但原先“每个实例至少20个fit高风险窗口”的门忠实失败，不能假装成八个独立风险数据集。风险是共享接触物理量，因此只为原型冻结pooled split：fit/calibration/validation分别有276/44/133个高风险窗口，物理初态槽为0--3/4/5且互不重叠；索引已独立逐episode重建通过。该自适应开发数据只能验证架构，不能静默成为论文最终checkpoint。正式确认前必须冻结并公开等预算的source/known-nominal训练与校准方案，或者把它明确记为共享离线预训练并完整报告数据预算。

下一版的因果对比也随之调整。transport增量必须比较相同substep-risk admission下的ours与task-only；风险门本身需要一个保持相同transport checkpoint的`no_substep_admission`消融，不能偷换现有anchoring-gate消融的含义。外部baseline保持作者机制和已核验adapter，不能接上ours的新risk head后仍称原方法。GPU1下一步只做风险模型原型训练与held-out校准；通过episode-block coverage、高风险召回、安全样本拒绝率、零fallback身份和公共sentinel门后，才接入现有T2 actor、重跑受影响行的CPU资格以及完整零物理/package/device审计。完整机器可读决定位于`results/_protocol/four_task_transfer/diagnostics/cpu_failure_diagnosis_and_risk_revision_20261021.json`。

理论同时收紧为多时标条件命题：共享task/executor、candidate transport和risk admission三层分开；风险定义在20 ms actor horizon内的1 ms峰值；episode-block conformal只在校准与部署block可交换时给边际覆盖，不能称普适force-safety certificate；零插值只保证回到公共base，不保证deadline内完成；沿可达轨迹还必须假设安全允许集合与正进展集合相交。transport性能贡献最终只由独立新roster上、相同risk admission与相同公共executor下相对task-only及冻结消融的增量支持。

### 2026-10-22 T2 在线风险接纳的 CPU 收尾与下一版修改

GPU1上的风险模型原型经过新的独立cohort后离线8/8合格；这只建立了记录指令分布上的episode-block coverage、高风险召回和拒绝率。随后CPU在线开发验证完整执行264个新episode并独立重审264/264份证据。结果忠实失败：risk-admitted ours为64/64、task-only为63/64且有一次33.52 N overforce，未加风险接纳的ours与task-only分别为57/64和62/64；新的public-geometry sentinel为7/8，也有一次31.94 N overforce。该验证没有打开confirmation状态，不能作为论文性能结果。

失败不是风险模型漏报。当前selector把最后一个零插值候选解释为公共base，并通过`safe[..., -1] = True`无条件覆盖模型判断。全部128个risk-admitted episode都出现过模型将公共base也判为超过20 N/8 N软限的状态；ours共有26,136个这种控制tick、task-only有25,412个，selector仍分别强制执行其中26,135和25,412个。task-only失败从tick 142起公共base预测上界已经超限，直到tick 199仍执行base；违规tick的预测上界为173.88 N/26.06 N，实际峰值为33.52 N/7.54 N。大量相同强制fallback仍成功，又说明模型在在线候选分布上明显过度拒绝。因而这不是单独缩小miscoverage或增加校准修正能解决的问题：当前离线数据来自记录行为指令，不能自动覆盖闭环中由transport生成的插值候选。

public sentinel与task-only失败具有同一个共享模式：都是mode-0、stage-2导面移动，x方向指令为0.0075 m/s，并在前一控制tick约4 N后出现单个物理子步的30 N以上冲击。公共sentinel失败重新打开的是共享mode-0执行器资格，而不是力限、deadline、mode、reset或evaluator。下一版先用公共状态设计并限定一个stage-2 reference brake/settling动作；任何共享修复必须同时用于全部方法和public sentinel，不能计为Cross-EBDM贡献。先在完全不相交的新sentinel cohort上通过，才允许重新资格化learned方法。

risk admission v2同时做四项结构修改。第一，删除末候选强制安全标记。第二，把有限的emergency reference-brake horizon作为显式候选并经过同一个风险模型；若接纳集合仍为空，返回`risk_set_empty`并按任务失败计数，不能继续执行被模型拒绝的动作。第三，训练、校准和验证数据必须由冻结后的部署候选生成器采集，覆盖transport插值、公共base和emergency horizon；只在旧行为策略指令上合格不够。第四，资格门同时检查episode-block coverage、高风险召回、安全窗口拒绝率、空集率、有限时间进展和新闭环cohort的约束违规，防止以全部停止换取表面安全。

当前actor不进入manifest，也不启动主表。放行顺序固定为：共享执行器诊断与新sentinel资格；risk-admission-v2零物理身份/空集检查；候选分布匹配的fit/calibration/held-out资格；同risk gate的ours/task-only新配对在线资格；完整零物理、package、device和行为predeployment；最后才生成四任务全新独立roster。外部baseline只共享公共executor，不能接上ours的risk head后仍沿用原方法名；risk gate的增量由新的`no_substep_admission`消融回答，不能复用anchoring-gate消融。机器可读证据与修改决定位于`results/_protocol/four_task_transfer/diagnostics/t2_substep_risk_online_failure_revision_20261022.json`。

### 2026-10-22 CPU 离线统计收尾与共享 T2 候选诊断

确定性入口`scripts/paper/cross_ebdm/analyze_shared_failure_closeout.py`把v11资格、264例在线风险验证和后续T2共享执行器诊断合并为单一机器可读工件。重复运行得到相同SHA。v11统计保持不变：1,440个资格episode中1,241次成功、102次timeout、97次约束违规、0次导面合同失败；36行行为合格、16行行为拒绝。这些数字只用于资格与诊断，不是论文性能比较。

三类资格失败的边界也保持不变。102次有限时间进展失败继续按方法诊断phase、issued/applied action、accepted-reference与终态误差；97次约束控制失败继续按1ms子步诊断预测/真实力裕量、reference和joint/support记账；v11中的导面合同失败仍为0。后来264例开发验证里的public sentinel为7/8并出现一次31.94N overforce，所以它只重新打开共享T2 mode-0执行器资格，不能回头改写v11分母，也不支持修改力限、deadline、mode、reset或evaluator。

共享T2的顺序诊断已经否决简单修复。6.5mm/s在已访问的8例中8/8，但新32例为31/32并有一次overforce；6.5mm/s加原始post-outlet lift为31/32、零违规、一次timeout；6.75mm/s加原始lift为30/32、零违规、两次timeout。把既有8/8/10mm三维reference lead governor接到6.75mm/s路线后，mode 1为16/16，mode 0却16/16全部timeout。mode-0约在global tick 750--753切换，stage 1、2、3的中位进入时刻分别为858、879、960，只剩约40个控制tick完成下降与15-tick终点保持。终态仍在clear-height附近且无接触。这证明完整三维lead限幅修复了reference runaway，却破坏了垂直路线的deadline liveness。

预先声明的平面lead候选已继续执行。第一版实现错误继承了完整三维lead控制器的0.02m/s三轴限速和事件逻辑，不满足“raw lift仅加x/y clip”的声明；32例证据保留，但run已用`INVALID_RUN.json`封存，禁止选择、资格和性能解释。修正实现通过逐位单测：z、旋转、阻抗和力通道与raw lift完全相同，仅在x/y lead超界时改变平面动作。修正后的32例仍只有30/32、零违规，故机制被否决且没有打开新种子。两个timeout中，clip确实把原先55--67mm的终端平面reference lead降到0.7--3.7mm，但一个到tick969才进入stage1，另一个到tick967才进入stage2；失败来自串行路线的阶段时间预算和垂直overshoot，而不是单一平面reference runaway。

因此下一版不再扫描速度、lead或deadline，也不再组合现有串行子类。唯一允许的结构修订是出口后的并行、单调公共路线：越过已声明出口后，平面横移与抬升在同一free-space segment内并行；随后执行单调下降和原15-tick终点保持。实现前冻结路线几何、事件、各轴issued-reference envelope、conditional integration及最坏阶段时间预算，解析预算必须为终点保持留足全部tick。该路线同样用于public sentinel、baseline、ablation和ours，属于benchmark共享能力。先在已访问32例要求32/32且零违规，再在至少64个全新seed-disjoint public-sentinel episode上资格化；任一门失败即否决该结构。

risk-admission v2仍排在共享executor之后：删除最后候选的强制安全覆盖；把有限emergency reference-brake作为同风险模型下的显式候选；空接纳集返回`risk_set_empty`并计失败；fit/calibration/held-out数据由冻结部署候选生成器产生；资格同时检查block coverage、高风险召回、安全窗口拒绝率、空集率、有限时进展和闭环违规。外部baseline只获得公共executor，不获得ours的risk head。完整收尾位于`results/_protocol/four_task_transfer/diagnostics/cpu_shared_failure_closeout_20261022.json`；当前状态是`revision_specified_not_qualified`，因此GPU训练、predeployment和新四任务roster仍关闭。

### 2026-10-22 CPU 权威收尾 v2：三类失败诊断与 common-protocol-v3 冻结

本节覆盖上一节的`revision_specified_not_qualified`状态。确定性CPU入口已经在不打开confirmation状态、不训练、不调用GPU的条件下重新生成权威工件`results/_protocol/four_task_transfer/diagnostics/cpu_shared_failure_closeout_20261022.json`。工件状态为`cpu_closeout_complete_common_protocol_v3_qualified`，全部输入均以SHA-256绑定，指针复核通过。旧v11资格统计保持为1,440个episode、1,241次成功、102次timeout、97次约束违规、0次导面接触合同失败；旧confirmation的21,200条记录继续标记为`descriptive_only_invalid_run_not_reinstated`。两套旧数据都不构成论文性能结果。

三类失败的诊断和判责规则已经冻结：

- **有限时间进展失败（旧资格102次）。** T2的串行出槽后横移与抬升消耗deadline，T3的10 mm insertion lead缺少进展裕量且retreat存在独立latch，T4在joint-margin约束下的固定底座可达集与终端集不相交。诊断单位是同一reset、同一requested mode和同一公共执行器下的配对episode；必须记录phase占用、accepted public-reference的逐轴进展、剩余路径/剩余deadline、终态误差、observed mode和hold count。任何共享候选只要在新鲜sentinel上出现timeout、约束失败、mode mismatch或依赖方法私有信号，就被否决。
- **约束控制失败（旧资格97次）。** T2的控制周期端点会混叠物理子步接触，旧接触单元允许单子步冲击；T4的legacy路径越过关节限制，而仅加joint margin会把joint-limit转成不可达timeout。诊断单位是同一post-reset状态上的最后安全与首次违规物理子步；记录总力/横向力裕量、issued/accepted/applied reference、joint margin、support count和WBC residual。公共sentinel在相同cell复现的失败归于benchmark基础设施；公共sentinel和共享执行器通过后，接口正确的方法专属闭环失败保留在该方法分母中。
- **导面接触合同回归门（旧资格0次）。** 这不是已修复失败的正证据，只是硬回归门。每个T2 episode继续用精确物理子步证据检查首次接触与出口、出口前轴向支撑覆盖、最长连续失联、mode crossing和task-owned request相等性。任何一项失败都会阻断对应方法行和公共协议，不允许调阈值、替换episode或使用fallback结果。

公共修改已经完成资格，不再继续扫描标量。T2的`concurrent_monotone_lift_v3`在已访问诊断集为32/32、新鲜seed-disjoint public-sentinel集为64/64，均零约束失败且全部证据独立重审；该路线使用6.75 mm/s，在出口后并行执行平面运动与抬升，再单调下降并保持终点。T3固定15 mm insertion-only lead与safe-settled release，开发能力门为64/64、零违规。T4固定`joint_margin_public_reference`与所有方法共享的平面body intent，能力门为16/16、零违规、四脚支撑，最小实测关节余量为0.0002256326 rad。三项均属于公共benchmark能力，不能计入Cross-EBDM的transport增益。

新合同已经落为T2 `task_contract_v3`、T3 `public_v6`和T4 `task_contract_v3`。`verify_shared_protocol.py`对52个main/baseline/ablation配置完成零物理审计，protocol validity和architecture equivalence均通过。公共合同从现在起保持不可变；v6行为预部署如果再次暴露公共失败，必须创建新合同版本并使本轮资格失效，不能原地修改配置或执行器。

下一版先执行全方法CPU预部署v6。每个main、baseline和ablation行都继承相同的T2/T3/T4合同、同一reset/mode请求和同一证据审计。source fixture、归一化、decoder、objective或adapter错误可修；这些检查通过后的方法专属timeout或约束失败保留为算法结果；缺少作者资产继续作为预声明的availability rejection；需要新learned工件的行明确记为`training_required`，不能伪造行为结果。只有公共sentinel和所有当前可执行方法通过这一行为门，才允许GPU1训练。

随后实现Cross-EBDM自己的risk-admission-v2：删除末候选强制安全，显式处理空安全集，emergency brake也必须经过同一个风险模型，并使用冻结部署candidate generator产生互斥的fit/calibration/held-out episode blocks。CPU先验证关闭时逐位身份与空集失败；公共行为门通过后，GPU1才收集或拟合冻结数据块并完成校准。之后要求episode-block coverage、高风险召回、安全窗口拒绝率、有限时间进展和ours/task-only同risk-gate的闭环配对门，再重跑全部受影响的v6 learned行。这个修改不能给原名外部baseline使用，也不能冒充共享执行器收益。最终行为绑定、package和device门全部通过后，才生成与开发、资格和旧invalid confirmation完全不相交的新四任务roster。

理论按这一证据边界表述：共享执行可行性、候选transport和risk admission是三个不同层；前向不变性不推出有限时间完成；安全允许动作集合必须与deadline正进展集合非空相交；hybrid release liveness依赖支撑重现和有界controller carry；T4只能主张给定公开body-intent下的受限可达性；经验收据不构成全局force-safety或robust-invariance证书。论文中的transport贡献必须来自新独立roster上相同公共执行器、相同risk admission、相同数据/优化预算和配对reset下，ours相对task-only及冻结消融的增量。

### 2026-10-22 v12 CPU 离线统计与联合力—接触接纳决定

共同协议v3下的全方法预部署已经完成并由独立路径重审。冻结plan包含60个逻辑行，其中52行实际执行；1,440/1,440个CPU episode均有可重审证据，1,263次成功、100次timeout、75次overforce和2次guide-contact continuity failure，基础设施错误为0。42行通过行为资格，10行按实际行为拒绝，ACP的4行算法级拒绝和Human2Sim2Robot的4行资源拒绝继续保留。四任务public-geometry sentinel为16/16。该运行只决定admission，不是论文性能样本。

177个失败被互斥且完备地分为三类。100次有限时间进展失败来自FADA 62、DIAL 32和public source prior 6；75次过力来自FADA 66、ordinary response MPC 7和public source prior 2；2次导面连续性失败分别记在main Cross-EBDM与`gated_full`，但两者绑定同一actor、同一instance、同一reset和同一requested mode，因此对应一个物理机制观察。该轨迹最终任务状态成功且峰值总力只有7.727 N，但最长连续失联为0.182 s，超过冻结的0.15 s上限。共同sentinel已通过，所以三类失败都不触发task contract、deadline、力限、route或evaluator修改。

下一版的baseline处理保持方法身份。FADA先做全部冻结实例的teacher-forced source检查、归一化/decoder/Box检查和闭环首次分叉定位；ordinary response MPC与DIAL保存优化前后目标、预测/真实力裕量、候选/reference差值和首次违规子步；direct source prior只检查单位、frame、mode和Box语义。只有可复现的fixture、normalization、decoder、objective或adapter错误允许修实现；接口正确后的timeout和overforce保留为方法限制，不能靠放宽共同合同消除。

Cross-EBDM的下一版由单一force-risk head改为**联合力—接触可行性接纳**。候选固定为transport fractions `[1, 0.5, 0.25, 0]`和一个有限emergency reference brake；五个候选全部经过同一预测器，任何候选都不能强制标为安全，空集合必须以`risk_set_empty`失败退出。标签同时包含十个控制步内每个1 ms物理子步的总力峰值、横向力峰值、最小轴向支撑覆盖和最长连续失联。选择规则同时要求校准后的力上界低于冻结限值，并要求接触可持续性下界满足冻结T2合同。

旧risk-v1行为策略窗口不再用于这项训练。CPU适配器已经支持完整MuJoCo积分状态、任务历史和episode ledger的checkpoint/restore，actor也支持状态恢复，因此数据单位冻结为同一个公共状态上的五条独立反事实分支；每条分支都恢复相同task/actor checkpoint、独立计费并保留失败。fit、calibration和held-out qualification按reset block划分，同一checkpoint lineage不得跨split。机器可读方案位于`configs/arm/guided_contact_alignment/training/joint_admission_counterfactual_data_v2.json`，当前状态为`frozen_cpu_design_unexecuted`。在CPU采集、独立审计和split检查完成前不启动GPU；随后仍只允许物理GPU1训练。

本轮CPU派生结果位于`results/_analysis/four_task_transfer/predeployment_20261022_v12_offline_closeout/`。`closeout.json`保存逐行、逐mode和三类失败统计，`diagnosis_plan.json`保存判责规则、baseline诊断与Cross-EBDM v2修改。分析从同一冻结输入重复生成后全部文件逐字节一致。

### 2026-10-22 正式 v7 后的 CPU 收尾与下一版决定

v13资格的1,440个CPU episode已经独立重审：1,256次成功、100次timeout、81次overforce、3次导面连续性失败、零基础设施错误。184个失败被互斥且完备地归为三类。有限时间进展失败100次，来自DIAL、FADA和direct source prior；力安全失败81次，来自FADA、ordinary response MPC、direct source prior及三个learned ablation尾部样本；导面连续性失败3次，来自task-only、band-only和ordinary response MPC。四任务public-geometry资格sentinel为16/16。资格集只决定部署兼容性，不是论文性能样本。

正式v7随后揭示了资格集未覆盖的泛化缺口。Cross-EBDM在T1为320/320、T2为79/320、T3为291/320、T4为512/640。相对task-only的成功率差分别为T2 -0.3个百分点、T3 +2.5个百分点、T4 +8.0个百分点，三个95%区间都跨零，当前仍没有transport任务性能优势证据。T2的240次timeout集中在latency两种mode和payload/mode-0；T4失败集中在latency与derated-support组合。

新的同reset CPU配对把T2与T4分开判责。T2中Cross-EBDM与public geometry在四个机制×mode条件上得到相同的失败形态：latency两种mode都timeout、payload/mode-0 timeout、payload/mode-1成功。因此T2是共享公共路线的有限时间可达性缺口，必须创建新task-contract版本，修复收益不能计为Cross-EBDM贡献。T4中public geometry为8/8，而Cross-EBDM为5/8；三个失败全部位于latency条件，说明共享body intent和joint-margin executor可完成任务，learned candidate在recovery gate后破坏了liveness，只允许修改Cross-EBDM方法侧执行包装器。

T2的三组有界探针已经停止标量搜索。固定切向速度下限达到15--20 mm/s时只能修复payload/mode-0；latency/mode-0仍timeout，latency/mode-1会overforce。把transit load降到0.75或1.0 N仍不能修复latency/mode-0，且1.0 N的一例达到30.31 N。把切向刚度请求乘以1.25、1.5或2.0产生逐位相同的issued/applied序列，因为共享13D接口中的log-stiffness已经按0.08/tick升到冻结上限1.0。失败轨迹中的速度命令确实被应用，但accepted reference达到固定lead envelope后，物理进展仍低于deadline要求。

T2下一版只实现一个结构修改：在现有`genedynamics/tasks/transfer/controllers/guide_contact.py`中增加公开的deadline-aware reference admission。控制器用剩余路径、为出口后路线和终点hold保留的时间、实测进展、reference lead、wrench与接触历史构造候选reference增量；只接纳同时满足力/接触可行性与正deadline进展的增量。若交集为空，显式返回`progress_set_empty`开发失败，不能延长20秒deadline或放宽力/接触判据。新行为通过新的task-local protocol版本绑定，并由所有方法及public sentinel共享；旧`task_contract_v3`和正式v7保持不变。

T4下一版在现有`genedynamics/tasks/transfer/controllers/recovery_execution.py`中增加Cross-EBDM专属的candidate/common completion admission。现有包装器每tick已经同时计算common和candidate；新版本在终端集外选择保持common方向进展与公开joint/support margin的最大candidate blend，在终端集附近还必须保持终端集进入和连续hold累计。集合为空时逐位回退common，并把两份动作、blend、终端状态、判定原因和计数纳入checkpoint状态与controller diagnostics。这个补充来自一条终态位置/姿态已在容差内但hold count只有7的latency timeout；只看位置方向导数不足以保证完成。共享reset、body intent、joint-margin executor、recovery gate、力限、支撑判据和deadline保持不变。该gate只能称为公开反馈启发式，除非后续给出模型误差界，不能称为安全或liveness证书。

CPU放行顺序固定为：先补两项控制器的状态恢复、逐位fallback、权限边界和失败退出单测；再用全新seed做T2 public-sentinel门和T4 candidate/common配对门；T2公共门通过后重跑全部可执行方法，T4单实例通过后扩展八个Cross-EBDM实例；最后重跑零物理architecture、package与device审计。只有这些门全部通过才生成与v7及本轮诊断不相交的新正式roster。GPU1仅在新的learned工件确实需要训练时使用；当前两项首版修改都可以先在CPU完成。

理论同步拆成三层：共享执行可行性、candidate transport和action admission。force safety、guide-contact continuity与finite-time completion是三个独立约束；前向不变性不推出有限时间完成，必须另假设安全/接触可行动作集与deadline正进展集非空相交。资格通过只说明部署兼容，transport贡献仍必须由新seed-disjoint roster上、相同共享执行器和相同admission下相对task-only及冻结消融的配对增量支持。

机器可读决定位于`results/_analysis/four_task_transfer/post_confirmation_cpu_diagnosis_20261022/report.json`。60个新CPU开发episode及审计证据按任务封存在`results/arm/guided_contact_alignment/validation/finite_time_progress_diagnosis/`与`results/quadruped/mobile_keyed_insertion_recovery/validation/candidate_progress_diagnosis/`；它们均标记为development-only、`confirmation_states_opened=0`，不能进入论文主表。

### 2026-10-22 控制器候选的 CPU 拒绝门与下一版实现决定

上一节提出的两个控制器候选已经完成CPU开发门，但都不能绑定到正式manifest。原始轨迹、逐控制周期诊断、独立evidence audit与SHA清单已归档到任务本地results层级；汇总结论位于`results/_analysis/four_task_transfer/post_confirmation_cpu_diagnosis_20261022/controller_probe_closeout.json`。这些运行均为development-only，`confirmation_states_opened=0`、`gpu_calls=0`。

T2的deadline-aware public reference admission先后运行了初版与修正writeback版，各8个episode。修正只消除了lead超限时遗留的反向速度写回，没有改变物理结论：latency/mode-0、latency/mode-1和payload/mode-0各0/2，只有payload/mode-1为2/2，总计2/8，零约束违规。latency/mode-0末端要求20 mm/s的deadline进展，而实测只有1.793和1.741 mm/s，并分别累计536和539个`progress_set_empty`控制周期。该候选因此被拒绝，不创建`task_contract_v4`。在冻结的20秒deadline、力限、导面连续性和共享13D动作支持下，这两个latency cell没有观测到同时满足约束与正deadline进展的动作；继续扩大lead或速度等价于修改benchmark合同。后续只能在“保留该失败”与“另建、重新资格化共享route/action-support合同”之间显式选择，公共修复不能计为Cross-EBDM贡献。

T4的completion-dominance admission在instance-01的8个新鲜episode上达到8/8，零约束违规且support gate全部通过，但它几乎完全退化为公共控制器。3,894个candidate-eligible控制周期中只有41次接纳非零candidate blend，接纳率1.05%；4/8个episode从未接纳candidate。该候选也不绑定到actor manifest，因为它只能证明common fallback恢复了完成性，不能证明transport产生任务性能。

下一版只修改Cross-EBDM的T4训练与选择语义。修改必须位于`genedynamics/solvers/single/cross_ebdm/learning`中的现有gated transport目标；任务参数放在`configs/quadruped/mobile_keyed_insertion_recovery/training`，入口继续使用`scripts/paper/cross_ebdm`。在现有task proxy、response matching、risk与change项之外，加入相对公共控制器的逐步完成进展项：候选对目标的平移与旋转方向进展不能低于同状态common action，并在终端邻域惩罚candidate residual，保护连续hold。损失只使用公开状态、公开任务目标、candidate/common action和冻结response模型，不使用evaluator结果或隐藏机制标签。

放行顺序也随之冻结。CPU模型门先检查关闭新项时旧目标逐位复现、梯度有限、checkpoint恢复一致，并在source/known-nominal validation上同时满足task-only相对proxy、平移/旋转进展分位数和非零candidate utility。之后才允许物理GPU1训练八个独立实例。新checkpoint先通过instance-01的8个CPU物理开发episode：8/8成功、零约束违规、candidate在eligible周期中的接纳率至少25%，且每个机制/支撑组都有非零接纳；随后扩展到八实例新鲜集合，要求每实例不退化并保留逐实例利用率。未达到这些门时保留当前正式v7结论，不用common fallback的8/8替换方法结果。

理论相应增加一个必要的归因条件。安全或接触不变性不能推出deadline完成；fallback恢复成功也不能推出learned transport有效。transport贡献要求安全/接触可行动作集与正deadline进展集存在非空交集，并且部署轨迹在非零测度的状态上实际采用candidate、产生相对common的增量完成进展。平均model task proxy的兼容性不足以保证hybrid terminal hold，因此训练、checkpoint选择和最终报告都必须显式覆盖平移、旋转、支撑、终端集进入与candidate利用率。

### 2026-10-22 T4 completion-aligned transport 与组合 actor 放行

首版completion-aligned训练在物理GPU1上完成八个独立实例、共16,000次更新，但raw checkpoint选择8/8被拒绝。任务代理和candidate utility均通过，utility为51.0%--61.4%；平移缺口p95仍为$0.80\times10^{-6}$--$1.46\times10^{-6}$，旋转缺口p95为$7.77\times10^{-6}$--$11.61\times10^{-6}$，高于冻结的$10^{-7}$门。失败不是训练崩溃，也没有打开confirmation状态、物理仿真或目标奖励。

诊断确认训练优化均值而checkpoint选择检查p95，少数困难状态被均值掩盖。现有`gated_transport.py`因此加入线向与角向分别计算的最坏10%尾部项，终端状态只由回归common hold的残差项约束；旧目标在不传completion spec时保持原路径。v2从八个失败continuation继续在GPU1训练12,000次更新。相对v1，平移p95降到$0.34\times10^{-7}$--$1.89\times10^{-7}$，旋转p95降到$0.83\times10^{-6}$--$10.86\times10^{-6}$，utility升到72.6%--83.6%；任务代理相对task-only只差-0.0047到+0.0046。raw v2仍未满足全部$10^{-7}$门，因此未被当作无包装器checkpoint放行。

随后冻结组合actor：终端邻域精确使用common hold；其他recovery状态在$\{1,0.5,0.25,0\}$中选择同时保持common平移与旋转方向进展的最大candidate比例。CPU零物理预检在347个独立验证状态上对八实例全部通过：正比例candidate admission为76.9%--92.2%，完整candidate使用为72.6%--83.6%，投影后两类最大缺口均低于$10^{-7}$。这满足“组合策略没有退化为common”的离线必要门，但本身不构成物理成功证据。

物理开发门先用instance-01和8个新seed运行，得到8/8成功、零约束违规、四个机制/支撑组均有candidate介入；3591个eligible非终端控制步中2755步为正比例candidate，admission为76.72%。随后八实例在另一组配对新seed上运行64个episode，得到64/64成功、零约束违规。逐实例admission为71.81%--80.43%，32个实例×机制/支撑组的最小组内admission为576步、最小组内比例为68.36%。全部失败替换关闭，confirmation状态打开数为零。

通过后，T4原有`actor_bindings.json`中的`main/cross_ebdm`已更新为`ours_gated_v2_completion_projected`，直接绑定八个v2 checkpoint和`public_completion_dominance_v1`；没有创建平行manifest。完整catalog验证通过，CPU和GPU1零物理package审计均为52个可执行row通过、8个既有资源/行为拒绝保留。GPU1物理UUID绑定正确且审计后释放。相关机器证据分别位于`results/quadruped/mobile_keyed_insertion_recovery/training/completion_aligned_transport/checkpoints/`、`training/completion_tail_aligned_transport/checkpoints/`、`validation/completion_projected_transport/preflight/`、`validation/completion_projected_transport/instance_01/`和`validation/completion_projected_transport/all_instances/`。

理论与方法身份必须按组合actor陈述。有限混合投影只保证公开动作代理上的逐步common-relative方向缺口，不保证真实接触动力学中的安全、支撑、有限时完成或任务优势；这些仍由共同物理合同与开发门经验检查。candidate在正测度状态上被实际采用是transport归因的必要条件，但76.06%的物理admission也不能替代新独立roster上相对task-only和消融的任务性能比较。T4方法侧已可进入新的共享行为predeployment；整篇论文仍不能启动新正式confirmation，因为T2共同路线限制与既有T3冻结合同漂移需要在最终协议冻结中分别处理。

### 2026-10-23 最终 actor v8 的 CPU 离线收尾与下一版冻结决定

最终 actor 的新鲜资格集已经在禁用 CUDA 的条件下完成离线重算。冻结输入为 `results/_qualification/four_task_transfer/behavioral_predeployment_final_actor_20261022/`；1,440/1,440 个物理 episode 均有独立审计证据，得到 1,264 次成功、99 次 timeout、77 次 overforce、0 次导面连续性失败和 0 次基础设施错误。60 个逻辑行中 43 行行为合格、9 行行为拒绝，另保留 4 行 ACP 算法级预部署拒绝与 4 行 Human2Sim2Robot 外部资源拒绝。四任务 public-geometry sentinel 为 16/16。该集合只决定部署资格，不是论文主表样本。

176 个行为失败按互斥且完备的三类口径结算：

- **有限时间进展失败 99 次。** FADA 61、DIAL 32、direct source prior 6；按任务为 T1 36、T2 36、T3 27。下一步只检查 phase/stage 占用、优化目标、issued/accepted/applied reference、剩余路径与剩余时间、终态误差和首次闭环分叉。只有 fixture、归一化、decoder、objective 或 adapter 缺陷可以修复；接口正确后的停滞保留为方法限制，不能延长公共 deadline。
- **力安全失败 77 次。** FADA 67、ordinary response MPC 7、direct source prior 2、`matching_no_band` 1；按任务为 T2 40、T3 5、T4 32。诊断必须对齐最后安全与首次违规的 1 ms 物理子步，并记录预测/实际总力与横向力裕量、候选/reference 差值、优化器或selector决定。共同力限和共享 executor 保持冻结。
- **导面连续性失败 0 次。** 这是零事件硬回归门，不是“修复了一个观察到的失败”，也不是接触安全证明。继续保留精确子步证据和 150 ms 冻结阈值，不修改公共 route。

三类只是共同统计口径，不代表共享根因。public sentinel 16/16、静态 T2 接口检查全过且基础设施错误为零，因此当前证据不支持修改共享 deadline、力限、mode、route、reset、evaluator、T3 合同或 T4 body intent。下一版按以下边界修改：

1. FADA 完成全部冻结实例的 teacher-forced source、归一化、decoder、Box 与首次闭环分叉审计；DIAL 完成 T2 有限进展归因；response MPC 完成预测力与首次违规子步对齐；direct source prior 只核对单位、frame、mode 与 Box 语义。只有明确实现缺陷才改代码，否则保留失败。
2. 当前 Cross-EBDM final actor、T2 公共接触反馈和 T4 completion projection 在 `confirmation_v8` 完整结束并独立审计前保持冻结。不得根据正在增长的部分结果修改 actor、绑定或 roster。
3. `matching_no_band` 的单次 T2 overforce 保留在该消融分母中，不放宽共同力约束。transport 是否有收益只能由完整、seed-disjoint、同 executor、同 admission、配对 reset 的正式 ours/task-only/冻结消融比较回答。
4. post-reset runner interruption 单列为基础设施结果，不能写成算法失败。v8 中已发生的收据只记录必须原位保留且不得 replay；receipt-to-outcome 的耐久性改进只能进入后续协议版本，不能原地修改正在运行的 v8。

理论随之固定五条边界：共享 executor 可行性、candidate transport 与 action admission 分层；force safety、guide-contact continuity 与 finite-time completion 相互独立；资格通过只表示部署兼容；transport 贡献要求正式配对比较中出现相对公共控制器的增量；runner interruption 不参与算法机制归因。机器可读结果位于 `results/_analysis/four_task_transfer/predeployment_final_actor_v8_offline_closeout_20261023/`，其中 `closeout.json` 保存统计与逐行结果，`diagnosis_plan.json` 保存判责、下一版修改和理论边界，`manifest.json` 已逐文件复核通过。

### 2026-10-24 confirmation v8 正式结果与 CPU 诊断收尾

`confirmation_v8` 已完整结束并通过独立执行审计。冻结计划的 21,200 条记录全部闭合，其中 15,360 次物理执行、5,840 个预部署拒绝计分槽位；44 个已写入 post-reset receipt 后发生的运行器中断原样保留为失败，重放和替换均为零。所有物理执行只使用 GPU1。CPU 统计重新核对 roster、schedule、outcome、receipt、aggregate 与各自 SHA，60 个统计 cell 全部闭合；旧 aggregate 中 24 个 worst-group 标量因为未联合包含 requested mode 而被离线纠正，其他计数和均值逐项一致。

正式 constrained success rate 如下。括号内为成功数/计划数；预部署拒绝仍在共同分母内。

| 方法 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|
| Cross-EBDM | 1.000 (320/320) | 0.250 (80/320) | 0.903 (289/320) | 0.998 (639/640) |
| task-only | 1.000 (320/320) | 0.250 (80/320) | 0.894 (286/320) | 0.702 (449/640) |
| DIAL transfer | 1.000 (320/320) | 0.000（预部署拒绝） | 0.856 (274/320) | 0.991 (634/640) |
| response MPC | 0.994 (318/320) | 0.000（预部署拒绝） | 0.956 (306/320) | 0.756 (484/640) |
| public geometry | 1.000 (40/40) | 0.250 (10/40) | 1.000 (40/40) | 1.000 (80/80) |

ACP、FADA 与 Human2Sim2Robot 的正式行均因冻结的资格/资源门被计为零；这些行证明的是当前协议下不可部署，不能解释成执行后的控制性能。不同 permission class 的对比只作描述。相同 B-log 权限下，Cross-EBDM 相对 task-only 的配对成功率差为：T1 0、T2 0、T3 +0.94 个百分点（95% bootstrap interval -5.63 到 +6.56）、T4 +29.69 个百分点（+21.72 到 +35.47）。全 main-family 的保守 Holm 校正后 T4 的 supplemental sign-flip p 为 0.1875，因此当前可以陈述幅度较大的描述性效应和区间，不能把 post-access 多重比较写成确认性显著性结论。T3 中 response MPC 与两个 public reference 均高于 ours；当前没有跨四任务的 transport 优势。

T4 的 final main actor 与消融 `gated_full` 已不再是同一 actor：main 使用 completion-projected actor，旧 `gated_full` 为未加 completion projection 的 earlier gated actor。T4 两者成功率分别为 0.998 与 0.791，说明 completion projection 是 T4 增益的主要候选机制。T1--T3 两个表角色仍复用相同 actor；除 2 个原样保留的 runner interruption 外，958 个可比较记录的评分、步数和 action hash 全部一致，不能当作独立重复证据。

物理失败在排除 5,840 个资格门计分槽位后共有 3,370 条。44 个 runner interruption 单列为基础设施失败；剩余 3,326 条按互斥口径分为三类：

- **严格导面接触合同失败 1,133 条。** 全部来自 T2 `latency_gain_shift`，两个 mode 均受影响。轨迹约在控制步 361--413 首次建立接触，但直到 1,000 步 deadline 仍未穿过 outlet；`public_geometry` 也出现同一形态。这是共享的延迟/有限时间可达性问题，不能归因给某一个 learned method。
- **通过接触门后的有限时间进展失败 2,192 条。** T2 有 568 条，全部位于 `payload_shift/mode-0`：接触、outlet 与连续性均通过，但终态仍约有 35 mm 位置误差。T3 有 314 条，主要是 `payload_shift/mode-1`，且强公共参照与 response MPC 优于 ours，应先诊断 ours 的 terminal release/hold。T4 有 1,310 条，几乎全部来自旧消融和 baseline；final Cross-EBDM 只有 1 条，说明 completion projection 应保持冻结。
- **约束控制失败 1 条。** 这是 T2 `ungated_full` 在 `payload_shift/mode-1` 的 speed-limit 违规。matched gated actor 没有该违规，但单事件不足以支撑一般安全性结论；冻结力限和速度限不变。

因此不立即启动下一次完整 confirmation。下一版按以下顺序修改并重新冻结：

1. T2 资格门扩展为完整 mechanism × mode 的 public-sentinel 覆盖，分别报告 contact acquisition、stable dwell、outlet crossing 和 post-outlet completion。现有 4 个名义 sentinel 不足以发现正式分布中的 latency 与 payload 失败。
2. 在现有 T2 公共执行器入口内记录 delayed public state、issued/accepted/applied reference、阶段进入时间、剩余路径和剩余 deadline。若 public sentinel 在新鲜开发 seed 上复现相同缺陷，再创建显式的新 task-contract 版本修公共 latency/contact 与 post-outlet liveness，并把同一修改应用到所有方法；不得修改 v8、放宽 deadline/力限/接触阈值或把公共收益算给 Cross-EBDM。
3. T3 只做方法侧配对诊断：同一新鲜 payload-mode-1 reset 比较 Cross-EBDM、band-only 和 response MPC 的 terminal-set entry、release、retreat 和 hold。只有证据定位到 transport/gate 才修改现有 Cross-EBDM solver；公共 T3 合同保持不动。
4. T4 final completion projection 保持冻结。论文消融中把旧 T4 `gated_full` 明确解释为 no-completion-projection actor，避免把它误称为 main actor 的重复行。
5. 完成受影响行的新鲜行为资格、零物理 architecture、package 与 device 审计后，才生成与 v8 状态不相交的新 roster。v8 的任何 reset、outcome 或失败均不用于调参、重放或替换。

理论需要同步收紧。严格接触理论必须把“到达接触/出口”与“接触建立后的连续保持”分开；v8 表明仅有不变性陈述不能保证 outlet reachability。有限时间结论必须显式依赖延迟观测下的剩余 phase budget 和正 liveness margin。T4 completion projection 可以陈述为 terminal-set reachability 机制的经验支持；现有证据不支持把 transport 写成四任务普遍性能优势。calibrated gate 目前只能称为 constraint filter，不能从一次 speed-limit 对照推出一般安全保证。

正式表格位于 `results/_analysis/four_task_transfer/confirmation_v8/`；`report.json` 的 SHA256 为 `2cad42441cd162d280d4a9ad59fa98b1c3b7f146e2997d2cc5c140b50d96903f`，三类失败与下一版决策位于同目录的 `diagnosis_plan.json`，SHA256 为 `7c9e81e23f22bad3f879457c03d563453af0a9649aaa3e11ea47cbe2a7a4a66e`。诊断从相同冻结输入二次生成后逐字节一致；本次只随离线分析入口扩展更新了分析脚本来源 SHA，正式计数、归因和决定未改变。

### 2026-10-24 fresh CPU 机制复现与下一版范围

正式统计之后又执行了两组不访问 confirmation 状态的 fresh CPU 诊断。所有运行均为 development-only、`performance_evidence=false`、失败替换关闭；它们只用于定位机制和否决候选，不能替换 v8 结果或进入论文主表。首个 deadline 候选计划错误地使用了 `delay_steps=2, payload_mass=0.2`，而正式 latency cell 为 `delay_steps=3, payload_mass=0.5`；该运行已通过 `INVALID_RUN.json` 和四个 SHA 原样封存，排除出选择与正式 cell 判责。综合工件为 `results/_analysis/four_task_transfer/confirmation_v8/development_diagnosis.json`，使用纠正后的 exact-formal-cell fresh 运行，并加入三类失败各自的可证伪假设、CPU筛查、比较对象、接收与拒绝门。SHA256 为 `75f38f18bfedc056f4b7a50cbfeb052d08cee1e7171acf5aab388e1fc14b6af2`；从同一冻结证据连续生成两次后逐字节一致。本次重建只更新分析脚本及上游正式诊断的来源 SHA，诊断数值与原决定未改变。

T2 的 public geometry sentinel 首先在 32 个 fresh episode 上重现 v8 形态：latency 两种 mode 与 payload/mode-0 均为 0/8，只有 payload/mode-1 为 8/8。纠正后的另一批 fresh reset 配对比较当前公共控制器和已有但未绑定的 `deadline_aware_v1` 前出口参考准入；两者成功率仍均为 8/32，32 对成功/失败完全相同。候选把 latency/mode-1 的中位 outlet tick推进到699、payload/mode-0从744.5推进到650、payload/mode-1从461.5推进到403，但 latency/mode-0仍不出导面，前两种已出导面的失败仍不能完成，并新增一个 payload/mode-0 overforce。该候选作为独立协议修改被拒绝，不创建 task-contract v4。下一版 T2 必须在同一个任务级版本中解决延迟观测下的接触后出导面进展和出口后的剩余时间完成；deadline、力限、接触阈值、reset 分布和 evaluator 保持不变。公共 sentinel 必须在全部 fresh mechanism×mode cell 上通过严格接触、完成和零约束门后，所有方法才可共同继承新合同。

T3 的第一次 fresh 配对中，Cross-EBDM、band-only 和 response MPC 分别为 31/64、49/64 和 64/64；所有方法侧失败都发生在 phase 3。第二批 fresh reset 比较当前 Cross-EBDM、终端邻域切回公共动作的候选以及 response MPC，结果分别为 40/64、24/64 和 64/64。候选把 16 个原本成功的配对 episode 改成失败，因此已从正式 controller 与 binding 中移除且从未进入 manifest。时序证据进一步定位到有限时间到达：当前 Cross-EBDM 的失败轨迹在中位 tick 744 才首次满足精确 hold 条件，总预算为 750 tick且还需连续保持 15 tick；response MPC 的对应中位 tick 为 695.5，最晚为 716。下一版 T3 不做“到达后切换”，而是在 phase 3 的更早阶段优化或接纳正终端进展，并在任何有状态 candidate 更新之前完成动作选择。新候选必须在同一 fresh reset 上与当前 Cross-EBDM、band-only 和 response MPC 配对，不得破坏任何原成功 episode，也不得增加约束违规。

正式三类失败据此分开处理。1,133 个严格导面失败归于 T2 共享任务协议；2,192 个有限时间进展失败拆为 T2 共享 completion 与 T3 方法侧 terminal arrival，T4 final completion projection 保持冻结；单个 T2 ungated speed-limit 事件不触发共享协议修改，只保留冻结约束并安排 fresh gated/ungated stress test。理论同步把 constraint preservation、finite-time reachability 和 terminal dwell 分开；liveness margin 必须包含观测延迟、接触获取时间、剩余路径、剩余 deadline 和保持时长。当前仍没有 T1--T3 transport 任务优势证据。

下一版 Cross-EBDM 只改 T3：在现有 gated transport 目标中加入 phase-3 finite-time terminal-progress 项，目标为公共 common action 进展与“预留15个hold controls后仍可进入精确终端集”所需进展率中的较大者；进入精确终端集后只收缩 candidate residual。输入限于公开 phase、当前/目标位姿、common action、剩余控制步、冻结 response model 与既有 task proxy，不允许 evaluator success、reward、隐藏机制标签或 confirmation 状态。CPU 数学单测已覆盖 phase gate、hold reserve、终端收缩和梯度作用域。

T2 不直接实现已被否决的 deadline candidate。先对 fresh public-sentinel 轨迹做 action-support CPU 筛查；只有在冻结 lead、force、contact 与 deadline margin 内存在经验正进展集合时，才允许创建新的共享 acquisition-plus-completion task-contract。否则保留 v8 失败，并把限制明确归于当前 benchmark action support。方法开发可以先完成 T3 CPU preflight，但新 confirmation roster 仍需按顺序通过 T2 共享门、T3 对 current/band-only/response-MPC 的 fresh 配对门、T2 gated/ungated 约束复现以及 architecture/package/device 审计。上述门通过前不绑定新 actor，也不启动新的正式 confirmation。

### 2026-10-24 T3 finite-time late-phase actor 放行

T3 方法侧 terminal-arrival 路线已经完成训练、独立配对资格和正式绑定。第一次 finite-time checkpoint 集在 fresh 配对中为63/64，并破坏1个当前actor原本成功的pair；把terminal residual改为仅按terminal state归一化后，第二组checkpoint仍为63/64。唯一失败不是约束事件，而是直到最后一个控制周期才首次进入精确终端集，无法满足冻结的15-control连续hold。因此这两组均保留为失败开发证据，没有进入actor manifest。

随后用现有统一runner从第二组actor收集64条完整、未按成功筛选的公开`shared13/B_log/fit`轨迹，共42,492个物理控制步。日志不含reward、evaluator outcome、隐藏机制状态或confirmation状态；其中62条轨迹提供了860个`remaining_controls<=150`的晚阶段训练状态，2条较早结束轨迹显式贡献0个晚阶段状态。v5从八个已选择v4 checkpoint初始化，使用新晚阶段公开状态训练八个独立实例，共12,000次GPU1 update。CPU preflight验证了64条来源的SHA、finite loss、exact optimizer restore和从selected checkpoint初始化；训练没有调用物理、目标reward或confirmation状态。

新鲜seed `4402000--4402007`的四路配对资格完整执行256个episode和170,527个物理控制步。冻结current、v5、band-only和ordinary response MPC分别为33/64、64/64、48/64和57/64，四行均为0约束违规、0执行错误；四行pair-set完全相同，v5没有丢失任何current成功pair。冻结门的七项检查全部通过：完整四路分母、64/64、零约束、零执行错误、保留current全部成功、优于band-only且不低于strong reference。该运行是development qualification，`performance_evidence=false`、`confirmation_states_opened=0`，不能进入论文主表或被描述为确认性优势。

通过后，T3现有task-local `actor_bindings.json`原位更新为`ours_gated_v5_late_phase`并绑定八个SHA固定checkpoint；没有创建平行manifest。完整catalog解析验证通过。CPU和GPU1零物理package审计均为52个可执行row通过、8个既有资源/算法拒绝保留；52个leaf config的shared-protocol与architecture-equivalence审计也通过。训练报告、配对资格报告、CPU package、GPU1 package和architecture报告分别位于`results/arm/keyed_insertion_recovery/training/finite_time_late_phase/checkpoints/`、`results/arm/keyed_insertion_recovery/validation/finite_time_late_phase_paired/gpu1/`、`results/_qualification/four_task_transfer/method_packages_t3_finite_time_late_phase_20261024/`、`results/_calibration/four_task_transfer/method_packages_t3_finite_time_late_phase_gpu1_20261024/`和`results/_architecture/cross_ebdm_shared_protocol_t3_finite_time_late_phase_20261024/`。

理论边界据此进一步明确。finite-time objective必须覆盖靠近deadline的公开状态，terminal residual不能被大量非terminal状态稀释，且reachability margin必须显式扣除连续hold预算；这三项是本次机制设计的经验依据。64/64开发资格仍不构成全局有限时间或鲁棒性证明，也不能单独证明transport贡献。论文中的任务性能结论仍需要新的seed-disjoint正式roster，在相同共享executor、相同admission、相同数据权限与配对reset下比较ours、task-only、全部主表baseline和冻结消融。下一份roster生成前还必须完成受影响T3正式行为predeployment，以及既定的T2共享能力门和gated/ungated约束复现；现有v8及本轮development seed均不得复用。

### 2026-10-24 T2 action-support CPU 收尾与三类失败修改决定

T2 action-support 分支已经用已有 exact-formal-cell fresh 配对证据完成离线收尾。分析重新核对 64/64 个 episode artifact 与 60,137 个已记录控制步，把每条轨迹划分为 contact acquisition、0.2 秒 stable dwell、outlet crossing、post-outlet completion 与 15-control terminal hold。该过程执行 0 个新物理步、0 次 GPU 调用，未访问 confirmation state，也未改变失败、阈值或 actor。机器报告位于 `results/arm/guided_contact_alignment/validation/deadline_reference_admission_formal_cells_paired/action_support_screen.json`，SHA256 为 `db6407e6a57f907c6036ad868463da0258cfb82eda9fc581e425add80221e832`；连续生成两次逐字节一致。

未绑定的 `deadline_aware_v1` 只有 payload-shift/mode-1 具有完整经验 action support。latency/mode-0 在冻结 hard margin 内的正要求控制周期共有 2,165 个，required speed 中位数为 20.0 mm/s，delayed-observation-derived achieved speed 中位数只有 1.80 mm/s、p95 为 3.81 mm/s，achieved/required 中位比为 0.090，8 条轨迹都未到 outlet。latency/mode-1 的前出口进展集合非空且 8/8 到 outlet，但出口后的 required net progress 中位数为 12.37 mm/s，实际只有 5.36 mm/s，比例为 0.434，8/8 都未进入 terminal set。payload/mode-0 有 7/8 到 outlet，出口后的净进展中位数为 -2.15 mm/s，说明 route 最终远离目标；另一条在 outlet 前 overforce。payload/mode-1 才满足前后两段 rate、8/8 completion 和零违规。

因此 CPU screen 为 empty：当前 `task_contract_v3` 下不允许再增加或绑定新的 T2 shared controller，停止 lead、gain 与 speed 的标量搜索，v8 失败保持原样。若论文决定继续修 T2，必须先创建显式的新 task-contract，并修改所有方法共同继承的 task-owned reference/action support：前段要在延迟观测下提供足够的接触后出口进展，后段要提供单调、目标导向且满足 hold reserve 的 completion primitive。deadline、总力/横向力/速度限、严格接触 evaluator、reset cells 和 mode 均保持冻结；public sentinel 必须在 latency/payload × 两个 mode 的四个 fresh cell 各 8/8、零违规，才允许该新合同进入全方法资格。该公共改动不能记为 Cross-EBDM 贡献。

三类失败现在按以下状态处理：1,133 个 strict-guide-contact 失败已经在当前合同上归因为共享 T2 latency/action-support 不足，该分支关闭且不接纳控制器修改；2,192 个 finite-time progress 失败中，T2 仍是共享 post-outlet support 限制，T3 已由绑定的 `ours_gated_v5_late_phase` 在 fresh 四路配对中从 33/64 提升到 64/64，并同时高于 band-only 的 48/64 和 ordinary response MPC 的 57/64，四行均零违规，T4 继续冻结；唯一的 constraint-control 失败仍只支持一个待证伪假设。下一项可执行工作是 fresh T2 gated/ungated stress：固定 payload-shift、xarm7 fixed-base、mode-1，用八个 actor instance × 八个未使用 reset slot，得到每个 comparator 64 条、共 128 条配对 episode；保存 candidate/issued/accepted/applied command、selector、预测与实测 force/speed margin，并对齐最后安全与首个违规 1 ms substep。只有配对违规减少可复现时才保留狭义 constraint-filter 陈述；零复现时删除安全收益陈述且不改 shared protocol。

下一版代码范围由此收紧为四项：保留已绑定的 T3 v5 actor，不再调参；不在 `task_contract_v3` 下添加 T2 controller；只在现有 `scripts/paper/cross_ebdm` 诊断入口和 task-local validation config 中加入上述 fresh paired constraint stress，不修改 actor binding 或共同阈值；T4 completion projection 保持冻结。新 confirmation roster 仍未放行。

### 2026-10-24 T2 constraint stress 收尾与下一版 actor 决定

预声明的 payload-shift、xarm7 fixed-base、mode-1 constraint stress 已完成。诊断链在现有 `qualify_method_behavior.py` 中记录 candidate/common/selected/requested/issued/applied action、selector，以及每个 1 ms 物理子步的关节速度、总力和横向力；这些字段只写入 evaluator evidence，不进入 policy observation，也不修改物理、判据或任务合同。

第一轮 128 个 episode 虽然两行均为 64/64 success、零约束失败，但离线动作链发现所谓 `ungated_full` 也执行了 public contact guard：gated 与 ungated 分别有 16,250 和 16,181 个 guard-selected controls。原因是当前冻结 manifest 给所有 Cross-EBDM ablation 行都附加了相同的 `contact_guard_version`。该轮 evidence 全部保留，report 已明确写为 `passed=false`、`comparator_semantics_valid=false`，禁止用于机制、性能或安全结论。

第二轮使用全新 seed，把完整 `gated_full` 与真正移除 runtime guard 的 `ungated_full` 做 64 对比较。两侧仍均为 64/64 success、零约束失败、零 force/speed hard-limit event；gated 有 16,108 个 candidate-to-selected change，ungated 为零。因为两行使用各自 checkpoint，这一轮只说明完整方法行在该 cell 上都可完成，不能单独归因 guard。

最后一轮再用全新 seed 和逐对完全相同的 `gated_full` checkpoint，只切换 runtime guard。64 对中两侧仍均为 64/64 success、零约束失败，所有 episode 都有精确 1 ms evidence，且 64 对都没有 force/speed event。guard-on 有 16,353 个 action change，在 61/64 对中完成更慢，平均多 61.25 个 control steps。guard-on 的关节速度峰值 p95 为 0.7245 rad/s，guard-off 为 0.2218 rad/s；总力峰值 p95 分别为 4.5149 N 和 4.7237 N，横向力峰值 p95 分别为 1.9714 N 和 2.1380 N。两侧都远低于冻结的 25 N 总力、10 N 横向力和 3 rad/s 速度限。配对事件数为 both/gated-only/ungated-only/neither = 0/0/0/64，因此没有 discordant pair，也不存在可报告的 McNemar 效应或置信区间。

约束分支的决定由此冻结：v8 的单个 ungated speed-limit 事件没有在两套有效 fresh stress 中复现，当前 public contact guard 没有任务成功或 hard-limit 减少证据，反而显著增加完成步数并提高部分实例的速度峰值。论文删除“transport/contact guard 带来 constraint-filter 或 safety 收益”的陈述；现有 heuristic guard 不能称为安全层、风险接纳或理论不变集。当前冻结 manifest 和历史结果不原地改写；下一版 actor manifest 才从 main Cross-EBDM 及其 ablation 行移除 `contact_guard_version`，随后用新 seed 重跑受影响的 CPU behavior/predeployment 门。若未来恢复安全陈述，必须另建具有公开候选风险预测、校准误差界、非空安全且正进展集合以及事件丰富 held-out stress 的机制，不能复用本 guard 名称或本轮零事件结果。

三类失败的最终处理现在是：1,133 个 strict-guide-contact failure 继续归于 T2 共享 action-support/latency 缺口，当前 `task_contract_v3` 下不加 controller；2,192 个 finite-time progress failure 中，T2 共享 completion 分支保持未解，T3 由 `ours_gated_v5_late_phase` 的 fresh 64/64 相对 band-only 48/64 和 response MPC 57/64 提供当前唯一正 transport 证据，T4 冻结；唯一 constraint-control 观察经 fresh stress 未复现，关闭为不支持安全贡献。下一步不需要 GPU 训练：先生成语义修正的 task-local actor manifest revision、执行零物理绑定审计，再重跑移除 guard 后受影响的 T2 CPU behavior rows。只有这些行仍满足现有门，才冻结新的 confirmation roster；T2 新公共 action-support 合同若要继续，必须作为独立公共协议工作重新资格化全部方法，不能计入 Cross-EBDM 贡献。

三轮工件分别位于 `results/arm/guided_contact_alignment/validation/gated_ungated_constraint_stress/`、`gated_true_ungated_constraint_stress/` 和 `same_policy_contact_guard_stress/`。最后一轮 report SHA-256 为 `ec7c528990870206bb829d0ce6e235325e489b1b39ffd15a98e9aca36ce560e6`，plan 为 `2ae7eca106aa84379cdec5cc498e4c2406b6cb424ca4915404369d87bcbac8aa`，progress ledger 为 `09aaa80c529fb0e57fe51b1322c9ea6ebffc15b8391801bd0e5b099db477dfa2`；128/128 evidence SHA 逐项复核通过，execution error 为零，进程的 `CUDA_VISIBLE_DEVICES` 为空。

理论需要把 liveness 写成条件命题。安全/接触不变性、有限时间可达性和 terminal dwell 是三个独立性质；有限时间结论必须显式假设延迟、force、reference lead 与 hold reserve 约束后的正进展可行动作集非空。T2 的 empty empirical support 是该假设不满足的边界证据，不是任一方法的性能证据，也不是连续动作空间上的数学不可行证明。T3 只支持“晚阶段覆盖与显式 hold reserve 能修复 terminal arrival”这一机制，不能单独证明普遍 transport 优势；fresh same-policy stress 已否证现有 heuristic guard 的 constraint-filter 证据，因此论文不再保留该安全陈述。

### 2026-10-24 正式离线统计与下一版修改收尾

使用同一 `scripts/paper/cross_ebdm/analyze_shared_failure_closeout.py` CPU 入口重新生成正式失败分类、development diagnosis、T2 action-support screen 与最终 next-revision closeout。四项工件连续重建两次后逐字节一致，`CUDA_VISIBLE_DEVICES` 为空，执行 0 次 GPU 调用、0 个物理步、0 次 confirmation state 访问。正式 v8 的 21,200 个记录、15,360 个物理执行、5,840 个预部署计分槽和 44 个保留基础设施失败均与封存 audit 一致；三类物理失败仍为 1,133 个 strict-guide-contact、2,192 个 finite-time progress 与 1 个 constraint-control。

最终决定不启动当前已封存且未访问的 confirmation-v10 包。该包已经正确包含 T2 `ours_transport_only_v3`、T3 `ours_gated_v5_late_phase`、通过的四任务行为资格与零物理 architecture/package/device 审计，但仍继承 `task_contract_v3`。已有 exact-formal-cell CPU screen 证明其 T2 经验 action support 只在四个 mechanism-by-mode cell 中的一个非空，而 transport-only 资格中的 public sentinel 只有 4 个 known-nominal episode；它不能替代每个 formal cell 各 8 次的共享能力门。现在消耗新的 confirmation state 会在已知共享 T2 缺口上重复花费正式预算。

下一版只保留三项方法侧决定：T1 不变；T3 保留已绑定并在新鲜四路配对中达到 64/64、相对 current 33/64、band-only 48/64、response MPC 57/64 的 late-phase actor；T4 保留 completion-projected final actor。T2 方法 actor 保留已经去除 heuristic contact guard 的 transport-only revision，但必须先建立显式的新 shared task contract：前段提供 delay-aware contact acquisition，后段提供 monotone goal-directed completion，同时保持 deadline、force/speed limits、strict evaluator、reset cells 与 mode semantics 不变。该公共修改应用到全部方法，也不能记为 Cross-EBDM 贡献。

放行顺序冻结为：先用 fresh seed 要求 public geometry 在 latency/payload × mode-0/mode-1 四个 cell 各 8/8 strict success 且零违规；再重跑所有受影响 T2 行及独立审计；随后重跑零物理 architecture、CPU package 与 GPU1 device audit；所有门通过后把未访问的 v10 标为 obsolete，生成新的 seed-disjoint package 并只启动一次。理论同步加入“安全可行动作集合与 deadline 内正进展动作集合交集非空”的条件，分开 contact acquisition、contact preservation、finite-time reachability 与 terminal dwell；删除 heuristic guard 的安全层/constraint-filter 陈述，并把 T3 development 结果限定为待正式验证的 late-phase mechanism evidence。

机器可读最终决定位于 `results/_analysis/four_task_transfer/confirmation_v8/next_revision_closeout.json`，SHA256 为 `f62344bd89d93c9cd58f29110cfa83eb33347eac5c775905ea76633c1c5152aa`。重建后的 formal diagnosis、development diagnosis 与 T2 action-support screen SHA256 依次为 `67d66002bba9cfd44746ff436f05126225952f6c999cba122f6b14a7b00a3561`、`0862b9c90beaaaad4450c99074de2324a8c4f43c03d2ad785c955330fc5d1549`、`5aff102c8bbe1bc2dfd83d2955c56ce1dd49edb788468760c3994a37830bf669`。

### 2026-10-24 T2 公共执行器候选离线收尾

在不启动新物理、不访问 confirmation state、`CUDA_VISIBLE_DEVICES` 为空的条件下，CPU closeout 又复核了 10 组公共执行器候选的 256 个诊断 episode；其中候选侧 128 个，成功 38 个，约束失败 23 个，执行错误为零。所有 episode evidence 的 SHA、progress ledger 与 report 均逐项校验。没有任何候选在 latency-gain/mode-0 成功；即使切向命令达到声明的 80 mm/s 上限，仍累积明显的 reference tracking lag，同时保有 joint margin 且未触发该 cell 的硬约束。这把剩余缺口定位为低增益延迟条件下的闭环跟踪能力，而不是再换一个 terminal route 就能解决的问题。

payload 两个 mode 提供了另一半证据：bounded-clearance completion、单轴 concurrent acquisition 及其后续候选均可达到 4/4，说明出口后的几何 completion 在控制权足够时可执行。相反，直接取消控制率模型或同时移动两个平面轴会产生 overforce；累计 23 个候选约束失败否决了全局无条件加增益、dual-waypoint 和移除 1 ms governor。staged acquisition 只把 latency/mode-1 提高到 1/2，latency/mode-0 仍为 0/2，所以它同样不能绑定。

下一版只实现一个新的、任务拥有的 T2 公共执行器合同：保留单轴 acquisition 和 bounded-clearance、单调朝目标的 completion；根据公开 TCP 位姿、公开 reference、实测 wrench 与时间估计 tracking lag，在正进展不足时有限提高声明的 impedance/reference authority；任何阶段的 reference 都必须经过现有 1 ms wrench 与 tangent-effort governor。执行器不得读取 `servo_gain`、delay、payload、evaluator support truth 或 method identity，并必须把 authority ceiling、适配状态以及 issued/admitted/applied action 写入证据。20 秒 deadline、25 N 总力、10 N 横向力、3 rad/s 关节速度、严格接触 evaluator、reset cells 和 mode semantics 全部保持不变。

验证分两级。先用全新 xArm7 latency-gain reset 做 public-sentinel tracking-authority screen，检查单轴接触/出导面进展、tracking lag、实测速率、joint margin 和每个 1 ms 子步的力/速度 margin；任何违规或没有正 outlet progress 都立即否决。通过后再跑 latency/payload × mode-0/mode-1，每个 cell 8 个全新 paired slot，要求每格 8/8 strict success、零约束、零执行错误和完整 evidence。只有公共门通过，才把同一个执行器绑定到全部 T2 main、baseline 和 ablation 行并重跑 behavior qualification。该改动属于 benchmark infrastructure，不能作为 Cross-EBDM 贡献。

三类正式失败的理论处理也随之冻结。strict contact 的 1,133 个失败要求把接触获取和接触保持分开；finite-time 的 2,192 个失败要求分别预算 acquisition、outlet traversal、goal completion 和连续 15-control hold，并明确满足 `tau_acq + L_out/v_out + L_goal/v_goal + 15*dt <= 20 s`；constraint 类的单个正式事件已被 fresh same-policy stress 否证为可复现安全收益，因此不保留 safety-layer 或 constraint-filter 陈述。有限时间定理必须增加“安全可行动作集合与 deadline 正进展动作集合交集非空”以及实测闭环进展下界的条件，单独的 commanded-velocity bound 不足以推出可达性。

失败候选的临时 controller 类、factory admission、白名单和 10 个 task-local 输入 JSON 已从工作树清理；不可变 report、plan、progress 与逐 episode evidence 继续保留。复用的 bounded-clearance route 留在 `genedynamics/tasks/transfer/controllers`，下一版仍从现有 `scripts/paper/cross_ebdm` 入口驱动，不增加新的顶层目录或平行入口。更新后的机器决定位于 `results/_analysis/four_task_transfer/confirmation_v8/next_revision_closeout.json`，连续两次生成 SHA256 均为 `9bdebe62fcc48af36342ee6767e17143ac5be799aa9f4e4837f4d46191eff014`。

### 2026-10-24 T2 response-normalized CPU 诊断与终端捕获修改

任务拥有的 response-normalized 原型又完成了 11 组 fresh CPU development screen，共 88 个 physical episode；候选侧 44 个，成功 8 个、约束失败 9 个、执行错误为零。这里的累计数包括逐步被否决的 terminal-force、surface-mode-1 和 direct-admittance 路线，不能当作最终候选成功率，也不是论文性能证据。所有运行均为 `diagnostic_only=true`、`performance_evidence=false`、`confirmation_states_opened=0`，失败没有替换。

诊断给出了稳定的 mode 分解。mode-0 保留 surface-following completion 后，在最终 fresh latency 两个 reset 上达到 2/2 strict success；mode-1 若继续贴面走第二轴会产生 27.97--30.79 N overforce，因此必须在 outlet 后切换到 collision-free lift/completion。采用该 mode split 的最终小屏幕为 3/4 success、零约束、零执行错误；四个 episode 都满足 strict contact，mode-1 为 1/2 success。唯一失败在 tick 509 出 outlet，20 秒末的 position error 为 2.144 mm，仅比冻结的 2 mm terminal set 多 0.144 mm；总力峰值 10.97 N、横向力峰值 4.45 N，且 terminal hold count 始终为 0。因此它是 terminal-set entry 前的有限时间捕获失败，不是接触失败、几何路线失败或约束失败。

轨迹进一步定位了代码机制。末段 TCP 位于目标 x 方向外侧约 2.09 mm，而 applied reference 已到目标另一侧约 0.15 mm；两者的三维距离仍为 4.25 mm。当前包装器把 reference 限制为不能越过单一 goal point，reference 到达该点后把 active-axis command 截成零，低增益延迟条件下便无法继续消除 TCP/reference lag。同时 required rate 使用到 goal point 的误差，并在扣除 3 秒 completion reserve 后把剩余时间夹到一个 control tick，末端持续报 30 mm/s required speed；这个量既不是到 terminal set 的所需进展，也不能证明还能完成 15-control dwell。

下一版只修改 response-normalized terminal capture。acquisition、mode-0 surface route、mode-1 outlet 后 lift route、20 秒 deadline、2 mm/2 degree terminal predicate、15-control dwell、force/speed limits、reset cells、mode semantics 和 evaluator 全部保持不变。terminal set 外按“到精确 terminal set 边界的距离 / dwell reserve 前的可用时间”计算 required progress；允许 reference 在声明的 lead envelope 内、按实测 TCP/reference lag 做有界越过 goal-point，只接纳单调减小 terminal-set distance 且通过既有 1 ms governor 的动作；进入 terminal set 后收缩切向 residual 并完整保留 15 个 control 的 dwell。不得加入 hybrid force、隐藏 mechanism label、method identity 或 threshold/deadline relaxation。

三类共同失败由此更新。1,133 个 formal strict-guide-contact failure 的共享 acquisition 机制在 4 个 fresh latency 小样本上得到支持，但尚未通过 latency/payload × mode 的 32-episode 公共门，不能称为已解决。2,192 个 finite-time failure 中，T2 现在收敛为上述 terminal capture，T3 继续使用已通过 development qualification 的 `ours_gated_v5_late_phase`，T4 不变。唯一 formal constraint event 的安全收益仍未复现；9 个开发约束失败只用于否决 terminal-force 和 surface-mode-1 路线，不恢复 safety-layer 或 constraint-filter 陈述。

放行顺序为：先用全新 latency/mode-1 reset 做 terminal-capture CPU screen，要求 strict contact、零约束、在 15-control reserve 前进入 terminal set并完成 dwell；随后执行四个 public-geometry cell 各 8/8、零约束的完整公共门。任何一项失败都不创建 task contract。全部通过后才版本化共享合同、重跑所有 T2 method/baseline/ablation 行及独立审计，再做 architecture、CPU package 和 GPU1 device audit。当前未访问的 confirmation-v10 继续保持 `hold_do_not_launch`。

CPU 收尾入口已把旧 10 组 route/action-support screen 与新 11 组 response-normalized screen 合并进同一机器决策。`results/_analysis/four_task_transfer/confirmation_v8/next_revision_closeout.json` 连续两次生成逐字节一致，SHA256 为 `a64ef71c5d030163e65fda3bb6fda0e67d9c70d64eb96d3634fef5bc292dfca8`；本次重建设置 `CUDA_VISIBLE_DEVICES` 为空，没有 GPU 调用、没有新物理执行、没有访问 confirmation state。

### 2026-10-24 T2 response-normalized 公共门最终 CPU 收尾

按照上一节冻结的公共观测边界，T2 response-normalized 执行器完成了最终四格 release gate。运行只使用公开 TCP pose、公开 reference、TCP twist、实测 wrench 与剩余时间；不读取 delay、payload、servo gain、evaluator 内部状态或 method identity。最终 `latency-gain/payload × mode-0/mode-1` 每格 8 个 fresh episode，共 32/32 strict success，零约束失败、零执行错误，32 个 evidence audit 和 exact-physics audit 全部通过。

| 条件 | mode | strict success | 最大总力 | 最大横向力 | 完成步数范围 |
|---|---:|---:|---:|---:|---:|
| latency-gain | 0 | 8/8 | 18.224 N | 4.289 N | 902--934 |
| latency-gain | 1 | 8/8 | 13.026 N | 4.464 N | 946--958 |
| payload | 0 | 8/8 | 14.384 N | 5.179 N | 848--879 |
| payload | 1 | 8/8 | 6.286 N | 2.952 N | 937--951 |

三类共同失败已经定位到可检验的控制机制。第一类是 phase-transition reference energy：mode-0 仅按几何位置切换 outlet 时，TCP 仍可能落后 reference，后端保留的阻抗投影会把 reference jump 变成力峰值；简单改成 measured-set transition 或固定 lead cap 会分别产生过力或停滞。最终修改在公开几何出口前限制 lead 与阻抗能量，到达端点后锁存状态，并按 20 mm/s 上限把 reference 向实测位置卸载；切向弹簧预算按横向力上限的 55% 限制为 8.8 N。

第二类是 finite-time terminal capture：被动等待 TCP 收敛会把 route completion 推到 deadline 附近，无法留下连续 15-control hold。mode-0 现在显式预留 3.0 s completion window，并用端点锁存避免 reference 在边界来回切换；mode-1 在已有 dwell 预算之外再保留一个完整 dwell window。第三类是 clearance descent/contact coupling：mode-1 固定高速下降会在重新接触时产生尾部过力，硬停又会造成 hover、二次接触和 timeout；最终下降上限连续乘以公开 wrench 导出的 force-release factor，在远离平面 terminal set 时也保留有限的安全下降。

这些修改支持把下一版共享 T2 合同版本化为 `task_contract_v4`。合同应冻结上述 controller version、公开输入、20 s deadline、2 mm/2 degree terminal set、15-control dwell、25 N 总力、10 N 横向力和现有 mode semantics，并把同一执行器绑定到 T2 的 main、全部 baseline 与全部 ablation。版本化之前还需完成 behavior qualification、架构审计、CPU package audit 和 GPU1 device audit；通过后才能启动新的 GPU1-only P6。

理论部分需要同步收紧三点：可达性定理必须显式要求安全可行动作集合与 deadline 正进展集合的交集非空；finite-time 界必须分解 acquisition、outlet traversal、goal capture 和 dwell，并以实测闭环进展下界替代 commanded-velocity bound；phase transition 还必须加入有界 reference energy/lead 和终端卸载条件。当前 32/32 只证明共享 T2 benchmark infrastructure 的开发门通过，不是论文主表性能证据，也没有证明 transport 对任务性能有优势。正式结论仍必须来自冻结 `task_contract_v4` 后 main、baseline、ablation 的全新运行。

最终可复现入口只保留 `configs/arm/guided_contact_alignment/validation/response_normalized_public_geometry_release_gate.json`；中间调参配置已删除，不可变 report 和逐 episode evidence 保留。最终 report 位于 `results/arm/guided_contact_alignment/validation/response_normalized_public_geometry_release_gate/report.json`，SHA256 为 `6eaba450c18559634d0049a5ab39770402cf2698a07b9cba22b5f69d0c4fadfb`。离线机器决定位于 `results/_analysis/four_task_transfer/confirmation_v8/t2_response_normalized_revision_closeout.json`，连续两次生成 SHA256 均为 `c6eb84199310cd4dc48c481b56e2a57c80252fce3bc23053916ad1f176c72d99`。

### 2026-10-24 T2 task contract v4 的 fresh-seed 修正与 CPU 冻结候选

上一节的 32/32 结果没有在后续不重叠 seed 上保持，因此不得继续作为 release 证据。绑定旧 position-impedance acquisition 的 v44 在 fresh 四格门中只有 15/32：latency/mode-0 为 1/8、latency/mode-1 为 0/8、payload/mode-0 为 6/8、payload/mode-1 为 8/8，并出现 3 次 overforce。该结果和随后 v45--v48 的失败链完整保留为 development evidence，但旧 report、旧 controller version 和旧 closeout 已被新合同候选取代，不能进入论文性能或协议放行判断。

backend 审计确认共享 13D 动作的第 13 维已经提供 backend-neutral 法向混合力通道：启用时只冻结法向位置 reference，接触前执行有界导纳接近，接触后闭合测力反馈，切向 reference admission 保持独立。v49 因此用 5 N 公共力目标完成导面获取和表面运动，在 mode-0 首次进入冻结的 2 mm terminal set 时切到 6 N terminal hold；它没有读取 delay、payload、servo gain、evaluator support truth 或 method identity。

mode-0 相变同时改为三段公开状态机。出口前 2 mm 开始把切向刚度从响应权限值降到基础值，为最大声明延迟留下执行余量；越过公开 outlet 后锁存端点，继续降低 reference spring energy，并要求 reference lead 不超过 2 mm、实测轴速不超过 4 mm/s 后才切换第二轴；内部 terminal latch 与冻结的 2 mm task set 对齐，避免 wrapper 已停止而 route 仍等待隐含 1.5 mm 条件的死锁。20 s deadline、2 mm/2 degree terminal predicate、15-control dwell、25 N 总力、10 N 横向力、3 rad/s joint-speed limit、reset cells、mode semantics 和 strict evaluator 均未改变。

最终 release gate 使用 seed anchor 4678000，与此前诊断及失败回归不重叠。四格均为 8/8，总计 32/32 strict success、零约束失败、零执行错误，32/32 evidence audit 和 exact-physics evidence 完整：latency/mode-0 最大总力 17.428 N、最大横向力 4.366 N、936--971 步；latency/mode-1 为 10.233 N、4.273 N、949--968 步；payload/mode-0 为 17.973 N、7.359 N、914--938 步；payload/mode-1 为 14.339 N、6.622 N、937--952 步。report SHA256 为 `42bae118e4718b19f0107d50dd1a341a0c0b4d8b317aee1a790c9b77a2cdeba6`，progress ledger SHA256 为 `68577cee5c9256ae08789cc0373a41d66fca100c36c32d228724fd2c5f767e2f`。

三类共同失败的修改决定随之更新。strict-guide-contact 类采用共享 hybrid-force acquisition，明确把“接触获取”与“接触保持”分开；finite-time 类采用 pre-outlet stiffness ramp、reference unload、task-set-aligned terminal latch 和提前 terminal force hold，有限时间条件必须覆盖 acquisition、两段平面进展及 dwell；constraint-control 类只得到 task-local 32 次零违规资格，不能升级为一般 safety theorem。相变理论必须约束执行延迟后的 reference spring energy，而不能只约束当前 requested action；liveness 条件必须使用实测闭环进展和 terminal-set dwell，不再由 commanded speed 推出。

`task_contract_v4.json` 已绑定 controller `public-T2-response-normalized-hybrid-force-v49`、5/6 N 力目标、2 mm pre-outlet ramp 和 8.8 N transition spring budget；task-local base 通过同一合同继承到所有 T2 main、baseline 和 ablation。81 个 contract/controller/adapter/evidence 测试通过，52 个配置的零物理 architecture verification 通过。CPU 离线决定位于 `results/_analysis/four_task_transfer/confirmation_v8/t2_task_contract_v4_cpu_closeout.json`，SHA256 为 `0ab53fe2d87c71c50e5ec3d3dbfbecae2134a371054090e22292eff58a0c0668`。当前状态只是“公共 sentinel 门通过、全方法 predeployment 待跑”；下一步必须在 fresh predeployment 中重跑全部 T2 main/baseline/ablation 行，再做 CPU package 和 GPU1 device audit。transport 仍没有任务性能优势证据。

### 2026-10-24 T2 outlet latch v50、v12 资格与 CPU 离线诊断

v49 的 fresh predeployment 暴露出两个 latency/mode-0 timeout。逐控制周期证据表明 route 已经观测到公开 outlet crossing，但 TCP 在 reference unload 后短暂退回阈值内侧，旧 `_outlet_ready` 又要求当前几何位置仍在 outlet 外，导致 stage-2 永久等待。route-v4 将已经观测到的 outlet crossing 改为单调锁存；controller-v50 随之完成 32 个不重叠 fresh public-geometry episode，四个 latency/payload × mode cell 均为 8/8 strict success、零约束、零执行错误。该修改只修正公开状态机记忆，不改变 deadline、terminal set、force/speed limits、reset、mode 或 evaluator。

绑定 controller-v50、route-v4 与 `task_contract_v4` 后，v12 在禁用 CUDA 的条件下完成全部 720 个 T2 predeployment episode。独立审计重新读取并核验 720/720 个 evidence，计划与结果一一覆盖、SHA 全部匹配、无 reset/execution error、无 confirmation state 访问，也没有复用旧 invalid outcome。总计 560 success、82 timeout、78 overforce、0 guide-contact-continuity failure；15 个逻辑行中只有 public geometry 行以 8/8 behavior-qualified，12 个可执行方法/消融行 behavior-rejected，ACP 与 Human2Sim2Robot 的原有两类预部署拒绝继续保留。Cross-EBDM 为 62/64，两个失败均为 mode-1 overforce；task-only 为 61/64；response MPC 为 60/64；FADA 为 60 overforce 加 4 timeout；DIAL 为 64 timeout；direct source prior 为 8 timeout。公共基础设施门通过不等于主方法可进入 P6，Cross-EBDM 仍未通过行为门。

160 个失败的 CPU evidence 形成三个互斥机制簇。第一簇有 24 个，覆盖 Cross-EBDM、task-only、response MPC 与六条 learned ablation：全部是 mode-1，终态均为 route stage 3、completion stage 2、`terminal_pose_active=true`，其中 18 overforce、6 timeout。公共几何在相同八个 reset 上 8/8，说明问题位于学得或优化动作在 terminal capture 已开始后仍保有 transport 权限；Cross-EBDM 与 task-only 的共同失败也否定了“先改 transport 训练即可”的解释。第二簇为 FADA 的 64 个早期闭环碰撞/停滞，需要检查 source fixture、normalization、decoder 与 Box 语义后再决定是实现缺陷还是方法限制。第三簇为 DIAL 的 64 个 timeout 和 direct source prior 的 8 个 timeout，需要对齐 phase occupancy、objective reduction、accepted-reference progress 与剩余 deadline；接口正确后的停滞保留为 baseline limitation。

下一版先做有界 CPU 配对反事实，不启动 GPU 训练：保留 v50/v4 及所有公共阈值，只在 public terminal capture 开始后让任务 completion controller 精确拥有完整 request，使所有方法返回相同 public common action。只有该单一修改在 v12 的精确失败请求上消除全部 terminal-handoff failure、保留全部配对成功，并在 fresh seed 上通过 public 与 learned-actor 门，才把它版本化为统一 T2 方法接口。若合法方法权限区间内仍有失败，再考虑重训。随后重新运行全部 T2 behavior qualification、独立 evidence audit、零物理 architecture、CPU package 和 GPU1 device audit；Cross-EBDM 主行合格前继续阻止 P6 performance roster。

理论也必须增加方法权限的有限生命周期：candidate transport 只在声明的接触迁移阶段有效，terminal completion 是任务拥有的 invariant regime。transport quality、finite-time completion、guide-contact continuity 与 force safety 分开陈述；资格结果只说明部署兼容。v12 没有提供 transport 的任务性能优势，最终贡献仍需在共同生命周期合同冻结后，以 seed-disjoint 的 Cross-EBDM、task-only 和冻结消融配对比较支持。

机器可读资格位于 `results/_qualification/four_task_transfer/t2_task_contract_v4_outlet_latched_predeployment_20261024/`：report SHA256 为 `2833bc9a5614ab2411fcff4ae442f59b6090e34b3ed48e7a1ebe5131b55a3426`，progress SHA256 为 `5d64401ec8b4efb58b5c79e34628d9c5396de31d67d6ee6b4b81752a3888e8c3`，independent audit SHA256 为 `3cb5789ae545cf3aa5ac2d549749fc2faad8a800ec42dd2e8e61baeaf86fafb2`。离线统计与诊断位于 `results/_analysis/four_task_transfer/t2_task_contract_v4_outlet_latched_predeployment_20261024_offline_closeout/`；`closeout.json` 与 `diagnosis_plan.json` SHA256 分别为 `368c1e9a535c56f7e8a50b5bd280348f809de4c130ed065b08e06f0382673de3` 和 `1cabcc6a42d90901698c36d5ecfb566b608ada81a8ee952b265b523dadbfb822`。绑定 v12 report 的零物理 architecture verification 检查 52 个配置并通过，report SHA256 为 `fd3721fe6f916160e52f91263c38c8505348a8a3c917d8e3bc28879a5c8c6da3`。

### 2026-10-24 T2 terminal authority 精确回放与 CPU 离线收尾

在不运行新物理、不调用 GPU、不访问 confirmation state 的条件下，现有入口 `scripts/paper/cross_ebdm/analyze_predeployment.py` 对 v12 与公共执行器候选做了 576 个方法 episode 的逐 reset 配对。v3 的 584 个诊断 episode 已通过独立 evidence audit，其中公共几何仍为 8/8。方法配对转移为：551 个 success→success、18 个 overforce→success、2 个 timeout→success、4 个 timeout→timeout，以及 1 个 success→timeout。因此 v3 虽把总约束失败降为零，也不能放行；任何共享修改都必须保留全部 552 个 v12 配对成功。

旧的 24 个 terminal-signature 失败不是单一根因。18 个过力在 v3 下全部成功，支持“mode-1 clearance/下降期间残差耦合”这一机制；2 个末段 timeout 也因权限衰减而成功；另 4 个 timeout 保留，其中 task-only 的重复主行/消融行在 tick 609 才出 outlet、tick 932 才进入 completion stage 2，response MPC 的两个样本到 tick 959/962 才进入 stage 2，已经没有足够预算完成 15-control dwell。这 4 个属于更早的进展预算损失，terminal wrapper 不能把它们算作已修复。唯一新回归是 ungated-full、latency、slot-1、mode-1：v12 在 990 步完成，v3 在 1000 步仍超时；两者 outlet 和 clearance 时间接近，差异定位到 v3 在 stage-2 将残差权限降为零的窗口，但仍需下一次隔离反事实确认具体必要分量。

v1、v2、v3 的否证链因此完整：v1 到 terminal capture 才切换，已经太晚，59 个已执行 episode 中仍有 2 overforce 与 1 timeout；v2 从 outlet 硬切 common，66 个 episode 即产生一个原成功样本 overforce；v3 用 planar remaining distance 连续衰减 clearance residual，完成 584 个 episode并消除全部过力，却产生上述一个成功回归。下一版冻结为 `public-T2-clearance-terminal-set-admission-v4`：mode-0、acquisition、outlet、lift 和 v3 的 clearance-traverse 距离包络不变；stage-2 只在 T2 transport 坐标 `{0,1,3,4,5}` 上测试 `{1,.5,.25,0}` 四个残差比例，task-owned 坐标 `{2,6,7,8,9,10,11,12}` 始终精确等于 common。候选比例必须使公开的一步平移和旋转 terminal-set 进展不差于 common，再乘已有 terminal force-release factor；进入精确 terminal set 后返回 exact common。该设计不增加新阈值，不读取 method identity、机制标签或 evaluator 私有状态。

v4 先在相同 576 对上做 CPU exact replay：要求保留 552/552 个 v12 成功、Cross-EBDM 64/64、零 overforce、零 guide-contact failure、public sentinel 8/8，并逐 tick 调用 candidate 与 common。任一成功回归立即早停封存。通过仍只算诊断；随后必须用 fresh seed 重跑完整 T2 behavior qualification 与独立审计，才允许创建新的共享 task-contract version。4 个上游预算 timeout 可以保留为各自方法限制，不能靠放宽 deadline 或替换 outcome 消掉。

另外两类共同失败保持独立诊断。FADA 为 60 overforce、4 timeout，49/64 未到 outlet；先核验 source fixture 重建、normalization round-trip、decoder frame/unit/Box 与 teacher-forced source error，再对齐首次闭环分歧和首个 unsafe 1 ms substep，只修可复现实现缺陷。DIAL 的 64/64 timeout 全部未到 outlet且力峰值仅 6.10--7.38 N，应审计 objective decrease、projected-gradient norm、active bounds 与 accepted-reference progress；direct source prior 的 8/8 timeout 全部到过 outlet，应单独核验 source frame/mode/unit 及 issued/accepted/applied progress。接口正确后的碰撞或停滞保留为 baseline 结果。

理论相应改为状态依赖的 residual admissible set，不再使用二元“terminal 前有权、terminal 后无权”。terminal regime 由到精确 terminal set 的单调进展和 invariant dwell 定义；有限时间条件必须在 terminal capture 之前显式预算 acquisition、outlet、clearance 与 hold，terminal projection 不能修复上游预算耗尽。共享执行器可行性、transport residual 的性能收益、force safety 与 finite-time completion 继续分别陈述；本轮仍没有 transport 带来任务性能优势的证据。

机器可读收尾位于 `results/_analysis/four_task_transfer/t2_terminal_authority_exact_replay_closeout_20261024/`。该目录的 `closeout.json`、`diagnosis_plan.json` 和 manifest 由同一 CPU 命令连续重建并逐字节核对；P6 仍保持阻塞，下一步是实现并只跑 v4 exact paired gate。

### 2026-10-24 T2 terminal-set admission v4 的 CPU 早停与因果收尾

v4 exact replay 在已知回归请求排到第 4 个 job 的顺序下启动，只完成了 5 个具有完整 evidence 的物理 episode：4 success、1 timeout。停止进程时 journal 另外保留了 2 条没有 episode evidence 的 `reset_or_execution_error`；它们被明确记为进程中断记录，不进入候选方法的 outcome 分母。已知的 v12 success→v3 timeout 请求在 v4 下仍然 timeout，因此 584-job 全量回放立即终止，v4 被封存为 `rejected_early_preserved`，没有用后续结果替换失败。

同一 reset seed 4682003 的三条轨迹给出了直接对照。v12 在 990 controls 完成，最终位置误差 1.853 mm，并在 tick 989 完成 15-control dwell；v3 在 1000 controls 时仍为 3.443 mm；v4 将误差降到 2.024 mm，但仍比冻结的 2 mm terminal set 多 24.2 μm，而且已经没有 dwell 预算。三者的 first outlet 都在 tick 597，v3/v4 的 completion stage-2 都在 tick 914，force peak 与 lateral-force peak 均保持 8.236 N 和 3.137 N。因此这次失败不是新的接触或过力问题，而是 terminal residual 接纳规则留下的有限时间缺口。

v4 在 stage-2 的 86 个 controls 中接纳 transport residual 56 次、回退 exact common 30 次。离线重新计算了每次回退时受同一个 public force-release factor 约束的 transport residual：30/30 都满足“在剩余 controls 中进入 terminal set并预留 15-control dwell”的绝对单步预算，却因相对 common 的预测位置距离差 0--9.26 μm 而被拒绝。该差异来自 command-space 单步代理；它没有覆盖 reference lag、耦合执行状态和 dwell reserve。现有 “no worse than common” 判据因此被否证为 closed-loop viability certificate，不能通过继续扫描 `{1,.5,.25,0}` blend 修复。

下一候选冻结为 `public-T2-clearance-terminal-transport-v5`，只做一次隔离反事实：保留 v3 clearance-distance envelope；进入 stage-2 descent 后以及 terminal dwell 期间，持续保留 transport 坐标 `{0,1,3,4,5}` 上的 residual，并继续乘现有 public force-release factor；任务拥有的 `{2,6,7,8,9,10,11,12}` 始终逐项等于 common；删除 v4 的相对 common blend search。controller-v50、route-v4、20 s deadline、2 mm/2 degree terminal set、15-control dwell、25 N/10 N/3 rad/s 限值、reset cells、mode semantics、evaluator 和 1 ms governor 均不改变。

执行顺序先做两个 exact-reset CPU preflight：一个是上述 success 回归 seed，另一个是 v12 的代表性 overforce seed。前者必须恢复 success，后者必须继续保持 force-safe，否则立即封存 v5且不扫描增益。两者通过后才运行 576 个方法配对和 8 个 public sentinel，要求保留 v12 的 552/552 paired successes、Cross-EBDM 64/64、零 overforce、零 guide-contact failure和 sentinel 8/8；4 个已经确认在 stage-2 前耗尽预算的 timeout 保留为方法限制。CPU exact gate 之前不训练、不调用 GPU；通过后仍需 fresh-seed behavior qualification 和独立审计，最后才做 GPU1 package/device audit。

另外两类共同失败继续使用独立修复门。FADA 的 60 overforce、4 timeout、49/64 未到 outlet，先离线核验 source fixture、normalization inverse、decoder frame/unit/Box 和 teacher-forced source error，再对齐最后安全与首个 unsafe 的 1 ms substep；只修可复现 adapter/decoder 缺陷。DIAL 的 64/64 no-outlet timeout 需要记录 objective decrease、projected-gradient norm、active bounds、accepted displacement 与单位剩余时间的 outlet progress；direct source prior 的 8/8 post-outlet timeout另查 source frame/mode/unit 以及 issued/accepted/applied progress。接口正确后的碰撞或停滞继续作为 baseline 结果进入分母。

理论同步修正为：坐标分区的 residual authority 与绝对有限时间闭环进展义务共同定义 admissible set；command-space 单步预测只作为诊断代理，不能充当 invariant-set 或 reachability 证明；terminal viability state 必须包含 reference lag、耦合执行状态和 dwell reserve；task-owned safety coordinates 与 method-owned transport coordinates 在 descent/dwell 全程分开；stage-2 前的 deadline exhaustion 不属于 terminal residual 能解决的范围。本轮仍是兼容性和因果诊断证据，没有提供 transport 相对 baseline 的任务性能优势。

机器可读工件位于 `results/_analysis/four_task_transfer/t2_terminal_set_admission_closeout_20261024/`，其中 closeout、三类失败 diagnosis plan 和 manifest 由现有 `scripts/paper/cross_ebdm/analyze_predeployment.py` 的 CPU-only 模式生成；v4 run 自身的 `REJECTED_CANDIDATE.json` 保留 5 个完成 episode 与 2 个中断 journal row 的不同口径。

### 2026-10-24 T2 task contract v5 的 fresh v13 否证与下一接口

v5 候选合同的 fresh v13 已在 CPU 上完成全部 720 个 behavior-predeployment episode，并由独立审计重新读取 720/720 份 evidence。结果为 577 success、82 timeout、61 overforce、0 guide-contact-continuity failure 和 0 infrastructure error。public geometry 仍为 8/8，task-only 与 frozen-warm 均为 64/64；Cross-EBDM 和 gated-full 各为 63/64，且二者唯一 timeout 具有相同 evidence SHA，属于同一条物理轨迹。因此 runner 的 sentinel-only `formal_evaluation_admitted` 不能用作论文门，`task_contract_v5` 被否决，P6 继续阻塞。

576 条可比 learned/model 轨迹中，mode-1 有 288 条，282 success、6 个逻辑失败和 5 条唯一失败轨迹。CPU 配对诊断将它们分成三类。第一类是 Cross-EBDM/gated-full 的 terminal convergence tail：两行共享同一 evidence，在 tick 899 才进入 exact common，剩余 101 controls，最终仍比 2 mm terminal set 多 0.303 mm且 hold 为零。第二类是 response MPC 的 late handoff：两个 payload 请求分别在 tick 957/963 才进入 exact common，只余 43/37 controls，最终位置误差为 8.094/15.383 mm。第三类是 matching-no-band 与 band-only 的 delayed impact：进入 exact common 时 tracking lag 为 7.94/8.96 mm，随后第 13/15 个 control 的 1 ms 子步超过 10 N lateral limit。exact common action 不能清除已经形成的 reference、velocity 和 delayed-execution state。

另一个 mode-0 ungated-full overforce 不属于共同 handoff 失败。它在 outlet 之前、selector code 0 和 full candidate authority 下超过 25 N；相同 request 的 main Cross-EBDM、gated-full 与 task-only 都成功。该事件保留为 gating ablation 的失败证据，不通过共享接口把它“修好”。FADA、DIAL 与 direct source prior 的既有失败也不并入本次 handoff 修改。

下一候选定义为 `public-T2-handoff-viability-v7`，未来通过后再创建 `cross-embodiment-t2-shared-execution-v6`，当前不绑定。它只作用于 mode-1：从 outlet 开始用允许的 public observation 计算 augmented-state handoff debt，包括剩余 deadline、15-control dwell、public reference 与 TCP 的位姿差、TCP twist、measured wrench、common completion stage 和 terminal-set distance；task-owned 坐标始终 exact common，candidate 只在 transport 坐标有残差权限，且 outlet 后权限单调不增。viability margin 用尽或 completion stage 2 开始后返回 exact common。v7 不再用 v4--v6 的 terminal one-step pose selector，也不读取 delay、payload、servo gain、method identity 或 evaluator state。

放行顺序冻结为四级：先做已有轨迹的 trigger/utility CPU screen；再在 5 条唯一 mode-1 失败及其 task-only 成功对照上做有界 exact-request counterfactual，任何成功回归、timeout、overforce 或 evidence 缺失立即拒绝；通过后才重放完整 576+8 exact v13 gate；最后用不重叠 seed 完成 public/Cross release gate、全量 fresh predeployment、architecture、CPU package 与 GPU1 device audit。20 s deadline、2 mm/2 degree terminal set、15-control dwell、25 N/10 N limits、controller-v50、route-v4、reset cells、modes 和 evaluator 全部冻结。v13 只能用于开发 v7，不能替代 fresh gate。

理论必须把 residual admissibility 定义在延迟系统的 augmented state 上。仅证明当前 action 朝 terminal set 前进，或最终切回 exact common，都不能推出闭环安全和有限时间完成；handoff 条件必须同时保证 delayed state 可刹停、公共 completion 与 dwell 的剩余时间债务可偿还。mode-1 shared compatibility、mode-0 learned gating、baseline mechanism limitation 和 transport 性能收益继续分开陈述；当前仍没有 transport 带来任务性能优势的证据。

机器收尾位于 `results/_analysis/four_task_transfer/t2_task_contract_v5_common_failure_closeout_20261024/`。`closeout.json`、`diagnosis_plan.json`、`paired_failure_profiles.csv` 与 `manifest.json` 的 SHA256 分别为 `feb4470deef005b52ddd8918bd951aa3b40fe9ff9a87d272e4a52f6bca662766`、`458db4deb8be4392431ce72761da33a776808f1d6001074b8811118cb73acc43`、`0a11e465b8b39df10e455d8ec39af69a5a825d9c8d9430e26584fbe98cb5e30b` 和 `e582db129c891f9ef547bd5792bfaf60170351241201d6f199177fff88f464b1`。v5 terminal-transport 与 v6 strict-progress 的两条有界预检也已分别用 `REJECTED_CANDIDATE.json` 封存；它们都保留代表性成功请求，但已知 regression 请求仍 timeout，因此没有运行剩余 582 个 job。

### 2026-10-24 T2 completion-budget capture-band 全量回放与 CPU 收尾

`public-T2-completion-budget-capture-band-v12` 的 80-episode exact-request 小门为 80/80，但随后完成的全量 comparable replay 否决了该候选。全量计划包含与 v14 完全相同的 576 个方法/消融请求以及 8 个未包装 public sentinel，共 584/584 episode、546,943 个控制步、0 个 reset/execution error。独立审计重新读取 584/584 份 evidence，确认 plan/result 一一覆盖、artifact SHA 匹配、confirmation state 未打开。总计 568 success、16 overforce；public sentinel 为 8/8，因此失败是行为回归，不是基础设施错误。

576 个请求与 v14 逐一配对后的转移矩阵为：550 个 success→success、8 个 timeout→success、2 个 overforce→success、15 个 success→overforce、1 个 overforce→overforce。v12 因而修复了 v14 的全部 8 个有限时间 timeout 和 2 个 mode-1 过力，却把 15 个原成功请求变成过力；可比成功数从 565/576 降到 560/576。各行结果为 Cross-EBDM 63/64、task-only main 61/64、response MPC 64/64、frozen-warm 64/64、task-only ablation 61/64、matching-no-band 62/64、band-only 61/64、ungated-full 61/64、gated-full 63/64。16 个逻辑失败按 evidence SHA 合并为 8 条独立物理轨迹。

三类失败的状态现在可以精确区分。第一类 finite-time completion-budget shortfall 在相同请求上从 v14 的 8 个逻辑失败、5 条独立轨迹降为 0，说明 45 mm/s far-field requirement 与 10 mm capture band 确实修复了已记录的上游进度债。第二类是新出现的 terminal recontact planar-reference energy：15 个逻辑失败、7 条独立轨迹全部为 mode 1，并且首个违规时 selector code 为 2、candidate authority 为 0、terminal ownership 已启用、completion stage 为 2；违规都发生在最后一个已记录控制周期的 1 ms 子步，前一控制周期总力和横向力均为 0。它们不是 residual 直接施力，而是 exact-common handoff 后仍留在延迟执行器中的平面 reference/action debt。第三类是同一个 mode-0 ungated-full pre-outlet collision；它仍在 outlet 前以 selector code 0 和 full candidate authority 超过总力限，属于预声明消融结果，不由 mode-1 公共 completion 修复。

第二类的代码机制已经定位。当前 terminal speed cap 用 `terminal_distance_m < 0.006` 触发，但该距离包含尚未完成的垂直下降；7 条独立失败在 completion stage-2 入口的这个值仍为约 35--37 mm，所以平面近终端 cap 没有启用。stage-2 到首次冲击还有 13--15 个控制周期，足以让 1/3-control 延迟队列刷新，但公共平面 issued/applied action 仍可接近 30 mm/s，随后产生 10.04--13.06 N 的横向冲击。单次 stage-entry pose、lead 或 twist 与部分保留成功轨迹重叠，因此不能把一个事后阈值分类器宣称为 viability certificate。

下一候选冻结为 `public-T2-completion-budget-planar-braking-v13`，先只做一个公共控制语义修正：保留 v12 的 45 mm/s far-field requirement、10 mm capture latch、垂直 completion、force release 与 dwell；mode-1 进入 completion stage 2 后单独计算平面 terminal distance，并把现有 10 mm/s terminal limit 应用于完整 x/y common action，同时让两维平面 reference 单调朝公开 goal 收敛。它不新增学习参数、不扫描 gain、不修改 actor/solver、deadline、terminal set、force limit、mode-0 或 evaluator。如果该单一修改不能通过有界门，下一版本才单独增加下降前 public planar-reference settling substage，不能把两项修改混在一次因果实验里。

验证顺序保持四级。CPU 先验证 v13 在 mode-1 completion stage 2 之外与 v12 逐项相同，并确认 cap 覆盖 7 条独立冲击轨迹。GPU1 有界 exact-request 门随后覆盖 15 个逻辑回归、与其状态重叠的保留成功样本、v12 修复的 10 个旧失败、persistent ungated mode-0 请求和 8 个 public sentinel；所有 mode-1/public 必须成功，旧成功必须全部保留，ungated mode-0 是唯一允许保留的声明失败。通过后才重放完整 584-job roster；完整门通过后才允许 seed-disjoint public/Cross release、fresh behavior predeployment、architecture、CPU package 与 GPU1 device audit。当前 P6 仍关闭。

理论同步加入两条边界。exact-common authority 不是延迟系统的安全证书，因为切换前形成的 reference spring energy 和 action queue 仍可在未来物理子步释放；terminal augmented state 至少需要平面 reference lead、TCP twist、phase、delayed-action debt 和 dwell reserve。finite-time reachability、guide-contact continuity 与 force safety 继续作为独立条件命题。本次修改属于所有方法共享的 benchmark infrastructure，不能计为 Cross-EBDM 贡献；transport 的任务性能优势仍需新的 seed-disjoint 主表证明。

CPU 可复现入口为 `scripts/paper/cross_ebdm/analyze_completion_budget_capture_band.py`。连续两次在 `CUDA_VISIBLE_DEVICES=-1` 下生成逐字节相同结果：`closeout.json` SHA256 为 `3d232074a6635bb2d88ce75bb802b11564f61c3cde6c9cdc17f6f6078883369e`，`diagnosis_plan.json` 为 `f572a65d2d2297e3375e710f22cf875dcb5040783bab86fb410d59ecbce38041`，manifest 为 `bab7b60f8c1a10deadcead1387228d03e00489ca2acd313e144785aad6cd9558`。目录位于 `results/_analysis/four_task_transfer/t2_completion_budget_capture_band_full_replay_closeout_20261024/`；原 full replay 已用 `REJECTED_CANDIDATE.json` 封存，SHA256 为 `337de5eb175b9f8bb7a3a35809177767b472ae6386099c5f51cdc90f17bb9470`。

### 2026-10-24 T2 planar-braking v13 的 CPU 离线收尾与下一修改

`public-T2-completion-budget-planar-braking-v13` 的 104-request exact gate 已完成且独立审计通过，但行为门明确失败：55 success、40 timeout、9 overforce、0 execution error，public geometry 为 8/8。96 个受包装 episode 中，47 个 mode-0 成功保持不变，声明的 ungated mode-0 overforce 也保持不变；48 个 mode-1 则为 0/48 成功。与相同请求的 v12 逐项配对后，转移矩阵为 55 success→success、39 success→timeout、2 success→overforce、7 overforce→overforce、1 overforce→timeout。v13 因而立即封存，不进入 584-job full replay。

49 个失败分成三类。第一类是 40 个 mode-1 finite-time liveness timeout，其中 39 个由 v12 success 回归而来，另 1 个把旧 overforce 换成 timeout。40 条轨迹都曾在平面上进入冻结的 2 mm task set，停留 7--12 controls，随后重新退出；最终仍比容差多 0.597--3.183 mm，terminal hold 的全程最大值为 0。永久的 goal-monotone projection 与 10 mm/s cap 因而不是 liveness-preserving terminal feedback：它禁止 v12 在低增益闭环中使用有限 reference overshoot 回收终端误差。

第二类是 8 个 latency-gain mode-1 descent/recontact overforce，6 个从 v12 保留，2 个是新回归。它们在 completion stage-2 入口的 measured force、issued planar action 和 applied planar action 都为 0，公开 planar reference lead 却仍为 8.20--8.95 mm；首次硬违规统一发生在 16 controls 之后。这否证了“只有高动作残留在 1/3-control delay queue 中”的解释。v13 的 pointwise cap 没有在下降前耗散 reference spring state，进入下降后才持续裁剪动作既太晚，也不能构成 stored-energy safety certificate。

第三类仍是单个 ungated-full payload mode-0 pre-outlet collision。它没有进入 completion stage 2，且 v12/v13 都为 overforce；这是声明的 gating ablation 结果，不属于 mode-1 公共 completion 修复范围，必须留在原分母。

下一候选固定为 `public-T2-completion-budget-pre-descent-settling-v14`，并从 v12 继承，不从 v13 继承。唯一新机制是在 mode-1 首次请求 stage-2 descent 时插入有限的 public-state barrier：先保持垂直 clearance 并冻结 x/y reference；只有当 planar reference lead 不超过冻结的 2 mm position tolerance、planar TCP speed 不超过已有 4 mm/s completion settling limit，并连续满足 4 controls（覆盖声明的最大 3-control delay 再加一次观察）后才允许下降。放行后恢复 v12 的平面 completion action，不保留 v13 的永久单调投影或完整向量限速。它不改 actor、solver、gain、deadline、terminal set、force limits、reset roster、mode-0、evaluator 或 confirmation-state 规则。

验证仍按因果门逐级进行。CPU 单元门先证明 settling interval 以外逐项等于 v12、mode-0 恒等、4-control admission 与 checkpoint/restore 正确。随后 GPU1 只重跑同一 104-request gate：要求 public 8/8、48 个受包装 mode-1 全部成功、47 个既有受包装 mode-0 成功全部保留、ungated mode-0 是唯一保留失败、零 execution/evidence error；任何回归立即拒绝且不扫描 settling window、lead threshold 或 gain。该门通过后才跑 584-job exact roster，再通过才运行 seed-disjoint public/Cross release 与 fresh full predeployment。

理论需要把 terminal 条件拆成两个义务：下降前的 delayed/reference-state dissipativity，以及下降后的 finite-time reachability/dwell。逐时刻 action 有界不能推出延迟闭环的 stored reference energy 有界；augmented state 至少要包含 reference lead、TCP speed、覆盖 delay 的持续计数和 dwell reserve。这个公共控制器仍属于 benchmark infrastructure，不能算作 Cross-EBDM 学习贡献；transport 的性能优势仍需 seed-disjoint 主表证明。

CPU 入口为 `scripts/paper/cross_ebdm/analyze_completion_budget_planar_braking.py`。连续两次在 `CUDA_VISIBLE_DEVICES=-1` 下生成逐字节相同结果：`closeout.json` SHA256 为 `1a5fb152822a1c6881121931a68f3d4d9e7cbdeb71fad512c728948f210cef95`，`diagnosis_plan.json` 为 `2282a4da29e96a132ce16cd471c2e4b98655db53eb1c9e21d866cee59198bc8e`，manifest 为 `072ed8882fa014bc361c7998842e674c8c285ca3a0ffe0753810ca449dbe7f50`。工件目录为 `results/_analysis/four_task_transfer/t2_completion_budget_planar_braking_closeout_20261024/`；v13 run 的 `REJECTED_CANDIDATE.json` SHA256 为 `921e9dd10b781a2554d83f81c7ae8138fb2314e29246aec048647de7e6741f36`。

### 2026-10-27 transport 因果归因与相对响应机制复核

T2 的 same-state handoff attribution 已完成。冻结源结果的六个 handoff 全部在预声明接纳域内复现，三条原成功轨迹在 normal 与 exact-common continuation 下都成功；三条原 timeout 在 normal continuation 下继续 timeout，而同一增广状态的 exact-common continuation 全部成功。十二条 continuation evidence 均通过独立审计。因此本轮没有 shared completion defect 证据，三个失败归于方法 continuation 或接口交接。报告位于 `results/arm/guided_contact_alignment/validation/handoff_state_attribution/gpu1_exact_state/report.json`，SHA256 为 `3665794ca218cfbc79d0aae017398e2fd9d3d268f4decb0cdbfc1adb7a43109d`。

T4 的合格 composite 与 task-only 因子对照已完成，而不是继续停留在未执行计划。四个 cell 各有 64 个 episode：transport 在 projection off/on 下均为 64/64；task-only 在 projection off 下为 44/64、projection on 下为 64/64；全部 256 条 evidence 零约束违规、零执行错误并通过独立重审。共享 projection 打开后，transport 与 task-only 的主成功率差为 0，精确 McNemar `p=1`；旧的 20/64 大幅成功差来自公共 completion feedback。projection off 的差异不能写成部署方法的独立 transport 收益。事后完成时间提示为 transport 平均少 9.31 个控制步，但它不是本轮预声明主指标，不能改写为正贡献。报告 SHA256 为 `52409f40f760ae526f308c8b3a6fd8feadac24195ee8e57060a7bbf53a82c7ec`，独立审计为 `aa9757e7295d5ebde1e662f735e338abdb0f670e660125acc2adcd0fbc5176f2`。

T3 随后完成了真正的同起点 anchoring 消融。两支从逐实例相同 task-only checkpoint 出发，使用相同公开数据、sample stream、fresh optimizer、1500 updates、模型容量和 terminal endpoint；唯一差异是 response anchoring weight 1 与 0。两支均为 128/128 success、零约束违规，成功率差 0、McNemar `p=1`。完成时间分别为 528.38 与 527.32 controls，anchoring 略慢 1.06 controls；总力差不显著，侧向力仅有很小的事后变化。报告 SHA256 为 `b92911efa1c369c4a719d8e7b058fe25ab9b0a757574c3e43d9831eeed2de65b`，独立审计为 `33ce6ad720eedd7ac0f6210c7bf3e1d129b460e0578eb32c5b3aa7e2979f235e`。

CPU autograd 诊断解释了第一版消融为何几乎等同 task-only：8 个实例的 response 梯度只占 task/progress 梯度的 0.037%--0.250%，中位数 0.123%。因此新增了同一 `cross_ebdm` solver 内的 task-only 相对响应归一化；没有新增顶层 solver、learning 算法目录或 P 编号入口。归一化后该比例为 10.8%--44.0%，中位数 32.2%。8 个 GPU1 训练实例使用完全相同的起点和 sample stream；实际 selector/fallback 输出中，7/8 的 held-out recovery response discrepancy 至少降低 10%，中位比值为 0.807，说明 transport 项已成为可辨识的训练机制，而不是数值上消失的正则项。

机制变强后仍没有产生任务收益。新冻结的 T3 对照包含 128 对请求：relative-response transport 为 124/128，task-only 为 128/128，零约束违规、零执行错误。4 个额外 timeout 全部来自唯一未通过 linear-progress selector 而回退的 instance 03；共同成功的 124 对中，transport 平均多用 5.22 controls，平均 reduction fraction 为 -0.98%，更快/更慢各 61 对、2 对持平，单侧精确符号检验 `p=0.536`，只有 5/8 实例方向非负。即使事后把 instance 03 换成与 matched task-only 完全相同的 fallback，最多得到约 0.58% 平均提速、61 对更快与 49 对更慢，`p=0.147`，仍过不了冻结的 2% 门。mode 0 的平均差只有 +0.53 controls，而 mode 1 为 -11.35 controls，说明当前 gate 没有建立 source response 与目标任务进展之间的正确方向条件。报告 SHA256 为 `8817efdc4d8e8a5cc37993c777fb963488912bcfafcf142373dee3a08c66cc17`，256/256 evidence 独立审计 SHA256 为 `fdfee6f5e38d8fe3d55905966703a7fb5ccafb823ff3b0f4fa557270300bd92b`。

当前结论是：已有实验不支持“transport 带来稳定任务收益、优于强 baseline”。把 response loss 放大或归一化只能证明它参与优化，不能把 response discrepancy 的下降变成任务效用。下一版若继续追求该核心贡献，方法必须先满足三个机制条件：以强 task-only endpoint 为共同 refinement 起点并把其等预算 continuation 作为精确 fallback；训练与 held-out selector 都按 mode 覆盖，不能只用 mode-0 validation 接纳在 mode 1 部署的策略；anchoring 只在 source response 相对 matched task-only 具有预先定义的任务/事件方向优势时启用。完成这一步后仍须先通过新的同起点 T3/T4 开发门，再冻结算法。

真正能支撑核心贡献的是冻结算法后的 seed-disjoint 四环境确认，而不是上述任一开发修补。因果子句要求 full 相对同容量、同预算、同公共反馈的 no-transport 消融在四环境给出预设效应量、跨实例方向一致性、统计不确定性和零额外约束违规；比较子句要求同一协议主表中的 full 优于所有已放行强 baseline。两项都通过后才能在论文中写性能贡献。在此之前不启动确认大表，也不把 T4 公共反馈收益、事后时间差或模型 response 指标写成 transport 任务优势。

### 2026-10-31 ACP 名义诊断止损与新 baseline 源码移植

ACP 的固定 reference 响应时标对照已结束：在 mode 0/1 各一个固定 reset、0.1/0.2 s 两个时标下，learned policy 四条均 timeout，而使用同一 adapter 的 perfect-future sparse-label oracle 四条均成功；8/8 物理 evidence 通过审计。这个对照只排除了“简单减慢 adapter 即可恢复名义行为”，并没有证明原论文 ACP 机制必然失败。它与先前第一步请求分歧及闭环偏离共同支持停止继续扫描 ACP 增益；该移植在当前四任务尚未通过名义资格，不能作为论文主表中被击败的强 baseline。工件位于 `results/arm/obstacle_transfer/baseline/acp/diagnostics/reference_response_time/report.json`。

按 `brainstorm.md` §8 的系统辨识/任务驱动适配路线，已把 ASID 与 AdaptSim 官方 GitHub 源码分别固定在 `baselines/ASID`、`baselines/AdaptSim`，来源和 commit 记入 `requirements/baselines/sources.lock.json`。对应独立 solver 包为 `genedynamics/solvers/single/asid` 与 `genedynamics/solvers/single/adaptsim`。ASID 上游提供 Fisher 信息探索目标及特定任务的探索策略训练入口，但没有可直接用于当前四任务的完整辨识/策略重训管线，且没有可见 license 文件；所以本包独立实现该数学目标，不复制作者代码。AdaptSim 在 MIT 条款下移植参数分布更新、PyTorch branching-Q 和训练损失。AdaptSim 现在也有模拟目标的粒子采样、内层策略训练/评估、replay Q 更新与完整交互计数；部署阶段冻结 Q，不使用目标 reward 更新。与作者源码的数值对齐和循环计数共九项测试通过；GPU1 上的单步 PyTorch 训练 smoke 也得到有限 loss。当前两包均只是方法核心：AdaptSim 尚无四任务长时策略训练器、上游外层粒子出生/淘汰与名义资格，ASID 尚无当前任务的完整主动探索/参数估计/策略重训，不能写成完整原方法，也不能进入独立主表。

TDC 的 CoRL 2026 官方项目页目前只有论文/arXiv 链接，没有官方 GitHub 代码、预训练 VLA 权重或仿真示范下载入口。完整 TDC 还需要视觉策略、任务帧/模式预测头、示范生成和人工刚度/力设定，因此此时不能把一个手写导纳控制器称为 TDC 复现。先保留为待源码/资产公开路线；若公开资源仍缺失，再按论文方法明确标记为 paper-based reimplementation，并独立记录额外视觉预训练与人工调参预算。

下一步从 **AdaptSim** 开始接任务，而不是把上述核心直接接入 actor 主表：先在 T1 的 source 机制范围声明可调参数及边界，建立“参数分布采样 → 同等预算训练任务策略 → 目标闭环 reward/feature → branching-Q 更新”的任务层循环，保留全部训练/目标交互账本。T3 要等任务参照在两种 mode 均通过源资格后才进入同一路线，再扩到 T2/T4。ASID 必须完整补齐主动探索、参数估计、重训策略三段才可作为强对照。两者对照条件不同：ASID 检验可辨识参数变化，AdaptSim 检验任务效果驱动的仿真分布调整；所有 target reward 和额外仿真训练计入预算。冻结公共协议、同容量 task-only/transport 消融及未接入的 TDC 状态均不因这次移植改变。

任务接线探索给出了两个止损信号，均为单次未封存开发预检，不能算正式 nominal qualification。T1 Panda 名义机制上，旧 `PublicTeacher` 的 mode 0 在 500 controls 超时、终点误差 241.6 mm，不能用其轨迹作为可靠的 AdaptSim 策略训练标签；现有 `public_geometry` 的 Panda mode 0/1 分别在 411/391 controls 成功，xArm7 mode 0/1 分别在 363/360 controls 成功，说明两种本体均有可用的 task-owned 参考路线。T3 Panda 名义机制上，`public_geometry` mode 0 在 320 controls 成功，而 mode 1 在 20 controls 发生 overforce；因此不能把该参照直接扩为四任务通用 teacher。AdaptSim 上游 `pretrain_push.yaml` 的策略环境只有一步、2D 动作，并要求 10,000 次内层适配、每次 96 个任务采样；本项目 T1/T3 为数百步、13D 执行，T3 的目标机制还有原作者 `apply_action` 不支持的离散 delay。需要明确报告策略训练器与预算的 task adaptation，不能把一个短预算浅层克隆称为“完整原方法”。

T1 xArm7 已形成第一条可运行的 AdaptSim 任务适配链：task-owned `obstacle_policy_search.py` 负责在真实 MuJoCo 机制中运行两种公共 mode、按任务成功/用时给 reward、只从公开终点误差与用时提取 8D 特征；solver-local bounded CEM 在每个参数分布下训练 6 参数路径/速度策略；GPU1 上的 branching-Q 先用模拟目标训练，部署时保持冻结。此 CEM 是明确声明的长时任务策略训练替代，不等于作者一步 2D NAF。第一份开发运行 `development_smoke_928500` 在 46.7 秒完成，但冷启动候选中重复评估了公共锚点，占掉 3 个候选中的 1 个；该工件原样保留，不能用于算法判断。修正后独立运行 `results/arm/obstacle_transfer/baseline/adaptsim/training/development_distinct_candidates_928501/report.json`：32 条物理 episode，模拟目标阶段训练 5,021 controls、目标执行 1,444 controls、Q 更新 1 次；部署阶段又训练 4,931 controls、目标执行 1,529 controls。4 次训练各有 3 个不同候选，checkpoint SHA 与全部账本求和独立核验通过，GPU1 已释放。

修正版的单一开发目标上，更新前后两个 mode 均 2/2 成功，平均 reward 由 0.6948 降至 0.6936；mode 0 从 381 增至 385 controls，mode 1 从 382 降至 381 controls。4 次 CEM 策略训练均选回公共锚点。固定 reset 的探索性速度扫描表明增益 1.0 时速度倍率 1.0 与 1.1 的控制步为 362/360 与 361/357，增益 0.7 时为 386/382 与 414/381；任务已接近饱和，而且更快请求未必带来更快完成。只用 1 个模拟目标 transition 的 Q 更新不能支持 baseline 性能或收敛结论。下一资格门不是盲目加大 T1 预算，而是先找到两种 mode 都有可训练、可验证优势空间的机制/任务条件；若继续 AdaptSim，应在冻结目标前预先声明多粒子元训练规模、完整任务训练预算、独立 source/nominal 检验和不含目标真值的特征。T3 先修复 mode 1 公共参照的 overforce，或采用不依赖该参照的真正任务策略训练器；离散 delay 作为固定 nuisance 或单列扩展，不能声称作者原方法已覆盖。

### 2026-10-31 新 baseline 多粒子接线与现有 T3 gate 分层

ACP 固定 reference 时标的 learned policy 四条 timeout，而同 adapter 的 oracle 四条成功；这足以停止继续扫同类增益，但不能证明 ACP 原方法的机制必然失败。ASID、AdaptSim 的官方 GitHub 检出与锁定 commit 再次验证通过；TDC 官方项目页仍无可核验的公开 GitHub 代码入口。AdaptSim solver 增加了作者多粒子比较所需的同一目标 reset 选择：每个初始粒子先各自训练任务策略，按公开目标 reward 选择，选择成本进入同一仿真和目标交互账本，选中策略用于后续训练 warm start。12 项 ASID/AdaptSim 源码数值与循环测试通过。

独立 GPU1 开发运行 `results/arm/obstacle_transfer/baseline/adaptsim/training/development_particle_selection_928502/report.json` 预先封顶 48 条物理 episode、180 s、2 GiB，实际用 68.95 s；全部 48 条完成，13,877 个仿真训练控制步与 4,549 个模拟/部署目标控制步逐项对齐，checkpoint SHA 通过独立核验。两个初始粒子在同一 reset 上均为 reward 0.6804，按预设 tie-break 选择索引 0；后续两个不同 reset 的目标 reward 为 0.6940、0.6932，不能把它们的差额当同起点任务效应。该运行只验证多粒子任务接线，不构成名义资格或主表 baseline。当前 T1 任务收益已接近饱和，继续加小预算不会提高可归因性；下一步须选择两种 mode 都有真实训练优势空间的机制和完整元训练规模，再做独立名义验证。

已封存 T3 方向优势 gate 的 256 条物理记录另做 CPU 离线配对：`results/_analysis/four_task_transfer/t3_directional_mode_closeout_20261031/report.json`。128 对同起点中 transport 快 54、task-only 快 68、平 6，transport 平均慢 2.59375 controls；mode 0 的 64 对平均慢 4.28125，mode 1 的 64 对平均慢 0.90625。只有 3/8 个实例的总体均值为正，mode 1 仅 2/8。训练日志最后一个 mini-batch 的 source-task-advantage 与 directional anchoring 分数均值分别为 0.2383 与 0.1367；这是模型代理量，不是目标任务收益。这个 gate 已经被训练并评估，不能再作为尚未尝试的新方案；下一理论修订需先核验 source 与 task-only 是否在相同目标状态下比较，再决定是否重训。离线分层是事后诊断，不替代预注册确认。

### 2026-10-31 AdaptSim 外层调度开发核验

在已有独立 `adaptsim` solver 内补充了作者式外层调度：固定模拟 meta-target 上逐粒子训练与评估，超过预置 reward 门槛时将最差粒子替换为最好粒子的受界扰动克隆；每轮在独立模拟目标上做内层 Q 学习。高斯粒子扰动与锁定上游源码的数值对齐，以及外层替换和逐项交互计数，共 14 项聚焦测试通过。运行后又将内层 replay 改为跨外层迭代保留，测试证明第二轮可以利用第一轮 transition；此前的物理开发工件保留原状，不能当作这一修正版本的验证。

物理 GPU1 开发工件 `results/arm/obstacle_transfer/baseline/adaptsim/training/development_population_928503/report.json` 预先封顶 96 条物理 episode、240 s、2 GiB；实际 96 条、148.96 s，全部账本和 checkpoint SHA 独立核验通过。仿真训练消耗 29,451 controls，模拟目标 6,250 controls，部署目标 3,060 controls，Q 更新 2 次。两轮固定 meta-target 的粒子 reward 分别为 0.6800/0.6800、0.6808/0.6808，部署同 reset 粒子 reward 为 0.6944/0.6944，故没有触发替换且选择按平局规则落在粒子 0。部署更新前后 reward 都是 0.6936，但两次使用不同 reset，不能解释为同起点无效应。所有候选策略、失败及交互仍保留；GPU1 已释放。

这组结果说明 T1 当前六参数 CEM 在不同仿真粒子上经常给出同一任务行为，目标 reward 无法区分粒子；进一步堆同类 T1 小预算不会检验 AdaptSim 的关键机制。下一步先用 CPU 诊断候选策略的粒子分歧与四任务的任务收益空间，选择 source 与两种 mode 均有非平凡梯度的任务，再冻结完整元训练、source/nominal 检验及等资源部署比较。没有这一步，AdaptSim 仍不能作为被 ours 击败的强 baseline 进入独立主表。

### 2026-10-31 AdaptSim 任务策略探索质量对照

三个既有 T1 账本的离线核对发现，第一份修正后 pilot 的 4/4 次、两轮外层 pilot 的 12/12 次任务策略训练都返回同一个公共锚点；此前的“粒子同分”因此先归于长时任务 CEM 搜索塌缩，不能归咎于 AdaptSim 机制。已在原 solver 的 CEM 加入预声明的速度坐标对称探针，保留随机候选和逐条仿真预算；15 项聚焦测试通过。独立配置 `configs/arm/obstacle_transfer/training/adaptsim_coordinate_policy_development.json` 在 GPU1 封顶 96 条物理 episode、180 s、2 GiB，实际 96 条、137.27 s；32,162 个仿真训练 controls、1,433 个模拟目标 controls、3,155 个部署目标 controls 均与报告账本、checkpoint SHA 对齐，GPU1 已释放。

新运行 `results/arm/obstacle_transfer/baseline/adaptsim/training/development_coordinate_policy_928504/report.json` 的 6 次策略训练给出 6 个不同策略，仅 1 次返回公共锚点。同一个目标 reset 上两个粒子回报为 0.6940 与 0.6804，差异由 mode 0 的 383 对 417 controls 贡献，mode 1 均为 382 controls；按预置规则选中较好的粒子。后续两个不同 reset 的部署回报为 0.6820、0.6816，不能解释为同起点 Q 更新效应。该对照说明任务层存在可区分的训练信号，但还不能证明 Q 网络收敛、AdaptSim 优于其他方法，或四环境名义资格。下一阶段应冻结更充分的模拟元训练与 seed-disjoint 源/目标条件，分别报告“多粒子选择”与“Q 更新”的增量，同时保留额外目标 rollout 成本。

固定上述 checkpoint 后，在训练采样域之外的 16 个 reset 上按相同初态比较 AdaptSim 策略与公共锚点，得到 `results/arm/obstacle_transfer/baseline/adaptsim/validation/nominal_diagnosis_928504/report.json`（SHA `8a094a47812f7857fe4319d2e76785770a5fdda5dd65d9cb44ce79213b51e7ff`）。32/32 条均成功且保存了全部控制步。名义增益 1.0 下，AdaptSim 在 8/8 对中各快 1 步；目标增益 0.7 下，mode 0 的四对步数差（AdaptSim 减锚点）为 +29/+28/+27/-3，mode 1 为 0/0/0/0。因而这份小预算策略只有很弱的名义优势，在实际偏移下并不稳健；不能把公共锚点当作可轻松超越的弱对照。进一步的多粒子元训练已按独立配置 `adaptsim_meta_coordinate_development.json` 冻结为 42 次策略训练、36 次模拟目标回放和 6 次部署目标回放，上限 672 个物理 episode／1500 秒／GPU1 2 GiB；这轮仍是开发而非四任务准入。

上述多粒子运行已在 GPU1 用 939.83 秒完成，`results/arm/obstacle_transfer/baseline/adaptsim/training/meta_coordinate_development_928505/report.json` SHA 为 `a827abd541d144043286871c02bb7c29a4e61ef906bd62e71de39150d334a4c6`，checkpoint SHA 为 `800b5d9b21d28c04159f7b775f0ff5770c3b966c34f7f030e070cfca1d638801`。42 次策略搜索各评估 7 个候选与两个 mode，36/6 次模拟／部署目标回放各评估两个 mode，总计 672 次物理执行；52 次 Q 更新，42 次搜索返回 20 个不同策略。四轮外层共享模拟目标的三粒子 reward 在迭代 0 为 0.7220/0.7096/0.7092，在迭代 3 为 0.7412/0.7096/0.7308；迭代 2 有一次 best-to-worst 粒子替换。部署阶段同 reset 的三粒子 reward 全为 0.6940，选择平局中的粒子 0；此处的外层趋势不能证明 Q 更新或分布选择带来目标收益。

用新 checkpoint、另一个训练外 seed 包执行的 `results/arm/obstacle_transfer/baseline/adaptsim/validation/nominal_diagnosis_meta_928505/report.json` SHA 为 `445cea87211b56ef827c15e9749779dc57b8bc2ad9f5abd48e1673aed57063c8`；32/32 条均成功。相对公共锚点，名义机制 mode 0/1 的控制步减少量分别为 -5/+3（各四对完全相同）；目标增益偏移 mode 0 为 -38/-37/-39/-39，mode 1 为 +2/+2/+2/+1。因此该训练规模产生了多样策略，却没有形成稳健的任务改进，尤其 mode 0 在增益偏移下明显变慢。不能用两套开发 seed 挑较好 checkpoint 冒充冻结基线；下一步要重新设计具有非平凡收益空间的任务层策略及独立资格。

本轮独立账本审计还发现一个记录缺口，原报告保持不变，旁置 `candidate_ledger_audit.json` 并绑定其 SHA：294 个策略候选都记录平均 reward 与总控制步，但 588 次候选 mode 执行没有逐条 outcome，21 个候选的平均 reward 非正；模拟目标与部署目标的 84 条 episode 则有逐条 outcome。原 `plan.all_failures_retained` 对训练候选而言过度表述，故本轮不能作为合格 baseline 的完整失败分解。后续代码已让 task adapter 返回每个 mode 的结果，并在 AdaptSim solver 的候选账本保留；18 项聚焦测试通过。任何新训练都须使用新版本独立运行，不能给旧结果补造失败明细。

ASID 的 T1 任务适配先做公开观测可辨识性试验，而不是把现有 Fisher objective 核心冒称完整 pipeline。官方提交 `cf2cb670fb983f67cf2af156d4e17b4a690787e2` 已重新核验；任务本地 `public_system_identifiability.json` 固定一个 xArm7 reset、两个 mode、三种公共速度倍率、20/60 控制步、增益／负载两维中央差分与 5×5 仿真参数网格。CPU 共执行 80 次仿真、3840 控制步，完整报告为 `results/arm/obstacle_transfer/baseline/asid/diagnostics/public_system_identifiability/report.json`（SHA `db3551bcc241340ada97e4adf93b4ed34ee7b1247e9fda758cf6988f152932c3`）。两维雅可比在这些候选下数值满秩，但负载列弱；mode 0、60 步、倍率 1.0 的列范数约 0.0471（gain）与 0.0060（mass），倍率 1.15 基本相同，表明仅提高同一路线速度没有增加有效激励。无噪声网格可精确找回模拟目标 (0.7, 0.5)，但邻近负载 0.4/0.6 的两时域位置差只有约 0.48--0.61 mm，不能据此宣称有噪声目标的可靠辨识。该试验不使用目标 reward、不训练探索策略或后续任务策略，因此仍不是可入主表的 ASID；下一步需要设计在公共动作边界内真正提高弱参数方向信息量的探索，再检验辨识误差如何影响重训后的任务效果和交互成本。

独立主表当前还不能冻结为可执行方法清单。逐一读取四个 task-local `actor_bindings.json` 后，ACP 和 H2S2R 在四任务均仍是历史 `predeployment_rejection`，且没有可供正式请求解析的 checkpoint set；后续开发 pilot 不能自动改写该冻结清单。FADA 虽均为 `instances_ready`，现有目标适配曲线与 source/nominal 资格仍不足以支撑强方法解释；AdaptSim 仅 T1 有单实例任务适配且训练外增益偏移明显变慢；ASID 仅完成 Fisher 核心与上述辨识诊断。T3 现运行时共享公共参照在高延迟 cell 只有 2/8，早期撤离试验虽净增成功仍新添一条过力。旧 `prepare_confirmation.py` 的 v14 全四任务逐格严格胜出门、方法拒绝占位与既有 invalid 结果不能充当新论文主表协议。下一版在冻结前须先形成每个外部方法的明确任务适配身份、独立训练实例／资源账、可执行 checkpoint 与 source/nominal 资格或透明的合格失败结论，并让所有方法共享合格的当前 T3 执行合同；主聚合固定 T1--T3、T4 单列形态扩展。

### 2026-10-31 T3 同状态方向判据与动态响应事件诊断

既有方向 gate 的实现把 `response(source_features, source_commands)` 代入目标 `current/task_reference` 计算 source 的任务优势，再和目标状态下的 task-only 预测比较；这两个代理量不在同一状态。CPU 只读脚本 `scripts/paper/cross_ebdm/diagnose_directional_source_state.py` 使用冻结公开验证状态与 8 份 SHA 校验的响应/策略 checkpoint，结果为 `results/_analysis/four_task_transfer/t3_directional_source_state_20261031/report.json`。2,896 个 recovery-enabled 状态里，旧判据接受 9.08%，把 source 命令在同一目标状态预测后的判据接受 56.18%；mode 0 为 16.67% 对 42.40%，mode 1 为 2.29% 对 68.52%。其中 1,537 个状态仅被同状态判据接受，173 个仅被旧判据接受。历史 actor manifest 已变化，脚本仅在内存中把该 pointer 更新到当前 SHA 并在报告保留新旧 SHA；全部公开数据和 checkpoint 仍按各自 SHA 核验。此结果说明 gate 定义有重大的状态混淆，但模型代理判定不是物理任务收益，也不能由接受率升高直接推断 transport 会变强。

`cross_ebdm` solver 的 `PublicDirectionalResponseSpec` 增加显式 `source_task_state`。旧配置缺省 `source`，因此旧训练与结果语义保持不变；新版本选择 `target`，只把方向资格计算改成同目标状态反事实，response matching 本身仍比较源/目标响应。独立配置 `finite_time_task_first_directional_same_state_anchored_preflight.json` 的 CPU preflight 已通过：8 实例、双 mode、共享 task-only 起点、严格 SHA、有限梯度，零 GPU 和物理调用；报告 SHA 为 `a3e338f8bfe3b8e32e8fbc201ef8c88a9738608ad5b03526a6eae090abab8a57`。完整 GPU1 训练与对照已在后续按授权运行，结果见下。

动态支撑响应模型的冻结验证集再按 10 μm 事件阈值分层，报告为 `results/_analysis/four_task_transfer/fixture_response_event_bias_20261031/report.json`。3 tick 的 5,480 个验证窗中，685 个真实位移事件窗里混合模型 MAE 为 28.13 μm，优于零位移的 37.37 μm；4,795 个近静止窗里模型 MAE 为 13.56 μm，远差于零位移的 0.36 μm，且预测均值为 -11.59 μm。6/10 tick 也出现类似的静止窗非零偏置。已在原 `fixture_response.py` 内增加明确版本化的事件概率头：由公开历史和已发请求预测各时域事件概率，只在事件窗口训练条件位移幅度，并用概率门控期望位移；旧 v1 checkpoint 与目标保持兼容。新配置 `segmented_response_event_gpu1.json` 对同一冻结公开 bundle 预检通过。真实 fit split 的事件率在 3/6/10 tick 分别为 11.55%/14.46%/16.78%，单个真实数据 minibatch 的 CPU 反传损失和梯度均有限；后续 GPU1 运行与验证结果见下。该模型仍不是 substep 安全证书。

### 2026-10-31 同状态 T3 配对与动态事件模型结果

T3 同状态 directional anchored 与新匹配 task-only 分别在 GPU1 完成 8 个实例、每实例 1500 次更新；训练报告分别为 `results/arm/keyed_insertion_recovery/training/finite_time_task_first_directional_same_state/anchored/checkpoints/report.json`（SHA `e97731c69d4c66ccdac1734290c2be3fbf4883171360651d82cb18f160307248`）及相邻 `task_only/checkpoints/report.json`（SHA `be5a87dd6b9120f9fd4c1b581251610e2e8c896503a86d09c56cfbab6a27ddb1`）。两分支的起点、公开数据摘要、八实例采样流、训练容量和预算一致，检查点均通过双模式冻结选择。旧 task-only 训练报告的数据元信息不同，因此没有借用它作本轮因果对照。新合同在 `configs/arm/keyed_insertion_recovery/validation/finite_time_task_first_directional_same_state_contrast.json`，保持旧 directional 对照的 16 个请求、两种机制 cell、8 个独立实例和预设判据；计划 SHA `a65991b40be742c619b323357c4a81f67f8a0583f31521fdf614c9e062854671`。

完整 GPU1 开发对照 `results/arm/keyed_insertion_recovery/validation/finite_time_task_first_directional_same_state_contrast/gpu1/report.json` 执行 256/256 个物理 episode，零 reset/执行错误；独立审计 `independent_audit.json` 对全部 256 条 evidence 重算通过。transport 成功 52/128、匹配 task-only 54/128；配对中双方成功 46、双方过力 68、仅 transport 过力 8、仅 task-only 过力 6。transport 的净成功率差为 -1.5625 个百分点，增加 8 次约束违规；在 46 对双方成功的请求上平均减少 4.30 控制步（约 0.80%），单侧符号检验 $p=0.779$。原冻结性能门失败，不能放行独立确认或声称同状态 gate 改善任务。transport-only 额外失败的 8 条均有非零 transport 接纳；68 条双方失败中有 38 条 transport 实际接纳为零，说明共同过力不能全归咎于新 gate。首个失败轨迹接纳 4 个 transport 步后回退到 task-only，数步后仍过力，提示瞬时回退不保证状态可恢复。

按预先存在的机制 cell 拆开，不改变主判定：`payload_shift` 的 transport/task-only 为 34/64 对 30/64（配对仅 transport 成功 5、仅 task-only 成功 1）；`latency_gain_shift` 为 18/64 对 24/64（相应 1 对 7）。在双方成功的请求里，transport 的平均控制步差分别为约 -3.97 与 -4.88 步。这是有方向相反的开发信号，不能事后仅选 payload 组宣称稳定优势，也不能把高延迟组的公共参照失败从分母移除。下一版应把响应模型对离散延迟的可辨识性、早期公共接触安全以及 gate 接纳后的恢复性列为独立机制问题，重新冻结协议后检验哪一类偏移确实受益。

跨运行版本不能直接和旧 directional 对照的 128/128 比较：新旧 task-only 的 8 份 checkpoint 的 `state_dict` 张量逐项相同，首个对齐请求的初始物理状态 SHA 和前 30 个请求动作也相同，但从接触后的第 30 个控制步起物理力/位姿分叉；actor manifest SHA 从旧计划的 `867bf9e8...` 变为本轮的 `94fdcb9e...`，evaluator 源文件 SHA 也变化。为区分版本漂移与同代码接触非确定性，另外固定 instance 01 的首 8 个 task-only 请求做复放；`replay_reproducibility_gpu1/replay_comparison.json`（SHA `46d533ae3f2eb141e7ab2bfef0bcd36afff40ceaf3678150cdff8f72b07618ca`）逐条核对初态、请求动作、物理轨迹、力峰和结果均 8/8 完全相同。复放仅是诊断，不进入 128 对分母。下一步先定位旧新运行时/物理变化，再在当前统一版本下处理双方共有过力与 transport 额外过力；不再把提高模型 gate 接受率当任务收益证据。

当前运行版的公共几何控制器另按冻结 `finite_time_current_public_geometry_sentinel.json` 在完全相同的 16 个机制/mode/初态槽上测试，`results/arm/keyed_insertion_recovery/validation/finite_time_current_public_geometry_sentinel/gpu1/report.json` SHA 为 `1bb0fbad1db7603a25ce471c0f0a6f05f8b661da9e2ef914a59e94baab72a6e8`。16 条原始证据均通过独立审计，且与 learned task-only 对应请求的物理初态 SHA 16/16 相同。公共控制器在 `payload_shift` 为 8/8 成功，在 `latency_gain_shift` 仅 2/8 成功；六条过力首发于 phase 0/1，最大力峰约 1804 N，轨迹仍为有限、有效的物理状态而非 reset/运行器错误。因此高延迟 cell 的共同安全机会本身没有建立，必须先审计/修复共享早期接触和延迟执行，再在新版本上重做全方法资格。不能把这个 cell 从已经冻结的对照中事后删除；这项共同故障也不消除 transport 相对匹配 task-only 的 8 条额外过力。

逐 tick 重查六条公共过力轨迹：按冻结判据的 25 N 总力或 10 N 横向力上限，全部在最后一个控制 tick 首次越限；前一 tick 的总力峰仅 9.84--16.73 N、横向力峰仅 2.63--9.05 N。首越限时四条处于 phase 1、两条仍在 phase 0，两条 z 请求为零，其余四条为 +0.02 m/s。总力峰包括 50.4、49.8、97.5、94.1、854.0 与 1803.9 N；其中两条峰值虽很大，控制频率末端 normal-force 读数却分别仅 77.6 与 -1.9 N。按当前证据，单凭控制频率的上一帧力阈值或只剪裁 phase 3 法向速度，不能保证预防这些子步冲击；早期阶段的接触几何、延迟执行及物理子步力证书须在新控制器版本中共同处理。

进一步离线检查发现，用公开 TCP 高度 `.5965 m` 作为早期撤离触发会命中上述 16/16 条轨迹，包括全部 10 条成功，因此它不是可选择性识别危险 episode 的判据。为直接检验“提前释放 phase 0 加载并从 phase 1 受限上移”这个结构假说，独立版本 `PublicEarlyContactMatingFeedbackPolicy` 在旧 actor 不变的前提下，按任务本地 `early_contact_public_guard_development.json` 对原 16 个机制／mode／初态完整运行。`results/arm/keyed_insertion_recovery/validation/early_contact_public_guard_development/gpu1/report.json` SHA `b61a05f2a87ad765097ac3b215e7b6bc338ede1e0f2e741f21adcb6f5840da13`，89.0 秒、16/16 完成；`independent_audit.json` 重新审计 16/16 evidence，通过且初始物理状态 SHA 全部匹配旧 sentinel。高延迟 cell 从 2/8 变为 5/8：旧过力→新成功 4，旧成功→新过力 1，另有 2 条仍过力；payload cell 维持 8/8。新过力三条发生在 phase 1、发出纯上移请求时，其中一条原成功轨迹在第 46 tick 发生 37.8 N 总力／35.3 N 横向力峰。这个候选说明早期时序可改变失败，但会伤害原本成功的请求，不能作为共享安全修复放行，也不能据净成功数直接推断 learned transport 的作用。该开发运行没有在执行前锁定代码快照；独立审计旁置了运行后源码 SHA，不能把它升格为独立确认。下一版应基于接触几何和可测延迟状态构造可审计的保护条件，并先证明不引入新增物理违规，再让所有方法共享新控制器。

动态响应 v2 的训练报告为 `results/arm/compliant_keyed_insertion_recovery/training/segmented_response_event_gpu1/report.json`（SHA `fc43da7f1469c95da9dbb5b0b696f0d894e148d226dd6bfcca9c7640dfa541e6`），同一冻结公开 bundle 上完成 600 次更新。训练对事件窗过采样，但事件概率 BCE 使用按采样概率修正的重要性权重；70 项相关回归测试通过。独立 CPU 逐时域复算 5,480 个验证窗的夹具 x 位移：旧模型到 v2 的总体 MAE 在 3/6/10 tick 为 15.38/23.84/28.52 → 7.65/9.17/13.15 μm，静止窗为 13.56/21.66/25.19 → 5.26/3.29/5.38 μm；真实事件窗为 28.13/35.75/43.86 → 24.38/41.24/48.89 μm，故 6/10 tick 事件子组反而退步，虽仍优于零位移预测。事件概率 Brier 分数为 0.0554/0.0605/0.0652，优于常数事件率基线 0.1094/0.1308/0.1467。模型改善总体响应但未证明动态任务收益或子步安全。论文 draft 只补了同状态反事实方向定义与事件混合误差分解，没有把开发结果写成实验报告或性能定理。

### 2026-10-31 T3 公开参考卸载的同初态机制对照

在早期撤离版本已固定的高度触发与上移命令上，开发版本改动撤离期间的平面请求：使用公开 `reference_position - tcp_position`，以覆盖三步执行延迟再多一观察步的 0.08 s 时标，将参考向 TCP 卸载，平面速度逐轴限于 0.025 m/s。该版本为 `PublicReferenceUnloadingMatingFeedbackPolicy`，不是 Cross-EBDM 的学习部件。配置 `configs/arm/keyed_insertion_recovery/validation/reference_unloading_public_guard_development.json` 在执行前绑定控制器、任务运行器及脚本三份源码 SHA；保持旧 sentinel 的 16 个机制／mode／初态请求，全部失败留在分母。**审计后发现本候选的 0.025 m/s 上限超过原公共 actor 的 0.020 m/s 平面上限；高延迟八条均实际越过原上限。因此它不是严格匹配的单变量机制对照，不能用它归因卸载效果。**

GPU1 发展运行的报告是 `results/arm/keyed_insertion_recovery/validation/reference_unloading_public_guard_development/gpu1/report.json`（SHA `8c257f0a6eb218d36da811dc79c4eb4a3a487a54c0c0b11908882e954626d414`）。16/16 物理 episode 完成、零运行错误、耗时 92.96 s；独立审计重新验算新旧共 32 份 evidence，16/16 初始物理状态 SHA 相同、三份预绑定源码一致。高延迟 8 个请求中，旧公共 sentinel 为成功 2／过力 6；只提前上移的候选为成功 5／过力 3；本次参考卸载为成功 4／过力 2／模式不匹配 2。相同初态相对旧 sentinel：三条过力转成功、两条过力转模式不匹配、一条仍过力、一条原成功转过力、一条原成功保持成功。另一个 payload cell 始终为 8/8 成功。两个模式不匹配均是 mode 0 任务到达物理终点，却以 mode 1 终态约 4.13 N 法向力结束，因此不能计入受约束成功。

机制上，在高延迟组的第 40 控制 tick，平面公开参考偏差的八条配对中位数由 2.197 mm 降为 0.387 mm；但冲击和模式错误仍存在。这个结果提示参考蓄能可能参与接触轨迹，严格的单变量归因须由匹配原上限的下一版检验。本版本明确拒绝放行，不能替换既有 T3 对照或用于独立主表。继续推进 T3 应先用公开姿态、接触力历史、参考状态和延迟动作构成可检验的子步冲击／终态模式预测，再在未用于模型选择的初态上同时检验过力与 mode；不再以单一高度、速度或卸载增益重复扫描。主表即使允许真实失败进入分母，也必须保留这些失败与版本差异，不能把模式不匹配当作完成。

随后只纠正这个超出原 actor 支持域的问题，新增明确版本 `PublicMatchedReferenceUnloadingMatingFeedbackPolicy`，其他卸载时标、撤离高度、动作、任务初态和截止时间不变，平面逐轴上限改为原 actor 的 0.020 m/s。新配置 `configs/arm/keyed_insertion_recovery/validation/matched_reference_unloading_public_guard_development.json` 同样在运行前绑定三份源码。`results/arm/keyed_insertion_recovery/validation/matched_reference_unloading_public_guard_development/gpu1/report.json` SHA 为 `18d197518cbb9c9b77f80f5602e4244b2b6138dce0f12e43f80ec86fba4ee8d4`：16/16 完成、零运行错误，88.48 s；独立重审新旧共 32 份 evidence、16/16 初态 SHA 相同，实际平面请求峰值为 0.020 m/s、零 episode 超出原上限。高延迟组为成功 4/8、过力 3/8、模式不匹配 1/8；payload 仍为 8/8 成功。相对旧 sentinel，三条过力转成功、一条过力转模式不匹配、两条过力仍过力、一条原成功转过力、一条原成功保持成功。相同的第 40 tick 平面参考偏差中位数由无卸载早期版本的 2.197 mm 降至 0.388 mm。于是现在有严格匹配动作上限的证据：压低可见平面参考偏差并不能单独解决高延迟下的子步冲击，也可能改变 mode-0 撤离轨迹并破坏模式合同。此候选同样拒绝放行；下一步先明确预测／保护的增广状态和独立初态验证门，不继续调卸载幅度或速度。

两次运行的 `source_snapshot/` 与 `source_snapshot_manifest.json` 均保存在各自结果目录，逐文件 SHA 与运行前绑定值一致；其中首版源码在第二版实现后已从工作树更新，因此复核首版时应读取其运行工件中的快照，而不是当前源码。

### 2026-10-31 ASID T1 三段开发链与辨识搜索审计

ASID 作者论文的完整语义是 Fisher 信息驱动的探索、目标轨迹与仿真同动作匹配的参数辨识、在更新后的仿真中训练下游任务策略；作者公开仓库只提供可直接移植的 Fisher 目标和其杆件环境的探索训练入口。现已在独立 `genedynamics/solvers/single/asid/learning` 中实现保留全部候选账本的有界 CEM，并在任务本地 `genedynamics/tasks/transfer/obstacle_asid.py` 接通 T1 xArm7。当前任务适配用 60 控制步的公共轨迹信息目标、CEM 搜索探索路线、同动作目标／仿真轨迹匹配及 CEM 下游路线训练；这不是作者原生 SAC 策略训练，也没有四任务名义资格。

首版任务本地配置 `configs/arm/obstacle_transfer/training/asid_full_chain_development.json` 执行 226 次仿真 episode／34,360 控制步、1 次目标探索和 2 次目标部署，耗时 135.61 s；`results/arm/obstacle_transfer/baseline/asid/training/full_chain_development_3817251/report.json` SHA 为 `54889f67acb926dba3613e23f71f58bd3eaba3fc108d1d85adff3b6959808c14`。它的目标部署 mode 0/1 均成功（384/383 控制步），但辨识给出 (gain 0.7143, mass 0.7358)，偏离模拟目标 (0.7, 0.5)。独立 CPU 同动作 11×11 参数地形审计额外花费 121 次仿真／7,260 控制步，并不回流训练；真实参数在无噪声公共轨迹误差上是唯一零误差最优，说明首版偏差来自搜索覆盖，而不能据此声称该设定下负载不可辨识。

为避免给 ASID 一条过弱的参数搜索路线，第二版事先冻结全域 11×11 粗网格，再以其最优解初始化相同 CEM 细化；探索政策、唯一目标探索轨迹、两次目标部署、任务搜索预算和随机种子均不变。`configs/arm/obstacle_transfer/training/asid_grid_seeded_development.json` 在运行前绑定六份源码。`results/arm/obstacle_transfer/baseline/asid/training/grid_seeded_development_3817251/report.json` SHA 为 `44c93745ffb77e904b6571ea401b1c6e7d3b3b3609f372ba077cc94cab2858af`：347 次仿真 episode／41,670 控制步、仍仅 3 次目标交互、166.16 s；121 个格点和 30 个 CEM 候选都保留在账本。独立审计核对 350 行交互、六份源码 SHA 及每阶段调用总量；格点和最终拟合均为 (0.7, 0.5)，误差为零。首版与第二版的探索候选、目标探索动作／观测、最终任务策略和两个目标部署结果逐项相同。正确辨识并未在这个 T1 开发 seed 上改变任务策略或成功表现；该结果只验证三段链能执行，不能证明 ASID 优于或弱于本论文方法。两个运行的六份绑定源码均有结果目录内快照；首版代码后来版本化，不能用当前源码冒充其运行时源码。

下一项准入工作是把 ASID 任务策略作为明确的公共 actor/checkpoint 接入独立 T1 source/nominal roster，并在未参与探索、辨识或下游搜索的种子上检验两种 mode、执行约束和与同容量公共路径的差异；随后才考虑 T2--T4 的任务参数化与完整四任务主表。所有目标探索、训练和失败均必须单列资源账；即使 T3 或其他任务失败，也保留在相应主表分母，不用开发 pilot 填充确认格。

该 T1 source/nominal roster 已实际完成，不再只是待执行计划。任务本地 `configs/arm/obstacle_transfer/validation/asid_nominal_diagnosis.json` 绑定了第二版 `checkpoint.json`（SHA `0bd5aad46be1223fafdca21b82f0c0513e7630ab428250a0b4d2178005548857`）、训练报告和三份运行源码，使用训练未访问的 24 个请求（source、deployment nominal、target gain shift 各四初态 × 双 mode），分别执行 ASID 和公共 anchor，共 48 个物理 episode。`results/arm/obstacle_transfer/baseline/asid/validation/nominal_diagnosis_grid_seeded_3817251/report.json` SHA 为 `5ca1d702f1c3a1eafe0ec7ed1bc93ae463370510f74bd009811a3f3d5dfdfd6a`：48/48 完成、零运行错误，每个 cell 的 ASID 与公共 anchor 均为 8/8 受约束成功。独立审计重算 48 份 evidence，24/24 配对初始物理状态 SHA、完整动作序列及完整物理轨迹逐项相同。这确立了 T1 source/nominal 开发资格，同时明确 **ASID 在当前 T1 参数化下没有独立任务行为或性能收益**；此结论不等于 ASID 方法在其他任务无效。该 checkpoint 仍为单训练实例开发工件，不自动放入八实例独立主表；后续优先在接触任务建立完整 ASID 任务适配，而不是继续增加 T1 同策略重复次数。

### 2026-10-31 T2 AdaptSim 策略空间预检

为避免把过弱的任务策略误称为强 baseline，在当前 T2 v7 shared execution 下，先对公共参照只调整接触路线出口前的 mode-wise 切向速度倍率（0.8、1.0、1.2），并在相同 reset 上执行 nominal 与高延迟增益偏移、双 mode 的 12 条 GPU1 可见物理 episode。`results/arm/guided_contact_alignment/baseline/adaptsim/validation/guide_speed_feasibility_4982301/report.json` SHA 为 `847da92b193bf98c5c81a25803b1d7ca19bc33b3a37668c72e3b061abc3c44d7`，运行源码在同目录：12/12 受约束成功，零违规；高延迟 mode 0 的三档为 976/938/932 controls，mode 1 为 952/952/960，名义两 mode 也全部成功。该小样本仅证明控制旋钮可执行、对时间有影响，**不是 AdaptSim 训练或名义资格**。它没有给受约束成功率提供可辨别的搜索空间，因此不将“两参数公共速度微调”包装为作者方法。T2 若继续接 AdaptSim，需要真正能改变闭环接触策略的任务训练器，同时保留共同 v7 控制器、目标 reward 交互账及源/名义独立资格；当前 T1 ASID 与 T2 公共参照饱和也意味着主表不能仅靠这两处证明 transport 的性能贡献。

### 2026-10-31 AdaptSim T1 完整候选账本与证据名义回放

原多粒子训练的 294 个候选只保存了平均 reward，未逐个保存双 mode 结果。使用原 seed、粒子、CEM 和 1500 秒预算在 GPU1 重跑后，`results/arm/obstacle_transfer/baseline/adaptsim/training/meta_coordinate_ledger_replay_928505/report.json` SHA 为 `c15995927702b88b2c33f4cf2d37db2b78b5a58785ee9bac584896b1684875f1`，用时 981.48 秒。42 次训练、294 个候选、588 个仿真物理 episode 的完整候选账本现含 506 success、67 timeout、8 mode mismatch、5 collision、2 speed-limit，82 条失败均留存。结果目录封存运行时 7 份源码。独立 `candidate_ledger_audit.json` 核对每个候选的 episode 计数、reward 均值、控制步数及源码快照；重跑与旧工件的候选分数、元训练/部署记录、最终策略与 Q 网络权重完全相同。它修复证据完整性，没有改变方法输出，也仍是单实例开发训练。

任务本地 `configs/arm/obstacle_transfer/validation/adaptsim_meta_nominal_evidence.json` 绑定新训练报告、checkpoint 和两份运行源码，以未在训练或旧名义诊断中使用的 16 个双 mode/双机制请求，将训练策略与公共 anchor 配对执行 32 个物理 episode。`results/arm/obstacle_transfer/baseline/adaptsim/validation/nominal_evidence_ledger_replay_928505/report.json` SHA 为 `f0f309df1d2de68bc1e46e5934295600d2d64d5de23b935084c74056fad558e9`：nominal 与 target gain shift 中两个方法均各 16/16 成功，零约束违规；32 份压缩证据共约 7.2 MB，独立 `independent_audit.json` 对全部 episode 重算物理判据并核对 16/16 相同初始物理状态。两策略的 16 对动作序列均不相同，所以这里有行为差异，但没有成功率优势。名义 mode 0/1 的 AdaptSim 相对 anchor 完成步数为 +5/−3；目标增益偏移 mode 0 平均 +37.75，mode 1 平均 −1.25。由此只能说该任务适配在 T1 名义与目标开发格可执行；不能把单个 CEM 策略、单个训练 seed 或饱和的 T1 成功率写成 AdaptSim 原方法的四任务性能结论。下一步需在可产生不同约束成功率的动态接触任务训练真正闭环策略，并保持作者参数分布更新机制、目标 reward 成本与现行共享执行合同。

上述 32 条回放后，为补齐明确的源资格，以新种子冻结了 source、nominal、target gain shift 三格，各四初态、双 mode、双策略，共 48 条 GPU1 物理 episode；旧回放不并入新 roster。任务本地 `configs/arm/obstacle_transfer/validation/adaptsim_meta_source_nominal_evidence.json` 绑定同一训练 checkpoint 与当前源码。`results/arm/obstacle_transfer/baseline/adaptsim/validation/source_nominal_evidence_ledger_replay_928505/report.json` SHA 为 `f17317f1abacfc38a35f2c7c52e665801269913f527bda779d40c746369113d1`，71.51 秒、17,946 控制步；两个方法均 24/24 受约束成功。独立审计 SHA `f3d5f282e03b2971ee205bcb27d8f4b133f19066a9f2baf7d2ed25b4e57d33cb`，核对 48/48 原始证据与 24/24 同物理初态。AdaptSim 相对公共路径的完成步数：source mode 0/1 为 +3/−2，nominal 为 +5/−3，target gain shift 为 +37.5/−1.5。因而 **T1 source/nominal 开发资格通过，但没有任务成功率优势**；方法仍只有一个独立训练实例，T2--T4 尚无合格任务策略，不能进入论文主表。

### 2026-10-31 动态响应的事件门控归因

对已有夹具动态响应 v2 的冻结开发验证 split 进行零物理调用的 CPU 只读分解；脚本及结果位于 `results/_analysis/four_task_transfer/fixture_event_gate_decomposition_20261031/`，报告 SHA `f120c33b7dc05aadce8eba55f465cdeecea80376146cd3eb4f620d4de73eaf17`。在 5,480 窗口上，3/6/10 tick 的事件概率均值在真实事件为 0.520/0.557/0.579、近静止为 0.055/0.062/0.068。条件位移头若不门控，静止窗口的误差高达 294/93/143 μm；当前概率加权后仍为 5.26/3.29/5.38 μm，而零位移为约 0.35 μm。真实事件上，6/10 tick 未门控条件幅度的 MAE 是 34.10/40.11 μm，概率加权后退至 41.24/48.89 μm；3 tick 的加权反而从 27.41 改至 24.38 μm。回看同一验证集，把概率硬阈值设为 0.5 可降低总体 MAE，却提高三个时域的事件 MAE；使用未来真实事件标签的 oracle gate 仅是不可部署的诊断下界。下一版模型应分别报告事件判别、条件幅度、静止假运动以及任务相关的漏检代价，并在新的 episode-disjoint 测试上检验；不能凭总体 MAE 改善或在旧验证集选阈值声称动态任务收益。理论中的响应误差项需要按事件与近静止模式分解，并显式包含门控误判项；ensemble spread 和控制频率预测仍不能充当物理子步安全上界。

### 2026-10-31 T3 AdaptSim 公共反馈任务适配及训练外诊断

同一 T3 shared execution、固定 mode 1 初态上的六条策略空间预检表明 `recovery_spatial_stiffness` 会改变 source 接触安全：公共值 0.25 成功，而 0 和 -0.25 均过力；nominal 三档都成功。原始 evidence 均通过重审，预检报告 `results/arm/keyed_insertion_recovery/baseline/adaptsim/diagnostics/feedback_policy_space_7319021/report.json` SHA 为 `bbae547fa55ad95f89aea089f16e3375bf0626ca327a6f8f3038371079cb261a`。据此在任务层 `mating_policy_search.py` 接入七维受界公共反馈策略，沿用同一 T3 actor、两种 mode 与共享执行；方法核心仍留在独立 `adaptsim` solver，不在高层 `learning` 增设算法目录。离散 delay 明确标为对作者连续参数更新的任务扩展，内层 CEM 仍是长时策略替代，不是作者原生 NAF。

GPU1 上预先冻结 72 条物理 episode／900 秒的单实例训练 `feedback_development_737215` 实际用 293.89 秒，六次任务策略训练、60 条候选模拟执行、四条模拟目标执行和八条部署目标执行全部入账；48 success、24 overforce，独立审计核对 72 条及 19 份运行源码快照。报告 SHA 为 `226b05d4ecacde1aad495af9ad459afbc6ff844a64483a0e3e3527332189ed2c`。高延迟目标上的两个粒子选择请求均过力，最终策略在该开发目标的双 mode 也均过力；中途一次成功不替代最终 checkpoint。训练外先做三机制各一初态的 12 条筛查，原始 evidence 和六对初态独立审计 SHA 为 `7776326b0bb95d5c74414dab67a6c21bce6a1af2a6919bbd40a77141ef05f1da`，其 nominal 双 mode 两方法都过力，说明小样本不足以判断资格。

随后在运行前固定另一组八个初态／机制、双 mode、AdaptSim 与公共锚点，共 96 条；任务本地配置为 `configs/arm/keyed_insertion_recovery/validation/adaptsim_feedback_heldout_diagnosis.json`，入口为 `scripts/paper/cross_ebdm/diagnose_adaptsim_feedback.py`。`results/arm/keyed_insertion_recovery/baseline/adaptsim/validation/feedback_heldout_diagnosis_737215/report.json` SHA 为 `7b11003cb09768ee07c51592fe9a4ad995bd5bd91d8952ecd213427821b18a59`，311.86 秒、26,880 物理控制步。独立审计 SHA `e57041317a47dd919a76925799325775e3af32a7f6e783b9c650911b08b66cfc`，96/96 轨迹重算、48/48 配对初态和五份预绑定源码快照通过。source 成功率 AdaptSim／公共锚点为 11/16 对 12/16，nominal 为 8/16 对 7/16，高延迟目标均为 4/16；所有 50 次失败都是过力并保留在分母。按同初态配对，source 的 AdaptSim-only／anchor-only 成功分别为 4／5，nominal 为 3／2，高延迟为 3／3，没有稳定方法优势。50 次过力发生在第 42--69 控制步，27 次前一控制步总力仍小于 10 N，31 次越限步的控制频率接触标志仍为 false；这支持早期子步冲击是两个 actor 的共同瓶颈，而控制频率上一帧力阈值不能充当安全证书。

因此不把这份单训练实例、多处 source/nominal 过力的开发工件升格为“合格 AdaptSim 四任务 baseline”，也不据此断言原方法机制必然失败。下一次训练前应先在统一共享接触合同下建立不增加新过力的公共保护条件，并把 source/nominal 准入、目标失败和额外训练／目标交互分别报告。若保护条件通过，再扩大每候选任务采样和元训练预算，使用新的训练 seed 与独立资格初态；旧 96 条只用于诊断，不可挪作主表或调参后确认。

沿这 96 条 evidence 做了一次零物理、零 GPU 的回顾性可观测性诊断，入口 `scripts/paper/cross_ebdm/analyze_t3_public_impulse_observability.py`，结果 `results/_analysis/four_task_transfer/t3_public_impulse_observability_20261031/report.json` SHA 为 `12f29dcb0e62b55ef6432bb4b72e2b8d3d3bb8fdb49404e9bece03c5f4425cc4`，重复运行逐字节一致。仅用公开 TCP 位姿及差分、公开参考偏差、控制频率法向力及差分、已经发出的动作，在早期接触 z 带对“下一控制 tick 首次过力”分类；同一初态的两方法始终同组，slot 0--3 训练、4--7 留出。训练／留出分别有 2,186／2,428 个窗口、29／21 个过力事件。线性风险分数在留出窗口 AUROC 0.941、AP 0.273，单一公开参考偏差范数为 0.856／0.182。按训练负例 95% 分位的阈值，前者在留出初态只提示 9/21 条过力轨迹，同时触发 4/27 条成功轨迹；后者为 7/21 与 2/27。窗口选择、特征和模型均来自已观察的开发结果，不能叫独立确认；这些数值只说明公开状态有部分可排序信号，**没有**证明阈值干预能阻止子步冲击，更没有覆盖未提示的 12 条过力。下一步若做共享 guard，须先冻结因果输入与单步干预、从新初态评估成功保持、mode、过力和成本四项，再让所有方法共用；不得用本次留出继续调阈值后冒充正式验证。

### 2026-10-31 T3 AdaptSim 多粒子预算与独立初态比较

鉴于 baseline 的机制性失败也需要按原样报告，本轮先在**完全相同的现有公共执行合同**下给 AdaptSim 更充分、但有界的训练机会，没有专为它修公共 guard。任务本地 `adaptsim_feedback_population_development.json` 先冻结两粒子、两轮外层 meta-target 评估、每轮两次内层模拟目标适配、两步部署更新及全候选账本。物理 GPU1 运行 `results/arm/keyed_insertion_recovery/baseline/adaptsim/training/feedback_population_development_8737217/report.json` SHA 为 `a69286eea4a86495d5755b30b0337de5a8dd794a3604d7cfb83b2d04932549fc`，用时 3491.90 秒；21 次策略训练、210 个候选、840 条候选物理 episode、16 次双 mode 模拟目标回放和 5 次双 mode 部署目标回放，总计 882 条。候选为 513 success、298 overforce、29 timeout；两类目标回放合计 21 success、20 overforce、1 timeout。独立账本审计 SHA `cb60ffb0b59abf15800514df14bf4b0f6a796aff4faa307819dd6f92da2a4520` 核对 882 条 outcome/控制步/reward、16 份源码快照、checkpoint SHA `d01c7f66459320e25ab1ac55114c400f74cda89e9c3f792d90cc6e6ba9cea28b` 和阶段资源账；它是账本与快照核验，不是 882 条逐步物理 trace 重放。模拟策略训练约 312,192 controls，模拟/部署目标共 12,613 controls，Q 网络更新 7 次。外层两轮未触发粒子替换，两个部署选择请求在同一目标 reset 分别为一成功一过力、双 mode 均过力；最终策略的最后训练目标回放是一成功一过力，不能用途中更好的目标请求替换最终策略。

随后用训练之外的 seed `3200400000`、source/nominal/高延迟三机制、每格八初态×双 mode、最终 AdaptSim 与公共 anchor 配对，执行 96 条 GPU1 episode；冻结配置 `configs/arm/keyed_insertion_recovery/validation/adaptsim_feedback_population_heldout_diagnosis.json` SHA 为 `cc825be4474543e7867e05250aaca4a20bc388ae40b67eef05181acb539a3a03`。报告 `results/arm/keyed_insertion_recovery/baseline/adaptsim/validation/feedback_population_heldout_diagnosis_8737217/report.json` SHA `a2e98a1ecf0904f4227dd051b99a3692f4dcb69547249ce65aae6c5b0b07e2e4`：328.21 秒、28,668 控制步。独立审计 SHA `537664a8a1a87f9037d2b3bc9344763842de49926f8e2da16429560156f3e58c` 核对 96/96 原始 evidence、48/48 相同物理初态、11 份源码快照及全部 outcome。source 的 AdaptSim/anchor 成功为 11/16 对 13/16，nominal 为 9/16 对 10/16，高延迟均 4/16；总计 24/48 对 27/48。AdaptSim 有 22 overforce＋2 timeout，anchor 有 21 overforce；过力首发在控制步 42--65，仍是双方早期接触共同瓶颈。配对仅 AdaptSim 成功/仅 anchor 成功在三格分别为 0/2、4/5、3/3。共同成功的 17 对中 AdaptSim 均更快，平均少约 39.1 controls；这是以双方成功为条件的描述性速度，不抵消成功率下降，也不能当成受约束总体优势。

与先前 72 条小预算训练的旧 96 条诊断使用不同初态，不能跨 cohort 直接算扩大预算的增益；但这份新训练外比较已说明当前更完整的任务适配仍未建立 T3 成功率优势。这里保留 AdaptSim 原多粒子与任务回报更新机制、冻结的 PyTorch Q 网络及额外交互成本，但内层长时策略学习仍是任务适配 CEM，而非作者原生 NAF，且只有一个训练实例。它可作为当前合同下**任务适配路线的有效失败诊断**，不能写成官方 AdaptSim 在四环境的强主表结果。后续若做公共早期接触保护，必须给 ours、task-only、公共锚点和全部 baseline 同一保护并重新测试；不再为 AdaptSim 单独扫反馈增益。独立主表仍待四任务可执行方法清单、资格和新确认集一起冻结。

### 2026-10-31 AdaptSim T3 公共 actor 接入与新请求验证

已将上述唯一真实训练实例的 checkpoint、训练 recipe 和报告以 SHA 绑定到 `configs/arm/keyed_insertion_recovery/baseline/adaptsim.yaml`；任务层 `controllers/adaptsim.py` 仅负责把七维策略解码为原 T3 公开反馈 actor，算法仍在独立 AdaptSim solver。该 actor 已注册到现有 transfer 方法入口，旧 v14 的九行／八实例 manifest 与确认集保持原状。新绑定在历史审计轨迹 `0_0_0_adaptsim_development` 的 325 个控制步请求动作逐值一致，报告 SHA 被篡改时拒绝构造 actor。

另以任务本地 `validation/adaptsim_public_actor_pilot.json` 预先固定四个新 seed、名义与高延迟两机制、双 mode，用统一 `evaluate_transfer_method.py` 在仅暴露物理 GPU1 的进程中执行；当前 actor 与 MuJoCo 物理仍在 CPU，不替换失败。报告 `results/arm/keyed_insertion_recovery/baseline/adaptsim/validation/public_actor_pilot_3300400000/report.json` SHA 为 `7d5dbc9898b37678b11fa293e2e18031220742ff7c809228b51792221a7ac67f`，四条原始证据经二次 CPU 重审及文件 SHA 检查均通过，配置 SHA 与运行报告一致，无 reset／执行错误。结果为名义 mode 0/1 两次过力（第 45/44 步，峰值 20.78/13.86 N），高延迟 mode 0 一次成功（408 步）、mode 1 一次超时（750 步）；总体 1/4 成功。它验证统一入口确实执行训练策略，并再次暴露早期接触失败；样本仅是开发 pilot，不能补入旧 96 条或论文主表。新评估计划保留唯一 `development_8737217` 训练实例身份，不能将其复制成八个独立训练实例。后续主表必须另冻结支持真实实例数的协议和全方法共同请求，先处理四任务方法可执行性及 source/nominal 资格，随后才生成全新、未访问的确认 roster。

### 2026-10-31 ASID T1 公共 actor 接入与新请求验证

ASID 的 `checkpoint.json`、训练 recipe 和完整报告已通过 SHA 放入任务本地 `configs/arm/obstacle_transfer/baseline/asid.yaml`，由 `controllers/asid.py` 将训练得到的路径和 mode-wise 平移速度倍率接到既有公共 T1 actor，算法的 Fisher 目标、辨识及有界搜索仍在独立 ASID solver。历史 `nominal_diagnosis_grid_seeded_3817251` 的首条 ASID episode 在新公共入口重放，369 个 13 维请求动作逐值相同，结果同为成功；修改训练报告 SHA 会阻止 actor 创建。

随后用 `validation/asid_public_actor_pilot.json` 冻结四个新 seed，nominal 与 gain-shift 机制各双 mode，统一方法评估入口在仅暴露物理 GPU1 的进程中跑完（actor／MuJoCo 仍为 CPU）。报告 `results/arm/obstacle_transfer/baseline/asid/validation/public_actor_pilot_3300500000/report.json` SHA 为 `5e1b54609bfba14450ed8b3b0d95636af11bb0a0c7bbfa0b2dea9734449c4b06`。四条均成功，控制步为 362、360、385、381；四条 evidence 经二次 CPU 审计、文件 SHA 和配置 SHA 检查均通过，零运行错误，运行 plan 与预检 plan 完全相同。这个小 pilot 仅建立统一执行链，**不改变**既有 T1 24/24 ASID／公共 anchor 成功且动作逐值一致的判断，也不将单个任务适配 CEM 训练实例冒充作者原生 SAC 或四任务强主表。

当前主表放行库存的零物理检查表明，旧活动 manifest 在 T1--T4 都把 FADA、DIAL、ordinary response MPC、task-only 标成 `instances_ready`，公共几何和直接 source prior 为单一固定身份；这只是工件解析状态，不能替代各方法在当前合同下的行为和源码资格。ACP 与 Human2Sim2Robot 在四任务均为 `predeployment_rejection`，不能记作真实零成功。新直接绑定的 ASID 目前仅有 T1 单训练实例，AdaptSim 有 T1 和 T3 各一单训练实例；T2/T4 尚无它们的训练策略。旧 v14 manifest 另强制所有 learned 行恰好八实例，因此本轮单实例工件不能直接加入旧包。独立主表冻结顺序仍是：先把真实可执行的任务适配和方法身份补齐，再按统一请求做行为资格；然后版本化支持各行真实训练实例数的 manifest、统计与成本账；最后生成与所有开发、训练和既有资格请求不相交的新确认 roster。不得以开发 pilot、逻辑复制的实例或 availability rejection 填补性能格。

### 2026-10-31 AdaptSim T1 公共 actor 接入与源码资格更新

T1 AdaptSim 的已审计多粒子训练 checkpoint、recipe、报告和候选账本审计现以 SHA 绑定到 `configs/arm/obstacle_transfer/baseline/adaptsim.yaml`。ASID 与 AdaptSim 的六维路径／mode 速度策略共用任务层 `controllers/obstacle_scaled_path.py`，各自训练和身份核验仍由独立方法入口负责；T3 AdaptSim 保留原 mating actor。旧 T1 AdaptSim source 证据首条回放 374 步，新入口与归档动作的最大差为 `4.85e-7`，结果均为成功，故按浮点容差而非逐位相等报告。现有 33 项相关测试通过，修改候选账本审计 SHA 会拒绝创建 actor。

任务本地 `validation/adaptsim_public_actor_pilot.json` 预先固定四个新 seed，在 nominal 与 gain-shift 两机制、双 mode 下经统一方法入口完整执行；仅暴露物理 GPU1，actor 与 MuJoCo 仍为 CPU。报告 `results/arm/obstacle_transfer/baseline/adaptsim/validation/public_actor_pilot_3300600000/report.json` SHA `e59ba443972b0e36207c59d99765580bb932fc41c23cd5927d374fb3c36e3580`：4/4 受约束成功，控制步分别为 367、357、421、381。四条压缩证据的二次 CPU 判据审计、文件 SHA、任务配置 SHA 均通过，执行计划与运行前预检逐字段相同，零运行错误。T1 的先前独立 source/nominal 对照仍显示 AdaptSim 与公共 anchor 都饱和成功，此 pilot 只核验统一入口，不证明任务优势或四任务复现。

按 2026-10-31 官方页面复核，[TDC 项目页](https://yifei-y.github.io/project-pages/TDC/)仍未提供可用作者代码入口；其早期 [Direction Matters 页面](https://yifei-y.github.io/project-pages/DirectionMatters/)虽显示“Code”，实际链接为 `github.com/YOUR REPO HERE`。不能依“从 GitHub 源码复刻”的要求凭空造一份 TDC 官方实现。[SGFT 官方仓库](https://github.com/WEIRDLabUW/sgft)已有代码，但公开示例是在 TDMPC2/DMC 上做源价值函数塑形和目标在线 fine-tuning；它需要目标 reward 与额外交互预算，应列独立权限组，不可当作现有 A-fixed 短 probe 主表的免费同信息 baseline。当前继续保留 ASID、AdaptSim 的明确任务适配身份，并优先补它们在可执行任务中的覆盖与独立训练实例；若 TDC 作者源码发布，再走固定源码 SHA、原生机制核验和任务适配准入。

### 2026-10-31 AdaptSim T4 公共移动执行可行性

T3 的固定臂搜索循环不能直接用于 T4：T4 每一控制步还要求按共享合同发出有限时长的公共底盘运动请求。任务层 `mobile_mating_policy_search.py` 通过原 `train_adaptsim_development.py` 入口对两个预先声明的七维反馈策略做有界检查；沿用同一 AdaptSim 策略参数表，但 **没有** 进行 AdaptSim 元训练或目标适配。两份 task-local 配置各在执行前校验 T4 共享底盘合同、四个新请求、两个策略、最多八个物理 episode、900 秒及物理 GPU1 身份；报告保留失败、不替换请求。actor 与 MuJoCo 计算仍在 CPU，仅运行进程暴露 GPU1。

源／名义检查 `results/quadruped/mobile_keyed_insertion_recovery/baseline/adaptsim/validation/mobile_feedback_feasibility_3300700000/report.json`（SHA `392149ec6c3f53b60fa0fcbe8990c70ec491b192f51cf52517d02496a7b858dd`）的 8/8 次执行成功，零约束违规，每次记录的最小支撑脚数均为 4。扰动检查 `results/quadruped/mobile_keyed_insertion_recovery/baseline/adaptsim/validation/mobile_feedback_shift_screen_3300700100/report.json`（SHA `1e3b79dda7a0822ab25f11010f8777df5857ec231626d94f0b35626ad2d5ba99`）的 8 次里 6 次成功：延迟／增益变化双策略双 mode 全部成功；负载变化 mode 0 双策略成功、mode 1 双策略均在第 115 步发生 `joint_limit`，且支撑脚数仍为 4。两份报告的 recipe SHA 已重算一致；总计 16 次、14 次成功、2 次共同 joint-limit 失败。相同机制、seed、模式下两个策略在同一步失败，提示共享可达性／执行合同可能比反馈增益更关键；这只是结构诊断，不能证明七维策略搜索无效，也不能据此宣称 AdaptSim/T4 已复现。T4 继续作为单列形态扩展；在共同 joint-limit 归因清楚并冻结一个不会饱和的策略学习目标前，不投入大规模 T4 AdaptSim 元训练。T2 仍缺训练过的 ASID/AdaptSim 策略，独立四环境主表仍未满足可执行条件。

配对的完整轨迹复测另以 task-local `adaptsim_mobile_feedback_joint_diagnosis.json` 冻结八次新请求，保存每条压缩 evidence；报告 `mobile_feedback_joint_diagnosis_3300700200/report.json` SHA `faceadafebe7a2f6207fc1cd43b33b96fe378a3bcc3e9fe75b6263d3cb5b4289`，CPU 独立重算八条均通过，`independent_audit.json` SHA `6e2207a6d06f5a4575e7d20f66ae891fee2d08debf1806df1275612eb25a5374`。同一负载／mode 1 请求的两策略 reset 物理 SHA 均为 `c03815a8463230957e60352cd825813a007ea444f58e60ee36f400c4509ce5d8`；第 115 步同为 `z1_joint4` 越界，末端关节余量分别为 `-0.000139`、`-0.000324 rad`。在这 115 步里，两个策略的 TCP 最大位置差仅约 `0.36 mm`，但请求动作的最大分量差达 `0.15`，说明没有理由把两条策略当成同一动作。底盘高度从约 `0.2776 m` 下降到约 `0.2675 m`，关节余量总体由 `0.0362 rad` 逼近零；越界前多数控制步仍请求 `+0.02 m/s` 的上移。该证据把共同失败定位到移动平台的具体关节边界和早期上移时段，但仍未隔离负载、底盘高度和参考轨迹的因果作用，不能直接将其归因于 AdaptSim 算法。

### 2026-10-31 AdaptSim T2 公共策略空间与共同失败

旧 T2 `guide_speed_feasibility_4982301` 只缩放分段动作的平移速度，同一小批请求 12/12 成功，不能支撑一个有竞争力的策略学习空间。任务层 `controllers/adaptsim.py` 增加独立的三维公共策略：进度不足时的控制权增长率、参考超前距离、接触阶段速度上限；它继承既有响应归一化 T2 公共控制器，保留冻结物理执行合同，只用公开观测。新 task-local `adaptsim_guide_policy_feasibility.json` 预先锁定三套参数、源／名义双 mode 共四个新请求、最多 12 个物理 episode、600 秒和 GPU1 身份；每条结果保存可 CPU 重算的压缩证据。这里没有运行 AdaptSim solver 的元训练，三套手选参数不能作为训练后的 baseline。

`results/arm/guided_contact_alignment/baseline/adaptsim/validation/guide_policy_feasibility_3300800000/report.json` SHA `2624aaae3b9ad258ff06485ed07d9756bd7234a3e6871a36ef4d88bed4541b5e`：锚点 3/4 成功，激进 3/4，保守 4/4；两条失败都在同一个名义 mode 0 初态过力，分别于第 706 和 676 步终止，保守策略在该请求 933 步成功。源条件下三套均双 mode 成功，但保守策略控制步较多。`independent_audit.json` SHA `af57c7b44b989e145f33c0e294b49299133ed9e75dda7b611112ff56547fa135` 对 12/12 条物理证据和结果重算通过。为了区分包装器错误与公共合同风险，另用未修改的公共 T2 actor 对失败的名义 mode 0 请求复放；`anchor_common_parity.json` SHA `5fd0c8fe1218a5a41971de4c5d2256d24b2cdd86f68dda06570a3bcfb35e99d2` 记录初始物理 SHA 一致、706 步动作逐值相同、同为过力。因此该开发初态上的失败属于当前公共参照本身，而非 AdaptSim 包装器偏差；不能外推到整个名义分布。

T2 AdaptSim 下一版应把约束违规放在完成时间之前定义奖励，并以分离的源训练机制／目标短 probe／全新验证请求执行作者式粒子推断和任务策略搜索；先审计训练候选账本与单个真实训练实例，再接入统一 actor 入口。当前敏感性检查只确认策略参数确实改变风险与完成时间，不证明 AdaptSim 能学出改进，更不把手选的保守策略计入主表。

### 2026-10-31 T2 AdaptSim 人口训练运行中与独立主表准入

`configs/arm/guided_contact_alignment/training/adaptsim_guide_population_development.json` 已冻结完整任务适配训练：两粒子外循环和 branching-Q 推断、每次策略调用的双 mode CEM、单个真实开发目标的粒子选择和两次更新。静态预算为 21 次策略训练、840 个候选物理 episode、16 次模拟目标双 mode 回放、5 次真实开发目标双 mode 回放，合计最多 882 个 episode、16,200 秒、GPU1 最多 2 GiB 显存；24 个相关源码文件各自有 SHA 并在运行目录快照。奖励首先区分受约束成功与违规，成功内再按控制步数排序；公开特征来自目标末端误差、实测力和已执行步数，不把构造机制参数传给 actor。四次候选加两次真实目标的端到端 smoke 已在物理 GPU1 上通过，公开特征有限，branching-Q 给出合法三维动作；smoke 不计入正式账本。完整训练当前由独立进程运行，`results/arm/guided_contact_alignment/baseline/adaptsim/training/guide_population_development_8764501/launch_receipt.json` 记录进程与 recipe/plan SHA，终局以该目录的 `report.json` 和后续独立审计为准，不能把启动当完成。首个 CEM 调用已经按计划记录 40 个候选 episode；训练目录的 24 份源码快照逐项匹配启动时 SHA。任务层 T2 checkpoint 入口在启动后加入统一 actor factory；当前源码因此与训练快照不同，但运行进程已加载并快照旧训练版本，后续 actor 必须用轨迹对齐验证，不可据源码名称假定等价。在真实 checkpoint／审计产出前没有创建任务本地 baseline YAML 或伪造主表实例。

CPU-only `scripts/paper/cross_ebdm/audit_main_table_admission.py` 对当前四任务旧 manifest 与 ASID／AdaptSim 直接任务 actor 形成新的 44 单元库存，报告 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v2/report.json` SHA `35e43c4b4ca1f0f56817ea61c63d8c567d83627413f5c71cf9646e1feb0afedb`。旧 ACP／Human2Sim2Robot 共八个单元为**未执行**，不能在任务性能列记零；新增方法另有五个任务 actor 缺失，三条已有训练 actor 仍仅具开发证据，28 个旧单元只有可解析工件状态。报告将 44 个性能值全部保留为空，主表当前 `admission_pending`，零物理、零 GPU 调用。新主聚合拟固定 T1--T3 等权、T4 单列形态扩展；主要报告受约束成功、约束违规、截断归一化完成时间和最弱机制组成功率。下一步先完成训练与各方法当前合同下的 source/nominal 行为资格，再生成全新不重叠的独立请求；不得把旧 v14 拒绝占位、开发 pilot 或单实例的逻辑复制并入论文主表。

### 2026-10-31 T2 新请求上的确定性公共反馈参照

在训练 checkpoint 尚未生成前，先以 `configs/arm/guided_contact_alignment/validation/adaptsim_guide_fresh_actor_roster.json` 固定 16 个全新 reset（源、名义、延迟／增益变化、负载变化各四个，双 mode 与两个初态槽位），再运行现行协议中 `public_geometry` 的确定性 finite-state T2 actor。`configs/arm/guided_contact_alignment/validation/public_geometry_fresh_actor_pilot.json` 直接引用同一 roster SHA，不需要伪造 checkpoint；`method_evaluation.py` 现在允许确定性公共反馈行使用与 learned baseline 相同的任务执行和证据格式，并提供 CPU 独立重算及同初态成对审计入口。

该公共参照在四组分别为 4/4、4/4、4/4、4/4，合计 **16/16 受约束成功、0 约束违规、0 执行错误**。报告 `results/arm/guided_contact_alignment/baseline/public_geometry/validation/fresh_actor_pilot_3300900000/report.json` SHA `d4101914de210d9570510d5bab8df39fae14c42a623a3a788019698b9b9eb588`；同目录 `independent_audit.json` SHA `4715a940f18623b61bf4ae58ddb48844d884439cc9760e1036833c7c713970fd`，16 份物理证据全部通过 CPU 重新评分，16 个 reset 物理哈希互异。各组平均完成控制步约为源 930、名义 894、延迟／增益变化 946、负载变化 935，均接近 1000 步上限，保留了完成时间的比较空间。这个小规模开发参照已经没有成功率提升空间，后续 AdaptSim 与它逐请求比较时应同时看违规、完成时间和更难机制组，并不得把该 pilot 升格为独立主表证据。T2 人口训练仍在原 GPU1 进程中运行；验证将等待训练报告、882-episode 账本审计和 checkpoint 入口对齐完成。

论文 draft 的实验设计已同步为十一行四任务名册：matched task-only、AdaptSim 与 ASID 明确出现在主表计划中，同时保留 task-only 的消融角色；四个共用指标及 T1--T3 等权、T4 单列的规则写入 `latex/cross_ebdm/sections/experimental_design.tex`。该处只写预期比较协议和未执行格子的处理，不复制本页的开发实验记录。AdaptSim 的来源条目按作者论文与公开项目添加到 draft 参考文献。

### 2026-10-31 T1 七方法同初态 source/nominal actor 筛查

为避免旧 predeployment 的“所有请求必须成功”资格门把真实弱 baseline 排除在比较外，新增任务本地 `method_source_nominal_actor_roster.json`：xArm7 上的源机制与名义机制各两个物理槽位、双请求模式，共八个先冻结的 reset；七个旧 manifest 可执行方法各使用一个真实训练实例或其确定性身份，最多 56 条、GPU1、失败不替换。这里的“source”仅指目标机器人上的源机制条件，不代替 Panda 源机器人正控制。`method_evaluation.py` 现在支持从现有 SHA manifest 直接绑定真正的实例，拒绝把 ACP/Human2Sim2Robot 的未执行占位当作 actor；所有方法仍使用同一物理执行和证据入口。

56/56 条物理 evidence 通过 CPU 重新评分，八个请求在七个方法之间的完整 reset 物理哈希逐项一致；汇总审计 `results/arm/obstacle_transfer/validation/method_source_nominal_actor_screen_3301000000/independent_audit.json` SHA `a8c0668194c1ca61c72b12e3971e2f722aa585cbae070ec549e203ce26abb8ac`。受约束成功为 Ours 8/8、matched task-only 8/8、DIAL-MPC 8/8、ordinary response MPC 8/8、public geometry 8/8、FADA 0/8、public source prior 0/8；全部方法 0 约束违规、0 执行错误。Ours 平均截断完成时间约 0.725，task-only 约 0.721，DIAL 约 0.713，普通 response MPC 约 0.720，公共几何约 0.730；**此小规模开发筛查没有显示 transport 的 T1 任务收益或最快完成时间**。FADA 和冻结源先验的失败均为完整 500 步 timeout，不是拒绝或崩溃。FADA 的 final position error 在 mode 0 为约 18--23 mm、mode 1 为约 225--236 mm；公共反馈约 3.5--4.2 mm。新请求上的旧 manifest FADA checkpoint 来自封存的早期开发链，不能单独代表最佳源修复版；既有 source-preserving/planner-repair 的 0/30/120 s 目标数据曲线 pilot 也都是 0/8，因此下一判断应核对这些 checkpoint 的源资格、训练链与命令图表，再决定是否只报告当前任务适配的失败，不做无界增益扫描。

旧的开发评估器仅汇总 policy `network_calls`，使 DIAL/ordinary response MPC 在报告里看似模型调用为零；审计轨迹的 controller diagnostics 实际保存了 `irreversible_model_calls` 和 DIAL 的 candidate rollouts。后续运行入口已分别计数 policy、response-model、candidate-rollout 与优化更新，旧报告不回写。CPU-only `results/arm/obstacle_transfer/validation/method_source_nominal_actor_screen_3301000000/resource_reconciliation.json` SHA `1044609233d3399aeef6aad8c17b33e398b0c09656bbed4970036df431372726` 从 56 条已核验轨迹恢复资源账：DIAL 共 5,768 次模型调用和 11,818,632 个候选 rollout，ordinary response MPC 共 17,292 次模型调用和 14,410 次优化更新；Ours/task-only 分别调用策略 2,901/2,885 次，公共几何为零模型调用。这里的“候选 rollout”是 controller 自身累计计数，不等于真实物理 episode；时间与计算成本比较还须统一测量其实际延迟和硬件。上述 56 条仍是单实例开发筛查，不是八实例独立主表或所有四任务的结论。

### 2026-10-31 FADA 修复链复核与 T3 配对筛查预检

按任务本地 FADA checkpoint 链复核，`repaired_source_positive_control` 使用的是各次修复训练输出的 `source_checkpoint.pt`，而训练报告的 `source_checkpoint` 字段指向更早的上游输入 checkpoint；两者 SHA 不同，不能混用。前者在 Panda 源机器人上的 T1/T2/T3 成功数为 4/8、8/8、8/8，全部零约束违规。T1 的修复后源能力仍不充分，因此其目标失败尤其不能归为单一跨形态机制问题。T1 的 source-preserving 与 planner-repair 两条目标数据曲线在 0/30/120 秒各为 0/8，全部 timeout；T2 的 planner-repair 曲线各为 0/8，0/30/120 秒约束违规数分别为 6/8、8/8、8/8；T3 的同一曲线各为 0/8，违规数分别为 5/8、2/8、0/8，其余为 timeout。各报告均标注 per-episode evidence audit 通过。源正控制、目标初始化、目标预算后 checkpoint 是三个不同工件；这些开发请求说明当前任务适配链未成功，不能证明 FADA 原算法普遍无效，也不应无限追加反馈增益搜索。后续主表须明确选择训练规则固定的 FADA 实例和独立请求；当前旧 manifest 的 FADA 运行只作诊断。

T2 的七方法 source/nominal 名册已在 `configs/arm/guided_contact_alignment/validation/method_source_nominal_actor_roster.json` 冻结，物理 GPU1 上串行执行，运行结果仍以完整 56 条审计为准。T3 的对应名册位于 `configs/arm/keyed_insertion_recovery/validation/method_source_nominal_actor_roster.json`；其七个现有 manifest actor 已全部通过零物理调用 preflight，配置固定 xArm7、源/名义机制、双 mode、两个物理槽位、共 56 条开发 episode、GPU1、失败不替换。T3 执行程序仅在 T2 的 56 条独立证据与资源审计全部落盘后启动；两份名单都只含单个真实 learned 实例与确定性参照，不开启正式 confirmation，也不能据此填独立八实例主表。

### 2026-10-31 T2 七方法汇总与 T3 native 启动缺陷封存

T2 七方法 source/nominal 开发筛查的 56/56 条物理证据与八个请求的全方法相同初始物理哈希均由 CPU 独立复核，`results/arm/guided_contact_alignment/validation/method_source_nominal_actor_screen_3301100000/independent_audit.json` SHA `6f55576803cbbeb248fd3389a13ddcfed2735d9d36eb46bc64ecb03b36b16658`。Ours、matched task-only、普通 response MPC 各 8/8 成功，公共几何 7/8（一次名义过力），FADA 0/8（七次过力、一次 timeout），DIAL-MPC 0/8（两次过力、六次 timeout），公共源先验 0/8（三次过力、五次 timeout）。所有方法的执行错误数为零。Ours/task-only 各消耗 7,371/7,376 控制步，成功率相同；相同初态审计的平均截断归一化完成时间为 0.92138/0.92200。普通 response MPC 为 0.90263、7,221 控制步，快于 Ours，但累计使用 43,326 次 response-model 调用和 36,105 次优化更新；DIAL 使用 15,152 次模型调用及 31,046,448 个控制器候选 rollout。公共几何在这一新小样本上不是全成功参照。资源报告 `resource_reconciliation.json` SHA `2ae726ee36de0af0d0fea6b2bf6d12700ea03ad2241b864514eb13cfa63735b9`，逐条对齐已核验的轨迹。初次 CPU 资源审计错误地只读诊断顶层；T2 的公共包装器将计数置于 `candidate` 子字段。审计器已修正并在 T1 原 56 条轨迹上重算得到与旧报告逐字段完全相同的结果。以上只是单实例、八个开发请求，不证明 transport 有稳定任务增益，也不把旧 FADA checkpoint 的失败上升为原方法的普遍结论。

条件启动的首份 T3 名册 `method_source_nominal_actor_roster.json` 因运行环境缺少 `CROSS_EBDM_NATIVE_LIBRARY` 而无效：七方法各八个请求均保留；Ours、task-only、DIAL-MPC、普通 response MPC、公共源先验共 40 条在 actor 创建时发生 runner error，没有物理轨迹；FADA 和公共几何各八条有实际物理证据。该批不能跨七方法配对或填任务性能表，`results/arm/keyed_insertion_recovery/validation/method_source_nominal_actor_screen_3301200000/invalid_run_audit.json` SHA `a1ab618aa55e3af015a3af25a352b140497fbc8034996bc4d57d48dc7978fcb9` 绑定原始报告与审计。已另冻结不重叠的全新 seed 名册 `method_source_nominal_actor_roster_native.json`（SHA `2ca1929aec90a5cf055cb30ba584072386f65f46f07d8d4fae9e2960116bdc16`），绑定现有 ABI v1 动态库 SHA；七个真实 actor 及 ABI 版本完成零物理调用 preflight，错误库路径在创建任何输出前被拒绝。新批在 GPU1 按方法串行运行，最多 56 条、2 GiB、2,400 秒，单方法审计出现 runner error 即停止。旧无效批不被覆盖、补跑或计入新批分母；新批是否通过以最终完整独立审计为准。

修正后的 T3 新批已经完成，`results/arm/keyed_insertion_recovery/validation/method_source_nominal_actor_screen_3301300000/independent_audit.json` SHA `d0cc54072b5411b28e76391fc9f4d5ebf180003203060e11a94424ff44b25b27`：56/56 条物理 evidence、每个请求的七方法相同 reset 哈希均通过。受约束成功为 task-only 5/8（源 2/4、名义 3/4）、公共几何 4/8、公共源先验 4/8、Ours 2/8（源 0/4、名义 2/4）、DIAL-MPC 2/8、普通 response MPC 2/8、FADA 0/8；没有执行错误，其余均为真实过力或 timeout。资源重算 `resource_reconciliation.json` SHA `1839122c679d5aa023983942d1ba4c9110ffa4e74e2dc6c03d52a9561b920f2d`：Ours/task-only 各 2,980/2,738 次策略调用，DIAL 3,136 次模型调用和 6,425,664 个候选 rollout，普通 response MPC 8,850 次模型调用和 7,375 次优化更新。此批是一个训练实例、八个源/名义开发请求，不能冒充八实例独立确认或全部目标机制泛化；但它明确否定了“当前 T3 选择器在这些条件下不损害 task-only”的叙述。

针对 Ours 失败而 task-only 成功的三个完全同初态请求，Ours 首次采纳 transport 分别在控制步索引 30、29、21；其后累计采纳 11、12、5 次，并在第 47、45、51 个控制步过力。首次采纳前，两方法的 13 维 issued action、TCP 位置、关节位置和控制频率力峰值逐值一致；首次采纳的动作最大坐标差约为 0.0988、0.0858、0.0800。对应控制器记录仍给出正的最差成员 task-proxy gain（末步 0.133、0.236、0.245），却未阻止后续过力。该可归因分叉说明仅凭局部代理任务收益和当前执行投影，不能保证接触阶段的未来约束；后续修订应在**同一公共状态**下对 candidate 与 task-only 的短时力风险做配对预测并在有校准覆盖时放行，否则精确回退到 task-only。新的风险门必须先在训练外接触事件/机制 holdout 校准，再用全新未访问请求评估；现有 T3 动态模型的短时位置 MAE 不能直接冒称力峰值安全证书。论文理论需把 transport 的承诺限制为校准覆盖内的选择性干预，显式区分预测代理收益与目标轨迹的受约束成功。

完整八对的首次干预审计由 CPU 入口 `scripts/paper/cross_ebdm/audit_t3_transport_intervention.py --roster configs/arm/keyed_insertion_recovery/validation/method_source_nominal_actor_roster_native.json` 重跑，输出 `results/arm/keyed_insertion_recovery/validation/method_source_nominal_actor_screen_3301300000/first_intervention_audit.json` SHA `ffabec26e792cb2437d80ff7ff38883c5cd15447e304202065661f3a1acb7fea`。八对中只有四对曾采纳 transport，首次动作差恰好发生在首次采纳步；采纳前完整物理轨迹逐值相等。采纳的四对中三对是 Ours 过力／task-only 成功，另一对双方都过力；未采纳的四对全程轨迹相同，双方各有两次成功和两次过力。因此这批开发样本**没有观察到有利的 transport 干预**，也不能由“拒绝所有候选”得出 transport 增益。四条采纳轨迹都在接近过力的末端继续采纳；首次采纳时横向物理子步力峰值只有约 1.1--2.2 N，末端为 12.6--29.7 N，超过 T3 的 10 N 横向限值。首次采纳距过力相隔 14--29 个控制步；现有响应模型的 10 步控制频率力预测既没有覆盖全部这段距离，也不等价于物理子步峰值。下一版需要分别验证风险校准能否避免已见过的伤害，以及在全新请求上是否存在通过风险门且有正任务收益的候选；两项都成立才可支持选择性 transport 的任务贡献。

理论修订应把局部 task-proxy 改成**提案评分**而非受约束任务收益保证。允许执行 transport 的证据应是同一公共状态下 candidate 与 task-only 两条短时执行分支的增量任务收益及物理子步力风险，训练与校准在 episode／机制层隔离，再用未访问请求验证闭环结果。风险模型需要覆盖滚动决策和干预后状态分布；仅凭固定 10 步边际误差或 ensemble 方差，不能声称整条轨迹的约束概率保证。若无法得到这种覆盖，论文只能报告经验性选择门、实测违规率和适用条件，不写形式化安全定理。通过风险门的候选还必须在新请求上相对 matched task-only 带来正收益，否则其作用只是公共回退机制。

### 2026-10-31 T4 七方法 source/nominal 移动形态开发筛查

任务本地 `configs/quadruped/mobile_keyed_insertion_recovery/validation/method_source_nominal_actor_roster_native.json`（SHA `70dd4758752e671372132e80d0890ac2d2e7bf18c6bffa00fcffd6d7caa5b50a`）在执行前冻结七个现有 manifest actor、八个从未使用的 reset seed、源／名义两种伺服与负载机制、两个物理槽位、双 mode 和共同 `body_servo_gain=1.0`。T4 的 folded stance 与公共 body intent 保持现行 task contract v3，ABI v1 动态库按 SHA 绑定；每个 learned 方法仅选真实 `instance_01`，两个公共参照各为单一确定性身份。所有 actor 已完成零物理构造预检，GPU1 与 T2 AdaptSim 训练并发，T4 单进程显存上限 2 GiB、总运行上限 7,500 秒，失败不替换。运行收据在 `results/quadruped/mobile_keyed_insertion_recovery/validation/method_source_nominal_actor_screen_3301400000/launch_receipt.json`。

56/56 条物理 evidence 经 CPU 独立重评分，八个请求在七个方法间的 reset 物理哈希逐项一致，零 reset／执行错误。总审计 `results/quadruped/mobile_keyed_insertion_recovery/validation/method_source_nominal_actor_screen_3301400000/independent_audit.json` SHA `3b02b3c13107e296bf84d08ca2a0e59673b4e1f758253216d973504212e4f5ab`；资源审计 `resource_reconciliation.json` SHA `2f664508730dbfc4224a4773d2591deb62668b654e087690d349ca7b21748116`。Ours、matched task-only、DIAL-MPC、ordinary response MPC、public geometry 和 public source prior 都为 8/8 受约束成功，FADA 0/8 且八次均为真实过力。六条全成功路线的平均截断归一化完成时间分别为 0.6610、0.7853、0.6713、0.6870、0.6738、0.6727；Ours 累计 3,966 控制步，task-only 4,712。DIAL 消耗 8,120 次模型调用与 16,637,880 个候选 rollout；普通 response MPC 消耗 24,732 次模型调用与 20,610 次优化更新。Ours/task-only 的单独成对审计 `results/quadruped/mobile_keyed_insertion_recovery/main/cross_ebdm/validation/source_nominal_actor_screen_3301400000/paired_development_audit.json` SHA `fbcffc5d931f9afb10e0e854a56296ca127bc8ca4d8703e5bc550c973a9a9da1`，八对均成功、零违规，源与名义子组各 4/4。这个单实例八请求的开发筛查支持 T4 移动形态可执行性和本批次完成时间优势；训练身份并非严格单因素，不能据此把速度差全部归因于 transport，亦不能替代八独立实例的正式主表。FADA 的失败仍需结合源质量与适配链解释，不能称为原算法普遍失败。

### 2026-10-31 T3 同状态物理子步力风险采集预检

将源／名义 mode-0 各一条新请求固定在任务本地 `configs/arm/keyed_insertion_recovery/validation/paired_force_fork_preflight.json`，只在首次 transport admission 分叉，各分支最多 32 控制步，公共前缀逐动作相同；先恢复完整物理及控制器状态，再重放首动作并保存两份可独立重评分的压缩证据。两条请求合计实际收费 122 个控制步，未超过冻结的 448 步上限；`results/arm/keyed_insertion_recovery/validation/paired_force_fork_preflight_3301500001/report.json` SHA `befaba6121425d87e57c7ebb3719644ba350ea784af7579d6bb2fb80d7be6c9f`。执行前另有不保留工件的单 seed 技术试跑，仅用于验证 checkpoint API，完全排除在此处的结果与下一版校准数据之外。

CPU 独立审计 `results/arm/keyed_insertion_recovery/validation/paired_force_fork_preflight_3301500001/independent_audit.json` SHA `18b84608e1d3a5e88a0298ca6fe3c1e4d7c2659e84f8d054527554dbe80b62e9` 重新计算四份 evidence 的判据、两分支完整共同前缀、首次动作与未来物理子步峰值。源请求在第 30 控制步首次分叉：Ours 继续 15 步过力（横向峰值 14.91 N），task-only 继续 5 步也过力（12.60 N）；名义请求在第 22 步分叉：双方继续 25 步都过力，横向峰值分别为 12.22/16.70 N。它们表明有些状态是两方法共同不安全，不能把每次 Ours 失败都归咎于 transport，也不能以这两条样本宣称风险门可校准。首次**策略请求**动作最大差为源／名义 0.216/0.020；经过公共执行器后的**实际 issued action** 最大差仅 0.107/0.017。未来风险模型必须区分提案、投影和真正执行的动作，标签取物理子步峰值；训练、校准、验证按独立 episode 与机制分组，并在闭环新请求上检验是否既减少额外违规又保留有益干预。

### 2026-10-31 T3 首次干预风险数据与候选方向诊断

将上述同状态分叉扩为任务本地 `configs/arm/keyed_insertion_recovery/training/first_intervention_force_risk_data.json` 的 64 个全新 mode-0 请求（SHA `fbb453ac4db2f21b25018199db68c430253fb01ab9dd49ea49776a386071fa59`）。拟合 24、校准 12、同机制验证 12、未见机制验证 16；前三区的 source/nominal 机制相同但 episode seed 完全不重叠，最后一区单列 delay/gain 与 payload shift。每个请求只在**首次**采纳处以完整物理 checkpoint 分叉，最多 160 步公共前缀、两条各最多 32 步的物理 continuation；无采纳请求保存前缀证据，不补样。总上限 14,336 个物理控制步、GPU1 单进程 2 GiB、2,500 秒外部超时，与 T2 AdaptSim 并发。运行报告 `results/arm/keyed_insertion_recovery/training/first_intervention_force_risk_data_3301600000/report.json` SHA `b94c086e5bef7517c0dc1d294254316fc0d4eef9c8f5c83e64f57997cdcc3cab`：64/64 请求完成，62 次配对分叉、2 次在前 160 步内无采纳，实际收费 5,029 控制步。CPU 独立审计 `independent_audit.json` SHA `33c1804ec7d2a3d8418121edf08fd253bad6a4d5ed20f9bb1f1aca4389b6ff02` 对 126 份 evidence 重新评分，并核对公共前缀、请求与 issued 动作、子步力峰值及收费账本。冻结运行器的原 `--audit` 分支曾把带 outcome 字段的 evidence 收据误当作仅含 path/SHA 的指针；独立审计入口仅规范该指针形状，运行器源码与其已绑定 SHA 未改，全部物理标签仍由原始 evidence 重算。

| 预先划分的集合 | 请求／有效分叉 | 仅 Ours 在 32 步内过力 | 仅 task-only 过力 | 双方过力 | 双方未观察到过力 | 无首次采纳 |
|---|---:|---:|---:|---:|---:|---:|
| 拟合 | 24／22 | 13 | 2 | 4 | 3 | 2 |
| 校准 | 12／12 | 5 | 0 | 4 | 3 | 0 |
| 同机制验证 | 12／12 | 7 | 0 | 5 | 0 | 0 |
| 未见机制验证 | 16／16 | 9 | 0 | 2 | 5 | 0 |
| 合计 | 64／62 | 34 | 2 | 15 | 11 | 2 |

这些标签只描述首次干预后**32 个控制步以内**的两条策略 continuation；未过力不等于最终任务成功，也不能把后续所有差异归因于单个首动作。尽管如此，34:2 的不对称表明当前 T3 v15 候选／选择组合在该采集分布下主要增加短时横向过力风险，不能通过调一条高回退率门就宣称 transport 提升任务表现。拟合集上，13 条 Ours 独有过力分支的首次采纳前横向力峰值中位数约 1.60 N，已有的正 task-proxy gain 中位数约 0.35；瞬时力阈值或代理增益均不足以区分未来伤害。task-only 也有 15 条共同过力分支，故回退本身不是安全证书。校准集合仅 12 个独立 episode，不足以给出非平凡的 95% split-conformal 上分位界。上述验证与未见机制标签已经在机制审计中汇总查看，**不能再当作未访问调参集**；下一版先检查候选族内有无更多安全且有正任务进展的动作（例如从同一状态比较预冻结的混合倍率），再扩大按 episode 隔离的拟合与校准样本，并另留全新闭环请求评估风险门和任务收益。

### 2026-10-31 T3 首次脉冲与持续 transport 的同状态诊断

任务本地 `configs/arm/keyed_insertion_recovery/validation/first_intervention_single_pulse.json`（SHA `c2c74fe030625285c5f262d845ea273a62cf259f6d2a81faf1a3ca0e1db02144`）预先固定了与上面 64 条完全不重叠的 16 个 mode-0 请求，source／nominal 各 8。每个请求在首次 transport 接纳之前执行相同物理及公开 actor 前缀，然后从完整 MuJoCo／控制器 checkpoint 派生五条最多 32 控制步的分支：首次请求为 task-only 与 Ours 请求的 0、0.25、0.5、1.0 比例混合，其后均执行 task-only；第五条从同一首次 Ours 请求开始并继续完整 Ours。混合发生在**策略请求**空间，实际 issued 动作仍由共同执行器投影和限速。该设计把单次脉冲及其诱发状态，与后续反复使用 Ours 区分开；它不是完整 episode 的成功率实验。

GPU1 与 T2 AdaptSim 并发的运行 `results/arm/keyed_insertion_recovery/validation/first_intervention_single_pulse_3301800000/report.json`（SHA `da01442df40919b658f5c17c4d47a981a2052ab6104cd4c52523a849c37ffafa`）完成 16/16 请求、80 条物理分支 evidence、收费 2,714 控制步、零执行失败。独立 CPU 审计 `independent_audit.json`（SHA `ea44878dd404e90577dec007dd66b5e4ec767cdddf58730d569adfa780d050a5`）重算全部 80 条物理判据，逐条核对共同前缀、首个请求混合、issued 动作、子步力峰与控制步账本；还验证单脉冲和持续 Ours 的首次完整物理转移 16/16 相同、下一次请求分歧 16/16 都始于第二次 transport 接纳。持续 Ours 在首次接纳后又接纳 1--11 次，首次后续动作分歧发生在第 1--26 个控制步。冻结运行入口内置的 `--audit` 因将策略的 `(1,13)` 请求与任务记录的 `(13,)` 单条动作作形状严格比较而拒绝等值动作；运行源码已由报告 SHA 绑定，故不修改它，另建独立审计入口只在比较时去掉 batch 维，不改原始 evidence 或运行报告。

| 32 步分支 | source 过力／8 | nominal 过力／8 | 合计过力／16 | 合计未观察到过力／16 |
|---|---:|---:|---:|---:|
| 首次零脉冲，随后 task-only | 1 | 4 | 5 | 11 |
| 首次 0.25 脉冲，随后 task-only | 1 | 2 | 3 | 13 |
| 首次 0.5 脉冲，随后 task-only | 3 | 5 | 8 | 8 |
| 首次完整脉冲，随后 task-only | 2 | 3 | 5 | 11 |
| 持续完整 Ours | 8 | 8 | 16 | 0 |

相对零脉冲，完整单脉冲有两条新增过力、两条避免过力；持续 Ours 则在零脉冲未过力的 11 条请求上全部产生过力。0.25 的 3/16 不是单调剂量关系或最优增益证据，样本量和短时域都不足以冻结新策略。当前更有力的机制信号是**后续闭环干预**而非首个动作的瞬时风险：单次脉冲的 32 步过力总数与零脉冲相同，持续 Ours 为 16/16。两种分支的后续控制器和状态演化不同，因此还要量化后续接纳次数、首次分歧时刻、累积子步冲击和进展，并在新请求上做 episode 级的预声明策略比较。未来风险门必须评价滚动闭环 continuation；不能用首次受力阈值或这 16 条的事后最优混合比例作为安全证书。该诊断只修正理论上的归因对象，不放行 T3 方法、不填四任务主表，也不改变已冻结的失败分母。

### 2026-10-31 T3 接纳次数与完整 episode 结果

为检查上述 32 步风险是否对应最终任务结果，任务本地 `configs/arm/keyed_insertion_recovery/validation/admission_budget_episode.json`（SHA `5b107dad0b9e96b5e16670226326c7d617815859e5c088e781d2e53ec6bf1505`）固定了另 16 个全新请求：source／nominal × mode 0／1，每格两个物理槽位、各重复两次。从首次接纳的完整状态分出 task-only、最多一次接纳、最多两次接纳、持续完整 Ours 四条分支，运行到原任务终止或 750 控制步上限；若整段均无接纳，则只保留共同物理轨迹并在四方法分母中作同一结果，不伪造四份重复 evidence。

GPU1 开发运行 `results/arm/keyed_insertion_recovery/validation/admission_budget_episode_3301900000/report.json` SHA `b05ca488b3b7dcaab7e05f9778688f8ff0a7b44ac944e7585e24cfbe7d289ec1`；16/16 完成、7 个有效同状态四分支、9 个全 episode 无接纳，合计 37 条独立物理 evidence、10,312 个收费控制步。独立审计 `independent_audit.json` SHA `5077ecb005e48147c55754642e151aa8388e6951401fdc2c943d123e814d9b2c` 逐动作复核一次／两次上限、共同前缀、首次候选的相同物理转移、所有判据与资源账，重复运行字节一致。有效分叉中，task-only／一次／两次／完整 Ours 的受约束成功分别为 6/7、5/7、3/7、1/7；全 16 请求纳入未接纳组后分别为 11/16、10/16、8/16、6/16，全部其余结果均为真实过力。相对 task-only，一次接纳新增 1 条过力，两次新增 3 条，完整 Ours 新增 5 条，均无任何反向的过力转成功。

| 机制和 mode（各 4 请求） | task-only 成功 | 最多一次成功 | 最多两次成功 | 完整 Ours 成功 | 有首次接纳 |
|---|---:|---:|---:|---:|---:|
| source, mode 0 | 4 | 3 | 2 | 0 | 4 |
| source, mode 1 | 2 | 2 | 2 | 2 | 0 |
| nominal, mode 0 | 2 | 2 | 1 | 1 | 3 |
| nominal, mode 1 | 3 | 3 | 3 | 3 | 0 |

这是单一 learned 实例的开发诊断，不能作为论文正式主表或跨实例效果。个体风险也并非按接纳次数逐请求单调：source mode-0 有一条一次接纳过力、两次接纳却成功。共同成功的五条一次接纳分支平均仅比 task-only 快 2.6 控制步，不能抵消新增过力。mode 1 没有 transport 接纳，四方法得到同一物理结果；它不构成该 mode 的 transport 收益。当前 T3 v15 实现即使将接纳限为一次，仍没有正的受约束任务收益。下一版应先验证公共候选在同状态、同执行器下是否存在兼顾正进展与低子步力的**新候选集合**，并用按 episode／机制隔离的训练和新请求检验滚动闭环策略；不能直接把这次 1/2 次上限选作论文方法或用调门槛掩盖失败。

### 2026-10-31 ASID T2 接触窗信息量预检

T1 的 60 控制步探索不能直接移到 T2：当前 T2 任务的已有审计轨迹首次超过 1 N 接触力约在第 235--263 步。任务本地 `configs/arm/guided_contact_alignment/validation/asid_public_information.json` SHA `6dd3733349c3944bfccab1d07cecc7e2f8ec1182c6efa4ec37bea3f395b21617` 固定 320 步观察窗、source 机制、两种 mode、三个原已测试的公共策略（anchor、conservative、aggressive）。每格先执行一次源策略，再对同一请求动作流分别以增益 ±0.1、负载 ±0.1 重放，使用作者 ASID 的中央差分信息目标；延迟为离散参数，本预检不对它做伪连续微分。无目标交互、训练或主表评分，最多 30 条物理 episode、9600 控制步，GPU1 与 T2 AdaptSim 并发。

运行 `results/arm/guided_contact_alignment/baseline/asid/diagnostics/public_information_3302000000/report.json` SHA `20e2c391f2cf1d622c4712aa367d87f28a94382cd487fe41abb0f7a022fa68fc`：30/30 轨迹执行到 320 步、零约束违规，6/6 单元的两参数雅可比均数值满秩。独立 CPU 审计 `independent_audit.json` SHA `7814c5ee3e7ebe33624ffef52e7698cdfd3d99d18b46b4bf64304617912b5b24` 重审全部物理结果、源动作的四次精确同动作重放、22 份运行源码快照和收费账，并从物理 TCP 轨迹独立重算**位置**部分的雅可比列范数；审计重复执行字节一致。完整 wrench 向量只在运行时公开观测中读取，未逐 tick 写入物理 trace，因此完整信息分数只核对与位置／wrench 列范数及奇异值的代数一致性，不能称为全量独立重算。

| 公共策略 | mode 0 信息分数／最小奇异值 | mode 1 信息分数／最小奇异值 | mode 0 首次 >1 N 接触 |
|---|---:|---:|---:|
| conservative | 269,311／145.6 | 103,753／131.2 | 第 240 步 |
| anchor | 693,002／234.0 | 107,596／142.1 | 第 237 步 |
| aggressive | 977,191／260.8 | 110,183／149.0 | 第 230 步 |

这些是一个源 seed、无噪声物理重复上的公共响应信息，不是辨识精度或目标任务性能。未经归一化的三个 mode-0 分数都超过 ASID 后端默认 1000 的 reward clip，若直接用于探索搜索将全部饱和；事先固定 normalization=100 可把其目标值变为约 26.9、69.3、97.7，并保持排序。mode 1 策略间增量很小，不能只凭 mode-0 最优策略声称两种 mode 均有强探索优势。完整 T2 ASID 下一步须冻结“安全公共探索 → 同动作目标／仿真拟合（增益、负载及 delay=0..3 枚举）→ 更新后仿真中重训公共任务策略 → 新初态名义验证”的资源和失败账本；本预检不能替代任何后续阶段。

### 2026-10-31 T2 AdaptSim 完整开发链与新鲜公共反馈比较

T2 任务本地 AdaptSim population 训练 `results/arm/guided_contact_alignment/baseline/adaptsim/training/guide_population_development_8764501/report.json` SHA `cc3b2e5b39bafe756426f9248f69f346ecfd03661df514c1ceaf81eb10be737d` 已完成。独立账本与源码快照审计 `independent_audit.json` SHA `dcb9fd9a6fe746f864ff25b1ca225b9f83d2ea4f34e3b2108dd7246e4ac72c76` 核对 21 次策略训练调用、210 个 CEM candidate、16 次仿真目标 rollout、5 次部署目标 rollout，总计 882 个物理 episode，开发样本内 773 成功、99 timeout、10 overforce。该审计没有逐条物理 trace 重新评分，训练内部结果不作为性能表数据。

训练输出作为唯一新实例，在与公共几何参照相同的 16 条新鲜请求上运行，`results/arm/guided_contact_alignment/baseline/adaptsim/validation/fresh_actor_pilot_3300900000/report.json` SHA `2c4edb9b1bf9027a43911e715e57fba64f843bfbe7af67e2cbed859ad6589524`；16/16 物理 evidence 独立审计通过。AdaptSim 受约束成功 15/16、一次名义场景 overforce；公共几何 16/16。配对 CPU 审计 `paired_development_audit.json` SHA `82ee7af5d8a8641f1d0e3d7a3a549c5c932b65d6ae4248f6241ee54c3cf62bd3` 核对相同初始物理哈希和每个场景格：source 4/4 对 4/4、nominal 3/4 对 4/4、latency-gain shift 4/4 对 4/4、payload shift 4/4 对 4/4。AdaptSim 平均截断完成时间 0.9361，公共几何 0.9261；该小规模开发批未显示 AdaptSim 的成功率优势，但给出了完整学习链和可执行 actor，后续可用冻结训练规则及独立实例进入主表资格筛选。

### 2026-10-31 T2 ASID 完整链已冻结并启动

根据上面的信息预检，任务本地 `configs/arm/guided_contact_alignment/training/asid_full_chain_development.json` SHA `a1c838079881c31efb3e33c756f0be9ad72729cf881bd85a88ab619d3b606ae3` 冻结 23 份源码 SHA、一个目标探索 episode、两个目标部署 episode、最多 232 个仿真 episode／103,760 个总物理控制步／4,800 秒。训练链使用作者 ASID 的中央差分 Fisher 目标，normalization=100；增益／负载连续辨识和 delay=0..3 离散枚举采用同一目标探索动作序列，随后在辨识的仿真分布内以任务本地公共反馈策略图做 CEM 搜索。由于作者发布代码没有可直接用于 T2 的下游 SAC 训练，这里的 task-policy CEM 是明确的任务适配，不能被描述为作者原论文的原样复现。

零物理 preflight 通过后已在物理 GPU1 启动，结果目录 `results/arm/guided_contact_alignment/baseline/asid/training/full_chain_development_8764601/`；运行中以 `progress.json` 为准，任何失败保留原始报告、不替换请求。完成后 CPU watcher 会检查全部有证据的探索、辨识和部署 episode，独立复算物理结果、同动作重放、位置 Fisher 下界、目标位置误差下界、CEM 选择和收费账；其余 40 个策略训练仿真 episode 仅能核对摘要账本，完整 wrench 时间序列也未写入物理 trace，因此不作全量独立信息分数重算。审计通过后还须绑定产出的 checkpoint，在全新请求上与公共反馈及其他强 baseline 配对；当前 ASID 没有任务性能结论，四任务主表仍无可填的正式独立单元。

ASID 训练结果尚未揭晓时，已冻结不重叠的 16 请求 `configs/arm/guided_contact_alignment/validation/asid_guide_fresh_actor_roster.json` SHA `ad3c165350dfbe4033649987f053a0293dc38267e60653bac8f1f731712abfcf`，包括 source、nominal、latency-gain shift、payload shift 各 4 次，两种 mode 和两种初始物理槽位；与训练目标探索／部署 seed 及前一份 AdaptSim pilot 均不重叠。公共几何先行执行 `results/arm/guided_contact_alignment/baseline/public_geometry/validation/asid_paired_pilot_3302200000/report.json` SHA `503994ebd9cf394cbd1452d95bc6b5b8bdddaeb207412042b25015a26c52e078`，16/16 物理 evidence 的独立审计 SHA `9625287d147fd1c685c02efb723e869b363a5a8b2b0da88901a80ce6c3434862`，受约束成功 15/16；唯一失败是 nominal、slot 0、mode 0 的真实过力。ASID 完整链若通过独立审计，已有 CPU watcher 会生成任务本地 actor 绑定、进行零物理 preflight，再在这份相同请求上运行 ASID 和配对审计；若训练或审计失败，则保留失败，不继续目标 pilot。

ASID 完整链随后完成，`report.json` SHA `5aa9737b86d2c5461efc6450d394bd2b01e16e532c4ac9b736d38735b6ed4848`：232 个仿真 episode／96,741 控制步、3 个目标 episode／2,147 控制步、1,596 秒；双 mode 探索 CEM 的最优公开策略为 `[0.02308, 0.00782, 0.03345]`、平均 Fisher 分数 44.76，离散网格加局部搜索拟合出目标参数 `[0.8, 0.6, delay=1]`、公开响应 MSE 为零。任务策略 CEM 的最优公开策略 `[0.02718, 0.00876, 0.03046]`，同一训练目标机制的新部署 mode 0／1 各一次成功，分别耗时 873／954 控制步。CPU 独立审计 `independent_audit.json` SHA `8881f29b478fd5d4ce87fcf623e177fb809267c7e776bc871c282a2bcf346401` 核对 23 份源码快照、195 条独立物理 evidence、40 条仅摘要账本的任务训练仿真、源／扰动同动作重放、目标／辨识同动作重放及选择／费用账本，结果 `passed_development_only`。无噪声物理轨迹中，准确网格点与目标 TCP 的 320 步序列逐值相等；相邻 gain／mass 网格点 TCP RMS 差约 1.77--2.62 mm，相邻 delay 只有约 0.09--0.11 mm。因此 delay 精确拟合是当前无噪声仿真的结果，不能直接解释为有传感噪声时的稳健辨识。经审计绑定的任务配置 `configs/arm/guided_contact_alignment/baseline/asid.yaml` SHA `88dfaf6b3bb75650b04fff5b0170035cbde96bf9349fc896810035ecc3edad7e` 已通过零物理 pilot preflight。

新鲜 16 请求 pilot `results/arm/guided_contact_alignment/baseline/asid/validation/fresh_actor_pilot_3302200000/report.json` SHA `cbdf121dd60c549a244ec393a1a1e5d8ffc691448023ef806e7c3d817403597c` 完成且无执行错误；全部物理证据审计 `independent_audit.json` SHA `7e4a5d8a8da1a709ccb1f1f005ef95f2469c4d78a2a2d7f14d5c2efbf366723e` 通过。ASID 和同初态公共几何各 15/16 受约束成功、各一次 overforce；配对审计 `paired_development_audit.json` SHA `203c19bd9ba70cf27ad07c99e5c213a2f5ad78b318186eaec1d1a5c77c255483` 核对了 reset 哈希和 16 对结果。ASID 救回公共几何在 nominal／slot 0／mode 0 的过力，却在 nominal／slot 1／mode 0 新增过力；其余 14 对均成功。平均截断完成时间 ASID 0.93606、公共几何 0.94006，两者当前没有受约束成功率差。这验证了 ASID 从信息探索到辨识、任务重训、checkpoint actor 的闭环执行能力，但不能宣称任务性能优于公共反馈，更不能把一个训练实例和 16 个开发请求当作独立主表。

CPU-only 四任务主表资格库存刷新为 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v3/report.json` SHA `3cc5665edb98d0f41c71f45a7d6f771f09b191249493c21e97fbb8619128f547`：44 个必需单元中，28 个旧 actor 可解析但待当前合同资格审查、5 个新增 baseline actor 只有开发证据、3 个没有任务 actor、8 个 ACP／Human2Sim2Robot 仍未执行。所有性能栏保持空值、正式独立主表合格单元仍为 0；T2 的 ASID／AdaptSim actor 完备不等于已取得论文级结果。

### 2026-10-31 T3 ASID 早期接触信息量探针

T3 现有完整 AdaptSim 在独立初态的共同首次过力主要出现于控制步 42--65，因此不能把 T2 的 320 步信息探索直接搬到 jammed 起点。任务本地 `configs/arm/keyed_insertion_recovery/validation/asid_preoverforce_information.json` SHA `663a2d6b1f6332afe6459aa56bb098c403fd35ecd52ddd5b09d4584182d89dff` 先冻结 32 步窗、与 T3 AdaptSim 相同的七维公共反馈策略图、anchor／guarded／fast 三种预声明策略、源双 mode、增益与负载各 ±0.1 同动作扰动；共最多 30 个物理 episode／960 控制步、GPU1、失败不替换。离散 delay 不做伪连续微分。本轮仅回答“越限前的公开状态是否含可辨识信息”，无目标交互、策略训练或任务表现评分。

GPU1 运行 `results/arm/keyed_insertion_recovery/baseline/asid/diagnostics/preoverforce_information_3302300000/report.json` SHA `ec7b96a467a7519b65eefcb937e2f81422c37602239eb85ec3dc7e297c9e4ced` 完成 30/30 条、960 控制步；六个源策略／mode 格与全部扰动都达到 32 步，无约束违规。源轨迹首次超过 1 N 的接触力位于第 12--14 步，故该窗确实覆盖早期接触而未混入已知 42--65 步的常见过力。六格增益／负载全量公开观测雅可比最小奇异值约 72--81，原始 Fisher 分数约 14,692--32,956；它们只反映该有限窗、固定源种子和无噪声仿真。CPU 独立审计 `independent_audit.json` SHA `d0a808b7f0338c266b114dadb8884f5f78acc96035b57050e40850a0ab80bff5` 逐条重评全部 30 条物理 evidence、核对 20 份源码快照、源与扰动的逐动作重放，并从物理 trace 独立重算**位置**雅可比列范数；运行时完整 wrench 时间序列没有保存，故全量 Fisher 只核对与奇异值的代数关系。首次审计的严格浮点容差在最大约 0.000135 的公开观测／物理 trace 列范数差上失败，随后仅把位置比较容差调到相应数值量级、保持冻结物理报告与判据不变，审计通过。此结果允许有界开展 T3 ASID 探索—辨识—任务策略链，但不能证明其能防止稍后的早期冲击或优于公共反馈；最终判断仍须新请求配对。

### 2026-10-31 T3 ASID 完整链与三方法同初态新请求

任务本地完整链配方 `configs/arm/keyed_insertion_recovery/training/asid_full_chain_development.json` SHA `05790e2cdfcfb24c71e9c0ad0891d5afdaa3952bdaab1ef2b475e786f89d8b5e` 冻结源探索、同动作增益／负载／delay 辨识、辨识后仿真中的公共反馈策略搜索及目标部署，GPU1 上限为 232 个仿真 episode、3 个目标 episode、2,400 秒，失败不替换。`results/arm/keyed_insertion_recovery/baseline/asid/training/full_chain_development_8764701/report.json` SHA `2b761ee26e1837f8c79ce4400d518c1c786925b1e62fe630ce28eacd0b632757` 完成 232 个仿真 episode／16,453 控制步和 3 个目标 episode／748 控制步，耗时约 196 秒；目标参数被无噪声公共观测拟合为 `[0.7, 0.5, delay=3]`。任务搜索最终选择了原有公共反馈 anchor `[0.25, 6, 2, 0.008, 0.004, 1, 0]`；同训练机制目标部署 mode 0 在第 51 步过力，mode 1 在第 665 步成功。CPU 独立审计 `independent_audit.json` SHA `20b05913a5554426ba4fdc1e437242038e7714f8f5f41066df50ff3e8a445cc5` 验证 21 份源码快照、195 条物理 evidence、40 条仅摘要账本的任务训练仿真及探索／辨识同动作重放，状态为 `passed_development_only`；完整 wrench 序列和这些 40 条仿真训练轨迹没有被逐条独立重评。已审计 checkpoint SHA `382dd9d6fa96a29772a7ca7ccd3b2cf6ffed3ce47b4b208fbcf4cedf6939ea4b` 绑定至任务本地 `baseline/asid.yaml`；公共 actor 的逐动作重放回归测试通过。

训练结果出现前冻结的 16 条全新 T3 请求涵盖 source、nominal、latency-gain shift、payload shift，各四条、双 mode／双物理槽位；三方法使用同一 reset 物理哈希，所有 48 条物理 evidence 均通过各自 CPU 独立审计，且无 runner error。ASID 的 `fresh_actor_pilot_3302500000/report.json` SHA `cdc018ec902b2dcc7487abb146761254fd49e94b628d9e4493fe4cb488e74429`，独立审计 SHA `bd0a91a1b93582c644f582c944d345a534839ee21d9e0f56a8c867737c7093d8`；AdaptSim 同名配对 pilot 的报告／审计 SHA 分别为 `8c8b712604bd920ba7c07c2492f523c2fa5b656763e920ee354617075c4a14b6`／`9fd3e37fc88f26096fc27f53549e887abc7bf9c0c28f7d17315aecf4ae781436`；公共几何为 `2745dbbba7ba68c4326995d16677abf135bd1d30729019763393e81a3f3b691a`／`c449143eab492ed5e6a731b8545c4435f2d4bfff8ad053f891431414a3fdfa42`。受约束成功 ASID 10/16、AdaptSim 8/16、公共几何 10/16；各自过力 6、8、6 次。ASID 对 AdaptSim 的六个不一致请求中救回四个、损害两个，平均截断完成时间 0.826 对 0.8208。ASID 与公共几何的 16 条 `requested_actions` 逐值完全相同，结果和完成时间也完全相同；逐请求动作摘要 `action_identity_audit.json` SHA `c962f1cb634584f9824350c370f7962841713809c8ef44c4cd548df4f16241e7` 绑定两份报告和全部 32 条 evidence。ASID 的参数辨识和完整训练链在这份 checkpoint 上**没有产生额外的任务策略收益**，不能把 10 对 8 解释成在线辨识的贡献。两个保留原始请求级结果的配对审计 `paired_adaptsim_development_audit.json` SHA `6569f97a3ce2502bda81347f09ae5b2d0c6acc4f26e5d762b7deda0ccf9d3fa4` 与 `paired_public_geometry_development_audit.json` SHA `f0a02fbe39ac017b88e03d8ce2c5cf78c500c7a8094a36d4a7391d0a9460e942` 均为 `passed_development_only`。这仅是一个训练实例的 16 条开发请求，不能代替独立主表。

四任务主表资格库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v4/report.json` SHA `a99cf8bfb148a6604d1b74146f0c079206f6a9e9104f396454deb48e96359131`：44 个必需单元中 28 个旧 actor 待合同资格审查、6 个新增 actor 仅有开发证据、2 个没有任务 actor、8 个 ACP／Human2Sim2Robot 未执行；正式独立合格单元仍是 0。下一步需要 T4 的 ASID／AdaptSim 任务适配及 ACP／Human2Sim2Robot 的可执行链，然后冻结所有方法共同的独立请求与训练实例、进行按实例配对统计。T3 的六次共同或交叉过力也提示理论必须明确区分可辨识性、辨识精度、后续策略实际变化及受约束收益；当前数据只支持前两项和链路执行，不支持最后一项。

T4 AdaptSim 先前的“共同 joint-limit 归因清楚前不训练”应改为训练设计约束，而非无限等待的准入门：同机制／mode 下两条预声明反馈策略共同在第 115 步触及 `z1_joint4`，说明这格存在公共执行瓶颈，但不证明所有任务策略都失败，也不构成删去该格的理由。后续任务本地元训练须冻结移动底盘 body-intent、joint-margin executor、支撑与力判据，明确将该格纳入训练／独立测试的完整分母；若学得策略仍失败，保留为真实 baseline 结果。ASID 的 T4 探索亦须先在不改变共享执行合同下检查接触前信息量，再完整记账，而不能把 T3 checkpoint 直接改名为移动形态训练实例。

### 2026-10-31 T4 ASID 完整链及新初态配对

T4 接触前 32 步信息量探针在源机制完成 42/42 条安全物理轨迹；公开 TCP-only 雅可比的最小奇异值约 1.85--2.61，说明在这个无噪声有限窗里有可辨识信号，不能推出下游任务收益。原始 T4 ASID 训练 `full_chain_development_8764801` 在源探索及一次目标探索之后，因某个参数网格候选的 Go2+Z1 初始 IK 不可达而 `failed_retained`。随后冻结的 64-seed 双机制／双 mode reset 预检通过，v2 配方 SHA `614ff0d4edd9177ab477b96557517c1eb381de228529a7f93d434673c5a372f1` 将候选 reset 失败记为零控制、显式惩罚，保留全部候选，未替换请求。GPU1 上 `full_chain_development_8764802/report.json` SHA `f9a016863b15b7918da7508073a7d4718dc9b06b71868efe9165cc0a30cd463c` 完成 308 个仿真 episode、3 个目标 episode，目标短探针辨识为声明的 `[servo_gain=0.7, payload_mass=0.5, body_servo_gain=0.9, delay_steps=3]`；两个目标部署均成功。首次 CPU 独立审计发现的是分母公式重复扣除 40 条策略搜索账本内的 10 次 `reset_unreachable`，并非物理证据缺失：311 条总账、40 条仅账本任务训练、41 条非任务训练 reset 失败、230 条物理 evidence。原训练源码快照 SHA `bac9dbe6d23058055424f7a7c329d7a76d1fa245db67803f132a1a972cb15833` 与运行时一致；仅在训练结束后修正审计分母，未改训练报告或 checkpoint。重审 `independent_audit.json` SHA `592f904ef586a8aa1452f94bb2401e34342e428338fcfbb21948ee97a7a810d9` 为 `passed_development_only`，并将 checkpoint SHA `f879937607da882b2fe863bb0dded85c13229fb1f7a4e53bbf8e144e69d1a995` 接到任务本地 ASID actor。

三方法训练结果出现前冻结 T4 全新 16 请求 roster SHA `a36bf14e347082aa375216c9487da94ce6123c301e33b8c71dba93002a9c07c2`，跨 source、nominal、latency-gain shift、payload shift，每格四条、双 mode／双物理槽位，reset 预检 16/16 通过。公共几何和 ASID 在各自独立审计通过的完整执行中均为受约束成功 12/16；四条 payload-shift 请求均 `joint_limit`，其余三格各 4/4 成功。ASID 报告／审计 SHA 分别为 `8e9e42f6d88e28fce73d5fd31b27a134477e1c650d307d3b3b4de8f75d6e1a10`／`c6d363c4febe8d3c4ab55056d410bacfbd02531909b464f6c76e2540771b96fe`；公共几何报告／审计 SHA 分别为 `516452ba8aee0d71e2411253f6e58b1f392fb59ebd4482e21cd84bea3822f6a9`／`02bbf153aae4934c452287c716ca90dd12d7441c2f8b605bf9cf5bbcbdb3c487`；配对审计 SHA 为 `e1d374a0d66d11bcb1acccf92fe0e5865a00e277437aed3490ef101359510966`。ASID 在 16/16 条请求的第一个控制步就产生不同于公共几何的动作，但成功／失败结果完全相同，平均截断完成时间 0.82625 对 0.81783；逐请求动作证据 `action_difference_audit.json` SHA `b2e14d799129a8541629926a42ec90341b6a9555679630b41eea676b0082817f`。因此本实例展示了辨识后策略**确实改变**，却未在该开发集带来受约束成功增益；不能把系统辨识成功直接写成方法优越性。AdaptSim T4 人口训练仍按冻结预算运行，三方结论待它完成。

CPU-only 主表库存 v5 把四任务各七个旧可执行方法的 source/nominal 开发审计接到当前 actor 配置：逐方法核对 28 份八请求报告 SHA、task config SHA、manifest SHA、方法身份、证据审计状态和零执行错误。`results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v5/report.json` SHA `424f09b1ecc9702e9712b7c64723effc3c85660937ba95ec7d8a6f3eb11b034b` 当前有 28 个带审计行为证据但待完整资格的旧单元、7 个新方法仅单训练实例开发单元、1 个 T4 AdaptSim 待训练 actor、8 个 ACP／H2S2R 未执行单元；正式合格仍为 0/44，全部性能栏为空。七个旧方法在 T1--T4 的开发源／名义结果均已可追踪，但例如 FADA 四任务这些请求均 0/8，说明其当前适配链还不具备可称为强外部方法的名义能力；不能把这些零分偷渡入正式表，也不能据此宣称原论文方法机制失败。当前结果是库存而非独立评估协议冻结。

上述已审计的八请求开发屏幕按每任务四条 source／四条 nominal 汇总如下；每格是两个成功计数，不是正式目标分布的成功率：

| 当前任务 actor | T1 | T2 | T3 | T4 |
| --- | --- | --- | --- | --- |
| ours | 4/4, 4/4 | 4/4, 4/4 | 0/4, 2/4 | 4/4, 4/4 |
| task-only | 4/4, 4/4 | 4/4, 4/4 | 2/4, 3/4 | 4/4, 4/4 |
| FADA task-adapted | 0/4, 0/4 | 0/4, 0/4 | 0/4, 0/4 | 0/4, 0/4 |
| DIAL-MPC task-adapted | 4/4, 4/4 | 0/4, 0/4 | 1/4, 1/4 | 4/4, 4/4 |
| response MPC | 4/4, 4/4 | 4/4, 4/4 | 1/4, 1/4 | 4/4, 4/4 |
| public geometry | 4/4, 4/4 | 4/4, 3/4 | 3/4, 1/4 | 4/4, 4/4 |
| public source prior | 0/4, 0/4 | 0/4, 0/4 | 3/4, 1/4 | 4/4, 4/4 |

T3 在这批请求上 ours 的 source／nominal 均弱于 task-only；主表统计代码保持真实全分母、允许弱方法留下真实失败，但这项开发结果不能支持 transport 的任务收益，更不能仅凭静态 T2 的四格近天花板成功推出动态接触优势。

### 2026-10-31 正式主表方法名单调整

按新的论文比较口径，正式四任务主表采用九行：ours、task-only、FADA、DIAL-MPC、response MPC、public geometry、public source prior、AdaptSim、ASID。ACP 与 Human2Sim2Robot 转为表外方法诊断；原四任务 actor manifest、预部署拒绝和开发记录原样保留，不把未执行算作零分，也不据此推断原论文算法本身无效。旧版 44 单元库存继续作为历史协议证据，后续正式结果只按九方法、四任务的 36 单元合同统计。主表的任务适配实现仍需分别通过来源一致性、当前 actor 和独立物理请求审查；名单调整本身不等于 baseline 已获得论文级成绩。

更新后的 CPU-only 准入库存为 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v8/report.json`，SHA `e7ce9dd8633557bbc1dd858c5d261418a3027605e7d5b583ea690a1864e25dd3`。36 个活跃单元中，28 个旧 actor 有已审计的 source／nominal 开发行为、7 个 ASID／AdaptSim 单元只有单训练实例开发证据、T4 AdaptSim 人口训练运行中且尚未绑定任务 actor；正式独立主表合格仍为 0/36，所有性能栏为空。另列出的 8 个 ACP／Human2Sim2Robot 历史拒绝单元均为表外记录。当前存在的 35 个主表配置均通过零物理共享执行／任务合同继承检查，缺失的 T4 AdaptSim 配置待训练与独立审计后补验。统计合同现为 `four-task-independent-main-table-statistics-v3`：拒绝表外方法 outcome，将 `worst-group success` 按机制×请求 mode×身体／支撑组计算，要求各组请求数平衡，并拒绝 outcome 组标签与冻结请求不一致；准入和统计六项测试通过。v6／v7 库存保留为逐步增加方法忠实度和共享合同检查前的历史快照。

ASID 的 solver 是作者 Fisher 信息目标的独立实现，当前四任务链使用任务适配 CEM，并非作者 rod 专用 SAC；AdaptSim 保留作者 PyTorch branching-Q 核心，但长时域任务策略优化器和部分参数更新属于任务适配。两者进入主表名单只确定比较对象，不代表已通过方法忠实度资格；正式对比前必须公开这些偏离、确认在共同任务合同下是合理适配，再由独立训练实例和新请求测量。现有 `verify_shared_protocol.py` 的 v3 等价检查固定旧的每任务 13 份 leaf 配置和旧 manifest，已不覆盖新增直连 actor；v8 准入清单只补上现有配置的共享合同继承检查，完整的新版本架构等价审计仍待完成，不能把 v3 结果当作新主表证明。

### 2026-10-31 T4 AdaptSim 完整链与主表资格刷新

GPU1 上冻结的 T4 AdaptSim 人口训练完成，训练报告 SHA `5722b0e80a4c21ddbae0c616b10b9784143fc5cb47590deb59dadc207a2d82fe`，checkpoint SHA `c64b42e62d4ab986b12fe823c57064e8d68039f9c8e44f238853ba6983bccde5`。CPU 独立账本审计 SHA `bddbdc839290a67828a48c57af367f02c209410aa7f47ec442d5d4780c43e785` 为 `passed`，核对 16 份源码快照、21 次策略训练调用、882 次总仿真／目标 episode 记账；其中 310 次 reset 不可达作为真实失败保留，审计明确未做逐条物理轨迹重放。绑定的 T4 任务 actor 配置 SHA `0bde6cc1de06f979a28a99aa0f41c42bc302f5c440299e475f67273787d52921`。这验证了训练链与账本，不等于论文级性能。

在训练结果出现前冻结的共同 16 请求上，AdaptSim 报告／物理独立审计 SHA 分别为 `667dbc115a05807700c8724f5c30e077e94824b9c2ad4118761639c35135c0ac`／`df09b580b3c6d256daa56d42257a8437681e4aa19c9933a78ee727c23f652a7a`：受约束成功 12/16，4 次 `joint_limit`，零执行错误；与 ASID、公共几何的配对审计 SHA 分别为 `fe378cd1605147a1fc7fd70425d492af10f01d989b96628a4dd014c6f9a6d7a2` 和 `b184f07720efa5841febf983be04a5497e6e3642541bd4c40e5c68440c7e56bc`，均为 `passed_development_only`。三者各 12/16，四条 payload-shift 请求共同失败；本开发集没有 AdaptSim 相对两者的成功率增益。新 CPU-only 九方法库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v9/report.json` SHA `3836f88e1f87ad8c364897c64438063536a2684968bcd43cb3faf954b1b2f1ee` 显示全部 36 个配置均通过共享合同继承检查，28 个旧方法单元仍待资格、8 个 ASID／AdaptSim 单元仅有单实例开发证据，正式合格仍为 0/36。

独立训练重复还发现 T1 ASID 的旧训练 recipe 绑定的任务适配代码已经变化，不能把旧 checkpoint 与当前代码训练当作同一冻结实例 cohort。现有 T1 任务训练器加入显式 `independent_main_table_training_v1` 模式，在新配方中绑定 16 份源码并在运行时复制快照、自动写出可追溯 checkpoint；旧配方和旧结果不改。首份新配方 seed 8801001 在系统 Python 缺 MuJoCo 时于首条仿真构造前失败，报告保留 0 控制步、0 目标访问。经 NumPy 1.26／MuJoCo／Torch 的混合环境预检，新 seed 8801002 的配方通过零物理检查并在 GPU1 启动；其结果尚未揭晓，不能提前算作有效重复。

T1 seed 8801002 随后完成，recipe／report／checkpoint SHA 分别为 `e2bbc1498aaee0c5b9c049a505693fc3eb1afca7bf034d4189d6c164f483f436`／`1bdf1aa067c57529699259b26da3cccc0b981c153aae25657ae1ef05584fe5e7`／`d12683d215f51d90bbd307449f9b4d76b457b68b6a346200a9e1f7532b97d2e6`；347 次仿真、3 次目标访问，两个目标部署均成功。新独立审计 SHA `e6dce0d1cc10ae222477586f6bf1ff225103465c85748784c93aa41bbf76e096` 核对 16 份源码快照、训练分阶段账本和全部三次目标访问的独立物理重放；347 条仿真候选只做账本核对，原训练未保存全部候选轨迹。它是当前代码 cohort 的**一个**有效训练实例，不是八实例估计或正式目标性能。

T3 训练器也把原固定开发 seed 保留在旧分支，另以 `independent_main_table_training_v1` 绑定 8803001--8803008 的训练 seed 与互不重叠的目标 seed。第一份新配方 seed 8803001 的 recipe／report／checkpoint SHA 分别为 `3259c0d60389721f6bdf797a1190272649437167f7d5f182f9b5931b8c2d3ab7`／`a5a35f66d6f5ee2f3d48df6443594b6e98dd08ca647c8f5a8ce27bab2e78a8c9`／`742cc1015c57e7af79cb4e882e6cc15190515cbaacccf5e306f57210098c2b9d`，完成 232 次仿真、3 次目标访问；两个部署均为真实 `overforce`。独立审计 SHA `81345dc7deebe289067a78faff1cebcb7087b46313c854c09247ad75cfb0ca30` 为 `passed_development_only`，核对 21 份源码快照和 195 条物理 evidence，并将 40 条只有训练账本的仿真单列。该失败不能因训练实例身份有效而删去，也不能从两个开发部署直接推断正式分布效果。

T2 第一份新代码实例 seed 8802001 已按同一 cohort 规则在 GPU1 运行；一次性 CPU watcher 只在完整 checkpoint 出现后调用原 T2 独立审计，失败保留。T4 在新的 3404000000--3404000063 块完成源／目标机制×双 mode 的 256 次零控制 reset，64/64 种子安全，独立审计 `results/quadruped/mobile_keyed_insertion_recovery/baseline/asid/diagnostics/independent_training_reset_feasibility_3404000000/independent_audit.json` SHA `ab86fb6bccba185d8d08744dd087a9cf2601f96b30afb76ce68a292e5a794c1b` 为 `passed`。引用该审计的 T4 seed 8804001 训练报告／checkpoint／独立审计 SHA 分别为 `c62247f4dcc61c8c696cf476adc611c646f2f2850998edcc444792155baa71ad`／`6db4965944f3e2646137b6db11f39623fb43d753118bc971c261a3115d860a7f`／`79eda19bd4566780d4050276d2d395ab665aa83ee348ad09a0833a866b6d9770`；308 次仿真、3 次目标访问、两个部署成功，228 条有物理 evidence 的训练轨迹通过独立核验。其余仿真条目只做账本核对，不能称为完整物理重放，也不能把两个目标部署记入正式主表。

四任务共 32 份 ASID 独立训练配方已在各任务 `configs/.../training/` 冻结，均通过零物理训练 `plan`；训练与审计队列只使用 GPU1，按任务内 seed 顺序运行，单实例失败保留并停止该任务后续队列，不重试、不替换。统一启动收据 `results/_analysis/four_task_transfer/asid_independent_training_cohort_20261031/launch_receipt.json` SHA `9cf3556662e84a1d66785c57cf6fa0840ade6c38cef9e94bdb1994470666888b`，声明最多 8,952 次仿真、96 次训练目标 episode、0 次正式请求访问；T1 先前 seed 8801001 缺依赖失败另存，不属于这 32 份有效配方。收据是启动身份与上限，训练实例是否有效以每份独立审计为准。

### 2026-10-31 九方法独立主表请求冻结

任务本地 `protocol/independent_main_table_requests.json` 按 T1/T2/T3 各 40 条、T4 80 条冻结，共 200 个从未用于旧结果的 reset seed：四个机制组（源、名义、延迟／增益变化、负载变化）×双 mode×每组五个物理槽位，T4 另按两种身体支撑设置分层。生成入口 `scripts/paper/cross_ebdm/freeze_independent_main_table_requests.py` 的复核已通过，跨任务收据 `results/_analysis/four_task_transfer/independent_main_table_request_freeze_v1/receipt.json` SHA `7e199dcc4ce56bbe177af7e73215b274c3243b9a13a20ab19f15d51ed1b61f23` 绑定四份任务本地名单 SHA、九方法顺序、T1--T3 主分析与 T4 形态扩展；历史 seed 冲突为 0，正式策略 episode 访问为 0。若七条学习路线都通过资格并各有八个真实独立实例，另加两个单身份公共参照，完整主表上限是 11,600 条物理目标 episode。该名单只是冻结请求，尚无任何正式性能数。

主表算术合同更新为 `four-task-independent-main-table-statistics-v4`：七条学习路线每任务必须声明八个不同真实训练实例，两个公共确定性参照必须声明唯一 `fixed` 身份；单实例开发 pilot 不能形成正式单元。原有机制×mode×身体／支撑组的 worst-group 指标、缺失单元不填零和表外 ACP／Human2Sim2Robot 拒绝得分的规则保留。对应统计与准入测试 7/7 通过。后续先完成 ASID/AdaptSim 的训练身份、方法适配说明、当前任务源／名义资格与 actor 绑定；只有完整合格的九方法单元才能在冻结请求上执行和计分。

### 2026-10-31 独立训练续跑与 T4 ASID 复位故障

ASID 的 T1 和 T3 各八个当前代码独立训练实例已完成并通过各自 CPU 开发证据审计；T2 的首个实例通过审计，后续顺序队列仍在 GPU1 执行。这些审计只确认训练身份、证据与账本，不是独立主表性能。AdaptSim 的四任务共 32 份 `configs/<domain>/<task>/training/adaptsim_independent_training_*.json` 已冻结并通过零物理计划检查；启动收据 `results/_analysis/four_task_transfer/adaptsim_independent_training_cohort_20261031/launch_receipt.json` SHA `cf726a0678b333793e8f15b3a5c2751e12ad24cd5e1aaa84feb8c1594e16b3ec`，上限 26,544 次仿真和 336 次训练目标 episode，正式请求访问为零。T1、T4 已开始首实例；T2、T3 在对应 ASID 队列结束后开始。T4 AdaptSim 与 ASID 故障独立，已单独放行训练。T1 新增训练源码快照和账本审计入口 `scripts/paper/cross_ebdm/audit_adaptsim_t1_training.py`，首实例尚需实测审计结果。

八实例的**训练内目标部署**另显示任务差异：T1 ASID 在同一训练目标机制的两次部署每实例均成功（16/16）；T3 ASID 只有 4/16 成功，其余为真实过力或超时，且后者集中于 mode 1。这些目标访问已计入训练预算，不能视作新鲜请求上的泛化率，也不能用 T1 的训练内成功推断正式主表优势；T3 仍需把接触过力作为任务适配的关键失败类型保留。

T4 ASID v1 的 seed 8804003 在完成 259 次仿真、1 次目标短探索后，于后续**仿真**辨识候选的零控制步复位遭到浮动支撑姿态求解器拒绝；失败报告 SHA `3f2e609cb2180bceda50e597488ecbcf8024aa3c2be1806cfce9a76029fe8686`。这不是策略执行失败，原实现也没有把该类复位拒绝计入候选搜索账本，故 v1 队列按预先冻结规则停止，前两个成功实例和失败实例均原样保留，不能拼成八实例 cohort。T4 任务适配层现仅将两类已知的零控制仿真复位拒绝记为收费的 `reset_unreachable`，目标复位仍会停止并保留失败；针对分类和 v2 新鲜安全 seed 的回归测试通过。新八实例 v2 配方 seed 8804101--8804108 绑定修改后的源码，独立启动收据 `results/_analysis/four_task_transfer/asid_t4_independent_training_continuation_v2/launch_receipt.json` SHA `0d5bcf92ec4fc9e1400c602d5cf641d301eaf8a7f89034bfa3f645242f4e5fba`，声明最多 2,464 次仿真、24 次训练目标 episode、零正式请求访问，引用并保留 v1 失败报告；GPU1 已启动首实例，单实例失败仍立即停止后续队列。

CPU-only 资格库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v11/report.json` SHA `17d2f35d768d84c138780843abf1c2bf89c0d637490422b9495c8ea8e3a332b1` 将 T4 ASID 活跃 cohort 指向 v2，同时列出 v1 失败 provenance。36 个主表单元正式合格数仍为零；28 个旧 actor 仅有开发行为审计，8 个 ASID／AdaptSim actor 仅有开发训练绑定。45 项相关协议与方法回归测试通过。下一步需完成全部训练审计、方法忠实度与当前合同下的 actor 资格，并建立正式请求的逐方法逐实例执行和独立证据统计入口；冻结的 200 个主表请求仍未访问。

新的论文入口 `scripts/paper/cross_ebdm/qualify_independent_baseline_actors.py` 只读取任务本地训练配方／报告／审计与现有源／名义开发请求，逐实例在双 mode 构造直接绑定 actor，检查源码／checkpoint 指针与零物理调用，不执行目标策略。当前 T1 ASID 的八实例资格报告 `results/arm/obstacle_transfer/baseline/asid/validation/independent_actor_package_v1/report.json` SHA `5fc4c1a61e2c5a04d43831cea18b49771a930e526a187ca616e81c830f271fc8`，T3 对应报告 SHA `2664dfd86086c9b4c6d3b0d6f091e554d9dfbef6ec6ee024f2cc1b5a1281b3da`，均为 8/8 actor 包构造通过、0 物理调用、0 正式 seed 访问。该检查证明 checkpoint 可加载和实例身份可解析，仍需源／名义物理行为与方法忠实度审查后才能放行主表。

准入审计 v5 现逐项核对八实例 actor 包的训练 seed、配方 SHA 和四类工件指针；CPU-only `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v12/report.json` SHA `c3af5e7b8a7f0bbad0905823dcb845e824500db00bbf5c34ae918d8234edd24b` 已记录 T1/T3 ASID 包通过。其余训练任务完成后，后台 CPU watcher 只调用这个任务本地资格入口，失败保留且不自动访问正式请求；最终主表合格仍为 0/36。45 项相关测试再次通过。

T1/T3 ASID 另用同一个 `evaluate_transfer_method.py` 入口增加了八实例直接 actor 绑定：每次评估先核对任务本地 `independent_actor_package_v1/report.json` 的 SHA 与对应独立 checkpoint、训练配方、报告和审计，旧开发绑定逻辑保持原有校验。原七方法的源／名义名单只准许 56 次执行，故为 ASID 新建任务本地 `asid_independent_actor_source_nominal_roster.json`，逐字沿用已有八个开发请求、写明 `instance_count=8` 和每任务最多 64 条目标物理 episode；T1/T3 名单 SHA 分别为 `6cbeaa9806bdd0e5e07de170b7d5596811f003b76e636432ff7326b82496709f`、`678d5456335be5f3bc1e1526ed09ba859580ec5bae4b551ec504740f41265ec4`。16 份任务本地评估配方均通过零物理预检，GPU1 顺序队列已启动，跨任务启动收据 `results/_analysis/four_task_transfer/asid_independent_actor_source_nominal_qualification_20261031/launch_receipt.json` SHA `175cd69bd0cb3e1d0f2b043c49576c3f46ed2e70829ddfa1f3eab97dbc66e40e`，总上限 128 条，遇首次执行／证据审计失败即保留并停止。源／名义成功率不作为排除 baseline 的资格门；零执行错误和独立物理证据才是此步检查目标。正式主表 seed 访问仍为零。

T1 ASID 八实例的 64/64 条源／名义开发请求已执行并分别通过物理 evidence CPU 审计；跨实例汇总 `results/arm/obstacle_transfer/baseline/asid/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `fb80b0a0506b67da2e14335bb0e9dc9718f617664adc1df7c7229ad458745e50` 再核对 16 份运行报告／审计、八个实例身份和 64 个与旧公共几何参照逐请求一致的初态物理哈希。结果 64/64 受约束成功、零执行错误；这是重复使用**开发名单**的源／名义条件，不是四机制独立主表，也不说明 ASID 优于公共几何或 Ours。T4 ASID v2 首实例另完成 308 次仿真、3 次训练目标访问及审计，44 次零控制仿真复位拒绝均按预算记账，两次训练内部署成功；v2 后续实例仍在 GPU1 运行。

CPU-only 主表资格库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v14/report.json` SHA `9cff372b0aac687e7fe95e49edd87764e0f7d8881f3acaba9fdeefbb84f1fd87` 已逐项绑定 T1 ASID 八实例源／名义汇总及 T4 v1→v2 训练 provenance。36 个正式主表单元仍无一放行；T3 ASID 八实例源／名义队列尚在执行，不能用首实例结果替代汇总。独立 actor 绑定、准入与主表统计的相关测试 21/21 通过。

T3 ASID 八实例源／名义队列随后完成，64/64 条物理 evidence 与旧公共几何参照的对应初态哈希逐项一致、零执行错误。任务本地汇总审计 `results/arm/keyed_insertion_recovery/baseline/asid/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `c07974be502c70bf38f6ca573f71428487bc4561fe10fdc66c7870fad7025442`：40 次受约束成功、24 次真实过力；source 条件 mode 0/1 分别 11/16、14/16 成功，nominal 分别 7/16、8/16。现有 Ours 旧单实例同八个开发初态只有 2/8 成功，因此 ASID 是 T3 上不可忽视的强对照；训练重复数不同、开发条件仅两种机制，不能据此宣称正式分布上 ASID 已优于 Ours，也不能再预设 Ours 在 T3 会领先。准入库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v16/report.json` SHA `3e42f03ac3f1d5aa0e958ae9eb65d2cb20454cbe1049f91f4009ccfcc10a8be8` 现在明确区分：28 个旧 actor 待资格、T1/T3 ASID 两格完成八实例源／名义开发检查、其余六个新 baseline 格仍只有旧单实例开发绑定；正式独立主表放行仍为 0/36。相关 21 项测试通过。

ASID 与 AdaptSim 的源码核验使用本地作者仓库固定提交 `cf2cb670fb983f67cf2af156d4e17b4a690787e2` 与 `9cf80fb4268749fddf21c39eb861498bc7b052c3`，工作树无修改；38 项方法测试通过。数值对齐的范围是 ASID 有限差分 Jacobian／Fisher reward、AdaptSim 参数动作映射和 branching-Q 前向输出，另有交互记账与 actor 行为测试。四任务长时域任务策略使用任务本地 CEM，AdaptSim 离散 delay 与 Go2+Z1 身体意图也是显式扩展；这些不是作者原任务训练器的逐行复现。论文和主表方法名必须标为**task-adapted ASID/AdaptSim**，不能仅凭核心数值对齐宣称完整作者原算法复现。全部八份 ASID/AdaptSim 的源／名义名单均已在各任务 `validation/` 冻结并验证与正式主表 reset seed 不重叠，余下六个方法／任务单元在训练审计完成后由后台队列先冻结八实例评估配方和收费收据，再运行最多每单元 64 条开发 episode，失败保留，不自动开启正式主表。

可复现的任务本地配方冻结入口为 `scripts/paper/cross_ebdm/freeze_independent_baseline_source_nominal.py`：只有八个独立训练实例的 actor 包资格已通过，才会绑定其 checkpoint、任务配置及每任务 64 条上限的源／名义名单；T1/T3 ASID 原有 16 份配方用它复核完全一致。后续六格的后台调度器只在该资格出现后调用同一 `evaluate_transfer_method.py` 物理入口和 `audit_independent_actor_source_nominal.py` CPU 汇总，单格首次执行或审计失败即保留并停止该格。两条后台 watcher 均已确认进程存活。这里的自动推进只针对开发资格，不包含独立主表请求或分数。

### 2026-10-31 独立主表执行边界与旧 actor 身份复核

正式表计分合同现为 `four-task-independent-main-table-statistics-v5`。只有带物理 evidence、独立复算通过、`target_physical_execution=1` 的执行可以进入任务指标；任何 `retained_runner_failure` 均保留在运行报告，但不可以被编码成算法失败或占据成功率分母。同一任务与请求跨方法、跨实例的物理复位哈希必须一致。新 `genedynamics/tasks/transfer/main_table_evaluation.py` 沿用任务级 `_episode` 执行和 evidence 审计；正式 episode 在复位后、首次动作前原子写入请求／实例／物理初态收据。论文入口 `scripts/paper/cross_ebdm/evaluate_independent_main_table.py` 提供零物理 `--plan-only`、GPU1 `--run` 和 CPU `--audit`。正式配方必须指向冻结请求、当前任务配置、八实例 actor 包或旧 manifest、任务本地单元准入文件、源／名义物理资格和方法忠实度审查；目前尚无单元准入文件，因此新入口不会打开正式请求。25 项主表与现有开发 evaluator 定向测试通过。

CPU-only 准入库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v17/report.json` SHA `3de2467fa2068a5032ebada2c7da6028fa90dfce83464634a1a97ff8b48d1667`：28 个旧方法单元仍为工件可解析但资格待审，T1/T3 ASID 两格完成八实例源／名义开发资格，其余六个新 baseline 单元仍待训练；36 个正式单元全部未放行，正式 seed 访问为 0。旧任务 manifest 的五种学习路线在四任务中每个相关 checkpoint／响应模型集合均解析到八个不同 SHA；这只证明字节身份不同，不能替代训练 seed／数据来源、方法忠实度与八实例物理开发资格。尤其 FADA、DIAL 与 response MPC 的若干实例位于历史开发归档，不能仅因当前 actor 可加载就把旧无效确认结果重新纳入论文主表。

两个确定性公共参照在 T1--T4 各有现行任务配置绑定的 8/8 条源／名义物理 evidence。CPU 入口 `scripts/paper/cross_ebdm/qualify_main_table_fixed_references.py` 对完整 manifest、方法身份、任务配置 SHA、原报告和独立审计逐项核对后，写出八份任务本地 fidelity、cell admission 与 `configs/<domain>/<task>/confirmation/independent_main_v1/` 正式配方。新库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v18/report.json` SHA `6f5fbe6c152b11190e97bc891f6827e44e2ac2f7b5d207bbf7bc336afe219569` 显示 8/36 个单元现在具备正式执行资格，另 28 个仍待审；这不是八个正式性能结果。T1 公共几何配方的 `--plan-only` 已通过，仍为 `planned_unexecuted`，零正式策略 episode。

旧五条学习路线的四任务八实例源／名义补充资格由一个任务本地 cohort 配方描述每格，沿用各任务现有八个开发请求及首实例的已审计物理证据，仅增加 instance_02--08；每格最多新增 56 条，共 20 格最多 1,120 条，成功率不作资格门，运行器故障停止且保留。冻结总收据 `results/_analysis/four_task_transfer/legacy_independent_source_nominal_cohorts_20261031/launch_receipt.json` SHA `9764eb16ec2b46459f6d71dae2a9817e2c22379262ca928309edf941358fd3d5`，声明 GPU1、零正式 seed 访问与失败不替换。论文入口 `scripts/paper/cross_ebdm/qualify_legacy_main_table_actors.py` 负责逐实例执行与 CPU evidence/配对复位审计；后台单队列已从 T1 Cross-EBDM 开始，首个 instance_02 已写入任务本地 plan/progress。训练来源独立性及方法忠实度仍需单独审查，此开发队列完成也不会自动访问正式主表。

### 2026-10-31 T4 AdaptSim 训练延续与主表审计边界

T4 AdaptSim 独立训练 v1 的 seed 8814001 完成并通过 CPU 账本审计；seed 8814002 在候选策略模拟器 `reset()`、任何控制动作之前，遇到 `floating reset stance preparation failed: 3`，失败报告 SHA `4dd9fee581ea659a704fd19efd46bc0b8af2faf4164d1f399304c1e37d451a0f`，v1 队列按规则停止。旧八配方、成功实例和失败报告全部保留，不将成功的 v1 实例拼进新八实例估计。T4 AdaptSim 任务适配器现在仅在模拟器候选或模拟目标阶段，把明确的 `Go2+Z1 source pose unreachable` 和 `floating reset stance preparation failed` 零控制复位拒绝计为 `reset_unreachable`，每次候选照常占用冻结的 episode 预算；真实部署目标的相同错误继续中止并保留。CPU 账本审计也拒绝把目标部署零控制复位当作完成 episode。

新 v2 八份任务本地配方 seed 8814101--8814108 各通过零物理训练预检，绑定修订后的 T4 适配源码 SHA `2fb6869604a6f12f186eb6f364e30824860d6b30c270fe0ebe65cb541bbdc1ea`，完整队列最多 7,056 次训练候选／目标 episode、40 次部署目标 rollout、零正式请求访问，遇首个未归类错误或审计失败即停。冻结收据 `results/_analysis/four_task_transfer/adaptsim_t4_independent_training_continuation_v2/launch_receipt.json` 引用旧启动收据和旧失败报告；GPU1 后台队列已从 8814101 启动。旧完成实例的只读 CPU 再审计仍通过，包含 882 次训练记账及 340 次零控制复位拒绝；这是训练账本验证，不是正式性能证据。

FADA 的四任务源训练来源另经 CPU 核对：每任务八个目标适配 checkpoint 都指向同一个经 4/4 source 请求合格的 Panda 源 checkpoint，八个目标适配 seed 不重复，四份任务本地报告均标为 `qualified_source_lineage_only`。这证明来源与 checkpoint 指针，不证明目标适配实现已忠实于作者方法。T1 的 FADA 八实例源／名义开发请求 64/64 均为真实 timeout，零 runner error，不能写成方法实现故障，也不能据此声称跨四机制的正式差异。T1 同一开发请求上 Ours 和 task-only 各 64/64 成功，尚无 transport 收益证据。

新 CPU-only 资格库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v20/report.json` SHA `bb3e9c0c8f51bf3fbf24ee9654ad7b542ff87d7f18d1238f156258530b3b2f4c`：36 格中 8 格固定公共参照可执行正式协议、6 格已完成八实例源／名义开发资格、16 格旧 actor 的补充资格尚在 GPU1 队列中、6 格 ASID／AdaptSim 仍在训练或后续资格队列中。正式主表物理请求访问仍为零，不能从开发资格库存生成论文性能表。新的 `scripts/paper/cross_ebdm/assemble_independent_main_table.py` 只在 36 格全部独立物理审计完成时重算 11,600 个受约束 outcome；现阶段 `--inventory` 应报告未完成。T4 复位分类、目标中止以及主表相关 CPU 测试为 21/21 通过。

在打开正式请求前，主表分析另冻结为 `four-task-independent-main-table-inference-v1`：对四任务九方法完整 11,600 条已审计物理结果做 10,000 次 crossed instance／request bootstrap，seed 267100，pointwise 95% percentile 区间。训练实例作为独立抽样层；每个任务在固定机制×支撑层内抽五个物理初态块，两个 mode 一起抽；同一请求块的抽样跨所有方法共享，不同学习方法的训练实例独立抽样，固定公共参照保持单 actor。主表四个指标及 Ours 对另八方法的逐任务和 T1--T3 macro 差值均报告区间；macro 的 worst-group 对每次重采样先取各任务最差组再取最差任务，不把“最差组差值”误当“差值的最差组”。这是条件于冻结模拟器机制／支撑混合的描述性不确定性，不宣称随机化因果检验。CPU 合成完整矩阵检验包含完全相同的 Ours／task-only 配对结果，其差值和区间均为零；22 项主表相关测试通过。

T4 ASID v2 的前七个 seed 8804101--8804107 完成独立审计后，第八个 seed 8804108 在预检阶段因共用的 `mobile_mating_policy_search.py` 已为 AdaptSim 修复而更改 SHA，被严格拒绝；原队列将失败写入日志并停止，未创建训练输出、未调用物理环境。这是并行修改对已冻结 ASID 配方造成的操作错误，不能把旧版第七实例的审计当作八实例完成。第七实例保存的原版源码快照 SHA `e827338c98b1f24e195597989e484e179757da474a79d7854bce19be4af14c19` 与第八配方绑定的 SHA 完全一致。已在 `/tmp/genedynamics-asid-t4-seed8-frozen-source` 建立隔离运行根，逐项核对第八配方的 16 个源码绑定，零物理预检通过；相同 seed、相同配方的 GPU1 运行与后续 CPU 审计已另行启动。任务本地恢复收据 `results/quadruped/mobile_keyed_insertion_recovery/baseline/asid/validation/independent_training_preflight_recovery_8804108/receipt.json` SHA `a61204ec9367b4a4e7120a4b3f0bdd08a24e08ead0eea96d5921c7b3891408ce` 保留原拒绝日志及隔离依据。只有完成此第八实例独立审计后，CPU watcher 才会生成 T4 ASID 八实例 actor 包；原 source／nominal 调度器仍需随后测满 64 条开发请求。此恢复不新增 seed、不改变冻结算法、不访问正式主表。

### 2026-10-31 DIAL 作者源码核验与 T4 ASID 恢复

DIAL-MPC 官方源码已固定在 `baselines/DIAL-MPC` 的提交 `871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c`，作者核心文件 SHA `a4471963c53b04ee2d8c67c7cefbe0aeced83fa3d2b97338c42e69d7238ba64c`；`requirements/baselines/sources.lock.json` 和 `genedynamics/solvers/single/dial/source_manifest.json` 记录来源。直接从该文件抽取作者反向更新，在相同采样噪声、样条和奖励下，分别对齐本地 JAX 实现及实际部署的 PyTorch 更新。带越界热启动的边界测试发现部署版本原先在固定首节点与追加 incumbent 之前裁剪候选，作者源码在两者之后统一裁剪，首动作可能为归一化 1.3 而作者为 1.0。现已按作者顺序修正；相关 13 项 CPU 测试通过，部署控制器当前 SHA `6d4d95202ab8b3c29d26cde0c42203d467ff98898da8b6d1ad9b0617afd7638a`。作者使用完整 Brax 动力学和力矩级 locomotion，这里用公共响应模型与任务动作接口，因此比较名仍是 **task-adapted DIAL-MPC**；单步数值一致不等于整个作者训练／任务系统等价。旧 T1 DIAL 八实例 64 条开发审计仍是原样保留的旧控制器证据，须在当前控制器下重新跑八实例源／名义资格，之后才可作正式单元准入。旧正式请求没有访问。

T4 ASID 被拒的第八实例 seed 8804108 已在隔离原版源码下完成同 seed 训练和 CPU 审计，训练报告／独立审计 SHA 分别为 `41c2f6f4a5ca4ba13403025950317874614b9e9b7ab9b5fe33ed390d50077aa3`／`a45c2a0f1c1626a3845b8828f368dc93c118e7a09e25d508447b1b6cc37a949d`。八实例任务 actor 包经 CPU watcher 生成，SHA `fe73fba4f2c3006a0de87b37cb416b39bd9d3a392efb90524572885eedf976d4`；T4 ASID 的 64 条源／名义开发请求现由既有 GPU1 调度器顺序执行。训练实例齐备和开发执行均不是正式主表性能。

四个任务先前的 DIAL 首实例开发屏幕均来自裁剪顺序修正前的控制器，故不把任何 v1 cohort 自动当作当前源码资格。现有论文入口 `qualify_legacy_main_table_actors.py` 加入任务本地 v2 刷新：冻结作者来源、当前部署源码、旧证据与相同的八个开发请求，再对每个真实 actor 实例重新执行与 CPU 物理审计；首次失败保留，不能替换。T1 v2 冻结配方 SHA `b662ace4d58b249edc498dbec67a4f6bacc324f57ec485f248388896b4f3c62c`，已在 GPU1 后台开始，最多 64 条开发 episode，零正式请求。T2--T4 在各自 v1 cohort 完成后按同一入口重验。CPU-only 库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v21/report.json` SHA `386f20b71eaea43ac384abea8de44ec19aa418170d0324c334732c294d811e45` 将四个 DIAL 单元明确标为 `controller_source_refresh_pending`，旧结果以 stale pointer 保留，8 个固定单元仍可执行正式协议；九方法 36 单元的正式物理请求仍为零。

T1 DIAL v2 随后在当前源码下完成八实例共 64/64 条真实源／名义开发请求，零执行错误、64/64 受约束成功；独立物理审计 `results/arm/obstacle_transfer/baseline/dial/validation/independent_source_nominal_cohort_v2/independent_audit.json` SHA `6cda23e076d4a77209df2dbb6997cd8735bdd45c807218c52fb578855a868199` 已由当前 CPU 入口重算一致，绑定当前部署源码及旧 v1 审计。此结果更新的是方法资格，不是正式主表成绩。T2--T4 的后续队列已启动等待各自旧 cohort；新 CPU-only 库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v22/report.json` SHA `b96d0283324a7a56cac41004b02ed2a857f5139129a4ea3039bbff2dfca29e9c` 显示 T1 DIAL 已恢复八实例开发资格、其余三个 DIAL 单元仍待当前源码复核，正式可执行仍只有 8 个固定参照单元。

T1 的四条共享模型路线另完成 CPU-only 训练来源审计：八份 Ours 策略、八份 task-only 策略、八份公共响应模型 checkpoint 均与训练报告 SHA 对齐，八个样本流彼此不同且每实例 Ours/task-only 精确共享同一样本流；Ours 报告记录 2,000 次主更新、任务代理选择通过、零新目标访问，响应模型各 2,000 次更新并具有源／名义 holdout。任务本地审计 `results/arm/obstacle_transfer/validation/independent_shared_training_lineage_v1/report.json` SHA `0f9154ad898ee045dcc8549f74b350ebb72b986100bc58c2f5eec0c310bcd6a5` 将 DIAL/ordinary response MPC 使用的同实例 task-only 策略及响应模型一并绑定；这是训练来源与配对性证据，不是方法忠实度或目标效果。T4 ASID 八实例的 64 条源／名义开发请求也已完成并通过独立审计，SHA `0aed4d8bf8a87fce133f0611cb05c3815597a132f573924a8cb18f3fad9a51ce`。最新 CPU-only 库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v23/report.json` SHA `7b49964f62c30213bebedc659422500d56729fb1e8806d2c714a061fe4d4ce42` 有 9 格八实例开发资格、8 格固定正式可执行、3 格 DIAL 当前源码待重验，其余任务继续排队；正式请求访问仍为零。

T2 现行主表绑定的训练来源经 CPU 核对：八个独立训练 seed、八个独立样本流、48 个已选 checkpoint 与 `policy_instances_v3` 聚合报告／artifact audit 一致；Ours、task-only、DIAL 绑定按实例的 task-only 策略及响应模型，而 ordinary response MPC 绑定一份固定 companion 策略及按实例的响应模型。任务本地审计 `results/arm/guided_contact_alignment/validation/independent_shared_training_lineage_v1/report.json` SHA `663b291302b6b99f2744f2e3a78ab8b17d4725a06905e3fa28c37813e63af03c` 只证明现行主表 actor 的训练来源，不把 response MPC 误写成同一策略的配对消融。T3 现行 Ours 主表为 mode-specific v21 策略集，mode 0 的八个 checkpoint 均不同于 task-only shared-start，mode 1 则有六个与 shared-start 相同；因此旧的通用 `cross_ebdm` 策略集不能用于审计现行 T3 行。T4 现行 Ours 主表使用 `cross_ebdm_completion_projected`，八个 checkpoint 均不同于 task-only；旧 `cross_ebdm` 的样本流配对也不能作为现行 T4 主表的单因素 transport 归因。T3/T4 的直接因果结论只能来自各自冻结的同起点／同执行器诊断，主表只报告现行完整方法表现。DIAL 作者对齐测试固定 GPU JAX 的完整矩阵乘法精度后，保持 `2e-5` 容差，三个源码对齐检查均通过；部署源码没有因测试环境精度差异而更改。CPU-only 资格库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v25/report.json` SHA `1e81af194ac4881cca46320bffa2b58c563a7a04e106348487087c7d4c282243`：九方法 36 格中 8 格固定参照具有正式可执行资格、10 格完成八实例源／名义开发资格；其余仍在 GPU1 训练、审计和 DIAL 当前源码复核队列中，正式主表请求仍未访问。

在不打开正式请求的条件下，现有论文入口 `qualify_legacy_main_table_actors.py` 又为 T1 的 Ours、task-only、task-adapted DIAL、ordinary response MPC，以及 T2 的 Ours、task-only，逐格生成任务本地训练来源／方法范围审查、正式准入和八份实例配置；48 份配置都由 `formal_plan` 独立重算绑定 SHA、冻结请求和 GPU1 物理标识后通过。这里的 DIAL 范围仍明确是公共响应模型下的任务适配实现，不冒称作者完整 Brax 系统。资格库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v26/report.json` SHA `92463269dbc2c3c57d82d39327acb7b6455bf27198964ad687fde6c3f59602da` 因此显示 14/36 格正式可执行资格（8 固定＋6 学习），5 格仅完成八实例开发资格；所有 36 格的正式请求执行数仍为零。T2 后续 DIAL 当前源码刷新、response MPC 源／名义审计、T3/T4 老 actor 补充审计及 ASID/AdaptSim 八实例训练和源／名义资格继续排队。尚未有可填入论文主表的独立成绩。

FADA 新增**有限范围**的任务适配忠实度核对：从固定作者源码抽取的 Planner-IDM／LoRA 定义、公共记录与联合动作接口、训练与归一化路径共 34 项 CPU 测试全通过、零跳过；收据 `results/_analysis/four_task_transfer/fada_task_adapted_parity_20261031/report.json` SHA `82a922d1ce48581adf64fe0357053029ed09b919b17fb79fd690f13611591eb6`。任务本地准入又逐项校验八个 target checkpoint 与现行 manifest、目标配方源 checkpoint、不同适配 seed、每实例 3,000 条既有目标控制、2,000 次完整 IDM 更新、冻结 planner／归一化和八实例 64 条源／名义物理审计。原作者 oracle PPO、DAgger 数据采集与完整模拟器仍不在复刻范围，故论文只能写 **task-adapted FADA Planner-IDM**，不能写成作者原始系统的逐字节复现。旧 `independent_source_lineage_v1` 中的 `target_method_fidelity_qualified=false` 是当时仅审过源链的结论；新增任务适配审查有单独范围和证据，不改写该历史报告。T1、T2 的 FADA 已各生成八份正式配置，保留 T1 64/64 timeout 等全部开发失败，不用成功率作准入门。新 CPU-only 库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v27/report.json` SHA `dd432a55a54b66e1f81d636b6f9818426e98bbc49dde0dae962cc998928c82e2` 为 16/36 格正式可执行资格；正式物理请求仍为零。

ASID／AdaptSim 的官方来源绑定、训练／适配和公共 actor 回放共 38 项测试通过、零跳过，收据 `results/_analysis/four_task_transfer/asid_adaptsim_task_adapted_parity_20261031/report.json` SHA `3d886d476b7bb0177c664d02f2a5059dbe2716d094924869511637036d3e3e05`。已完成八实例训练包及 64 条独立源／名义物理审计的 ASID T1、T3、T4 三格，通过任务本地方法范围审查和八实例 `formal_plan`，用一个任务本地 cohort 配方绑定八个训练身份；完整作者模拟器不在此任务适配端口范围。CPU-only 库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v28/report.json` SHA `9151171de144a99fe560db1ae145b48de207caeed2f12b62dab146d4f39fba86` 为 19/36 格正式可执行资格。其余 ASID／AdaptSim 格待训练包与源／名义审计；零正式请求执行。

T3／T4 的现行 actor 来源不再用旧通用策略集冒充：T3 任务本地 `results/arm/keyed_insertion_recovery/validation/independent_current_actor_lineage_v1/report.json` SHA `94b34e1bd980f1050ce9eec43edbfbda7559099c92d2e30c2c688cfbcdfe82e8` 核对 shared-start、两份 mode 训练报告、两个物理 selector、最终资格报告及八实例最终 checkpoint；mode 0 八个全部来自 selector，mode 1 七个来自 selector、instance-05 回退 shared-start，最终 mode 1 六个 checkpoint 与 task-only 相同。T4 `results/quadruped/mobile_keyed_insertion_recovery/validation/independent_current_actor_lineage_v1/report.json` SHA `48aee7a5087fb2bbea0ee151460623bdfa0cc7b8990c54ebd9959aacc7805bb3` 明确保留 raw 未投影 checkpoint 的 `selection_failed`，并分别校验组合投影 actor 的八实例离线预检与另一批 64/64 物理开发成功、零违规、每实例非零 candidate 使用；现行主表身份是这个组合 actor。两份报告均只认证现行 actor 来源和方法范围，不把 T3/T4 主表性能差额写成 transport 的单因素因果效应，也不替代新的独立主表请求。库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v29/report.json` SHA `6e95bed621465c3c23a9ea47604d5dc53146c485f20379833ad9e90c7e9eed6a` 将两份审计接入待资格单元；正式可执行仍为 19/36。三个 fail-closed CPU 后续准入进程只在各自新的源／名义独立审计到齐后产生配置；GPU1 继续训练和开发筛查，正式请求未开启。

主表 CPU 装配器已按现行配置布局读取：固定参照每格一份 `*_fixed.json`，旧学习方法每格八份 `*_instance_XX.json`，ASID／AdaptSim 每格一份含八个身份的 `*_cohort.json`；不再假设所有方法都有同名的单一 `<method>.json`。`--inventory` 现在正确区分 19 格“正式配置齐备、物理结果未执行”和 17 格“配置待准入”，而不会把已有配置误判为缺失。正式 GPU1 后台门先要求九方法四任务 36 格全部可执行、232 个 actor 实例的 `formal_plan` 全部通过并冻结总分母 11,600 条；只有这时才逐实例运行已封存请求、立即做独立物理审计，任一 runner／审计错误原样保留并停止。全部审计通过后才允许 CPU 重算主表和 10,000 次 crossed bootstrap。门当前处于 `WAITING_FOR_ALL_36`，19/36 合格，正式物理结果 0/36；DIAL T2 的旧 v1 八实例审计已完成，当前源码 v2 刷新刚冻结并由既有 GPU1 队列执行。17 项主表准入／装配／统计测试通过。

后台门在等待期间增加了正式启动时的共享源码、solver、公共学习组件及四个任务配置的 SHA 冻结；每个 actor 作业开始前和物理审计完成后再次校验，字节变化即保留现场并停机。门脚本位于 `/tmp/genedynamics_independent_main_table_gate_and_run.py`，状态与日志分别位于 `/tmp/genedynamics-independent-main-table-gate-state.json` 和 `/tmp/genedynamics-independent-main-table-gate-and-run.log`。修订后重新挂起时仍为 `WAITING_FOR_ALL_36`、19/36 合格，未生成正式启动收据，也未执行正式 episode；主表合同测试仍为 17/17 通过。

T1 AdaptSim 八份新独立训练及各自审计完成后，CPU actor 包资格发现一个兼容性错误：新独立训练审计的真实状态是 `passed_development_only`，旧 T1 pilot 绑定只接受 `passed`。失败的封装日志保留于 `/tmp/genedynamics-independent-actor-package-watcher.log`，T1 AdaptSim 继续记为未合格，不能填主表。直接修改 `controllers/adaptsim.py` 会破坏仍在执行的 T2 ASID／AdaptSim 冻结训练源码 SHA；尝试改动已完全撤回，原 SHA `ac2a11c2d0531e386769958c4aad7085533bea87ad0b421ae8e427f3340de52c` 恢复，38 项来源／公共 actor 对齐测试重新全部通过。严格保护的 CPU 后续守护 `/tmp/genedynamics_adaptsim_t1_posttraining_finalizer.py` 只有在其余 35 格先合格、T2 两条训练队列均完整结束、三个源码／测试入口的原 SHA 均保持不变时，才修正新旧审计状态分支、重验测试、另存 v2 parity 收据并重新封装 T1 actor；任何前提不符即保留失败并停机。其后现有源／名义调度器和正式准入守护可继续该格，正式请求仍需等待全部 36 格合格。

延后补丁还在独立 Python 进程中做过不落盘预检：候选源码逐一加载八份真实新 checkpoint 和旧 pilot，各自的两个公共 mode 均能构造可调用 actor；这一检查没有修改正在训练所引用的仓库源码，也不替代届时的正式八实例包资格、64 次源／名义物理审核或独立主表结果。

DIAL T2 的当前控制器源码 v2 刷新随后完成：任务本地八实例源／名义独立审计 `results/arm/guided_contact_alignment/baseline/dial/validation/independent_source_nominal_cohort_v2/independent_audit.json` SHA `4f5cb529fff343ba7c1cf0a73d2cf707cb287bca6f899cd30299010d3bf377f9`，CPU 方法资格审查返回 `passed`。重新运行九方法准入库存得到 20/36 格正式可执行；T2 DIAL 的八份正式配置分别通过 `formal_plan`，共同绑定 320 个冻结请求实例结果位（每份 40 请求），但仍未进行任何正式物理执行。

该里程碑的库存快照为 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v30/report.json`，SHA `6598a3369ce6e3560087d62f986ef09ecba30dc172d462269919d45188f87803`：20 格已合格、9 格待旧 actor 完整资格、2 格 DIAL 当前源码重验、5 格新 baseline 八实例与源／名义链待补齐。

T2 DIAL v2 的 64 条源／名义开发请求全部有可审计物理 evidence，但任务结果为 16 次 `overforce`、48 次 `timeout`、零次受约束成功；旧 v1 同名单也是这个分布。方法资格通过只说明部署身份、作者来源范围和执行证据有效，不以任务成功作筛选，也不把任务适配 DIAL 的开发失败泛化为作者完整 Brax 系统的结论。

T2 ordinary response MPC 的八实例源／名义开发 cohort 随后 64/64 次受约束成功、64/64 份物理 evidence 审计通过；汇总审计 `results/arm/guided_contact_alignment/baseline/response_mpc/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `4d840aa7418e2b2873143d605a275fba34c0f782c9dc483236089053d2463cfc`，不含正式性能证据。CPU 方法资格审查通过，八份正式配置逐一通过 `formal_plan`，绑定 320 个实例×请求结果位。新库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v31/report.json` SHA `ee20631e2c6d79bf7c27ffd90afc2681b26a4f046c8713e7efc9a6b7e4716293` 为 21/36 格已合格，余下 8 格旧 actor 完整资格、2 格 DIAL 当前源码刷新、5 格 ASID／AdaptSim 链。旧方法 GPU1 队列已转入 T3 ours；正式请求仍未执行。

T3 现行 ours（mode-specific v21 actor）随后完成八实例 64 条源／名义开发请求，独立物理审计 `results/arm/keyed_insertion_recovery/main/cross_ebdm/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `467e6957a0131b7026dce3a0d89eafa4233c49f8db610bcc3de1e29304771ac6` 为 `passed_development_only`：25 次受约束成功、39 次真实 `overforce`，全部 64 条有物理 evidence。结合现行 actor lineage 的 CPU 审查后，该格八份正式配置和 320 个冻结请求结果位逐一通过零物理 `formal_plan`；资格通过并不掩盖开发失败，也不能预告正式分布性能。库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v32/report.json` SHA `c138a352066358376592ada4331925a7c2787caf82fcd35bdd08828f7f11e55f` 为 22/36 格合格；T3 task-only 已由 GPU1 旧方法队列接续执行，正式请求仍未开启。

T3 task-only 随后也完成八实例独立物理审计，汇总 `results/arm/keyed_insertion_recovery/baseline/task_only/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `b2feaf147748148f16c0b5c2b8673383715c7a66570d57f7ac35f07dabd169a8`；同样为 25/64 成功、39 次过力，CPU 方法资格审查通过。逐实例、逐请求核对两方法报告和物理初态哈希后，54/64 对成败相同，ours 独赢 5 对、task-only 独赢 5 对；source 条件 ours/task-only 为 13/32 对 14/32，nominal 为 12/32 对 11/32。失败罚时归一化完成时间的 ours 减 task-only 均差为 -0.00708；按八个训练实例重采样 10,000 次（seed 267132）的描述性 95% 百分位区间约 [-0.0617,+0.0549]，未显示稳定时间优势。这是开发名单的完整方法配对，T3 两行的 mode-specific 选取及共享起点不满足单因素 transport 归因；正式性能仍须等待独立四机制请求。

T3 task-only 的八份正式配置也逐一通过 `formal_plan`，库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v33/report.json` SHA `7734b7d89da09ca35fd6b066b3c1a6a835847a172e8e739055aa877a1ce7022f` 增至 23/36 格已合格。余下 6 格旧 actor 资格、2 格 DIAL 当前源码刷新、5 格新 baseline 链，正式主表仍未开启。

T3 FADA 八实例源／名义开发 cohort 随后完成独立物理审计，`results/arm/keyed_insertion_recovery/baseline/fada/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `e54ee8387df3c73a09eae90a779a7dc59db32fa0052070574f7217a1a2ffed63` 核实 64/64 份物理 evidence，结果为 51 次过力和 13 次超时、零次受约束成功；按已冻结的非成功率准入规则，任务适配方法资格与八份正式配置仍通过 CPU 审查。T2 ASID 八个独立训练实例及各自训练审计完成，actor 包 `results/arm/guided_contact_alignment/baseline/asid/validation/independent_actor_package_v1/report.json` SHA `b17e04424e0062eee24477c36db1a55cbff072308501bcfbf00a521c04be4348` 已生成，现由 GPU1 顺序执行八实例的 64 条源／名义开发请求；T2 AdaptSim 八实例训练队列也按原冻结配方开始。库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v34/report.json` SHA `98109d50273aa15ca10a81daa1502eb4ec1fc52990dd72629ce54ff036c1288f` 为 24/36 格正式可执行，余 5 格旧 actor 资格、2 格 DIAL 当前源码刷新、1 格 ASID 源／名义资格、4 格 AdaptSim 链。正式主表物理请求仍为零。

T3 DIAL 在旧八实例源／名义 cohort 之后，按冻结的当前控制器源码执行同一批 64 条开发请求，v2 独立审计 `results/arm/keyed_insertion_recovery/baseline/dial/validation/independent_source_nominal_cohort_v2/independent_audit.json` SHA `1046174e076788392b23ff21478098b4d6c637b43e5f8e9512dd5d6571fac922` 记录 15 次受约束成功和 49 次过力、64/64 份物理 evidence；旧 v1 的 17 次成功不能替代当前源码结果。CPU 方法资格与八份正式配置通过，库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v35/report.json` SHA `c48abfe74d504dbfed1ec5cafed5f673747326469a55463a2a3fb4ff53a675bd` 为 25/36 格已合格。T4 AdaptSim 修订训练的第二实例 8814102 亦完成独立训练审计，SHA `3fe25eba8d15e72588ea4ca8bce90cf5d8752ec8d378273f2aeba815416d22c4`，第三实例已启动；T2 ASID 源／名义 cohort 在继续，正式主表仍未执行。

T3 ordinary response MPC 的八实例源／名义 cohort 独立审计 `results/arm/keyed_insertion_recovery/baseline/response_mpc/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `61434c8da18b8be4b09794c94fd9740c3c6f558f88c9bc44db9dee881f660666` 核实 64/64 份物理 evidence，23 次受约束成功、41 次过力；CPU 方法资格与八份正式配置通过。T2 ASID 八实例源／名义 cohort 亦完成，独立审计 `results/arm/guided_contact_alignment/baseline/asid/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `26522a7c8b6578838dbb76936804a6891e20ff22e77bdc4c6d0995b421f48b44` 核实 64/64 次开发受约束成功与全部物理证据，task-adapted 方法审查和八身份正式 cohort 配置通过。这些结果只说明源／名义可执行性，不能预告冻结机制偏移下的独立性能。库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v36/report.json` SHA `f1abfbc987009c200a5d04324893ca94694d2cf7b19329a2dc3d7faf256cb3b0` 为 27/36 格已合格，余下 T4 五个旧方法／当前源码资格及四个 AdaptSim 单元；正式主表请求仍为零。

T4 现行 Ours 的八实例源／名义开发 cohort 独立审计 `results/quadruped/mobile_keyed_insertion_recovery/main/cross_ebdm/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `770d9b007d847efe58416df4117a8167d551393975b7ca4ffdfd6266940a631f` 核实 64/64 条物理证据、64/64 次受约束成功，方法资格和八份正式配置通过；它认证当前组合投影 actor 的来源与可执行性，不把主表差异归因于 transport 单因素。新库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v37/report.json` SHA `60f721bdd94e5ef8e78d22cf0a77d53d6d6e433894b69533c1de222b3ce7bde3` 为 28/36 格合格。余下 T4 task-only、FADA、DIAL、response MPC 及四格 AdaptSim；T4 task-only 开发筛查已由原队列接续，正式主表仍未开启。

T4 task-only 的八实例源／名义开发 cohort 也完成独立审计，`results/quadruped/mobile_keyed_insertion_recovery/baseline/task_only/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `edea22378ac44dd4b98b5325c019d323a757840bf7172b40d7678c0b48891430` 核实 64/64 份物理 evidence 和 64/64 次受约束成功。它与现行 Ours 在该开发成功率上均饱和，不能据此声称 transport 有 T4 任务收益；两行现行 checkpoint 的续训与选取也不满足单因素因果归因。CPU 方法资格及八份正式配置通过；库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v38/report.json` SHA `519d9cb5c189a69d1f59db1c435b87315c579908610faa374e084db334439a5c` 为 29/36 格已合格，T4 FADA 队列已接续，正式主表尚未开启。

T4 task-adapted FADA 八实例源／名义 cohort 独立审计 `results/quadruped/mobile_keyed_insertion_recovery/baseline/fada/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `340ed449db63f4281f99908ae6ab6f7d80563b0ff648c3220b67d3168c9ec63c` 核实 64/64 份物理 evidence；全部 64 条为真实 `overforce`，零次受约束成功。按冻结准入合同，真实任务失败不作排除理由，任务适配方法忠实度审查及八份正式配置仍通过；不能把这个开发现象概括为作者完整 FADA 系统的四任务结论。库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v39/report.json` SHA `2f2202c2392d52825730bff890b18decf3165fb47ded1f75bdf385defa4c7d64` 为 30/36 格合格，剩余 T4 DIAL 当前源码重验、T4 response MPC 与四格 AdaptSim；正式请求仍为零。

T4 task-adapted DIAL 按当前控制器源码完成八实例源／名义重验，`results/quadruped/mobile_keyed_insertion_recovery/baseline/dial/validation/independent_source_nominal_cohort_v2/independent_audit.json` SHA `dd6a6abde9e9e411a568b4c3276ee8dcc4d22b84577bce49f81cb495b7015dc4` 核实 64/64 份物理 evidence 与配对复位哈希，64/64 次受约束成功；旧 v1 结果仍保留但不替代当前源码资格。CPU 方法资格和八份正式配置通过。库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v40/report.json` SHA `082c700da38338a300a4c7638b866531d6a4928e5df4e14ffd39aebc33640efe` 为 31/36 格合格，剩余 T4 response MPC 与四格 AdaptSim；正式主表请求仍为零。

T4 ordinary response MPC 八实例源／名义 cohort 独立审计 `results/quadruped/mobile_keyed_insertion_recovery/baseline/response_mpc/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `2e02a2343504106f09d3820f302da15ac02e63889cf279865fd52b03fb270889` 核实 64/64 份物理 evidence 和配对复位，64/64 次开发受约束成功；CPU 方法资格及八份正式配置通过。这是资格筛查，不是正式机制偏移下的性能。库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261031_v41/report.json` SHA `ab77ac760da7bc5b14968649e79dd022c71e760d92ef1f4e1c6ef8afc3a7815f` 为 32/36 格合格，仅余四格 AdaptSim；正式主表请求仍为零。

AdaptSim 独立训练继续按冻结配方运行。T3 第五种子 8813005 的训练账本审计 `results/arm/keyed_insertion_recovery/baseline/adaptsim/training/independent_training_8813005/independent_audit.json` SHA `5e9dcc9308299874aec1979cca0e6cc3b7933faccfc9684762320fdf285a69a1` 通过，核对 21 次策略训练调用、16 次模拟目标和 5 次部署目标回放，现进入第六种子。T4 v2 第三种子 8814103 的审计 `results/quadruped/mobile_keyed_insertion_recovery/baseline/adaptsim/training/independent_training_8814103/independent_audit.json` SHA `628ce3a932387c176826ae4ee04f43106fd8b15046ba64e085fa57562d970725` 亦通过，882 次账本物理尝试中 170 次候选训练阶段的零控制复位不可达被单独保留，模拟目标与部署目标阶段均无此项，现进入第四种子。两份审计只认证训练来源和账本，不是主表性能；四格 AdaptSim 仍待完整八实例 cohort 和后续源／名义资格，主表仍为 32/36。

动态夹具 response model 的独立预测 holdout `results/arm/compliant_keyed_insertion_recovery/validation/fixture_response_model_holdout_20261031/report.json` SHA `68b281ff688cb45b07f8f92867ed83e1fad545e70dcea5ef5b011954375cf3b7` 为 `complete_independent_prediction_holdout`：冻结 v2/v3 checkpoint，使用八条新公开 episode、3,623 个窗口，零新训练更新。对应的 v2/v3 GPU1 训练报告分别完成 600 次更新。全窗口位移 MAE 在 3 tick 为 7.72/4.47 μm，在 6 tick 为 9.16/9.02 μm，在 10 tick 为 13.79/13.81 μm；v3 的 10-tick 事件窗口 MAE 为 49.38 μm，v2 为 50.63 μm。证据支持短时预测改善及长时总体误差近乎持平，只是模型预测审计，不能据此推断 transport 或任务成功率收益。

T3 AdaptSim 第六种子 8813006 的独立训练账本审计 `results/arm/keyed_insertion_recovery/baseline/adaptsim/training/independent_training_8813006/independent_audit.json` SHA `32603f4e3864610853a236611a024222603ef7d025c117a75547a5c5e2cf53a8` 通过，核对 16 份源码快照、21 次策略训练调用、210 个候选、16 次模拟目标和 5 次部署目标回放，共 882 次账本物理尝试；队列已进入第七种子。当前 T3 为 6/8 个独立训练种子审计完成，仍未生成八实例 actor 包或源／名义资格，正式主表准入仍为 32/36。

T4 AdaptSim v2 第四种子 8814104 的独立账本审计 `results/quadruped/mobile_keyed_insertion_recovery/baseline/adaptsim/training/independent_training_8814104/independent_audit.json` SHA `98003c27dd2c9f1bb408dde8781c0461b37f1cd58019fe8de968dd138cb9d6a6` 通过，核对 16 份源码快照、21 次策略训练调用、210 个候选、16 次模拟目标和 5 次部署目标回放。882 次物理尝试中有 140 次候选训练阶段的零控制复位不可达，模拟目标与部署目标阶段均无此项；队列已进入第五种子。T4 现为 4/8 个独立训练种子审计完成，仍待完整 cohort、八实例 actor 包及源／名义资格，正式主表准入保持 32/36。

T3 AdaptSim 第七种子 8813007 的独立账本审计 `results/arm/keyed_insertion_recovery/baseline/adaptsim/training/independent_training_8813007/independent_audit.json` SHA `78ad943103659c53995d910aac462911ec11345bcabf675d6d2ce0990ad61791` 通过，核对 16 份源码快照、21 次策略训练调用、210 个候选、16 次模拟目标和 5 次部署目标回放。882 次物理尝试全部含动作并入账，零次复位不可达；队列已进入第八种子。T3 现为 7/8 个独立训练种子审计完成，仍待完整 cohort、八实例 actor 包及源／名义资格；正式主表准入保持 32/36。

T2 AdaptSim 首个独立训练种子 8812001 的账本审计 `results/arm/guided_contact_alignment/baseline/adaptsim/training/independent_training_8812001/independent_audit.json` SHA `2153a3c1c1807e66817ca4ae2d29e14c4b94602332f634d01f8df069013141b1` 通过，核对 24 份源快照、21 次策略训练调用、210 个候选、16 次模拟目标和 5 次部署目标回放。882 次物理尝试全部含动作并入账，零次复位不可达；GPU1 队列已进入第二种子。T2 现为 1/8 个独立训练种子审计完成，后续仍需八实例 actor 包及源／名义资格；这份审计不是正式任务性能，四任务主表准入仍为 32/36。

T4 AdaptSim v2 第五种子 8814105 的独立账本审计 `results/quadruped/mobile_keyed_insertion_recovery/baseline/adaptsim/training/independent_training_8814105/independent_audit.json` SHA `41afed8528f0eacdeaafc7eec2b2b28a441e50a2eac98a4c409637a2b2f60db7` 通过，核对 16 份源快照、21 次策略训练调用、210 个候选、16 次模拟目标和 5 次部署目标回放。882 次物理尝试中 260 次候选训练阶段的零控制复位不可达被保留；模拟目标与部署阶段均无此项，第六种子已在 GPU1 启动。T4 现为 5/8 个独立训练种子审计完成，仍待完整 cohort、八实例 actor 包及源／名义资格；正式主表准入保持 32/36。

正式主表 CPU 汇总链路的现有测试使用 `/tmp/genedynamics-gpu-062/bin/python -m pytest -q test/transfer_methods/test_main_table_evaluation.py test/transfer_methods/test_main_table_admission.py test/transfer_methods/test_main_table_statistics.py` 运行，17/17 通过。测试覆盖评估计划、准入和交叉 bootstrap 统计；系统 Python 自带 pytest 7.4.4 低于仓库规定的 8.3，故使用已有隔离环境的 pytest 9.1.1。该检查验证代码路径，不替代正式物理结果审计。

T3 AdaptSim 第八种子 8813008 的独立训练账本审计 `results/arm/keyed_insertion_recovery/baseline/adaptsim/training/independent_training_8813008/independent_audit.json` SHA `00a3586ae7bc985f80516b15b1c3dd45d9862d92868546ae067fdeaf5e2fd575` 通过，八个种子现均通过相同范围的独立审计（每个 16 份源快照、21 次训练、210 个候选、16 次模拟目标、5 次部署目标回放和 882 次入账物理尝试）。八实例 actor 包 `results/arm/keyed_insertion_recovery/baseline/adaptsim/validation/independent_actor_package_v1/report.json` SHA `b1ca03ad00b75d8234a69a4a015d1a494daf39c348b07da4fe59d5ce7ba2ed31` 为 `passed_development_only`，实例恰为 8813001–8813008。源／名义开发审计启动回执 `results/_analysis/four_task_transfer/adaptsim_t3_independent_actor_source_nominal_qualification_20261031/launch_receipt.json` SHA `4832c4f35b2e8014c32d816f740435f1c01423ee80f1b156c52202047bdd77a2` 固定 GPU1、八实例、最多 64 条物理请求、失败不替换且零正式请求种子访问。T3 尚待完整源／名义 cohort 和正式准入，主表仍为 32/36。

T3 AdaptSim 八实例源／名义开发 cohort 现完成，独立审计 `results/arm/keyed_insertion_recovery/baseline/adaptsim/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `02e58990498b68cba4c05f23c7895e40ff3ae441be2d9b54d4ee57a1c600d704` 为 `passed_development_only`，核对 64/64 份物理证据及共同初态哈希。开发结果 48/64 受约束成功、16 次过力；源条件 28/32 成功，名义条件 20/32 成功。方法保真审查 `results/arm/keyed_insertion_recovery/baseline/adaptsim/validation/independent_method_fidelity_v1/report.json` SHA `aa8ca6cac8ee48a1a07997a605d9e4f9a1a6006883668b010c5fe8d9f523bad8` 和正式配置资格 `results/arm/keyed_insertion_recovery/validation/independent_main_table_admission_v1/adaptsim.json` SHA `fd7754153f8828765caf40bf2d3b699a1ebca67a61bec88c0d868884adb0a61a` 均通过；方法身份为任务适配的 AdaptSim CEM，而非作者完整系统等价复现。新主表库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261101_v42/report.json` SHA `53403250aa8abd2415a04d2896ec8c01c2880d3bb5c099d9a43607e520b2861a` 为 33/36 格合格，仅 T1/T2/T4 AdaptSim 待准入；正式性能结果仍为零。

T4 AdaptSim v2 第六种子 8814106 的独立训练账本审计 `results/quadruped/mobile_keyed_insertion_recovery/baseline/adaptsim/training/independent_training_8814106/independent_audit.json` SHA `188d46e4a47922b5db9d29c11742820d595249f0ef6b1b2789c372a649b08af8` 通过，核对 16 份源快照、21 次策略训练调用、210 个候选、16 次模拟目标和 5 次部署目标回放。882 次物理尝试中 190 次候选训练阶段的零控制复位不可达被保留；模拟目标与部署目标阶段均无此项。队列已进入第七种子，T4 现为 6/8 个独立训练种子审计完成；尚待完整 cohort、八实例 actor 包及源／名义资格，主表仍为 33/36。

正式运行前对当前 33 个已合格单元执行了 CPU-only、零物理 `formal_plan` 全量预检：共解析 208 个 actor 作业、10,320 条冻结请求，配置来源、actor 绑定及请求 roster 均通过，没有计划构造错误。其余 3 个 AdaptSim 单元仍未合格，不在此次预检范围；本检查不产生正式性能证据，也不放行主表门控。

同一 33 格的额外零物理审查确认：208 个正式输出路径两两不同且均不存在；每个任务内各方法／实例使用完全相同的请求 ID 顺序，四个任务的配对请求 roster 均一致；每格实例数符合冻结统计合同（固定参照 1 个，其余方法 8 个）。正式请求仍未执行。

在 T2/T4 AdaptSim 队列继续运行期间，对尚未启动的 T2 种子 8812003–8812008 和 T4 v2 种子 8814108 做了只读配方预检：六份 T2 配方各自的 24 个源码绑定、T4 配方的 16 个源码绑定均与当前 SHA 一致，种子身份正确，未来输出目录均不存在。此检查只确认后续队列的当前可启动性，不代替训练账本审计或 actor 资格。

T4 AdaptSim v2 第七种子 8814107 的独立训练账本审计 `results/quadruped/mobile_keyed_insertion_recovery/baseline/adaptsim/training/independent_training_8814107/independent_audit.json` SHA `8942cfa592a41207fc7734c689bb795cd9732557994657464a42643225372eec` 通过，核对 16 份源快照、21 次策略训练调用、210 个候选、16 次模拟目标与 5 次部署目标回放。882 次物理尝试中 200 次 `reset_unreachable` 均在候选训练阶段；模拟目标与部署目标阶段为零。部署目标的 10 条开发 episode 均成功，但模拟目标阶段只有 8/32 条成功，另有 18 次关节限位及 6 次过力，说明训练分布表现并不均一。这是来源与账本审计，不是正式任务性能；队列已启动最后一个种子 8814108，T4 现为 7/8 个独立训练种子审计完成，主表资格仍为 33/36。

正式主表门控静态预检发现其 `/tmp/genedynamics_independent_main_table_gate_and_run.py` 来源清单额外引用了不存在的 `genedynamics/experiments/framework.py`；实际 framework 是目录包，已被 `genedynamics/experiments` 递归来源扫描包含。此错误原本会在 36 格合格后、正式启动回执写入前导致确定性失败。确认旧门控仍为 `WAITING_FOR_ALL_36`、33/36 且没有正式启动回执后，仅从 `/tmp` 门控脚本移除该多余路径；抽取实际 `source_roster()` 做零物理预检通过，核对 1,492 个来源文件及 framework 包的 5 个文件。旧等待进程退出后，新进程 PID 2989196 以独立 session 启动，日志 `/tmp/genedynamics-independent-main-table-gate-and-run-v2.log`，再次显示 `WAITING_FOR_ALL_36`、33/36；没有访问正式请求，也未改变训练、评估或统计源码。未来正式启动仍须审计 36/36、232 作业和 11,600 条物理结果。

T2 AdaptSim 第二个独立训练种子 8812002 的账本审计 `results/arm/guided_contact_alignment/baseline/adaptsim/training/independent_training_8812002/independent_audit.json` SHA `0e1dd05f3b6d3a9fd5b900c3de292505a4cb559e83e4ca8e94497d2c6070cfa8` 通过：21 次策略训练调用、210 个候选、16 次模拟目标和 5 次部署目标回放，共 882 次物理尝试，全部含动作，零次复位不可达。训练账本中有 786 次成功、47 次过力、49 次超时；这些是训练分布内的开发结果，不是冻结主表性能。GPU1 队列已进入第三种子 8812003；T2 现为 2/8 个独立种子审计完成，主表仍为 33/36 格合格，正式请求未启动。

T4 AdaptSim v2 的最后一个独立种子 8814108 完成训练，独立账本审计 `results/quadruped/mobile_keyed_insertion_recovery/baseline/adaptsim/training/independent_training_8814108/independent_audit.json` SHA `95ba37a109bdc670dd32ae02e3915e59f85802c0c36bea03e544537982a220d9` 通过：21 次策略训练调用、210 个候选、16 次模拟目标与 5 次部署目标回放，882 次物理尝试入账。其中 110 次 `reset_unreachable` 全部发生在候选训练阶段，目标两阶段均无此失败。八个独立种子共 7,056 次训练物理尝试；逐条汇总为候选阶段 6,720 次（含 1,630 次复位不可达）、模拟目标阶段 256 次（137 次成功）、部署目标阶段 80 次（全部成功）。这些都是训练分布内的开发账本，不是冻结机制偏移下的性能。八实例 actor 包 `results/quadruped/mobile_keyed_insertion_recovery/baseline/adaptsim/validation/independent_actor_package_v1/report.json` SHA `5c8bd1f798177d201e4a70259d9120bf95a5550039bd549f997b4cc7135a2896` 已生成并核对种子 8814101–8814108、零正式请求访问；下一关仍是八实例源／名义开发审计及当前方法资格，主表此时维持 33/36。

T4 AdaptSim 八实例源／名义开发资格随后完成。冻结启动回执 `results/_analysis/four_task_transfer/adaptsim_t4_independent_actor_source_nominal_qualification_20261031/launch_receipt.json` SHA `52053c30e2bd859c136bc2a396999cd0c0bf12744c55606e2a9b713c2029d70e` 限定 GPU1、八个训练身份、64 次开发物理请求、失败不替换、零正式请求访问。每实例独立物理审计均通过；cohort 审计 `results/quadruped/mobile_keyed_insertion_recovery/baseline/adaptsim/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `5a93ef3db90fbec07219d7ab477a69fce3edcd1065aff495747f1a32919cbe14` 核实 64/64 份证据、62 次受约束成功和 2 次超时，两次超时保留在种子 8814103。当前方法资格与八身份正式配置通过，CPU-only 库存 `results/_analysis/four_task_transfer/independent_main_table_admission_20261101_v43/report.json` SHA `1d126df887c8cb8424ede11b673e3e514cb8b601ee3b72a5163c84723d5230c9` 为 34/36，余 T1/T2 AdaptSim。新增 T4 单元的零物理 `formal_plan` 预检确认八个不同实例、每实例同一组 80 个冻结请求、640 个唯一待运行物理结果、正式输出路径均不存在；主表门控仍未启动正式请求。

进一步对全部 34 个合格单元重做零物理正式计划预检：216 个 actor 作业、10,960 条冻结请求，任务内各方法／实例请求 ID 配对一致；T1/T2/T3 每实例 40 条，T4 每实例 80 条，所有正式输出路径唯一且均不存在。T2 训练队列继续运行第三个独立种子；T1/T2 AdaptSim 尚未合格，不能把这次预检解释为 36 格主表放行。T1 训练后终结器的三份冻结源码 SHA 与 patch 预检再次通过，待 T2 cohort 完成后执行。

T2 AdaptSim 第三个独立训练种子 8812003 完成并通过来源／账本审计；`results/arm/guided_contact_alignment/baseline/adaptsim/training/independent_training_8812003/independent_audit.json` SHA 为 `fc35ee661fcc7848c374618dd0019bf0b470495bc0031ed6649ea658e52d1a52`。审计核对 24 份源码快照、21 次策略训练调用、210 个候选、16 次模拟目标回放、5 次部署目标回放及 882 次有动作的物理尝试；账本结果为 679 次成功、35 次过力、168 次超时，零次复位不可达。这是独立训练来源与账本核验，不是逐轨迹物理重放审计或冻结主表性能。GPU1 队列已自动进入种子 8812004；T2 为 3/8 个独立种子完成，主表仍为 34/36 格准入，正式评测尚未启动。

T2 AdaptSim 第四个独立训练种子 8812004 的同类审计 `results/arm/guided_contact_alignment/baseline/adaptsim/training/independent_training_8812004/independent_audit.json` SHA `8c2cceac60bb23f1608b8dc7ceb63430b525f521d1dade0a4153255ddac9cf7f` 通过，核对 24 份源码快照、21 次策略训练调用、210 个候选、16 次模拟目标与 5 次部署目标回放。882 次有动作的尝试中，777 次成功、28 次过力、77 次超时、零次复位不可达。前四个独立种子各自审计通过，共 3,528 次训练账本尝试，合计 2,867 次成功、125 次过力、536 次超时；不同阶段和训练分布的尝试不能合并解释为正式泛化率。GPU1 队列已进入第五种子 8812005；T2 当前 4/8，正式主表仍为 34/36 格准入。

T2 AdaptSim 第五个独立训练种子 8812005 已完成并通过同一来源／账本审计；`results/arm/guided_contact_alignment/baseline/adaptsim/training/independent_training_8812005/independent_audit.json` SHA `ef00927a1d2297b0a2847f18f30bbedad796ba436852727a56ff826026014927`。审计核对 24 份源码快照、21 次策略训练调用、210 个候选、16 次模拟目标与 5 次部署目标回放；882 次有动作的训练尝试中为 726 次成功、13 次过力、143 次超时、零次复位不可达。前五个独立种子共 4,410 次训练账本尝试，合计 3,593 次成功、138 次过力、679 次超时；这些仍是分阶段开发账本，不能作正式泛化率。GPU1 队列已进入种子 8812006；T2 当前 5/8，主表仍为 34/36 格准入，正式评测尚未启动。

T2 AdaptSim 第六个独立训练种子 8812006 的来源／账本审计 `results/arm/guided_contact_alignment/baseline/adaptsim/training/independent_training_8812006/independent_audit.json` SHA `a28ed75ffa8bf4854d131def07b82f7fc38751ae28676a2ee6bc56ec4daaa59d` 通过：24 份源码快照、21 次策略训练调用、210 个候选、16 次模拟目标与 5 次部署目标回放，882 次有动作的训练尝试为 768 次成功、20 次过力、94 次超时、零次复位不可达。前六个独立种子共 5,292 次训练账本尝试，合计 4,361 次成功、158 次过力、773 次超时；审计只证明来源与账本一致，不是逐轨迹物理重放或正式性能。GPU1 队列已进入第七种子 8812007；T2 现为 6/8，主表仍为 34/36 格准入。

T2 AdaptSim 第七个独立训练种子 8812007 的同类审计 `results/arm/guided_contact_alignment/baseline/adaptsim/training/independent_training_8812007/independent_audit.json` SHA `855400f10eedfe9475d80b04ed7dc4043657bef1e74bb0a1e6505d00a65b088e` 通过，核对 24 份源码快照、21 次策略训练调用、210 个候选、16 次模拟目标与 5 次部署目标回放。882 次有动作的训练尝试中为 582 次成功、10 次过力、290 次超时、零次复位不可达；它明显比其他若干训练种子的同类账本更常超时，须保留，不能凭开发分布剔除该实例。前七个独立种子共 6,174 次训练账本尝试，合计 4,943 次成功、168 次过力、1,063 次超时；这仍不是正式主表成功率。GPU1 队列已进入最后的种子 8812008；T2 现为 7/8，正式表门控仍为 34/36 格。

T2 AdaptSim 最后一个独立训练种子 8812008 的来源／账本审计 `results/arm/guided_contact_alignment/baseline/adaptsim/training/independent_training_8812008/independent_audit.json` SHA `0670bd3d90a5ddde4d148e04c552aa962e5fef08b2818bc66ed9d4f6a251174f` 通过，核对 24 份源码快照、21 次策略训练调用、210 个候选、16 次模拟目标与 5 次部署目标回放。882 次有动作的训练尝试为 718 次成功、61 次过力、103 次超时，零次复位不可达。八个独立种子共 7,056 次训练账本尝试，合计 5,661 次成功、229 次过力、1,166 次超时；来源／账本审计不等同逐轨迹重放，也不是冻结主表性能。八实例 actor 包 `results/arm/guided_contact_alignment/baseline/adaptsim/validation/independent_actor_package_v1/report.json` SHA `671c1598b6342e24de3a61882e06132e75d581c12bee867c7d4f321e1008b923` 已核对全部种子身份及零正式请求访问。源／名义开发队列的 GPU1 启动回执 SHA `524b2f0a403b5f36993d3992f6f65f545d22a46d58eacbc60f1c2a582b96a255`，已完成前两实例独立物理审计并进入第三实例；主表门控在该项资格完成前仍为 34/36。

T2 AdaptSim 八实例源／名义 cohort 随后完成，`results/arm/guided_contact_alignment/baseline/adaptsim/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `bba366742049d6b763ee65c34c16c3b0305f7a711187312dd60727196eaf3e38` 核实 64/64 份物理 evidence 与公共参照的 64 个配对复位哈希；62 次受约束成功、2 次真实过力。T1 AdaptSim 的训练后修正通过 38 项方法 parity 测试，v2 报告 `results/_analysis/four_task_transfer/asid_adaptsim_task_adapted_parity_20261031_v2/report.json` SHA `ea112e151143373779822f6c7b7c849f6aed3718c8e195cde06a96cbf5d070fb`；八实例 actor 包 SHA `6255fbced92618446a38986cfe93e378464f73d33219504fd0599d69582e87af`。T1 的源／名义 cohort 审计 `results/arm/obstacle_transfer/baseline/adaptsim/validation/independent_source_nominal_cohort_v1/independent_audit.json` SHA `addba0354185df3e74f6f6439d6b3bc9bd546eea4fc681f5b97c23dc8829b4c1` 核实 64/64 份物理 evidence、64/64 次开发受约束成功与 64 个配对复位哈希。两格方法资格和正式八身份配置均通过，正式请求访问在准入时仍为零；这些是开发资格，不能充当正式性能。

独立 CPU 准入重审现为 36/36 格合格，无缺失单元。GPU1 正式启动回执 `results/_analysis/four_task_transfer/independent_main_table_execution_v1/launch_receipt.json` SHA `20b10fbb55dd1d97d8d218b664de01ec80da2c1889a9c8bb28a736cc21270849` 在首个正式请求前冻结 1,519 份共享源码／任务配置哈希、九方法×四任务的 232 个 actor 作业与 11,600 条预定物理结果，声明失败不替换、启动前零正式策略 episode；源码哈希只读复核无差异。门控已在物理 GPU1 开始第 1/232 个作业（T1 Cross-EBDM instance_01）；首次检查时其 40 条请求中的 29 条写入进度，但作业尚未完成独立 CPU 审计。当前不能报告四任务主表指标；全部 232 作业的物理证据独立审计和 CPU 组表仍是后续必要工作。

正式执行的首个完整作业 T1 Cross-EBDM `instance_01` 已在 GPU1 跑完 40/40 条冻结请求，独立 CPU 审计 `results/arm/obstacle_transfer/main/cross_ebdm/confirmation/independent_main_v1/instance_01/independent_audit.json` SHA `110491ff8da4b61b10878ec54a86e22f9187e2bd39583f072bb2ade252db0b2c` 为 `passed`，核实 40 份物理 evidence、正式性能标记和 verified outcomes SHA；运行器失败为零。门控随后启动第 2/232 个作业。单个实例只证明正式执行与审计链贯通，不能填充四任务完整主表或单独推断方法效果。

GPU1 正式门控连续完成并独立审计前 24/232 个 actor 作业：T1 的 Cross-EBDM、matched task-only、task-adapted FADA 各八个真实实例全部结束。只读复核逐一核对 24 份审计的 `passed` 状态、每份 40/40 条物理 evidence、零运行器失败和 verified outcomes SHA，共 960 条正式物理结果；队列已进入 T1 DIAL 的首个实例。这里尚未跨九方法／四任务完成矩阵，也未提前计算或报告主表比较。

GPU1 旧正式批次随后在第 35/232 个作业 T1 response MPC `instance_03` 触及单作业 14,400 秒上限，门控按冻结合同停止；前 34 个作业共 1,360 条物理 evidence 已独立审计，第 35 个完成 21/40 条，但第 22 次尝试留有动作前复位收据、没有结果文件，自动续跑将违反失败不替换。旧批次完整保留并在 `results/_analysis/four_task_transfer/independent_main_table_execution_v1/invalidation.json` 封存为无效、不计分（SHA `ecaaa391e23b26a85dd49ff735d5272333e9c92b8ee46335a00a3080dd265c40`）。第 35 个作业前 21 条的累计 wall time 约 13,878 秒，每条约 605--785 秒；同方法前两个实例各 40 条的累计 wall time 约 4,804／5,376 秒。超时期间 GPU1 上有其他作业并发，但不能仅凭同时发生把减速因果归于它们。旧结果不混入新的 GPU0 主表。

按新的 GPU0-only 设备要求冻结了全新四任务请求：任务本地 `protocol/independent_main_table_requests_v2.json` 使用与旧版相同的四机制组、双 mode、每组五个物理槽位和 T4 双支撑分层，200 个新 reset seed 与旧正式名单及历史记录不重叠；跨任务收据 `results/_analysis/four_task_transfer/independent_main_table_request_freeze_v2/receipt.json` SHA `8e722db8f0988c0f1d083690fdb43a646166b9b25f4587b8bce4a99a5cd08f4a` 锁定物理 GPU0 UUID `GPU-253f0813-4432-2d51-a149-3e73b38bf77a`、失败不替换及旧批次无效指针。新 `validation/independent_main_table_admission_v2/` 与 `confirmation/independent_main_v2/` 保留原八实例 actor、源／名义审计、方法忠实度、任务配置和请求结构，只改新请求、设备及输出路径。零物理全量预检通过 36 格、232 个 actor 作业和 11,600 条待执行结果；新表 CPU inventory 为 0/36 已审计，19 项定向回归测试通过。GPU0 CUDA UUID smoke 通过，尚未启动新版正式请求；旧批次保持封存。

GPU0 新版正式启动回执 `results/_analysis/four_task_transfer/independent_main_table_execution_v2/launch_receipt.json` SHA `ca8c0344392e23b4d7adba410a6532110aae004177b237869d970891e560c6f8` 已在首个请求前冻结 1,703 份共享源码／任务配置哈希、36 格、232 作业与 11,600 条预定物理结果；每个 actor 作业的超时上限为 43,200 秒，失败不替换。后台门控进程 PID 3936948 的父进程为 1，已启动第 1/232 个 T1 Cross-EBDM 作业；GPU 进程查询确认子进程 PID 3937021 位于物理 GPU0 UUID `GPU-253f0813-4432-2d51-a149-3e73b38bf77a`，首次计划与进度文件已写入任务本地 `independent_main_v2` 路径。此时尚无新版完整 actor 审计，不能报告 GPU0 主表分数。旧 GPU1 结果和无效记录不参与新版分母。
