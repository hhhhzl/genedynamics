# 四环境强 baseline 比较与论文贡献规划

更新：2026-10-16。跨任务主表的机器合同已冻结在
`configs/cross_embodiment/four_task_benchmark/protocol/main_table_v1.json`；本文较早的
按任务候选叙述若与该合同冲突，以机器合同为准。
用户已明确“dense”指**成熟、完整的强 baseline**，并新增要求：优先最近两年
顶会论文，最好有 Karen Liu 和 Guanya Shi 的工作。
先建立竞争方法在四环境中的真实表现，再决定本文方法与理论应解释的增量。
从 P3 开始的实施顺序已重新编号为 P3–P7，见
[阶段执行计划](cross-embodiment-p3-onward-plan.md)。本文件继续定义方法候选与比较协议，
阶段编号和完成门槛以新计划为准。初版尚未启动训练；目前已实际实现并运行原生/改编
训练链，最新事实见 [P3 执行记录](cross-embodiment-p3-execution.md)。资格训练不等于强方法的充分训练。

## 1. 研究顺序调整

下一阶段不以扩大有限映射枚举、升级 flow 或追加 source anchoring 权重为主线。
顺序调整为：**冻结可比较的任务 → 接入强 baseline → 建立开发集性能与成本曲线
→ 确定待验证贡献 → 修正方法和理论 → 冻结独立主实验**。

现有 P0–P2 工件用于工程参考和机制诊断。有限 affine maps 的 full/task-only
对照保留为内部消融；它不能代表连续策略、完整辨识系统或在线模型规划的
成熟竞争能力。目标最终评估继续使用独立 CPU 参考；GPU 搜索资格不自动扩展
到新的策略、任务、接触拓扑或 WBC。

文献与原生实现核验见 [baseline 筛选](cross-embodiment-baseline-survey.md)。
接口、现状与代码冲突见 [四任务协议](cross-embodiment-four-task-protocol.md)。
两份文档中的“已实现”只指明确列出的代码和资格，不代表方法已经跑过四任务。

## 2. 近期主会论文优先，作者与复现范围同时核验

主候选以 **2025/2026 的 CoRL、RSS、ICRA、ICLR、NeurIPS 等主会论文**为主，
优先纳入 C. Karen Liu 与 Guanya Shi 参与、与任务直接相关的完整方法。
2026 已录用但尚未出版 proceedings 的条目标注录用依据；不将 arXiv 年份、
workshop 或期刊论文冒充主会。按当前日期 2026-10-14，2024 年底会议仍在滚动
两年内，但只作边界候选；PMLR 出版于 2025 不会把 CoRL 2024 改成 CoRL 2025。

| 主候选 | 正式状态与指定作者 | 竞争解释与任务位置 | 接入优先级/边界 |
| --- | --- | --- | --- |
| **FADA** | CoRL 2026 已录用；Guanya Shi；作者公告与官方仓库核验 | 冻结 planner，仅用目标动作/观测适应 IDM；首要排除普通执行适应已足够的解释 | 首轮 B-log。原生是人形同身体 dynamics adaptation；T1–T3 要重新训练任务 oracle/Planner-IDM，T4 要接口资格；不是现成跨臂 checkpoint |
| **Human2Sim2Robot** | CoRL 2025 正式 PMLR；Karen Liu | 物体中心演示奖励＋目标模型 RL 是否足以跨 embodiment 完成任务；T1/T2 优先 | 首轮方法路线。人 RGB-D 视频原生监督与共同源物体轨迹改编分开；数字孪生构建和探索初始化需保留 |
| **ACP** | ICRA 2025；Shuran Song 等，非两位指定作者 | 参考位姿、virtual target、可变刚度的完整接触策略；T2/T3 | 首轮接触对照，但需要带力示范，放 C-supervised；补足只有指定作者时欠缺的直接 compliance 竞争者 |
| **Play2Perfect** | CoRL 2026 已录用；Karen Liu；作者/Stanford IPRL 公告核验 | 多对象 play 预训练＋精密装配 RL 微调；T3 优先、T2 次之 | 优先做原生/适配资格审计，再定主表。原生灵巧手 in-hand 控制是核心，固定夹持接口不能删掉它后仍声称完整原法 |
| **ASAP** | RSS 2025 正式 proceedings；Guanya Shi | B-log 目标轨迹学习 delta-action model，加入仿真后微调源策略；排除常规 simulator alignment 已足够 | 第二批关键适应对照。T4 近原作，固定臂是显式方法改编；完整目标记录、状态对齐/额外测量、delta model 和 policy finetuning 均声明 |

主事实依据：[FADA 录用公告](https://angchenxie.github.io/)、
[FADA 官方代码](https://github.com/LeCAR-Lab/FADA)、
[Human2Sim2Robot 正式论文](https://proceedings.mlr.press/v305/lum25a.html)、
[ACP](https://adaptive-compliance.github.io/)、
[Play2Perfect 录用公告](https://iprl.stanford.edu/)、
[Play2Perfect 官方代码](https://github.com/kushal2000/play2perfect)、
[ASAP 正式论文](https://roboticsproceedings.org/rss21/p066.html)。

首轮集中 **FADA、Human2Sim2Robot 的任务适配路线、ACP**，它们分别比较执行
适应、源语义驱动的目标策略学习、柔顺接触控制，分别涉及 B-log、源轨迹改编/
原生人视频和 C-supervised，不能混排
成一个同预算排名。Play2Perfect 先判断当前固定夹持任务是否能保留原方法核心，
ASAP 在验证 simulator alignment 是主要竞争解释时补齐。正式主论文收敛到
3–5 个可忠实复现的方法；候选有顶会/作者标签不等于已获得主表资格。

### 2.1 按任务补充的近期完整方法

- **SPI-Active（CoRL 2025，Guanya Shi）**：优先服务 T4 的 Go2 主动辨识。
  官方发布 SPI、主动探索和下游训练，数据衔接/硬件部分仍有缺口；不能称任意
  arm/contact 参数可开箱使用。[正式论文](https://proceedings.mlr.press/v305/sobanbabu25a.html)、
  [官方代码](https://github.com/LeCAR-Lab/SPI-Active)。
- **DIAL-MPC（ICRA 2025，Guanya Shi）**：近期强规划候选。使用相同校准模型
  优化任务参考时标 planner-adapted；原生全身 torque-level 控制另列完整系统比较，
  不用共享旧 WBC 替换其核心后冒称原生。[官方代码](https://github.com/LeCAR-Lab/dial-mpc)。
- **XMoP（ICRA 2025）**：T1 的直接跨身体避障规划参照，官方代码/模型/数据
  已公开；大规模程序生成规划数据单列，持件碰撞与动态机制仍需适配。
  [作者项目](https://prabinrath.github.io/xmop/)、[代码](https://github.com/prabinrath/xmop)。

Karen Liu 的 **SimToolReal（RSS 2026）**、**Cloak（CoRL 2026）**保留为候选：
前者仍有灵巧手自由度边界，后者需要完整视觉/VLA 及外部预训练协议，不能删掉
核心视觉 masking 后放入状态/力主表。SPARR（ICRA 2026）与 LAD（ICRA 2026）
相关性高，但本次未核到完整公开执行链，作为等待 release/忠实复现资格的候选。

### 2.2 基础参照与消融保留，退出近期主会论文排名

DR/PPO、经典 Residual RL、MuJoCo MPC、ASID（ICLR 2024）、Mirage（RSS 2024）
继续承担工程 sanity check、普通校准、控制或历史方法参照，不再占据近期论文
主 baseline 名额。**FORGE 是 RA-L 2025，不因年份新而改称顶会主会**，仍可作为
强接触参照。Direct、经典 hybrid force/position 与限力搜索保留。

同容量 task-only、deterministic transport、去 response/mode/risk、
reweighting-only、arm-only/coupled calibration 是内部消融。它们和成熟外部
方法分表，也不把所有对手限制到现有 24 个 affine maps。

## 3. 四环境如何比较

| 环境 | 近期论文候选与比较位置 | 主性能与增量问题 |
| --- | --- | --- |
| **T1 ObstacleTransfer** | Human2Sim2Robot-source-trajectory 改编、FADA/ASAP 执行适应；XMoP 与 DIAL-MPC 几何/规划专项 | 全臂/持件无碰撞完成、SE(3) 误差、间隙、时间；普通源语义 RL、几何规划与动力学对齐后是否还有增量？ |
| **T2 GuidedContactAlignment** | ACP 的示范监督赛道；Human2Sim2Robot 的对象轨迹赛道；FADA/ASAP 执行适应；Play2Perfect 资格后 | 限力定位、接触保持、滑移/停滞、力峰值；响应校准是否优于充分接触反馈与 compliance 学习？ |
| **T3 KeyedInsertionRecovery** | Play2Perfect 优先资格候选，ACP、FADA/ASAP；SPARR 代码/复现资格通过后加入 B-online | 全分母插入、卡滞恢复、重试、超力、拒绝与 timeout；预训练后常规 RL 或 compliant policy 是否已经解决恢复？ |
| **T4 MobileKeyedInsertionRecovery** | FADA/ASAP 身体适应；SPI-Active Go2 主动辨识；DIAL-MPC 原生系统/同 WBC 改编分列 | 插入且支撑满足的联合成功、滑移、WBC/身体失效、端到端时延；同控制能力下耦合校准是否有价值？ |

最终选入主表的方法必须铺满四任务；未运行不能记为失败、零分或 N/A。
v1 选定 FADA-task-adapted、ACP-task-adapted、Human2Sim2Robot-source-trajectory、
DIAL-MPC-task-adapted，并保留 ordinary response MPC、public geometry 与 public
source prior 三个强公共控制。每行显式报告数据权限，跨权限只解释资源–性能
前沿，同权限才作算法优劣结论。原生完整系统证据另列身份，改编只保留小模块
时不能代表原论文。Play2Perfect、XMoP、SPI-Active 与 ASAP 暂不称为本版
baseline；只有完成保留核心机制的四任务适配后才能进入针对性论文分析。

T1/T3 先建立比较入口；T2/T4 并行完成环境与接口资格。当前草稿/代码中的
时限、力限与恢复标签冲突先在开发集统一，再冻结正式协议。平面 caster
代理不能支持真实轮地稳定性结论，Go2+Z1 当前站立 WBC 也不等于导航/步态。

## 4. 数据权限必须分表

| 协议 | 允许的目标信息 | 训练方式 | 可支持的比较 |
| --- | --- | --- | --- |
| **A-fixed：同一短探针缓冲** | 同一计费 probe buffer、已知几何/接口，无完整任务演示 | 各自模型/context 拟合；仅在校准模型或公开开发域更新策略 | 分离辨识、策略学习与响应目标的增量 |
| **A-active：完整主动系统** | 各方法自行选择探针；总次数、总时长和失败计费一致 | 保留主动探索→辨识→模型内训练完整链 | 比较原生主动辨识系统，不能假装共享完全相同 buffer |
| **B-log：完整自采目标记录** | 计费完整 action/observation 轨迹；无 target reward、专家成功示范或纠正 | FADA 原生适应、ASAP 数据对齐等；额外状态测量逐项声明 | 同轨迹/观测权限下比较执行适应，不与 A 短探针混排 |
| **B-online：目标任务适应** | 明确计费的完整任务 rollout 与 reward，无额外示范 | 原生 target-online residual RL、finetuning 等 | 性能对目标任务 transitions/分钟的曲线 |
| **C-supervised：额外监督参考** | 目标成功示范、人工纠正、aligned actions 或外部预训练，逐项标注 | 方法原生监督训练，目标在线交互另开计费列 | 更高资源下的可达性能参照，不与 A 混排 |
| **D-oracle：封存真值诊断** | 真实机制参数、真值模型规划或高预算可行性搜索 | 仅在冻结后诊断，结果不回流训练 | 解释机制/支持，不是同预算主排名或数学上界 |

任务目标、公共训练 reward 和已知身体模型不视为额外秘密信息。
主协议隐藏的是实际执行机制和目标完整任务训练数据。
B-log 不含 reward 并不等于 A：完整轨迹的覆盖和状态信息权限也不同。
同组控制方法获得相同可用记录/测量；仅用短探针的 FADA/ASAP 版本标为
probe-adapted，不代表其原生完整数据协议。
模型内 reward 来自校准模型，不能查询隐藏目标真值作为免费 GPU 训练环境。
训练期 privileged critic 可使用其公开/校准训练模型状态；actor 部署只读
公共观测，隐藏目标参数禁止进入任何主方法的训练。

“不需要目标示范”不代表“低目标数据量”：例如大规模随机交互仍需独立计费。
已知目标名义身体上训练的策略与 frozen unseen-body 泛化也分别报告。

## 5. 公平性与可复现实现

### 5.1 同信息、同控制能力，不把强方法削成候选选择器

- 共享可测任务状态、wrench、同长度历史、已知几何、控制周期、命令边界和
  低层 decoder/WBC。Full 使用人工 phase/事件特征时，基线也获得相同特征，
  或统一移除。物理 checkpoint 中的 warmstart/隐藏 delay queue 不属于 actor 观测。
- 成熟方法可保留合理网络架构、连续动作或 chunk 和完整在线规划能力。
  参数量和模型调用量公开；仅内部消融要求同容量。
- 原生只输出受限 pose 的方法保留一个原生动作子空间版本；扩展到 13D 的
  版本标为 adapted，并检查增加 stiffness/force 自由度带来的影响。
- 同一状态 benchmark 中使用同一参考物理与成功 evaluator。先做原生算法的
  小规模复现检查，再做本项目接口适配；不混合 IsaacLab 原生分数与 MuJoCo
  改编分数。对 FORGE 的移植不要求先把整个项目搬到 IsaacLab。
- 同一物理步的力尖峰、失败与拒绝都保存。MPC 超时/fallback 规则预先固定，
  不能只在算完规划的成功试次统计。

原生 FORGE 已有 [IsaacLab 官方实现](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/forge)，
但其力/动作/身体接口与本任务不同。ASID 官方发布主要是 FI 探索目标示例，
四任务完整 pipeline 仍需适配。这些是接入工作量，不能写成已完成的复现。

### 5.2 训练与目标成本分开记录

目标访问以实际秒数、transitions 与 reset 为主轴，短 probe 次数作为补充。
原 `B ∈ {0, 5, 10, 30}` 可保留为短探针子协议；对 FADA 等原生约两分钟
动作/观测记录，应包含相应时长的数据点，不能以极短且不适合的 buffer 代表
其完整性能。每次时长、fit/validation 分配在开发集冻结，接口调整、验证失败
和尝试均计费。需要完整任务记录的方法单列记录权限，即使不使用 reward，
也不能偷偷算作短 probe；目标成功示范另列 C-supervised。

训练另报公开源/多机制数据量、模拟 transitions、模型 rollout、更新步、
调参试验数、wall time 和 GPU time。既画同模拟数据量曲线，也画训练饱和前后
的表现；不能靠一个统一小更新数让某种成熟方法始终欠训练。
开发调参预算与正式训练预算预先固定，检查学习曲线和失败原因后才解释差异。

部署报告全部预处理、source query、策略/transport、模型规划、低层控制/WBC
与同步的端到端 p50/p95 时延及 deadline miss。首次编译、稳态运行和重置分开。
GPU kernel 时间或神经网络前向时间不能代替机器人每次控制决策的完整成本。

### 5.3 机制与身体交叉，独立分层统计

先在 Panda/xArm7 做身体×执行响应交叉；接触摩擦/柔顺变化单列，再测试
预声明组合。UR5e 分别做 frozen/context-only 与模型内 few-shot adaptation。
T4 固定/释放同臂底座只用于隔离支撑影响，锁定底座不混入移动主成功率。

公开开发机制用于 reward、随机化范围、模型类和超参数选择。正式机制组合、
目标初态与扰动列表在方法冻结前封存；最终目标轨迹不回流选择或再次调参。
旧 CPU 工件不复用，P0–P2 已访问目标属于开发证据。

独立训练/校准实例是统计重复层；同一实例中的配对 reset 用于降低比较方差。
样本量由开发集方差与最小有意义效应规划。原先“5 repeats × 50 episodes/cell”
只能作为资源估算起点，不是尚未估时就承诺的完整规模。
按任务和身体分别给区间；四任务归一化汇总只能补充，不能掩盖某个任务失效。

## 6. 贡献应该看到哪些 performance

| 候选贡献 | 主图/表 | 成立所需证据 | 不足以成立的现象 |
| --- | --- | --- | --- |
| **少目标数据的迁移效率** | 受约束成功率对目标 probe 秒数/次数；达到固定成功水平的数据量 | 同权限近期强方法下更省目标数据，或相同数据下更可靠，跨独立校准成立 | 仅比未校准 direct 好；只少模型内训练样本 |
| **接触/恢复鲁棒性** | 力/身体约束下整体成功、恢复、超力与 timeout；机制 OOD | 不能由ACP/Play2Perfect、普通动作对齐和任务 residual 解释的改善 | 低平均力但更多卡滞；只报成功恢复子集 |
| **任务有效的行为选择** | task cost–模式请求完成/可行模式覆盖/响应质量 Pareto | 在预声明任务容差内改善有明确任务意义的行为，且对手获得同模式请求 | 任意保留源比例；source 距离下降但更多拒绝或失败 |
| **摊销在线执行** | 成功率对端到端时延、模型调用与 deadline miss | 与校准 MPC 接近的任务表现下更低在线成本，并与强策略 baseline 比较 | 仅 transport 前向更快；忽略 WBC 或训练费用 |
| **身体耦合迁移** | 固定/移动同臂、arm-only/coupled、UR5e frozen/adapted 分表 | 同一低层控制器与相同预算下，耦合校准带来联合任务成功增量 | 换了更强 WBC；只在平面代理或静止支撑上成功 |

**模式公平性：** 无模式请求的任务允许任意合法策略完成，不能因 task-only
选择更可靠模式就判失败。有明确绕行/接触/恢复请求时，全部方法获得同一请求，
再比较按请求完成、可行覆盖与自主目标完成。响应 discrepancy 是解释性指标，
不能替代任务表现，也不能自动把算法拒绝视为优点。

具体成功率门槛、容差和最小有意义提升由任务需求及开发集方差确定后预声明，
不预写期望优势，不在最终数据中寻找获胜指标。

## 7. 如何由 baseline 结果决定方法与理论

| 开发集观察 | 研究判断与后续动作 |
| --- | --- |
| FADA/动作 grounding 已解释收益 | 检查是否确有跨接触阶段、非线性延迟或行为选择需求；不能把普通动作对齐包装成新 transport |
| 近期接触策略或 DR-RPPO 已在其数据权限下达到同等任务表现 | 检验数据效率、未见机制与有任务意义的选择；若均无增量，收缩方法贡献 |
| SPI-Active/ASID 完整链同样有效 | 校准/主动探索归因于既有方法，论文必须证明后续行为迁移的独立价值 |
| 校准 MPC 高成功但在线成本高 | 检验摊销策略在同延迟约束下的表现；同时与充分训练的 dense policy 比较 |
| Full 比 task-only 更常拒绝或失败 | 先分离模型偏差、支持限制、选择规则与响应目标偏置；不靠只统计已部署子集修饰表现 |
| 随机/flow 版本未改善有效性能 | 采用更简单映射；额外随机性与生成架构不列为必要贡献 |

理论保留校准误差、表达支持、采样误差与停止成本的条件分析。
待贡献方向明确后，再把主要结论对准相应比较：例如 task–response tradeoff、
模型误差下的可行性，或在线规划与摊销策略的成本关系。理论不预设 Full 必胜，
也不从某个小实验成功倒推 uniform 误差界或通用安全保证。

## 8. P3–P7 阶段执行顺序

详细任务、复现等级、完成门槛、方法改进闭环和资源估算见
[新阶段计划](cross-embodiment-p3-onward-plan.md)，替代旧 P3.0–P3.3 安排。

| 阶段 | 实施内容 | 出口 |
| --- | --- | --- |
| P3 | 逐方法原生复现、独立 runtime、接口/数据合同、本任务适配资格 | 合格近期 baseline、完整链条证据和实测资源，不以 smoke 当复现 |
| P4 | 四任务开发比较，先 T1/T3，T2/T4 资格后接入 | 强竞争水平、数据/计算曲线和失效归因 |
| P5 | 连续反馈、模型/校准、可行性/停止、耦合的诊断驱动改进 | 同预算任务/效率增量及可区分的贡献消融 |
| P6 | 冻结方法与协议，新机制/初态、独立训练/校准重复 | 正式表图、区间、资源与全分母审计 |
| P7 | 根据最终方法和证据完成论文 | 主张–证据闭环，开发记录不回填 draft |

P4/P5 交错推进，不能等全部外部方法跑完才开始增强我们的方法；正式 P6 前
必须完成保留四任务的比较资格。主候选按适配性/完整性选择，不能按最终胜负删选。

**共同权限主比较补充：** B-log 不仅是 FADA/ASAP 的额外资源表。ours 与它们
获得相同目标时长、可测字段、reset 与无奖励记录权限，各自选择采集方式，
比较完整适应管线；另做完全共享记录的控制实验，分离采集与适应收益。
A-fixed 保留为短探针机制比较。与 ACP 同权限比较时提供相同示范并标 ours+demo；
否则只作不同资源下的系统 Pareto 比较，不据 A 窄权限获胜宣布整体最强。

新算法/框架按独立可删 dependency profile 安装，现有 P2 runtime 保持冻结。
目前环境管理器的 `--name` 只改变目录，尚不能代替新的 baseline 安装器。
新连续策略和物理拓扑各自资格；配置/结果仍镜像 domain/task/category/method/
level/seed，开发前缀沿用已有约定，不引入 `results/<project>/<run_id>/`。

## 9. 论文与开发记录分离

`latex/cross_ebdm` 保留论文文字、方法、一般性理论、实验协议和最终选定证据。
本轮移出 13 个实验报告附件与旧 PDF；主稿中的开发成绩、P1/P2 说明、工程
验收和执行路线同样清理。原文快照、迁出文件及 SHA 清单见
[开发报告归档](cross-ebdm-experiment-reports/README.md)。

`results` 中历史运行和原报告不改写。实验报告导出默认留在实验 `_reports`；
显式 `--report-export-dir` 也拒绝整个 `latex/`，旧 `--paper-dir` 仅兼容参数名。
后续将正式论文结果引入草稿时，应选择与论点对应的最终图表并撰写科学结论，
不自动拼接开发报告。
