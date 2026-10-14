# P3–P7：强 baseline 复现、方法改进与论文验证

更新：2026-10-14。按用户最新要求，从 P3 起重新规划：复现近期完整 baseline，
建立公平且有竞争力的比较，依据差距改进方法，验证贡献，完成论文。
本文件替代旧计划中压缩在 P3.0–P3.3 的实施顺序；P0–P2 仍保留原执行事实。
这是阶段计划。已开始按官方源码将原生 PyTorch 核心移入 solver，并建立独立
CUDA 对齐环境；最新实现范围、验证与缺口见 [solver 接入记录](cross-embodiment-p3-solver-integration.md)。
核心移植不代表原生任务定量复现或四任务适配已完成。
本轮新增原生资源/训练/续训、四任务物理接口和改编 learner 的实际实现，见
[P3 执行记录](cross-embodiment-p3-execution.md)。原生资源缺口和欠训练结果分别列出，
未以少量更新或 actor forward 将完整 P3 标为完成。
随后在用户接受NVIDIA许可后，Play2Perfect发布checkpoint已完成本机原生tight-insertion
seed42/8环境完整任务8/8资格；这是固定checkpoint的功能闭环，不是全训练复现、
原论文aggregate score或T1–T4自动准入。FADA/ACP源稳定性等缺口继续保留。

方法候选与一手来源见 [近期文献筛选](cross-embodiment-baseline-survey.md)；
任务、预算和比较设计见 [比较协议](cross-embodiment-baseline-comparison-plan.md) 与
[四任务审计](cross-embodiment-four-task-protocol.md)。后续阶段编号以本文为准。

## 1. 把“方法最强”转成可验证目标

主目标是在**同等可用信息和目标交互预算**下，最大化满足碰撞、力、时间及
身体约束的整体任务成功。再比较目标数据效率、未见机制/身体泛化、恢复和
端到端执行成本。期望四任务均达到合格强对照水平，并在预声明的核心性能
或性能–成本前沿上取得可重复增量；这是一项目标，尚不是实验结论。

目标行为更接近源行为是机制指标。只有任务表现满足预声明容差、行为选择
有任务意义时，才将其列为独立贡献。单纯增大网络、增加噪声或采用 flow
不作为优势证据。最强/SOTA 的表述限定于实际验证的方法、任务、身体、数据
权限和资源；不以四任务平均值掩盖某项失效。

当前方法的有限 phase prior 与 affine maps 保留为开发参照。它们尚不代表
完整的连续闭环 transport；P2 的 GPU admission 也不覆盖新神经策略、源查询、
T1/T2/T4 或新的 WBC。先补比较能力，再决定如何增强方法。

```mermaid
flowchart LR
    P3[P3 原生复现与比较资格] --> P4[P4 四任务开发比较]
    P4 --> P5[P5 方法改进与贡献消融]
    P5 -->|开发集复测| P4
    P5 -->|方法与协议冻结| P6[P6 独立正式验证]
    P6 --> P7[P7 论文定稿]
```

P3 按任务/方法路线放行；P4/P5 交错推进，不等待全部任务、全部论文都做完才开始改进方法。
T1/T3 先建立对比，T2/T4 的环境与控制资格并行推进。

## 2. 阶段总览与完成门槛

| 阶段 | 主要工作 | 可审阅交付 | 完成门槛 |
| --- | --- | --- | --- |
| **P3 复现与资格** | 原生复现、资源隔离、公共评估、适配审计 | 逐方法复现卡、入口、checkpoint/训练证据、资源 profile、适配差异 | 核心链条可执行且指标可解释，原生/改编及缺口明确；不是 import/smoke 即通过 |
| **P4 开发比较** | 近期强法在适合任务中充分训练；比较数据/计算曲线 | 方法×任务×权限矩阵、开发成绩、学习曲线、失效归因 | 每任务有合格近期强对照及连续策略/规划参照，关键性能差距可定位 |
| **P5 方法改进** | 连续反馈、模型/校准、可行性和耦合的诊断驱动迭代 | 可复现方法版本、同容量消融、贡献证据、理论修订 | 在共同权限下出现稳定的任务/效率/Pareto 增量，至少一个核心机制获得归因证据 |
| **P6 独立验证** | 冻结协议与方法，新目标集和独立重复 | 正式表图、区间、资源、失败及独立审计 | 全分母、分层统计、无最终测试回流，结果支撑明确范围内的主张 |
| **P7 论文** | 按最终方法和证据重写论点、实验、理论与局限 | 可编译 draft、正式图表、复现索引 | 每项贡献有最终证据，开发记录与论文分离，结论与实际范围一致 |

## 3. P3：先让 baseline 完整且可信

### P3.0 选定复现对象与合同

主池继续优先最近两年顶会、Karen Liu 与 Guanya Shi 的相关工作。
首批接入路线为 FADA、Human2Sim2Robot、ACP；Play2Perfect 先做 checkpoint、
核心机制与资源资格。ASAP、SPI-Active、DIAL-MPC、XMoP 按任务问题补齐。
主论文最终收敛到 3–5 个近期完整方法，每个核心竞争解释至少有一个合格对照。

选择依据是任务相关性、方法忠实度、可取得资源及竞争解释覆盖，不能在查看
最终胜负后删除强对手。每个任务至少一个适合的近期完整方法，另有充分训练
的连续 task policy 或模型规划参照；不要求每种异质方法硬铺四任务。

每条路线建立复现卡，至少写清：

- 正式论文/录用状态、代码 commit、许可证、已取得资产和 checkpoint SHA。
- 原生身体、观测、动作、控制器、训练/适应/评估阶段及核心机制。
- 原生复现入口、项目适配入口、信息权限、数据与计算预算、选择集与评估集。
- 逐项接口差异，保留/替换/缺失的模块，当前资格和未完成原因。

| 方法 | 必须保留的核心 | 首批复现资格与边界 |
| --- | --- | --- |
| **FADA** | oracle → Planner-IDM DAgger → 冻结 planner → IDM 目标数据适应 | 先核原生 checkpoint/训练/适应链；原生完整无奖励记录归 B-log，约 120 秒的数据点不能省；arm/task oracle 要重建 |
| **Human2Sim2Robot** | 示范处理、数字孪生、物体轨迹 reward、探索初始化、RL | 先确认原生数据/资产能实际取得；源机器人轨迹替代人视频是 adapted，不是原论文成绩复现 |
| **ACP** | 带力示范、compliance/virtual-target 标注、策略与控制器 | 原生视觉＋力数据归 C-supervised；state-only 或模拟示范标 adapted，不能只留固定刚度控制 |
| **Play2Perfect** | dexterous play 预训练 → 装配微调，保留 in-hand 能力 | 先用公开 checkpoint 验证原生任务；固定夹持删掉核心时只作受限改编或原生扩展参考，不为凑方法扩大整个 benchmark |

一手实现入口：[FADA](https://github.com/LeCAR-Lab/FADA)、
[Human2Sim2Robot](https://github.com/tylerlum/human2sim2robot)、
[ACP](https://github.com/yifan-hou/adaptive_compliance_policy)、
[Play2Perfect](https://github.com/kushal2000/play2perfect)。
资产未实际取得时记 `assets_pending`，其他路线继续；不将表单入口视为资源已具备。

### P3.1 可装可删的环境与运行基础

保留现有 `.runtime/cross_ebdm/`；新 baseline 按依赖族建独立 runtime。
官方源码放在仓库根 `baselines/<upstream>/`，固定 commit，保留可删除、可重建
的参考 checkout。移植实现纳入 `genedynamics/solvers/single/<method>/`，
训练/适应辅助模块放 `genedynamics/learning/baselines/<method>/`。
安装包、模型/数据缓存、编译缓存、临时文件和 native 库归独立 `.runtime/`；
结果、必要 checkpoint 与复现快照留在配置对应的结果树中。
例如 FADA 的 IsaacSim、MuJoCo、inference 环境分开，避免覆盖 P2 依赖。

现有 `scripts/research/environment.py --name` 只换目录名，安装仍使用固定
Cross-EBDM requirements。新建的 `scripts/baselines/environment.py` 已提供
独立 `core-pytorch`/`tasks-pytorch` profile、CUDA/CPU 安装、解析版本快照、缓存隔离运行和所有权清理。
ACP 视觉/力与 FADA mjwarp 另有独立 native profile。IsaacGym、IsaacSim 与硬件
原生部署仍须各自验收，不能把这些 profile 当成所有原生系统的安装器。

运行层仍需补齐独立原生进程入口、统一轨迹/预算导出、原生发布 checkpoint
加载、资格报告和中断恢复；按方法保留原生 learner，不先重写为一个削弱版
通用算法。第三方训练库与项目公共 evaluator 通过明确 adapter 连接。

### P3.1a 源码移植与 solver 的边界

用户选定先保留原生 PyTorch，再做对齐，之后按实际需要考虑 JAX。
源码下载/校验入口为 `python3 -m scripts.baselines.sources fetch|verify`，
固定来源记录在 `requirements/baselines/sources.lock.json`。
移植文件保留许可证、原文件路径、SHA 和修改说明。

FADA、Human2Sim2Robot、ACP 作为闭环策略 solver，使用原生模型和 checkpoint
语义，不继承候选采样 planner 伪造算法身份。`PolicyExecutionController`
只连接 task-owned public observation 与真实执行，记录实际施加的动作。
ACP 的 reference pose/virtual target/stiffness 保留结构化语义；任务控制器
资格完成前不强行转换成当前 13D skill 后宣布原法复现。

`scripts/baselines/qualify_core.py` 对固定上游源码做同权重、同输入、同随机数
的核心数值对齐，分别在 CPU/CUDA 执行。源码核心、发布 checkpoint 原生闭环、
训练链、原论文定量复现、项目任务资格分别记录，不能相互替代。

配置路径继续按 domain/task/category/method 组织。以下是待创建的设计示例：

```text
configs/arm/obstacle_transfer/qualification/fada_adapted.yaml
configs/arm/obstacle_transfer/validation/fada_adapted.yaml
configs/arm/keyed_insertion_recovery/validation/acp_adapted.yaml
configs/arm/keyed_insertion_recovery/validation/cross_ebdm_continuous.yaml
```

结果镜像配置的 domain/task/category/method 与 level/seed，开发前缀沿用
已有 `_development` 约定；不新增 `results/<project>/<run_id>/`。
原生人形/灵巧任务按真实 domain/task 记录，不能挂到 T3 后当 T3 成功。

### P3.2 先定义、按任务冻结比较契约

先建立版本化临时契约供 adapter 开发；资格修订后、P4 计分比较前，
冻结对应任务的契约，后续变更须新版本并重跑该版本的比较。契约包括 public observation、动作/原生子空间、控制时钟、reward、
task cost、独立成功 evaluator、碰撞/力/身体约束、模式语义和数据权限。
先解决草稿/代码的阈值冲突，在开发集验证可达、可恢复和源行为支持。
物理 checkpoint 内部状态不自动成为策略可测信息。

原生视觉或专属 WBC 不在共同接口内时，完整系统单列；同状态/同 WBC 的
adapted 版本另列。共同任务比较必须使用相同目标物理与判据，训练模拟器、
后端误差和模型验证机会显式登记并对称开放。

P3 按方法/任务路线分别验收。T1/T3 达到各自门槛即可进入 P4/P5，T2/T4
继续独立资格；四任务全部资格与开发比较齐全是 P6 前置，不阻塞先行任务。
每条合格路线交付原生/适配证据、对应任务契约、可删环境、运行入口与资源估算；
资格阶段不宣布 baseline 已充分训练或四任务比较已完成。

### P3.3 两道资格，三种证据等级

先做**原生资格**，再做**本任务适配资格**。每条路线依次检查：

0. 移植核心对固定上游源码的输出、梯度/更新及 checkpoint 对齐，CPU/CUDA
   设备行为可核验。这是先决检查，尚未通过下面的原生任务资格。

1. 发布 checkpoint 在原生任务闭环运行，输入归一化、动作和公开指标可解释。
2. 核心训练/适应链能执行完整更新、保存/加载，optimizer/RNG/归一化可恢复。
3. 一个项目任务、一个身体和机制的完整 episode，经独立 evaluator 重算。
4. 多 reset、历史状态、终止/partial reset、重排及预算记录正确。
5. 两种 workload 的端到端资源 profile，为后续规模提供实测依据。

分别记录 checkpoint 验证、训练链验证、原论文定量复现。少量更新只证明链条
可执行，不等于达到论文成绩；达到原生数字也不证明本任务改编忠实。
进入 P4 的版本必须保留核心机制、可正常学习并有可解释评估，未定量复现的
部分明确披露，不能冒称已复现论文优势。

新神经 source-query/prior/response/transport 和新物理拓扑各自做设备资格。
CPU target 保持统一参考；GPU 可作合格的近似训练模型，但不能自动替代目标。

## 4. P4：建立真正有竞争力的开发比较

### P4.0 不把强方法隔离到补充表后宣称最强

保留已有六类权限；增加明确的**共同权限主比较**：

| 比较 | 目的与规则 |
| --- | --- |
| **A-fixed，共享短探针** | 同一计费 buffer，隔离校准/适应模块。短 probe 版 FADA/ASAP 明确是 adapted |
| **B-log，共同完整记录权限** | ours 与 FADA/ASAP 都可在相同目标时长、观测、reset 和执行约束下自行采无 reward 记录；ours 可选短探针，但不禁止对手原生采集；比较完整适应流程 |
| **B-log，共享记录控制实验** | 再提供完全相同的可用记录，分离“采集更好”与“学习/适应更好”；额外状态测量对同组方法一致 |
| **C-supervised，共同示范或系统前沿** | 与 ACP 比较时给 ours 相同可用示范并标 ours+demo；否则只比较不同资源下的系统性能–成本前沿，不写同预算胜出 |
| **A-active/B-online** | 分别保留原生主动探索和目标 reward 训练权限，全部采集/失败计费，不混入 A 或 B-log |
| **D-oracle** | 封存目标只在冻结后做真值/可行性审计，不进入 actor、拟合、选模型或调参；公开开发机制的真值误差诊断另列 |

同秒数不是同信息：来源、状态测量、示范、完整任务覆盖与外部预训练都要列。
协议内比较与跨资源系统比较同时存在，不能只在最窄权限组寻找容易的对手。
所有扩展版本单独命名，不能用有示范的 ours+demo 证明原来的无示范主张。

### P4.1 先 T1/T3，再扩展已合格 T2/T4

| 任务 | 首要强对照及补充 | 主要开发问题 |
| --- | --- | --- |
| **T1 持件绕障** | Human2Sim2Robot 任务路线、FADA-adapted；XMoP/DIAL-MPC 按几何与规划缺口补齐 | 无碰撞成功、间隙、位姿、跟踪与时序；是否只是几何/动作对齐？ |
| **T2 接触对齐** | ACP、FADA/ASAP-adapted；适合的源对象轨迹方法 | 限力定位、滑移/接触丢失、停滞；强 compliance 后还有何增量？ |
| **T3 插入恢复** | ACP、FADA/ASAP-adapted；Play2Perfect 保核心资格后；SPARR 资源齐全后 | 全分母插入、卡滞恢复、超力、拒绝/超时；控制表达还是模型/选择瓶颈？ |
| **T4 移动插入** | FADA/ASAP、SPI-Active，DIAL-MPC 系统/同 WBC 改编分列 | 操作＋支撑联合成功；arm-only 与耦合模型、控制器贡献能否分开？ |

每方法保留原生合理容量、连续动作/chunk 和历史，不锁到 24 个 maps。
同时报告相同模拟数据/计算预算的曲线和合理训练机会后的表现。
开发调参预算提前固定，检查学习曲线、训练稳定性和独立重复，避免把欠训练
当算法失败。原生 checkpoint 不能替代在本任务的训练/适应质量证据。

### P4.2 形成能指导改进的结果

逐任务/权限输出成功–目标秒数、任务表现–模拟训练量、成功–部署时延曲线，
以及力/碰撞/身体失效、拒绝、超时、恢复和最差机制组表现。
给基线与 ours 相同模型验证机会、可观测输入和规则；失败不从分母中删除。

将差距分成：源支持、身体几何/控制、校准/模型偏差、数值后端、策略表达、
优化、选择/接受规则和在线计算。保留同容量 task-only、普通动作适应及模型
规划，确定哪个竞争解释已足够。开发重复用于稳定性与方差规划，不称正式胜出。

P4 交付不是一张最好数字表，而是可信的竞争水平、成本曲线和瓶颈排序。
T1/T3 有明确诊断后即可进入 P5，T2/T4 继续接入；正式 P6 前四任务都需完成。

## 5. P5：把我们的方法做强，并证明为什么变强

当前优先级是连续反馈与合理选择，数值/模型诊断同步。连续 transport 也可能
只是常规 residual policy；立项与消融需解释其超过普通动作适应的机制。每轮只改变能归因的
模块，固定该轮公共数据、开发 split 和资源合同；保留失败版本与消耗。

| 优先级 | 待实施研究 | 必需对照与判断 |
| --- | --- | --- |
| **0 模型与后端** | 配对 CPU/GPU 模型、同初态/噪声/命令下的预测、事件和候选排序；校准参数与后端误差分离 | 相同模型/选择的 CPU 搜索；名义/拟合模型；公开开发机制真值仅用于误差诊断。封存目标真值不回流。先判断优化器是否在优化错误响应 |
| **1 连续闭环 transport** | 用可测 pose/wrench、自己的命令历史、源技能、身体/校准 context 输出连续有界 action/chunk 修正 | 同结构/数据/优化预算 task-only、无历史、普通动作适应、有限 affine。GPU/source query 独立资格 |
| **2 可行性与停止** | 减少少数训练 reset 全成功过滤导致的脆弱拒绝；用独立模型验证、不确定性与共享 recovery/fallback 选择 | 同接受规则和验证预算的 task-only，point/ensemble，拒绝成本与全分母比较；不放宽物理阈值 |
| **3 校准与响应模型** | 有证据时比较多步/event-aware response、posterior/ensemble、固定/主动干预 | 同 buffer 比模型，同总交互比完整探针流程；独立访问支持验证，不能只比较训练 RMSE |
| **4 身体耦合** | T4 增加 base/support–arm/contact 联合响应与校准 | 同 WBC、传感、actor 容量、数据预算，只改 arm-only/coupled；身体失败进入整体任务 |
| **5 模式与随机性** | 有任务需要时比较条件模式、重加权、确定/随机 transport；flow 只有表达不足时升级 | 请求同时给所有方法；无请求允许任意合法成功模式；保真不替代任务性能 |

先核源 prior 的成功质量和支持。只有其表达不足被明确定位时才升级源模型；
共享源数据，单列 prior 改变，避免把更好源先验误记成 transport 增量。
不默认同时开发新 prior、大型 world model、flow 和新 WBC。

### P5.1 贡献必须有可区分的消融

| 拟主张 | 归因实验 |
| --- | --- |
| 校准有效 | 同 actor/优化器：名义与校准响应；固定 buffer 与主动采集分开 |
| transport 有效 | 同可用模型和预算：完整方法对执行适应、目标模型 RL 和规划 |
| response anchoring 有效 | Full 与同结构 task-only 仅改变响应目标；源、历史、context、容量、训练、接受规则相同 |
| 历史反馈有效 | 同容量移除历史/当前 wrench，公共低层反馈保持一致 |
| 耦合有效 | 同 WBC、预算和传感：arm-only/coupled，操作与身体联合评估 |
| 摊销有效 | 与合格近期规划和强神经策略比较完整 p50/p95 延迟、deadline、模型调用及任务表现 |
| 模式有任务价值 | 预声明请求/约束下的完成与可行覆盖；不能只保持任意源比例 |

理论跟随这些问题，继续分离校准、后端、采样、表达与停止成本。
任务预算/响应保证仍需相应假设；少量模型验证全通过不能充当 uniform 认证。
当 task-only 一直更好时，研究 anchoring 是否服务任务、如何条件化或调整目标，
不能用新名字或更小成功分母维持主张。

### P5.2 进入正式验证的门槛

- 合格近期强法有合理训练曲线，关键失效与后端/接口问题已解释。
- 四任务开发比较齐全；在共同权限下出现独立重复支持的核心性能/效率增量，
  四任务其余项达到预声明可接受水平；至少一个关键机制有同预算消融证据。
- 任务阈值、主要指标、最小有意义效应、非劣容差和样本量规则在最终测试前固定。
- 方法、基线、数据处理、选择/接受、fallback、checkpoint 选择与调参均冻结。

门槛未通过则按失效类型回 P4/P5：模型规划成功而 actor 失败时改摊销/反馈；
CPU 模型也无可行策略时改控制支持；源语义无增量时调整贡献。
不以预算将尽或跑完脚本作为方法研究成功的标志。预算耗尽只记录进展和未解项。

## 6. P6：独立正式实验与贡献确认

封存新的机制组合、初态、扰动和适用身体；P0–P5 已访问目标均属开发集。
正式比较集合、任务权重、预算点、主要终点、区间算法和停止规则预先冻结。
固定已知形态上的适应与 UR5e frozen/context-only 泛化分别报告。

以独立训练/校准实例为重复层，以配对 reset 降低方差；同 checkpoint 的许多
episodes 不能冒充许多次校准。样本量由开发方差、配对效应和最小有意义提升
确定，再根据资源 profile 固定。原 5 repeats × 50 episodes/cell 仅是估算起点。

CPU 参考 evaluator 独立重算任务成功、子步峰值、事件和全分母成本。
四个任务均保留。全部适用请求中的拒绝、支持失败、超力、超时、身体失败计入结果。
真正不适用的算法/任务先在资格矩阵中声明，不编成零分，也不只取有利任务
宣称全四环境领先。汇总权重预声明，并保留每任务、身体和机制组结果。

正式表图至少包括：

1. 四任务的受约束整体成功、任务成本、失败类型与置信区间。
2. 共同权限下目标数据效率曲线及训练/预训练资源。
3. 未见机制/身体、恢复与最差组表现。
4. Full/task-only 和关键模块的贡献消融。
5. 端到端部署时延、deadline、模型调用和资源前沿。

非劣需要相应区间相对容差的证据，不能把“不显著”解释成相等。
最终数据不回流修改方法后继续称同一套确认；新开发版本需要另留新确认集。
若结果不支持优势，保留并相应修订主张，而不是继续选择有利统计口径。

## 7. P7：以最终方法和证据完成论文

先在 P3/P4 更新实验协议与 related work，在 P5 收敛方法与理论，在 P6 后才
写正式成绩和优势。主线围绕最终得到证据的 1–2 个核心贡献，不预定所有
模块都必须成为创新。

论文写清近期方法的原生/改编、资源权限、四任务职责、基线和因果消融。
`latex/cross_ebdm` 继续只承载 draft、科学协议与最终选定图表；复现日志、
失败调试、版本台账、GPU 工程验收留在 `docs/development` 与结果树。

当前机器的目标参考是仿真。真实 sim-to-real 结论需要相应实机验证；若没有
硬件数据，标题和结论按实际 cross-embodiment/sim-to-sim 证据定位。
实机验证可设独立 P6-R 里程碑，不能由软件 GPU 资格替代。

## 8. 资源、工时与恢复

两张 24 GB GPU 默认运行两个独立 worker/重复，不把它们当作一个 48 GB
设备，也不默认 DDP 已验收。Isaac Kit 按其环境约束独占对应卡；CPU target
并发实测，避免与 data loader/reset 争抢 CPU。新 runtime/资产先测展开与缓存
占用，再设限额，不能机械复用 P0 的 30 GiB runtime/10 GiB 每配置默认。

原生 checkpoint/少量数据资格优先，先完成一个完整采样–更新–保存 block 与
一个长 episode 的 profile，记录冷启动、稳态、reset、IO、有效/padding steps、
显存、磁盘增长和部署延迟，再承诺训练规模。
OOM 优先调环境并行数/minibatch；改网络、历史、horizon 或核心模态要登记方法版本。

用实测完整 block 估算每个 job：

\[
T_j=T_{\mathrm{setup}}+T_{\mathrm{compile}}+
N_{\mathrm{blocks}}t_{\mathrm{block}}+
N_{\mathrm{eval}}/r_{\mathrm{CPUeval}}+T_{\mathrm{IO}}.
\]

block 已包含采样与更新，不能重复累加采样时间。共享源预训练、外部预训练、
开发调参和正式重复分别记账；GPU 工时由实际卡占用累加，wall time 由排程和
CPU 实测吞吐估计，不简单除以卡/核数。存储按轨迹、checkpoint 和缓存估算。
P3 profile 后给乐观/保守范围，目前不凭空承诺若干周或固定全部矩阵规模。

中断前 durable pending，完成时 atomic commit；记录 optimizer、scheduler、
RNG、归一化、采样器和原生支持的环境状态。无法精确恢复时标新尝试，原尝试
和成本保留，不能免费重放。checkpoint 采用 last/best/预声明里程碑，日志与
磁盘限额随 profile 固定。

## 9. 下一次执行的具体队列

1. 完成 P3.0 逐 baseline 复现卡及资源取得清单，冻结首批范围与权限。
2. 实现 dependency-profile 隔离环境、原生运行入口及统一预算/evaluator 导出。
3. 并行做 FADA、ACP 原生资格与 Play2Perfect checkpoint/资源资格；
   Human2Sim2Robot 资源未取得时保留等待，source-trajectory 改编独立命名推进。
4. 冻结 T1/T3 比较契约，完成首个本任务 adapted baseline 闭环和 profile；
   T2/T4 进行各自场景、身体和控制资格。
5. 开始 P4 学习/预算曲线；取得可解释差距后进入 P5 连续反馈与选择改进。

这一步的目标是可信而可扩展的竞争比较，不是立刻跑四任务全部主表。
后续以阶段门槛推进，减少候选扩张，把主要投入用于训练质量、方法改进与
独立验证，使最终优势经得起强对照。
