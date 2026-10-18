# 四任务比较协议：接口、权限与验收设计

本文是研究设计与代码审计。审计日期为 2026-10-14；P3 的新实施与实验事实单列于
[P3 执行记录](cross-embodiment-p3-execution.md)，不将资格样本写成正式比较。
旧 CPU 工件按研究决定不再保留；旧文档中的小样本成功数不构成本机结果，
也不能作为正式统计重复。当前机器的 P0–P2 开发证据保留在对应
`docs/development/` 与 `results/_development/` 中。论文草稿用于定义研究问题、
方法、实验协议和最终选定证据，不承担开发日志索引。

baseline 文献选择与实施决策见 [比较计划](cross-embodiment-baseline-comparison-plan.md)。本文先固定它们必须共享
的环境、数据权限和比较口径；初版发布时尚未启动训练，现在已有独立记录的训练链和物理资格。
文献主池已按用户偏好改为 2025/2026 顶会工作，优先 Karen Liu 和 Guanya Shi；
各论文的原生身体、监督、动作与完整流程不能被本协议的共享接口悄悄删减。

## 1. 四项任务的证据角色和实际状态

| 环境标识 | 任务与需要隔离的问题 | 身体阶梯 | 现有实现与当前资格 |
| --- | --- | --- | --- |
| `obstacle_transfer` | 持方向性负载绕三维障碍进入工位；分离已知形态几何与未知执行增益、延迟、负载响应 | Panda 源，xArm7 主目标，UR5e 跨 DoF 扩展 | P3 已有公开接口、默认 10 s 内成功源轨迹和精确 CPU 重放；Human2Sim2Robot 改编 PPO 在名义 xArm7 模型训练并执行真实目标。充分训练的比较仍待完成 |
| `guided_contact_alignment` | 沿真实三维导面接触、滑动和对齐；检验自由空间跟踪相近时，摩擦／柔顺变化是否需要新的接触策略 | Panda → xArm7，之后 UR5e | P3 已有默认 20 s、总力25 N/侧向10 N合同的双路线成功执行与精确 CPU 重放；新语义 GPU admission 和近期训练基线仍待资格 |
| `keyed_insertion_recovery` | 匹配键槽插入、卡滞与恢复；分离普通动作补偿、接触控制、任务优化与分布目标 | Panda → xArm7；UR5e 分别做 few-shot adaptation 与冻结泛化 | P0–P2 有限phase/affine和CPU/MJX不等价结论保留；P3 新增成功源恢复与严格重放、公共历史/原生控制接口，以及三条显式改编 learner。近期强基线表现尚待充分训练 |
| `mobile_keyed_insertion_recovery` | 保持 T3 connector、插入判据与恢复语义，检验身体支撑与接触反作用耦合 | 固定／平面 xArm7 诊断；真实轮式候选；Go2+Z1；G1 视接口另行接入 | P3 新跑真实floating Go2+Z1/OSQP stance WBC的插入和两模式卡滞恢复，严格CPU重放一致；导航/步态、真实轮式、新GPU拓扑和近期T4基线训练仍待资格 |

相关代码相对仓库根路径：

- `genedynamics/tasks/transfer/{runtime,specs,protocol}.py`：13D skill、执行时钟、任务与机制合约。
- `genedynamics/envs/domains/manipulation/transfer_workcell/queries/`：源语义查询和支持标记。
- `genedynamics/tasks/transfer/contact.py`：contact profiles、终止判据与 backend admission。
- `genedynamics/tasks/transfer/{behavior,contact_reference}.py`：显式行为 phase、reference table 与反馈 decoder。
- `genedynamics/envs/domains/manipulation/transfer_workcell/`：工位、数值后端与身体组合。
- `scripts/paper/cross_ebdm/`：统一 paper 入口；任务、物理和算法实现不放在脚本层。
- `configs/arm/{obstacle_transfer,guided_contact_alignment,keyed_insertion_recovery}/` 与 `configs/quadruped/mobile_keyed_insertion_recovery/`：P3 已建立四任务 qualification 配置；结果继续镜像 domain/task/category/method/level/seed。

旧 legacy MJX 能运行，不等于新的 spatial guide、TCP wrench 或移动组合已合格。
`ContactRunner` 目前对新 JAX 接触语义只接纳固定 Panda/xArm7 的 matched keyway；
Go2+Z1 显式限定 NumPy/CPU。新的 topology、接触 profile、controller 和 source
query 均需各自验收。

## 2. 先冻结共同接口，再选择算法实现

### 2.1 观测、历史与信息权限

四任务的主比较使用状态／力观测接口，暂不引入视觉作为额外变量。具体字段、
单位、坐标、采样率、噪声模型和缺失值 mask 在开发资格后写入配置与 schema。
拟定的公共观测为：

| 字段组 | 允许提供的内容 | 公平性要求 |
| --- | --- | --- |
| 任务语义 | TCP 与工件相对工位的位姿、twist、目标位姿、已知工件／夹具几何 | 全部方法使用相同可测状态；不能把隐藏摩擦、真实有效增益或未来可行性标签放入观测 |
| 身体与执行 | 目标 encoder q/qdot、已声明的关节边界／形态、历史发出的指令、可观测的饱和与执行余量 | 不同 DoF 用 mask 或身体专属编码；不要求将 Panda 与目标关节向量直接对应 |
| 接触 | TCP 上同一坐标定义的 measured wrench、允许接触／轴向支撑的可测事件、接触历史 | command force 与 measured force 分开；真值内部接触标签若无法由声明传感器获得，只用于评估 |
| 时间与行为历史 | 经过时间、同长度观测／动作历史；由公共观测计算的 phase／事件可作为共享特征 | Full 使用人工 phase memory 时，dense baseline 和受控消融也获得同样事件信息，或统一去掉它 |
| 移动身体 | 基座／躯干位姿和 twist、支撑／滑移观测、WBC projection／saturation／failure | 同一 body 的所有方法使用相同 WBC 和反馈；去掉高层身体条件时，公共低层 WBC 仍保留反馈 |

源 prior 只接收其冻结时声明的任务语义与合法 source virtual state；目标 joints、
目标 controller memory 和身体真参数不能直接塞进 Panda query。现有
`SourceQueryAdapter` 对 ObstacleTransfer 只支持 phase 0；`ContactSourceQuery`
使用 TCP pose/twist、normal force 与 contact phase，并检查源接触兼容性。
合法 source query 的覆盖需要单独验收，不能将 IK 成功等同于接触历史可复制。

现有代码中的全部 carry 不自动成为“可测观测”：MuJoCo 内部 warmstart、隐藏
delay queue、真实模型参数只用于物理复现或审计。需要维护 public observation
与 simulator checkpoint 两个明确定义的结构。

P3 `CpuTaskAdapter` 的 `previous_action` 和 `reference_position/rotation` 来自公开发出的
有界指令，在隐藏 actuator delay 之前；plant 真正延迟动作和积分参考只供执行与审计。
历史 P2 `cpu_mechanism_validation.policy_step` 仍读取 backend reference carry，不能直接
加入同信息主比较。P4 前必须将 ours 的策略/源查询也迁入相同 public adapter，并重跑
开发比较；历史 P2 结果保留原权限事实，不作为公平新排名。

### 2.2 动作、低层控制与策略容量

公共 skill 维度为 13：world spatial twist 的 6 维、Cartesian log-stiffness
ratio 的 6 维、压缩法向 force 的 1 维。坐标变换、限速／加速度／刚度变化率、
force rate、投影和低层控制由共享 adapter 负责。当前常见时钟为 physics
2 ms、control 20 ms、skill horizon 10；它们是现有实现起点，四任务与不同
body 的最终适用值还需各自资格。

T1 无需法向接触力时统一屏蔽该通道。T4 不把底盘速度或步态标签新增为某一
方法的独占高层动作：相同 13D skill 由同一 WBC 实现。身体 probe 可以使用
预先声明的专门辨识接口；部署时的 body-intent／站位模块如存在，应对全部
方法固定并公开。

当前 T3 的 `PhaseGaussianPrior` 是 8 类 phase×mode 条件到 5D reference 的
小模型，配合反馈 decoder 与有限 affine map 搜索。它不是完整 dense policy
研究结果。强基线应能够连续输出合法 13D action／chunk，并接收公共历史和
任务目标；不能将所有外部算法限制在现有 24 个 map 后声称比较充分。

同容量 task-only、去掉 response matching、deterministic transport 等用于
归因；不同原生结构的外部方法可以保留合理架构，但需披露参数量、训练数据、
计算与接口改编。所有方法共同遵守碰撞、力、时间和身体约束，不能靠放松
adapter 或延长超时取得优势。

### 2.3 reward、任务 cost 和成功判据分开

拟冻结三个对象，而不是用单个训练 loss 代替 performance：

1. **可观察的公共任务 reward／cost**：目标进展、姿态、时间、可测 force、
   碰撞／执行风险及必要恢复事件。dense RL 与模型规划均可使用；具体归一化、
   权重、terminal shaping 和累计方式只在开发集调节后冻结。
2. **执行成功与安全判据**：由独立 evaluator 从真实轨迹重算，统一用于所有
   方法；停止、拒绝、超时、source-support 失败和身体失败计入全部 reset 分母。
3. **方法特有响应／分布目标**：source anchoring、response discrepancy、mode
   coupling 等；它们可以成为 Full 的训练项，但不自动并入公共任务成功定义。

当前脚本中的 `.5/.25/.25` task cost、mode/time 加权 joint cost 和 P1/P2 的
response signature 属于开发实现，不能无审计地固化为新的四任务 benchmark。
若主张“任务成本容差内改善响应”，需要预先固定 task cost、容差、停止成本
以及统计比较方式。成功条件上的 final force band 与全程 peak force limit
是两个不同约束，必须分别定义。

## 3. draft 与代码数值冲突：开发集资格后统一

下表是审计时的起点与缺口，不是新的最终阈值。不得在看过最终测试后，选择
对某一方法有利的版本。最终应由已冻结 config/schema 生成论文协议表。

| 项目 | draft 起点 | 当前代码／开发入口 | 需解决事项 |
| --- | --- | --- | --- |
| T1 timeout | 10 s | `ObstacleTransferConfig` 默认 10 s；`validate_obstacle_mechanism.py --deadline` 默认 30 s | 重新验收冻结源 prior 在拟定共同期限内的支持；不能混合 30 s 历史结果与 10 s 新表 |
| T1 几何／负载 | L 件包围盒 120×80×20 mm；质量 0.3–1.0 kg；工位包围盒 0.8×0.6×0.6 m | `workcell.py` 构造实际碰撞代理／障碍；config 默认名义 payload 0.5 kg | 逐项导出实际 XML 尺寸／质量、collision exclusions 与共同可达域；draft 尺寸尚不能当作已实现资产规格 |
| T2 force／timeout | 合力 20 N、10 s | `spatial_guide_config` 继承总力限 25 N、timeout 20 s，侧向限 10 N | 选定可执行版本并对所有方法一致；不能静默更改或将不同协议汇总 |
| T2 初态／形状 | 工件 80×50×20 mm、偏移 ±10 mm／±10°、单边间隙 2–4 mm | 当前 spatial guide 为 12 mm tip、16/18 mm 通道、双轴 0.1 slope；reset task noise 0.3 mm | 这是不同的场景参数化；需确定姿态变化、reset 抽样及真实可恢复区，而非宣称完整实现 draft 的扰动范围 |
| T3 深度／保持 | 名义入座 15 mm、成功深度 ≥14 mm、2°、保持 0.3 s | `research_contact_config` 的目标位移为 16 mm；evaluator 检查目标 TCP 欧氏距离 <2 mm、2° 与 force band，保持 0.3 s | 当前实现不是独立的“插深 ≥14 mm”标量判据；需定义 socket frame、工具尖端深度与完整姿态／包络条件并交叉审计 |
| T3 工件／扰动 | keyed 件 30×18×12 mm、单边间隙 0.5–1 mm、偏移 ±2 mm／±3° | 当前 matched keyway 声明 peg 横截面 20×12 mm、key x extent 15 mm；`mating_contact_config` clearance 0.8 mm、reset task noise 0.5 mm | 当前 composite tool 不能直接等同 draft 实体；要核对 reset noise 的实际分布及旋转支持，不能把数值当完整扰动覆盖 |
| T3 接触成功 | draft 强调深度／姿态／保持和 25 N 总力、10 N 侧向力、15 s | 代码另要求 normal force 距 desired 5 N <2.5 N；matched profile 还有 near-seat 轴向支撑丢失判据 | 力保持及支撑失败必须公开进共同 evaluator；command force 不作为 measured-force 证据 |
| T4 身体稳定 | 平台专属稳定、滑移、关节边界；15 s 操作期限 | 历史 `validate_mobile_mechanism.py` 用 min support≥2、tilt≤0.5 rad、QP residual≤0.01 的开发检查；Go2 noisy task-space reset 显式不支持 | 开发阈值未自动成为新主表规格；需验收站立、接触、滑移定义、QP residual 单位／scale 与 reset 支持 |
| probe budgets | draft 不同位置出现 0/10/30/100 和 0/5/10/30 | P0–P2 开发校准复用了 5-probe 单元 | 新协议建议先研究 0/5/10/30；是否扩展 100 在开发后、测试前固定，正文与 config 只保留同一版本 |

当前接触峰值需保留 P2 的实际定义：single-contact norm 与 net tool force norm
均检查，物理子步采样；侧向力定义、坐标、力矩中心和 contact whitelist 同样
冻结。不得用控制周期均值掩盖子步尖峰。时间步减半／solver 收敛是环境资格，
不是给某个算法额外 target tuning 的机会。

## 4. body×mechanism 与 held-out 划分

先做能够回答归因的问题的紧凑矩阵，不立即铺开四任务×所有身体×所有参数的
笛卡尔积：

| 对照轴 | 必需比较 | 可支持的解释 |
| --- | --- | --- |
| 同身体、不同执行响应 | Panda 或 xArm7 的名义／偏移 controller，已知几何保持相同 | 未知 execution response 的校准，而非 IK 差异 |
| 不同身体、响应尽量匹配 | Panda → xArm7，公共任务／工具与接口一致 | 已知 morphology／adapter 影响；若 direct 已解决，仅作为接口证据 |
| 身体与执行响应同时变化 | xArm7／UR5e 加 controller deviation | cross-body adaptation 与 response calibration 的联合效果 |
| 接触单因素与组合 | 先 controller，再 fixture friction／compliance，再预声明组合 | 接触模型是否提供普通动作补偿之外的信息 |
| 移动身体耦合 | 固定／释放同臂，arm-only／coupled calibration，同一 WBC | 反作用力与身体响应耦合的独立贡献；基座锁定只作诊断 |

增益、延迟、摩擦数值不在此定为最终机制等级；draft 的 gain 0.7/1/1.3、delay
0/20/60 ms 与 fixture friction 0.15/0.35/0.6 仅是 pilot 起点。轮地／足地
摩擦和 body response 需专门资格，不能复制 fixture 的范围。机制等级在身体
间交叉，不能把某一机器人固定绑定最难机制。

数据切分按用途与组共同冻结：

- 源 prior 数据：Panda 成功轨迹与真实模式证据；train/validation 按 episode
  group 切分，归一化仅使用 train。公开 response development 与 validation
  数据同样按 episode／mechanism group 切分。
- 开发集：用于环境资格、reward、模型结构、超参数与 probe 分配。已经用于
  P0–P2 调试的机制与 reset 只能列开发数据，不能重新称 untouched final test。
- 泛化集：分别留出初态／目标位姿、机制组合、机制参数范围及身体；within-support
  与 outside-support 分开。只留出随机 seed 而复用全部机制并非机制 OOD。
- 最终目标集：各方法、模型、停止规则、mode classifier、预算分配与超参数
  冻结后才访问；其轨迹／失败不能回流选模型，再以同一集确认。

UR5e 两种协议独立报告：冻结 prior／shared response／transport 后只拟合
context；或者在相同 calibrated-twin 预算内更新 transport。后者不能叫 frozen
generalization。UR5e 若参与公共 response 训练，则另标 seen-body 工程条件，
不能并入未见身体主协议。T4 同样披露目标 body 是否参与 response 预训练。

## 5. 目标访问、模拟预算与统计

### 5.1 信息权限分层

| 协议 | 目标信息／交互 | 用途 |
| --- | --- | --- |
| A-fixed：少量校准、零目标任务训练 | 已知形态与控制能力、同一计费短 probe buffer；无目标成功示范，无目标完整任务 RL | 四任务主要比较。离线更新只查询 calibrated twin；普通 residual、dense policy、MPC/MPPI 与 Full 服从相同目标访问 |
| A-active：原生主动辨识系统 | 可按本方法选择短 probes，但总次数／时长／失败成本与 A 相同 | 单列完整系统，不能仅摘辨识模块代表原方法；与共享 buffer 的机制消融分开 |
| B-log：完整自采目标记录 | 计费完整 action/observation 轨迹，无 target reward／专家示范；额外状态测量单列 | FADA/ASAP 原生适应；同组方法共享记录权限，完整轨迹不能因无 reward 混入 A |
| B-online：目标任务适应 | 明确计费的完整任务 rollout 与 reward，无额外示范 | 强在线适应和数据效率曲线；不得将完整任务交互隐藏为短 probes |
| C-supervised：额外监督参考 | 目标成功示范、人工纠正、aligned actions 或外部预训练，逐项声明 | 按原生监督训练，目标在线交互另计；不得混入 A |
| D-oracle：封存诊断 | 目标真参数、真值模型规划、高计算预算可行性搜索 | 解释任务能力与模式支持，不称已证明上界；不提供主方法训练样本、mode mask 或调参依据 |

每个 probe 除次数外还报告最大 duration、实际 duration、reset／预热时间与
总分钟。接口调整、fit、held-out physical validation、失败与 interrupted
attempt 均计入 B；不因失败免费补采。B=0 不得读取目标真参数。
对 FADA 等需要约两分钟动作/观测记录的方法，除短 probe 子协议外，还需原生
时长的数据点。数据是否包含完整目标任务执行也单列，不能因不使用 reward
就把任意完整任务轨迹当作短 probe；示范和外部预训练仍按 C-supervised 声明。

校准后 GPU／CPU 模型训练调用与真实 target 交互分开计账。模型 validation
reset 是 calibrated-model 调用时，不冒充新增 target probes；反过来，CPU
目标上的 validation／调参也不能被记成免费模型调用。P2 的六个复用校准单元
不因增加 resets 而成为更多独立校准重复。

共同披露模拟 transitions、有效／padding steps、更新次数、超参数尝试、
参数量、编译／训练／reset 时间及计算设备。主比较给出目标数据预算曲线，
再以匹配模拟预算与部署时延的版本比较算法。MPC/MPPI 的在线模型查询与时延
计入部署成本；amortized policy 不能只与无计算上限的规划器比较而省略成本。

### 5.2 统计与整体失败口径

沿用原研究目标作为资源评估起点：正式 cell 尽量包含至少 5 个独立校准／训练
seed，每 seed 至少 50 个冻结评估 episodes。是否缩减 cell 数与具体样本量，
应先估算 wall-clock 和方差，再在查看 final test 前固定；本文不将它们写成
已经完成或无成本的承诺。

方法使用共同 reset／扰动列表与物理初态，random stream 按全局 sample
identity 定义；不同设备或 batch 重排不改变该世界的随机序列。用于拟合、模型
验证与最终 target test 的 resets 独立。统计按独立校准／训练 seed、机制组
聚类，保留方法配对；单次训练后的 100 条轨迹不是 100 次训练重复。

报告总体、每组和最差组的安全整体成功率、任务成本、停止／拒绝／超时、
完成时间与置信区间。失败／停止是任务表现的一部分；仅成功样本的速度与
force 可作为条件指标并公开其分母，不能替代整体指标。T4 的稳定但未插入、
倾覆、支撑丢失和 WBC 失败均为整体失败。source-support failure 也不剔除。

CPU target 继续作为当前物理参考。不同训练后端对所有方法一致，并单列
calibrated model 到 CPU target 的误差。只有需要用 GPU 取代 target reference
时才要求对应物理替代资格；CPU 确认的研究不必等待所有 GPU 长轨迹等价。
不能将后端导致的不同终止结果用作同物理端到端加速证据。

## 6. 模式语义与分布性能的公平规则

主任务成功由环境目标与安全定义。若没有明确的用户模式要求，task-only
选择更可靠的成功模式是合法行为，source-mode mismatch 不自动变成任务失败。
若模式有语义请求，例如指定绕行侧或禁止某种恢复接触，必须预先写进任务
输入与成功条件，所有方法都得到相同 mode request。

T1 的实际路线、T2 的接触导面／事件序列、T3/T4 的恢复事件分类器，需要从
真实 rollout 标注，并在 final test 前冻结。当前 T3 的
`classify_contact_modes` 先要求载荷后撤回，再按 y 方向位移阈值区分两个分支；
不能把这个现有粗标签自动描述成“搜索推进”与“回撤恢复”两个完整策略族。
新增更丰富事件语义必须先在源成功数据和目标开发数据验证可辨识性。

distribution 指标分开回答：

- 无 mode request 时，成功行为的实际频率、已验证可行模式覆盖、无效／停止
  概率质量与经验响应距离；同时报告全部 reset 和未知标签质量。
- 有 mode request 时，每个 request 的任务成功、实际行为匹配与失败；不能
  只报成功样本的 mode match。
- 相同任务质量约束下的 response／law fidelity，配合 task-success–time–force
  tradeoff。随机噪声提高 entropy 本身不是贡献；deterministic transport
  仍可因 stochastic source prior 产生多模式行为。

模式不可行时不强制保留源比例。封存 oracle 只有找到并执行成功行为才标为
verified feasible；搜索失败记未验证，不作为不可行证明。oracle 不向训练
提供 mask、演示或目标真值。跨方法对响应距离的比较必须处理不同拒绝质量
和成功分母，不能只看各自接受样本的均值。

## 7. T4 的范围和资格出口

当前 `contact_workcell.py` 的平面组合使用 x/y/yaw 驱动 joints、低摩擦球形
caster 代理，并机械约束 height/roll/pitch。它适合隔离基座自由度与共同 TCP
控制问题，不能支持真实轮胎摩擦、倾覆稳定或通用轮式机器人结论。

真实轮式平台需要先选定并冻结底盘模型，验证安装、质量／惯量、轮地接触、
驱动／速度约束、全身碰撞与共同力接口。Go2+Z1 在
`floating_workcell.py` 有真实浮动基、脚接触和 stance inverse-dynamics QP，
WBC 4 ms 的 host OSQP 成本需计入 end-to-end 时延。当前支持站立操作而非
步态生成；G1 尚需实际夹持／力接口、组合资产与 controller，不能提前填入
结果表。

T4 主协议保持 T3 connector、工具、force、timeout 与源 prior，起点位于
Panda 与目标共同支持域，接触历史通过公共 warmup 建立。远距离到站或源不
支持的新工作空间另列 extended-workspace：同一个导航／站位模块，保留
到站失败与总耗时。T4 操作 mode 继承 T3，支撑事件单列，不能把步态变化
当作操作模式覆盖。

先验证固定／移动同臂，再真实轮式，再一种腿式。只有同 WBC 下 coupled
calibration 比 arm-only／普通动作校准改善 held-out 响应，并改善受约束任务
或模式质量，才支持 coupling 贡献。若公共 WBC 已解释全部收益，如实限定
为接口扩展。至少轮式与一种腿式有效结果才主张跨移动支撑类型；人形不作为
未经接口验证的必要承诺。

## 8. P3–P7 中的任务交付

阶段编号与完整门槛见 [新执行计划](cross-embodiment-p3-onward-plan.md)。
旧的把全部工作压到 P3 小步骤的安排不再沿用。

- **P3：原生复现与比较资格。** 四任务 schema、阈值冲突、资产/source 支持、
  evaluator 与预算合同先明确。每 baseline 的原生与本任务改编分别验收，
  代码可 import 或 checkpoint 可加载不等于完整复现。
- **P4：四任务开发比较。** T1/T3 先接入合格近期方法，T2/T4 各自环境资格后
  扩展。B-log 给 ours 与强法同等记录权限，并做共享 buffer 控制实验；C 同示范
  或系统 Pareto 分表，不能把强对手全部放入额外资源后宣称最强。
- **P5：方法改进。** 用失败分解决定连续反馈、模型/后端、可行性/停止、校准与
  耦合的优先级。与 P4 交错迭代；同容量 task-only 和同 WBC 的身体消融用于归因。
- **P6：正式验证。** 四任务资格与开发证据齐全后，冻结方法/协议，生成新机制、
  reset 与独立训练/校准重复。全分母、分层区间、资源和每任务结果同时报告。
- **P7：论文。** 用最终证据写贡献；原生复现记录、工程排错和未成功版本留开发
  目录。任务语义、身体范围、模式与预算沿用本文件，不因最终成绩改口径。

T4 的平面代理不当真实轮式，Go2+Z1 的站立操作不当导航/步态；同 WBC 高层
比较与原生全身控制系统分开。跨移动支撑类型的主张仍需有效轮式与腿式证据。
