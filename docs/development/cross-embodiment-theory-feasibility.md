# Cross-embodiment 理论可行性与 CPU 研究验收

日期：2026-10-12。分支：`codex/cross-embodiment-transport`。本计划承接六项论文证据缺口；旧 `mechanism_revision` 工件不可覆盖。新实验位于 `tmp/cross_embodiment/theory_feasibility/`，开发 pilot、校准/验证和封存评估分开。

最新逐关修复与验收移至 [CPU机制闭环](cross-embodiment-validation-closure.md)，工件根目录为 `mechanism_closure_v2/`。下文保留先前协议及失败，不将旧86/86回归追认为当前版本结果。新增正证据为状态条件短/多步response、有限联合响应law、5次独立校准＋优化50/50，以及障碍/真实3D导向/Go2+Z1站立操作；身体×机制矩阵修订验收、最终轨迹复核和完整tensor回归分别登记。

## 要验证的主张

冻结源行为 law，通过计费目标探针校准任务相关响应，再优化目标可执行的源条件技能映射。核心链为：探针覆盖 → 校准误差 → transport 访问支持内的联合响应误差 → 任务/响应成本与接触事件。每个箭头独立验收；成功率不能反向证明结构模型、辨识或 uniform kernel 假设正确。

目前理论允许结构 twin 和确定性 transport：随机源 prior 已可产生多峰输出。额外噪声、conditional flow、neural ensemble、主动探针优越性都需要独立证据，不能因名称预设必要。若普通控制优化解释全部收益，应收窄贡献，不通过削弱 baseline 或调 final-test 阈值追求优势。

## 六项工作与完成条件

| 顺序 | 工作 | CPU 产物与通过条件 |
| --- | --- | --- |
| F1 物理前提 | 匹配键槽、真实卡滞、TCP wrench、时间步检查 | 正确方向可入座；180°错误键位实际碰壁；reset无穿透；加载后实际停滞且撤回能释放；两种源恢复独立成功；接触符号和TCP力矩通过独立物理测试；dt细化不改变核心结论 |
| F2 方法必要性 | 同模型控制搜索、task-only residual、短时MPC及名义响应对照 | 共享 prior/decoder/限速限力/初态/probe buffer；搜索使用同模拟预算；在线MPC单列模型调用/步数/时延；不将inverse gain等同完整system-ID；保留强baseline胜出或持平 |
| F3 模型偏差 | 未知摩擦/顺应性、载荷偏差、观测噪声；模型访问支持检查 | contact-only参数用实际接触探针；fit/validation分开；支持外或验证失败返回unsupported并计失败；probe误差和实际task occupancy误差分开；结构twin与learned surrogate分别记录，不继承成绩 |
| F4 分布主张 | 两种分支均可行/代价不同/一支受限；deterministic与stochastic | 模式从实际轨迹分类；采样源mode并报告频率；额外噪声消融固定source样本与共同随机数；不只报告指定mode成功；允许得出额外噪声无必要 |
| F5 跨身体与统计 | Panda机制变化、xArm7匹配/偏移响应、UR5e身体留出，多次独立校准 | source/response/transport训练权限清单；frozen context-only与few-shot adaptation分表；训练/校准/机制/reset分层；先5次独立校准pilot，再按误差/效应量决定正式reset量；旧600条条件于单次校准不能充当重复 |
| F6 贡献与边界 | 对照原理论、相关工作、Task1/2及移动扩展 | 提交claim→实现→证据→限制矩阵；插入先通过F1–F5，再复制到绕障/导向；Task4共享WBC，对照arm-only/coupled calibration，操作与支撑联合成功；Go2失败继续保留，不能以固定臂成功宣布移动成功 |

## 理论审计

1. `ker A_B ⊆ ker B_g` 仅适用于声明的局部线性响应；自由空间探针不能分离仅在接触中出现的摩擦/柔顺性。通过有限机制对照构造不可辨识反例，而不把所有增益/延迟都称为接触不可辨识。
2. `(r+eta+epsilon_p)/kappa` 需要真实结构族覆盖与逆分离；拟合残差小不能代替二者。参数边界、模型外机制和无法分离的候选必须拒绝或明确unknown。
3. 源耦合成本依赖联合law。只验证目标marginal均值或TCP终点不能认证源/目标coupling，更不能认证模式分布。
4. 连续Wasserstein/Lipschitz界不能直接控制接触成功指示函数。事件误差还依赖到碰撞、入座、力限和保持时间边界的margin；分别记录几何和力事件。
5. `2 epsilon_F + epsilon_opt` 是同一可行集合、同一包含源anchoring目标下的次优界，不是相对任意task-only控制器的regret。若baseline目标不同，比较任务/载荷/时间Pareto关系，不引用该式证明必胜。
6. 有限冻结策略可用独立配对reset验证有界成本差并给出有限类置信界。这只覆盖预声明策略、校准和reset分布；不能换成无限参数类uniform kernel证书。

## 实现规则与 GPU/backend 迁移

- 公共层只声明config、mechanism support、policy参数、预算、指标与provider能力；MuJoCo/SciPy放在NumPy provider。新功能未实现的backend直接拒绝，不能静默返回旧wrench语义。
- 固定控制carry、命令队列、参考、事件phase和积分。预测从已观测qpos/qvel与已知controller carry重建，不能复制目标隐藏模型或未来轨迹。
- 数值轴明确为候选C × 独立初态B × 机制M × 模式R，时间T为scan；完整机器人拓扑单独分桶。CPU串行provider是reference，未来GPU实现沿同一frontend替换为vmap/scan；不用host SciPy作为GPU主循环。
- portable模型保留现有JAX/Torch/native inference接口。阶段特征由张量backend扩展；新增对齐阶段纠偏不能依赖隐藏mechanism真值。
- 数据/推理/恢复artifact保持命名张量＋JSON、明确checksum。训练来源/公开模型、计费target访问、post-calibration模拟成本和封存评估均分列。
- 首次source可行性、物理反例、CPU优化与小模型可在CPU完成；正式大规模neural ensemble与整条collector/learner的GPU设备、吞吐和内存验收仍独立进行。

## 当前执行记录

- 已增加独立 `matched_keyway` 开发配置，0.8mm单边间隙；保留legacy场景。
- 已增加 `tool_at_tcp_world` 力/力矩语义与接触friction/time-constant建设参数。显示/旧aggregate不等同新传感语义；需通过后续物理测试才能认定实现正确。
- 初始pilot：nominal Panda教师seed230/mode0成功；旧冻结prior在seed231/232、两种模式共4/4成功。旧xArm7 transport在seed231两模式仅1/2成功，表明旧成绩不能直接转用紧配合场景。
- 新阶段transport允许对齐阶段纠偏，与最终插入力补偿分开；处于开发pilot。任何用于此调试的目标配置/seed均不得再次称为未触碰测试。
- F1–F6仍按各项真实产物更新；本计划不宣称已完成论文主表或全局理论条件认证。

### 已完成的匹配键槽与强对照

`derated_contact/`：冻结 Panda prior，目标 xArm7 gain .65 / delay60ms，已知 .5kg 工具，匹配键槽、±.5mm任务扰动，所有方法共用16mm/2mm/2°/.3s/15s及25N/10N验收。5个计费探针中3 fit、2 validation；自由空间拟合gain/delay，接触记录在9个friction/time-constant候选中拟合。搜索为12个候选×2个训练reset×2模式=48条；task-only和joint目标共享同一缓冲。名义模型另有同额度48条。

| 冻结方法 | 新reset任务成功 | 实际mode匹配 | 末段力RMSE / N | 平均时间 / s |
| --- | ---: | ---: | ---: | ---: |
| Direct | 0/10 | 10/10 | 6.003 | 12.818 |
| Calibrated source-conditioned map | 10/10 | 10/10 | 1.707 | 9.710 |
| Same-model task-only controller search | 10/10 | 10/10 | 1.707 | 9.710 |
| Nominal-response map | 5/10 | 10/10 | 3.437 | 13.574 |

两种校准目标均选中 force scale2.5、阶段xy/angular log-stiffness1、插入xy bias(-.004,.002)m/s。**这是持平结论：目前没有response anchoring超越普通控制搜索的证据。**所有40条actual与40条estimated-model轨迹独立重算入座、保持、姿态、力和碰撞，0 mismatch。5个reset clusters的uniform model-cost置信界全部未认证，不能以近零模型误差替代样本量。

独立物理测试包括真实lip stall/卸载、180°错误key位穿透反例、TCP wrench到generalized constraint force恒等式，以及完整工具包络的JAX可编译检查。`timestep_refinement/`中Panda/xArm7×2模式×2/1ms共8/8成功，峰值合力最大11.302N。测试支持这一开发点，不认证任意接触事件。

### 理论与实现的必要修订

- 对齐判据必须包含工具全长、key和roll/pitch投影；只约束TCP xy会在紧配合孔口提前切换。
- 区分插入途中的侧壁擦触和近入座轴向支撑丢失，保留统一限力与hold；不能把任意短暂侧触永久锁存成已入座接触。
- 阶段transport激活原13D中的刚度通道，恢复/对齐/最终插入力补偿分开。portable16D特征→13D仿射模型已在JAX/Torch/C++核对。
- `ContactPlanner`为solver式frontend，NumPy provider从encoder q/qdot与已知controller carry重建，不能复制隐藏模型、warm-start或未来轨迹。候选×预测时间在provider内部；GPU/Torch物理provider当前显式unsupported。
- 连续kernel/参数界仍为条件定理。另增加有限冻结策略模型误差界及直接观测配对成本界；二者不能互相冒充，所有独立模式在reset内聚类。
- 随机源prior经过确定性映射即可保留多峰，额外transport noise不列为已验证必要组件。

### learned-response 独立验收

`learned_episode_response/`以calibrated-twin候选轨迹训练3-member、32×32小网络，310训练、311独立验证。signature RMSE .06413（阈值.05）、task-cost RMSE .17812（阈值.15）、failure error1/24（阈值.1），因此 **unsupported**，没有部署。其actual occupancy误差与ensemble disagreement分别保存；disagreement未校准。该模型只预测有限候选的完整episode response，不是共享autoregressive dynamics，不能继承结构twin成绩。下一阶段需要增加独立初态/机制数据、明确初态条件与概率预测，而不是放宽阈值或增加更新次数直至通过。

### 独立成本、分布、偏差与身体证据

**新配对统计。** `fresh_paired_audit/`在开始前固定25个新的伪随机独立reset clusters、两种模式在cluster内平均、固定两方法及有界task cost。Direct0/50、calibrated50/50，实际模式50/50保留；task cost平均下降.80203，单侧95% Hoeffding下界为.31248，故这一**实际成本比较**通过。该结论条件于一次校准和声明的reset分布，不认证模型uniform kernel，也不是25次独立训练或校准。同模型task-only已选择相同参数，其行为相同，不能将重复命名计作独立实验。

**实际采样law。** `behavior_laws/`每方法12次实际执行，使用共同uniforms和源prior随机流。模式由轨迹分类：

| 对照 | 成功 | 实际模式0:1 | 源mode对应 |
| --- | ---: | ---: | ---: |
| 确定性map | 12/12 | 6:6 | 12/12 |
| 加小幅独立transport noise | 12/12 | 6:6 | 12/12 |
| 打乱source-target coupling | 12/12 | 6:6 | 0/12 |
| 模型成本重加权 | 12/12 | 5:7 | 12/12 |
| 刚度纠偏受限、原source law | 6/12 | 6:6 | 12/12 |
| 同一受限类、按模型support重加权 | 12/12 | 12:0 | 12/12 |

最后两行限制的是已声明零刚度纠偏类，不是机器人全局不可能执行mode1；support来自训练twin，不从最终成功标签推导。重加权恢复成功但丢掉一支；开放阶段刚度通道的map能恢复两支。成功率和target marginal无法辨别第三行的错误coupling；额外噪声暂无优势。12次采样不认证精确mode概率。

**接触参数与读数噪声。** `contact_shift_readout_noise/`真实friction .7/time-constant .04，探针TCP位置噪声20µm、法向读数噪声.03N；5探针估计gain .650154/delay3/contact .7/.04。独立probe位置RMSE19.16µm、forceRMSE.03056N。`noisy_contact_occupancy/`的4次新实际执行和两模型各4次预测：实际4/4成功；校准contact模型signature RMSE.001082/task-cost MAE.000265，名义contact模型分别.017680/.015076。这个诊断只改变probe readout，未加闭环反馈噪声。

同一观测初态下，friction .15/.7两模型对free probe的position/force差精确为0，对两条contact probe的force差RMSE .414/.474N。这是实际执行机制的不可辨识反例：自由空间拟合再精确也不能辨识仅在接触激活的参数。

**模型族外。** `outside_contact_family/`真实.55/.03不在9点contact族中，选择.4/.04；probe forceRMSE.19248N仍通过原.25N阈值。`outside_family_occupancy/`实际4/4，估计模型signature RMSE.01489、task-cost MAE.00334、status agreement1。只能说明这个访问集合内近似有效；真实参数在族内的假设不成立，因此不能援引参数恢复/κ定理，更不能将验证通过说成族覆盖证明。

**载荷假设负对照。** `payload_misspecification/`真实工具.8kg、响应模型固定已知.5kg。拟合gain到上边界1.3、delay4，独立position RMSE1.104mm、force RMSE1.233N、peak RMSE1.347N，超过原1mm/.25N/1N阈值，验证拒绝。`payload_rejection/`保留请求的4个任务trial为unsupported失败，实际任务执行0次；5个probe仍计费。没有把未建模载荷误差解释成正确actuator参数，也未放宽阈值或部署该模型。

**独立校准与6DoF身体。** 原校准加`calibration_repeats/r1..r4`共5个独立probe随机初态，均恢复delay3、friction.4、time-constant.02，gain范围[.650000245,.650000366]，各2条独立validation通过。这是无噪声正确结构族下的辨识重复，不是5个独立transport训练结果。`ur5e_frozen_map/`同一源prior下，UR5e nominal/derated direct各2/2，而xArm7冻结map0/2超时。`ur5e_calibrated/`按公开UR5e形态构造response、独立5探针和同48条候选预算；独立3reset×2模式，direct、calibrated、task-only、nominal均6/6。UR5e校准map不放大force，体现身体条件化；其成功率没有比direct提高。

**MPC成本。** `.6s horizon/1s replanning`诊断2/4，额外16,103个模型控制步、190.0s planner wall；`1s horizon/1s replanning`另4条新reset诊断2/4，完整模型调用与时延保存于`one_second_mpc/`。这都是有限候选短时MPC，不是全局最优/充分调参的MPC，不能据此声称普遍优于MPC。核心强对照仍是同模型、同48-rollout池的task-only最优候选，其与方法持平。

### 本轮 go/no-go 与贡献边界

| 主张 | 状态与证据 | 仍不允许的外推 |
| --- | --- | --- |
| F1 真实匹配装配和恢复 | development pass；几何/力矩/卡滞、8条dt检查 | 工业亚毫米装配与任意事件稳定 |
| F2 校准后可实现冻结源行为 | 25个新cluster真实成本下界>0；强同类控制搜索持平 | anchoring、flow或本方法普遍优于system-ID/MPC |
| F3 task-relevant probe覆盖 | free/contact反例；未知contact+readout noise的小样本occupancy通过 | 全局κ/εK、未知mass可辨识、neural dynamics已成功 |
| F4 联合law区别于marginal | 实际打乱coupling反例、受限模式重加权、noise消融 | 精确模式频率、额外随机性必要 |
| F5 身体条件化 | UR5e冻结map失败、独立校准后可行；5次辨识重复 | body-agnostic冻结map泛化、5次独立训练显著优势 |
| F6 论文贡献 | 条件理论与可复现研究基线成立 | 新颖性已确立、Task1/2正式机制统计、Task4 learned成功、实机成功 |

下一阶段的验收顺序：扩大带真实初态条件的response数据并校准不确定性 → 对新initial-state/mechanism与真实闭环sensor noise进行支持检查 → 冻结多校准训练与强对照 → 再复制到ObstacleTransfer、3D导向接触和MobileKeyedInsertionRecovery。移动身体仍需独立WBC/support耦合响应；旧Go2 learned0/2保持失败。GPU用于吞吐与规模，不能替代这些建模和验收条件。

完整CPU回归`qualification/`为86/86通过、0 failure/error/skipped；最后可选backend构造器修改后，相关15项复测通过，去重总数仍86。核心新研究共332条保存轨迹独立验收、0 mismatch，其中252条actual评估、80条模型occupancy预测，不将后者算作目标交互。旧冻结Panda源轨迹与最终源码下匹配键槽两模式逐通道回放最大误差均0，见`legacy_replay/`、`final_replay/`。统一验收为`acceptance.json`，17个启动合约及最终代码/测试按实际SHA内容归档于`source_versions/`。本轮结果图`theory_feasibility.png/.svg`与机器可读`summary.json`由实际报告生成；旧600条不混入这些分母。

### 工件与复现入口

```bash
python -m scripts.cross_embodiment.run_theory_feasibility --output tmp/cross_embodiment/theory_feasibility/new_run
python -m scripts.cross_embodiment.evaluate_contact_mpc --study STUDY --output NEW_OUTPUT --horizon 5 --replan 5
python -m scripts.cross_embodiment.evaluate_behavior_laws --study STUDY --output NEW_OUTPUT --samples 12
python -m scripts.cross_embodiment.audit_paired_improvement --study STUDY --output NEW_OUTPUT --clusters 25
python -m scripts.cross_embodiment.audit_episode_surrogate --study STUDY --output NEW_OUTPUT
python -m scripts.cross_embodiment.audit_response_coverage --calibration CALIBRATION --reference-study STUDY --output NEW_OUTPUT
```

Docker环境与旧协议一致；这些命令使用CPU。保存ledger、fit/validation、candidate缓冲、frozen evaluation、真实trace、模型occupancy与producer版本。开始合约记录启动时源码；本轮补充可选MPC hook时已运行进程的原始rollout源码及SHA核对保存于`runtime_source_snapshot/`。Python已加载函数与后续lazy import需区分；每episode fingerprint读取序列化时磁盘，不回写为启动版本。新增helper/CLI没有改变这些冻结策略默认数值流程，原版本与改动说明一并保留；正式复现须先冻结所有源码再采集。已有输出严格拒绝不同合约/源码的恢复；重新设计协议使用新目录。
