# Cross-embodiment：近期顶会强 baseline 文献筛选

检索日期：2026-10-14。本文是文献与复现资格筛选，未安装第三方环境、未运行这些方法，也不包含新实验结果。比较总协议见 [四环境比较计划](cross-embodiment-baseline-comparison-plan.md)。论文 draft 只接收文献定位、方法和正式评估设计；复现过程与实验报告留在开发目录和结果工件中。

用户要求成熟、完整的强 baseline，优先最近两年的顶会及 **C. Karen Liu、Guanya Shi** 的工作。首轮从具体论文的完整方法出发；DR/PPO、旧 residual RL 和通用 MuJoCo MPC 保留基础参照，不再作为近期文献主比较。

复现、开发比较、方法改进和正式验证已拆分为 P3–P7，见
[阶段执行计划](cross-embodiment-p3-onward-plan.md)。首批方法池与实际执行队列
分开：资产未取得的路线保留待办，其他原生资格继续，不冒称已完成完整复现。

## 1. 会议年份与复现资格

- 主池优先 **2025/2026 的 CoRL、RSS、ICRA、IROS、ICLR、NeurIPS 主会**，须有正式 proceedings 或可核验的主会录用信息。期刊、workshop 和 under review 明确区分。
- 滚动两年窗口为 **2024-10-14 至 2026-10-14**；CoRL 2024、NeurIPS 2024 的会议日期仍可能符合，但只作边界候选。使用会议年份，不把 2025 出版的 CoRL 2024 proceedings 改称 CoRL 2025。
- CoRL 2026 若依据作者录用公告、实验室与官方仓库互证，标为“**已录用，作者公告核验，正式 proceedings 待核**”；不写会议已举办，也不将 arXiv 年份当录用证据。
- “有仓库”不等于完整可跑。分别核验训练、适应、数据/资产、checkpoint、评估与部署接口。未取得或未运行的资源不称已复现。
- 原生身体、传感器、动作或控制器不同，使用 `*-adapted` 并保留改动清单。删除核心阶段后不能继续用原方法名称代表完整系统。

## 2. 五个主候选，首轮只做前三条路线

主池收敛为 **FADA、Human2Sim2Robot、ACP、Play2Perfect、ASAP**。首轮只接通前三条方法路线并完成小规模资格，不同权限分表；Play2Perfect 先检查原生 in-hand/play 核心和资源，ASAP 第二批。SPI-Active、DIAL-MPC、XMoP 按任务和贡献问题补齐，不默认同时大规模实现。

| 方法与作者关系 | 会议身份 | 主要竞争解释 | 首轮位置 |
| --- | --- | --- | --- |
| **FADA；Guanya Shi** | CoRL 2026 已录用 | 保留高层意图、只微调逆动力学，是否已足够处理执行机制变化？ | 核心路线；原生数据权限归 B-log；arm/task Planner-IDM 改编，短 probe 版另列。 |
| **Human2Sim2Robot；C. Karen Liu** | CoRL 2025 | 物体中心示范奖励＋目标模型充分 RL，是否已足够跨身体？ | 核心路线；T1/T2；原生人示范归 C，source-only A 改编共享源轨迹监督。 |
| **ACP；Shuran Song 等** | ICRA 2025 | 完整视觉/力反馈与可变柔顺策略，是否已足够解决接触？ | 核心路线；T2/T3；视觉＋力示范资源，state-only 改编另标。 |
| **Play2Perfect；C. Karen Liu** | CoRL 2026 已录用 | 大规模 dexterous play 先验＋模拟 RL 微调，能否解决精密装配？ | T3 优先资格；不能删除 in-hand/play 核心后代表全法。 |
| **ASAP；Guanya Shi** | RSS 2025 | 模拟器动作对齐＋任务策略重训练，是否已足够？ | 第二批；完整 delta-model→simulator alignment→policy fine-tuning。 |

某候选因资源或问题不适配未通过资格时，公开原因并以回答同一问题的方法替换；不能削弱实现来获得较容易的对照。成熟外部方法不受当前 24 个 affine candidates 限制。

## 3. Karen Liu 方向

### 3.1 Human2Sim2Robot：首轮跨身体路线

**Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration**，Tyler Ga Wei Lum、Olivia Y. Lee、**C. Karen Liu**、Jeannette Bohg；**CoRL 2025**。[正式 proceedings](https://proceedings.mlr.press/v305/lum25a.html)、[官方项目](https://human2sim2robot.github.io/)、[官方代码](https://github.com/tylerlum/human2sim2robot)。原法从一个 RGB-D 人类视频提取物体轨迹和预操作手位姿，在数字孪生中训练 RL，零实机策略微调。训练/处理代码已发布，数据、资产和模型下载需要作者表单，尚未取得并运行验证。

**适配判断。** T1/T2 优先，T3 可资格；原生 KUKA＋Allegro 灵巧操作，无固定工具插入或移动 WBC 的原生验证。使用源机器人轨迹替代人视频时标 `Human2Sim2Robot-adapted`，保留物体轨迹奖励与预操作初始化。成功示范版本归 **C-supervised**；source-only 改编另设同信息子协议，双方共享同样源轨迹、目标名义/校准模型和监督，不能称原生无示范方法。

### 3.2 Play2Perfect：精密装配优先资格

**Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly?**，Tyler Ga Wei Lum、Kushal Kedia、**C. Karen Liu**、Jeannette Bohg；**CoRL 2026 已录用**，依据 [第一作者主页](https://tylerlum.github.io/) 和 [Stanford IPRL 录用公告/列表](https://iprl.stanford.edu/)，正式 proceedings 待核。[论文](https://arxiv.org/abs/2606.26428)、[项目](https://play2perfect.github.io/)、[官方代码](https://github.com/kushal2000/play2perfect)。原法保留大规模模拟 dexterous play 预训练，再针对装配 CAD/任务进行模拟 RL 微调，实机零微调；仓库公开两阶段训练、装配评估、checkpoint 和部署参考。

**适配判断。** T3 优先、T2 次之，不强铺 T1/T4。原生 KUKA＋22 DoF Sharpa 手，in-hand object reorientation 和 play pretraining 有核心作用。当前固定夹持若完全消除这些能力，仅剩普通 RL 微调，不能代表完整 Play2Perfect。先判断可保留的核心，或放入扩展灵巧任务。论文使用单 RTX A6000 的长时间预训练；当前 24 GB GPU 先测资源，不提前承诺耗时。外部 checkpoint/play 数据归 **C-supervised 的外部预训练列**；从头模拟训练量独立报告。

### 3.3 SimToolReal：物体中心工具泛化备选

**SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation**，Kushal Kedia、Tyler Ga Wei Lum、Jeannette Bohg、**C. Karen Liu**；**RSS 2026**。[正式 proceedings](https://roboticsproceedings.org/rss22/p151.html)、[官方代码](https://github.com/tylerlum/simtoolreal)。原法通过程序生成工具训练通用物体目标位姿策略，部署使用物体位姿/几何和目标轨迹，不做目标工具/任务专项训练。训练、DexToolBench、checkpoint 与部署代码已发布；论文的 Isaac Gym 配方与当前推荐 Isaac Sim 路径须区分。

**适配判断。** T1/T2 可比较物体中心泛化，T3 是困难迁移参考；固定工具会删减原生灵巧手能力，必须说明。它与 Play2Perfect 可形成“零样本工具泛化→精密装配微调”的问题链，不必同时做主基线。T4 不匹配。外部模型和预训练量单列。

### 3.4 Cloak：视觉协议备选

**Cloak: Zero-Shot Cross-Embodiment Manipulation by Masking the End-Effector from the VLA**，Michael Piseno、Guy Tevet、**C. Karen Liu**；**CoRL 2026 已录用**，依据 [Stanford 官方项目](https://tml.stanford.edu/cloak/)，[OpenReview 条目](https://openreview.net/forum?id=7KV78k8rcv) 的独立页面读取尚未完成。[官方代码](https://github.com/Stanford-TML/cloak) 有训练、部署、处理、测试及 checkpoint 入口。原法用末端几何遮罩及指尖 FK/IK 处理外观和运动学差异，依赖 VLA 和机器人数据预训练。

**适配判断。** 适合 T1 视觉扩展。当前状态/力协议不能只摘取 FK/IK 便称完整 Cloak；π0.5/DROID 等资源归 **C-supervised**。是否增加视觉比较应由论文问题决定，不为作者或年份扩张实现范围。

## 4. Guanya Shi 方向

### 4.1 FADA：首轮少量数据执行对齐路线

**FADA: Few-Shot Domain Adaptation via Dynamics Alignment for Humanoid Control**，Angchen Xie、Nikhil Sobanbabu、Ishayu Shikhare、Alan Wang、Max Simchowitz、**Guanya Shi**；**CoRL 2026 已录用**，依据 [第一作者 2026 年 9 月公告](https://angchenxie.github.io/)、[LeCAR 官方列表](https://lecar-lab.github.io/publications.html)、[官方代码](https://github.com/LeCAR-Lab/FADA)，正式 proceedings 待核。[论文](https://arxiv.org/abs/2606.28476)。原法训练 PPO oracle，经 DAgger 蒸馏为 Planner-IDM；冻结 planner，仅用约 **120 秒目标 actions/observations** 监督微调 IDM，不用目标 reward/专家示范。主要训练、LoRA 微调、MuJoCo/硬件采集和 G1/T1 oracle checkpoints 已发布；whole-body tracking 尚未发布。

**适配判断。** 与“保留意图、适配执行”直接竞争。T1–T3 需新 arm/task oracle、Planner-IDM 和动作接口；T4 与原作更近，但人形 checkpoint 不能直接控制 Go2＋Z1。`FADA-adapted` 保留冻结 planner、IDM 学习和目标无 reward 的核心。原生完整自采 action/observation trajectories 归 **B-log**，至少包含约 120 秒数据点；不能因为无 reward 或与 probes 同秒数就混为 A-fixed。仅使用既定短 probes 的版本是 **A-fixed 改编**，须验证激励覆盖并单列，不代表原生采集收益。训练 oracle 的 privileged state 仅来自公开训练模型，不能读隐藏目标机制。

### 4.2 ASAP：第二批模拟器对齐＋重训练

**ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills**，Tairan He 等，含 **Guanya Shi**；**RSS 2025**。[正式 proceedings/完整作者名单](https://roboticsproceedings.org/rss21/p066.html)、[官方代码](https://github.com/LeCAR-Lab/ASAP)。原法收集目标轨迹，学习 delta-action model，在加入模型的模拟器中微调源策略，部署不携带 delta model；训练、delta model、微调、动作数据、MuJoCo 和硬件流程公开。

**适配判断。** T4 最接近原作；T1–T3 的 arm/contact 版本标 `ASAP-adapted`，保留完整对齐→策略微调链。原生目标完整动作/状态轨迹归 **B-log**，并声明 full-state 对齐与测量权限，真实硬件的 MoCap 等不能作为免费共同观测。仅共享既定短 probe 的版本是 **A-fixed 改编**并单列，专家跟踪示范等额外监督另注 **C-supervised**。模拟策略微调不等于查询目标 reward。不能只拟合 gain/delay 改名 ASAP，也不能隐藏目标采集或模拟微调量。

### 4.3 SPI-Active：T4 主动辨识专项

**Sampling-based System Identification with Active Exploration for Legged Sim2Real Learning**，Nikhil Sobanbabu、Guanqi He、Tairan He、Yuxiang Yang、**Guanya Shi**；**CoRL 2025 Oral**，题名按 [正式 proceedings](https://proceedings.mlr.press/v305/sobanbabu25a.html)。[官方代码](https://github.com/LeCAR-Lab/SPI-Active)、[主动辨识说明](https://github.com/LeCAR-Lab/SPI-Active/blob/main/active_sysid.md)。原法为并行采样辨识→Fisher-information 主动探索→重辨识→下游 RL；主要训练模块已发布，sim2real、数据 replay 和主动采集衔接仍有未完成内容，不能称端到端资源齐全。

**适配判断。** 优先 T4 的 Go2 质量/CoM/惯性/电机变化。固定臂接触需新增参数和可执行激励。完整系统归 **A-active**，所有目标时长、失败、reset 和 transitions 计费；固定 buffer 的辨识版本只是消融，不能代表原生主动收益。arm-only 模型不承担移动身体耦合结论。

### 4.4 DIAL-MPC：近期完整模型规划专项

**Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing**，Haoru Xue、Chaoyi Pan、Zeji Yi、Guannan Qu、**Guanya Shi**；**ICRA 2025**，会议日期 2025-05-19–23。[IEEE proceedings](https://ieeexplore.ieee.org/document/11127320/)、[官方奖项页](https://2025.ieee-icra.org/program/awards-and-finalists/)、[官方代码](https://github.com/LeCAR-Lab/dial-mpc)。原法无需策略训练，用全阶动力学进行在线采样 MPC，有 Brax/MJX 环境、实时模式和 Go2 部署。

**适配判断。** 共享 probes 校准模型后保留连续轨迹、warm start 和充分规划，归 **A-fixed** 加在线计算预算。T1–T3 新增 arm/contact model 与 task cost；T4 原生全身 torque 控制替换 WBC 时单列系统比较，同 WBC 上优化 reference 的版本标 `DIAL-MPC-planner-adapted`。报告每步模型调用、时延和 deadline miss；真值模型仅归 **D-oracle**。

**MBD 边界候选。** Guanya Shi 同作者的 [Model-based Diffusion for Trajectory Optimization，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6a5c38a730b17f3827c09cf6b192be04-Abstract-Conference.html)、[官方代码](https://github.com/LeCAR-Lab/model-based-diffusion)可作 DIAL-MPC 算法来源/离线规划参考。会议日期在滚动窗口；原作 open-loop 结果不直接混入闭环比较，不必同时将两者列为核心。

## 5. 任务专项与待资格文献

| 方法 | 一手来源和资源 | 四环境位置与边界 |
| --- | --- | --- |
| **ACP：Adaptive Compliance Policy: Learning Approximate Compliance for Diffusion Guided Control，ICRA 2025**；Yifan Hou 等、Shuran Song | [项目](https://adaptive-compliance.github.io/)、[官方代码](https://github.com/yifan-hou/adaptive_compliance_policy)。发布训练/执行、虚拟目标与刚度标注、数据及 checkpoint；ICRA 主会论文与另获 workshop 奖项分开。 | **T2/T3 首轮**。原生视觉＋力示范学习参考位姿、虚拟目标和可变柔顺。目标示范归 **C-supervised**；state-only、模拟生成示范或替换控制器标 adapted。视觉无法共享时原生系统单列。力采样率不支持 FFT 时可用作者 temporal-convolution 版本并说明。 |
| **XMoP: Whole-Body Control Policy for Zero-shot Cross-Embodiment Neural Motion Planning，ICRA 2025**；Prabin Kumar Rath、Nakul Gopalan | [项目](https://prabinrath.github.io/xmop/)、[官方代码](https://github.com/prabinrath/xmop)，有模型、数据、ROS 和新身体接口；预训练使用大量程序生成身体规划示范。 | **T1 专项**，持件几何、载荷和执行误差需适配；外部 checkpoint 归 **C-supervised**，从头预训练量另报，不称原生接触/移动插入算法。 |
| **SPARR: Simulation-based Policies with Asymmetric Real-world Residuals for Assembly，ICRA 2026**；Yijie Guo 等 | [NVIDIA 官方项目](https://research.nvidia.com/labs/srl/projects/sparr/)、[论文](https://arxiv.org/abs/2602.23253)。仿真低维基础策略＋目标视觉在线 RL residual，自成功轨迹启动；未确认完整代码发布。 | T2/T3 的 **B-online** 高优先备选，完整任务训练计费。资源资格通过前不承诺可直接复现，不拿旧 Residual RL 代替全法。 |
| **LAD：Latent Action Diffusion for Cross-Embodiment Manipulation，ICRA 2026**；Erik Bauer、Elvis Nava、Robert K. Katzschmann | [项目](https://srl-ethz.github.io/lad/)标注 ICRA 2026；[公开代码](https://github.com/srl-ethz/robot_clip)目前声明对比动作编码器/解码器模块，而非完整 diffusion policy。 | **C-supervised** 备选，需要配对动作和机器人示范，不列完整可跑主基线或短 probe 无示范方法。 |
| **TACTIC: Tactile and Vision Conditioned Contact-Centric Control for Whole-Arm Manipulation，RSS 2026**；Rishabh Madan 等 | [官方项目](https://emprise.cs.cornell.edu/tactic/)：分布触觉/视觉、contact Jacobian、混合预测和 contact-aware MPPI；未确认完整学习/MPC 代码入口，硬件发布分开判断。 | T2 接触模型规划文献备选。原生 whole-arm 分布触觉不同于 TCP wrench，不能免费用隐藏接触真值或删核心模态。作者含 Angchen Xie，**不含 Guanya Shi**。 |

## 6. 基础与年份边界参照

| 方法 | 准确身份 | 保留位置 |
| --- | --- | --- |
| DR＋Recurrent PPO；旧 Residual RL；MuJoCo MPC/predictive sampling | 通用算法或较早论文 | 强通用下限、内部机制对照、诊断工具；仍充分训练/优化，不替代近期文献完整系统。 |
| FORGE | [IEEE 正式记录](https://ieeexplore.ieee.org/abstract/document/10925874/)为 **RA-L 2025**；[Isaac Lab 官方实现](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/forge)可用 | T2/T3 接触 RL 参考；不标顶会主会，也不因作者网页 Code Coming Soon 误称无代码。 |
| Mirage；ASID | RSS 2024；ICLR 2024，会议均早于滚动窗口 | [Mirage](https://robot-mirage.github.io/)、[ASID](https://weirdlabuw.github.io/asid/)保留 related work/传统参照，近期主角色由新方法承担。 |
| TRANSIC；CrossFormer；UMI on Legs | **CoRL 2024**，2024 年 11 月会议属于滚动边界；PMLR 2025 出版不改会议身份 | [TRANSIC](https://transic-robot.github.io/)人类纠正、[CrossFormer](https://crossformer-model.github.io/)多机器人预训练归额外资源；[UMI on Legs](https://umi-on-legs.github.io/)为 T4 系统备选，专属 learned WBC 与同 WBC 高层改编分开。 |

## 7. 数据权限与公平性

| 协议 | 可用目标数据/信息 | 方法归属与比较方式 |
| --- | --- | --- |
| **A-fixed** | 相同计费短 probe buffer、已知几何/接口；无完整目标轨迹、成功示范和 reward rollout | FADA/ASAP 的 short-probe 改编单列；Human2Sim2Robot 的同源轨迹改编；DIAL-MPC 的校准模型。固定数据不限制模型训练或规划能力。 |
| **A-active** | 自选探针，目标时长、次数、失败、reset 和 transitions 一致计费 | SPI-Active 完整探索→辨识→下游训练；不宣称与 A-fixed buffer 完全相同。 |
| **B-log** | 计费完整自采 action/observation trajectories，无目标 reward、无专家示范；额外状态测量另标 | FADA 原生采集权限、ASAP 完整轨迹/对齐权限；按交互秒数和覆盖比较，不因为同秒数就与短 probes 混排。 |
| **B-online** | 计费目标完整任务 rollout/reward，无免费额外示范 | SPARR/在线 RL 类；画性能对目标交互秒数/transitions 的曲线，不与 A/B-log 混排。 |
| **C-supervised** | 示范、纠正、配对动作、外部预训练/checkpoint 逐项标注 | Human2Sim2Robot、ACP、Cloak、LAD、外部 XMoP/Play2Perfect 先验；source-only 改编另设子协议，不称数学上界。 |
| **D-oracle** | 封存目标参数、真值规划或高预算可行性搜索 | 冻结后解释机制/可行性，不回流训练，不进入同预算排名。 |

目标交互以**秒数为主**，同时列 episodes/reset、transitions、失败尝试。B-log 必须覆盖 FADA 原生约 120 秒数据量；数据来自指定 probe 还是完整任务 rollout 明确登记，无 reward 不自动符合 short-probe 限制，也不因交互秒数相同就共享同一排名。Human2Sim2Robot 的一条示范、FADA 的两分钟无奖励记录、Play2Perfect 的零实机微调是不同资源声明。

外部方法保留成熟架构、历史反馈、连续动作、chunk、完整规划和必要训练量；仅内部 Full/task-only 要求同容量。共享成功/风险判据、目标访问权限、可用信息和兼容低层接口，分别报告模拟 transitions、更新次数、网络容量、源预训练、GPU 时长和部署时延。原生视觉若不属于共同观测，单列系统/额外观测比较，state-only 改编不冒充完整原法。目标名义身体训练与 frozen unseen-body 泛化分表；隐藏目标机制禁止作免费 actor/model 输入。

## 8. 四环境如何检验论文增量

| 环境 | 近期方法位置 | 主性能 | 贡献问题 |
| --- | --- | --- | --- |
| **T1 ObstacleTransfer** | Human2Sim2Robot 路线、FADA-adapted；XMoP 专项 | 全臂/持件无碰撞完成、SE(3) 误差、间隙、耗时、时延 | 物体中心训练、执行对齐和强几何规划之后，是否更可靠或更省目标数据？ |
| **T2 GuidedContactAlignment** | ACP、FADA/ASAP-adapted；Human2Sim2Robot 接触迁移 | 限力定位、接触保持/释放、滑移/停滞、力峰值与持续时间 | 成熟柔顺与历史反馈之外，响应校准还有何增量？ |
| **T3 KeyedInsertionRecovery** | ACP；Play2Perfect 核心资格；FADA/ASAP、SPARR 分预算 | 全分母成功、卡滞恢复、超力、重试、拒绝/超时、模式与耗时 | 完整预训练/微调和接触学习之后，校准与恢复是否独立有效？ |
| **T4 MobileKeyedInsertionRecovery** | SPI-Active、FADA/ASAP 身体接口；DIAL-MPC 系统比较 | 操作＋支撑联合成功、滑移、身体偏移、WBC 饱和、端到端时延 | 耦合辨识是否超过 arm-only；高层方法与控制器贡献能否区分？ |

不要求每种方法都原生覆盖四环境。公共适配方法形成跨任务比较，专项在擅长任务提供强对照，换 WBC 的完整系统另表。T4 当前站立 Go2＋Z1/平面 caster 代理不等于人形 locomotion；有效支撑动力学与接口先资格。

论文贡献由预声明指标决定：相同目标权限下可靠性、达到同等性能所需交互、未见机制/身体泛化、任务性能相当时在线成本。模式/响应保真须在任务成本满足预声明容差且模式有任务意义时作为额外贡献；拒绝和超时计入全分母，不只报 accepted 子集。

实施前逐方法冻结复现合同：论文/代码版本和许可、已取得资产、原生阶段、接口改动、预算组、训练/选择集和独立评估。先最小原生闭环与本项目资格，再开发预算曲线，最后根据增量问题冻结正式实验；不预先承诺胜过所有近期方法。
