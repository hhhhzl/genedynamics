# P3 实施与资格记录

日期：2026-10-14。本文记录实际工程与开发实验，论文 draft 保持独立。
本轮按 P3.0–P3.3 实现原生 PyTorch、训练/适应、公共任务和审计链。
**完整 P3 尚未通过**：原生 Human2Sim2Robot 资产/IsaacGym 未具备，
FADA/ACP部分任务改编策略尚未通过源端稳定性门槛，原生资格仍有缺口。
Play2Perfect公开checkpoint的本机原生in-hand闭环已通过本轮8环境资格，范围见下文。
这些缺口不由核心 parity、checkpoint forward 或小样本成功替代。
用户后续确认目前没有 Human2Sim2Robot 资产，先推进其他路线；原生路线维持
`assets_pending` 并暂缓，不阻塞 FADA/ACP 的源端训练资格。该决定不等于原生复现通过，
保留于 `_reports/route-decisions.json`。新增源端实验继续使用配置对应结果树。

实时证据索引由 `python3 -m scripts.baselines.p3` 生成：
[P3 审计 JSON](../../results/_development/p3_20261014/_reports/p3-audit.json)。
它逐条保留 qualification、失败 attempt、SHA 和费用，不自动批准论文比较。

## 1. 已实现的公共基础

- `genedynamics/solvers/single/{fada,human2sim2robot,acp}/` 保留原生 PyTorch 推理和模型结构；
  `genedynamics/learning/baselines/` 提供各方法训练/适应及完整 continuation。
- `genedynamics/experiments/baselines/` 提供四任务 CPU 真实物理、公共观察、历史、
  partial reset/reorder、物理 checkpoint、结构化 ACP 执行和独立 evaluator。
- public reference/previous_action 由自己发出的有界动作积分，位于隐藏执行延迟之前。
  隐藏 plant command、延迟 reference、模型参数和 MuJoCo warmstart 只用于物理和审计。
- logical accounting 随 checkpoint 恢复；lifetime accounting 永不回滚。
  续训/replay 是新的实际调用。episode 的拒绝、部分执行、超时和失败均保留。
- 所有结果镜像配置的 domain/task/category/method/level/seed：
  `results/_development/p3_20261014/results/...`，未采用 project/run_id 结果结构。
- ACP 保留 reference、virtual target、tool-frame SPD 6×6 stiffness 与 waypoint 时钟，
  在真实控制中执行各向异性刚度，未压成固定刚度 13D 输出。

## 2. 四任务物理资格

下表为公开几何教师的**可达与重放资格**，不是训练基线的成绩。
默认物理阈值/期限未放宽；重放使用独立 CPU 实际执行且额外计费。

| 任务 | 当前成功工件 | 物理用时与边界 |
| --- | --- | --- |
| T1 Panda 绕障 | source_interface seed 18204 上绕、18201 侧绕 | 8.26 / 7.80 s，默认 10 s；上绕 18204 classifier=0 且当前逐字段 replay 精确 |
| T2 Panda 导向接触 | task_interface 18400 x-first；task_interface_yroute 18403 y-first | 10.10 / 11.98 s，默认 20 s；后者峰值 12.90 N、侧向 6.07 N，精确重放 |
| T3 Panda 卡滞恢复 | source_interface 18212 直接恢复、18211 侧向恢复 | 5.14 / 8.46 s，默认 15 s；当前 18212 逐字段 replay 精确，旧模式轨迹另做只读模式核验 |
| T4 真实 Go2+Z1 卡滞恢复 | task_interface_recovery 18412/18413 | 8.76 / 7.28 s，另各计 1 s/50 controls 的实际 stance 准备；两者逐字段 replay 精确 |

T4 是实际 floating Go2+Z1 和 OSQP stance WBC，不代表导航、步态或真实轮式任务。
另有 clear 初态插入 seed 18410，不能用它替代卡滞恢复证据。

失败未清理：T1 seed 18202 上绕超时；18203 虽成功但实际模式 -1，不能当 mode0 示范。
T2 seed 18401/18402 触发侧向力限制；18401 及 T4 18411 的写出失败仍保留物理费用。
T2 18403 补齐已有 spatial-guide 教师的公开出口事件和终点积分捕获逻辑后另跑，
未修改力/成功阈值。所有旧轨迹与新轨迹分开保存。

## 3. 逐方法复现卡

### FADA

来源为 `290b3a9c5c0ebe2e9851b61350617544be05e9bc`，Apache-2.0。
G1 发布 oracle 及 251 个 checkpoint 的实际 SHA 取得；原生 PPO 更新、
Planner-IDM 两 pass、20 个弱策略 checkpoint、严格 oracle-shadow DAgger 已实际运行。
使用独立 mjwarp 环境，属于**模拟器改编**，不是 IsaacSim 原生定量复现；未关闭不支持的 randomizer 蒙混通过。

较大源训练实际收集 18,304 optimal、16,384 weak、3,539 online train transitions，
3000 warmup + 2×500 DAgger updates；585.65 s，峰值 owned GPU 1604 MiB、RSS 5.55 GB。
完整原生两 optimizer/RNG/normalization 的下一步 continuation 已精确一致。
原生 evaluator 的 long done-mask 历史 reset 问题已定位，受影响的旧评估/记录需标为
superseded 并保留费用；不得沿用受影响的学生稳定性数字。修复后单一固定目标重新采集
完整 120 s/6000 transitions（83.72 s wall），73 trajectories/5490 windows；released
100-step、batch4096 LoRA 与精确续训用时20.53 s、峰值 reserved4.75 GiB。
修复后 source4×10 s有15次失败，目标10 s前后9/8次失败，稳定性仍不合格。
原release默认各2.56M optimal/weak数据、32000×4096更新，本轮只用了1/256采样更新预算。

任务改编保留 released 128×3 Planner / IDM encoder3+decoder2、history30/horizon6、
两 pass 两 AdamW、冻结 planner 和 LoRA。源为公开 Panda 几何教师/真实 shadow labels；
portable Cartesian observation 去掉跨臂无法一一对应的 q/qdot。B-log 完整自采 120 s、
无 reward/目标 expert，并按 episode 划分留出集。初次改编的 source-std action/obs chart
放大近常数通道，已用公开物理尺度和完整硬件 Box 修正；旧 18501 的记录和失败不覆盖。
原始观测包含 issued action/reference 时会给 IDM future 提供退化捷径；旧18501/18502
单列 fidelity audit。最终仅保留35D真实物理state，goal进入14D command、action只进独立history。
最终容量1,765,087参数，T1/T3分别实跑1000 warmup+500 DAgger+300 LoRA、完整续训和冻结planner验证。
T3 seed18503复用旧collector的120s raw B-log（SHA+86→35映射、显式off-policy，0新增采集），
新目标为timeout/overforce，948 controls+948 replay；T1 seed18510新采120s/6000 controls，
12次timeout全部保留，新目标2 timeout，1000 controls+1000 replay。两任务完整replay一致，
尚不能作为充分训练的强基线分数。
最终source_policy的独立Panda源闭环也均失败：T1为2 timeout，T3为2 joint_limit，
四条CPU重放精确，额外2001 source controls+2001 replay、0 target。这定位为源student
尚未稳定，不能解释成跨身体迁移失败。完整执行日志位于各最终seed/source_closed_loop_gate。
两任务再分别实际执行batch1/4、各11次policy+history+Panda CPU物理工作负载，各新增55源controls、
550物理步，0目标调用；并发GPU上下文明确记录。正式FADA任务路线累计目标20696 controls、
源5072 controls（含采集/shadow/重放/workloads），与回归夹具物理调用分开，不把复用旧buffer重复收费。

后续source-only attempt18600按固定预算完成1500 warmup + 三阶段各1500 updates，
DAgger teacher比例依次0.85/0.5/0；原released网络与两pass保留。
新6条完整教师轨迹全部成功，以完整episode固定train/validation；teacher bank的实际force
支持为{0,5}N，据此比较continuous13D与2.5N source-gap off deadzone。
两decoder共用同一权重、训练数据、采集预算和评估seed；deadzone只把小力置零，
正力仍连续，未修改物理控制器的0.01N切换条件。不得把该公共解码对照当成transport贡献。

0.85 teacher的两条完整DAgger primary成功并覆盖phase0/1/2/3；0.5及纯student阶段
均超时，说明完整episode长度并不保证成功阶段occupancy覆盖。
最终两decoder均**0/2成功，2 timeout**；continuous模式[-1,-1]、off-deadzone模式[1,1]，
均未满足请求模式[0,1]。最终held-out teacher-state normalized MSE为0.001514，
阶段中曾低至0.000413，仍不能代替完整闭环资格。四阶段共16条评估全部未成功，
所有已完成评估逐条CPU replay精确，未按阶段挑一个好模型宣布通过。

本次source实费30296 controls：teacher2118、primary3657、expert shadow3657、
evaluation10432、replay10432；新增target为0。36 reset calls/44 reset worlds、
375.39s wall、CUDA peak allocated122869760 bytes。训练预算6000步之外，
每阶段恢复probe还实际计算2步，额外8次执行单列，未假装免费。
证据为
`results/_development/p3_20261014/results/arm/keyed_insertion_recovery/qualification/fada_source_train/level_source/seed_18600/qualification.json`。
该实验没有晋级P4；后续恢复边界/中断保存的代码修订与原始source-overlay身份分开记录。
运行后另外核验6 train / 2 validation的path、原始NPZ SHA和模型实际输入/动作的
canonical SHA均无跨split重复，初始TCP随seed实际变化；相同706帧配对长度并非重复模型数据。
审计保存在同leaf的`episode_split_post_execution_audit.json`与`training_accounting_audit.json`。
运行后修复阶段训练末尾CLI恢复的metrics缺失、中断shadow后缀/零步reset行保存、
全部captured genedynamics执行代码与data SHA核对，以及checkpoint decoder绑定契约。
新增12项针对性回归，含真正CLI恢复边界两case；相关26项CPU测试通过。
最新修订源码/验证在`post_run_correctness_revision_v3.json/.zip`，v1/v2历史仍保留。
这些小型source测试夹具每轮至多85 controls，独立于30296正式开发实验账单；
修复后没有重跑该长实验，也不追认旧CLI已经获得新边界验证。

### Human2Sim2Robot

来源为 `894eae2ec3ae39a573b81bd1860d14cc6bdfa6df`，MIT。
实际调用作者 PpoAgent 和 512/512 MLP + LSTM1024；原始三关键点/轴物体轨迹 reward、
reference clock 停止条件及归一化 action smoothing 保留。源机器人状态轨迹替代人 RGB-D，
公开 xArm7 名义模型替代数字孪生，held-object/shared controller 替代人手探索/FABRICS，
均明确标 adapted。共同任务终止替代原生 reference-end/fallen-object reset，轨迹末帧保持；
这也是明确的适配差异。demo读取绑定源task/robot/control_dt并检查实际timestamps，
environment continuation v2同时绑定示范SHA和采样钟，不能静默换轨迹/时钟续训。

T1/T3 已实际训练、保存和精确续训，比较 model/optimizer/RNN/normalizer/RNG/meters/物理状态。
真实 lifetime audit annotation 不参与算法等价比较，实际费用持续增加。
两任务各追加至 128 PPO epochs（另 1 epoch continuation audit），各 4128 controls / 41280 physics steps。
首次两条目标均保留：T1 seed18305 为 2 timeout，T3 seed18312 为 2 collision。
这些结果说明尚需充分训练，不能用来宣布方法弱于 ours。

原生资源入口需要作者签发下载 URL；inventory/download/ZIP/SHA/所有权 clean 工具已实现。
[资源 manifest](../../results/_development/p3_20261014/_reports/human2sim2robot-native-assets-manifest.json)
目前 `assets_pending`；没有提交表单、猜测私有链接或将状态轨迹当人视频复现。
Python3.8、IsaacGym Preview4、Torch2.1.2cu118、FABRICS/cuRobo 的原生环境与闭环继续待验。

### ACP

来源为 `74d68eca3bfb5cb59da131c6351d9e1c9b129ff6`，encoder submodule
`e4bd73e3e6ff0d4334d12aabbe0e9575669b63e7`；均保留 MIT notice。
实际作者 flip-up checkpoint 1,569,750,882 bytes、真实示范 episode、CLIP+force-ViT、
Mod1 U-Net、16 DDIM steps、EMA 和 native VT labels 全链运行。
7566 robot / 105906 wrench samples，原生 7000 Hz FFT 输入不是本项目 500 Hz 模拟传感器。

原生 source/local 推理、真实 loss/gradient 对齐；model/EMA/AdamW/LR/RNG 下一步精确一致。
旧2,853,016,190-byte continuation保留为v1证据。随后发现作者optimizer遗漏参数的累计
`.grad`也应恢复，已在gradient_continuation_v2重新实跑精确恢复，含model_gradients；
2,996,304,270-byte最新checkpoint SHA `99437c0d7bcf97e23cf2d867188bc17e2913cd9f6d0fa693fd9b5e58d8884d1a`。
不完整的upstream-mode v1训练状态明确拒绝恢复，原生optimizer数学未改变。
作者 optimizer 原生遗漏 89 个 force/fusion trainable 参数，upstream 模式忠实保留；
all_trainable 是任务改编选项，未静默改变原生结论。确定性 FP32 数值环境未放宽容差。
原始硬件成功率和完整 300-epoch 原训练没有复现。

T3 state/history 改编保留 canonical U-Net 256/512/1024、19D ref/VT/stiffness、diffusion/EMA。
初次 300 updates 的 loss 下降仍伴随采样方向严重误差，两条首次 xArm7 目标均在 2 controls 超力；
完整费用、真实结构化执行、resume 与失败已保留。时间对齐源采样误差证实还不具备充分模仿质量。
GT19D 回放没有初始暴冲，但后期仍触发力限，说明 hybrid-teacher 示范不能自动当成
native ACP-controller 成功示范。公开反馈通过实际 ACP 控制器重新采集的两模式源示范
18450/18451 均成功（3.6/3.8 s、峰值15.74/14.72 N、精确CPU重放），已保留实际
reference/VT/stiffness与public sensors；后续训练改用这些控制族一致的源示范。
这些控制族一致源示范加2000 source-only updates的18422精确续训通过；采样姿态误差
已降至0.0009–0.0234rad，但VT平移最差仍20.9mm，未宣称稳健策略成功。
GT诊断还实际执行了2条xArm7目标（235+238=473 controls），不能称全部source-only：
ACP本轮已有目标policy cohort2resets/4controls，加GT diagnostic2resets/473controls，
共4resets/477controls，cohort分别保留，不合成基线成功率。新live示范/训练/workloads为Panda源端。

原生H2S2R延期后的source-only精度实验已完成：新采6条Panda示范均成功并精确重放；
连同旧2条，以完整episode固定为6 train / 2 validation，训练文件SHA与验证文件无重叠。
后续canonical审计对原始与冻结normalizer之后的实际state-history输入/19D标签、
逐episode及逐window都检查跨split重复，均为0；4条冻结物理核验初态也彼此不同且非训练初态。
详见同leaf的`canonical_source_split_audit.json`，仍只称development validation。
6000步uniform后，加4000步低/高刚度平衡采样warmstart，再精确续训2000步。
三阶段optimizer/EMA/RNG恢复均精确，峰值CUDA allocated约1.58GB。
最终p95 reference误差0.500mm、rotation 0.00363rad、stiffness 294N/m；
VT为5.315mm，仍未通过预先声明的2mm门槛。验证集反复用于开发诊断，不是最终独立测试。

未降低采样门槛，也未按闭环结果挑模型：先冻结最终预算EMA
`457c405f449b5f9382a25aac9224983ced82ac5e9617022aba1dd27c35403c85`，
再执行预先固定的新seeds18522–18525。完整源闭环**0/4成功，4条overforce**；
404 collection controls与404 replay controls均保留，所有重放trace exact且状态一致。
在同一TCP下审计联合K×VT的期望wrench，normal误差p95为4.814N、max23.911N，
force-vector误差p95为6.550N；部分5N意图被采成25–32N。该误差具有实际控制含义，
不能仅因姿态MSE下降而晋级。增加DDIM steps或取消clipping未解决尾部问题。

本轮实际source费用为20 resets / 3028 controls（含expert与冻结actor重放）；
累计incremental source 4313 controls。新增target调用为0，历史target仍为4 resets / 477 controls。
统一证据为
`results/_development/p3_20261014/results/arm/keyed_insertion_recovery/qualification/acp_source_balanced_continue/level_state/seed_18507/source_precision_closure.json`。
本轮预算已结束，source采样与物理门槛均未通过；下一轮应先诊断接触条件学习和VT/K联合响应，
不能把欠拟合或控制误差解释为跨身体迁移的损失。

### Play2Perfect

来源为 `ff2fc62873f6c6cd93164123c8358e9ce2d65c9b`。公开 play 与 tight-insertion
checkpoint 均取得并按固定 HF revision/SHA 校验；实际原生 rl_games actor 严格加载，
140D observation、额外 coefficient/32D learned embedding、LSTM1024、29D action 保留，
batch1/8 CUDA forward profile 已跑。前一轮只证明发布网络可加载；本轮已新增真实原生episode。

配置位于 `configs/dexterous/tight_insertion/qualification/play2perfect_native.yaml`，
没有塞入固定夹持 T3 后删掉 in-hand 核心。IsaacLab2.3.2.post1 的 Python3.11 / IsaacSim5.1
运行栈已在独立runtime实际安装：CPython3.11.14、Torch2.7.0cu126、IsaacLab2.3.2.post1、
IsaacSim5.1.0.0、numpy1.26.0。metadata与GPU1 tiny Torch doctor通过；
准备阶段仅做metadata/Torch检查；随后用户明确接受许可，本轮已实际启动Isaac和Kit，
本机driver570.153.02在此固定任务/软件栈上的运行得到实际验证。
按作者安装指南覆写typing_extensions4.15.0，与kernel的4.12.2精确pin冲突单列；
其余pip check缺项和wheel/packaging冲突已修复，不能把该显式例外称作完全无冲突。
原先稀疏checkout漏掉的392个公开资产已展开并按固定commit/Git blob/SHA逐一核验，
实际机器人为IIWA7+Sharpa22，共29个actuated joints。公开资产不再是这条路线的阻塞项。
安装累计700.56s、峰值runtime allocated19.34GiB，全部缓存与managed Python归runtime，
安装日志、失败尝试、依赖闭包和资产SHA均已复制到配置镜像结果目录，删除环境后仍保留。
官方 `eval_offline` 排除早期短失败；项目比较必须另保留全 requested denominator。
新native runner已准备GPU1、8环境、2500控制步、600s含setup/10GiB GPU/20GiB RSS边界，
保留early drop、未完成、第一episode分母与中途费用。许可确认前只运行plan-only和守卫回归；
本轮记录用户明确授权后才设置EULA接受，没有通过原作者自动接受许可的worker绕过确认。
本次固定选择published checkpoint的exploitation分支，8槽均为训练过的coef0，
逐槽记录实际系数/parameter ID；不将它称作原作者六策略平均或论文aggregate score。
上游B8 linspace与六个训练coefficient ID不能精确匹配，显式source-mixed只允许匹配的B6。
最终peer review已修正完整任务必须包含release/retract的成功判定，并保留插入成功单列。
新native runtime另外实际严格加载同一公开tight actor，batch1/8各11次Torch forward通过；
NumPy1/2与Torch2.5/2.7 restricted scalar加载兼容已修正。该进程实际断言未导入
isaacsim/isaaclab/isaacsimenvs，EULA保持NO，新增物理调用为0；证据在
`level_source/seed_0/native_runtime_network/`，未覆盖旧network工件。

用户接受许可后的实际原生核验完成：固定seed42、8环境，**8/8完整任务成功**，
每条都完成2个goal及release/retract，没有early-drop过滤，也没有未结束槽位。
首episode长度分别167/248/201/603/420/187/160/107 controls，control_dt=1/60s。
实际vector control调用603次，全部槽位共4824 transitions；确认的SimulationContext.step
调用1206次，对应9648 world advances，包含部分槽位首episode结束后的额外自动reset执行。
这些单位重叠，不相加。内部Kit app.update启动ticks不在此step counter中，
不能由setup counter=0推断启动阶段零物理推进。

首次启动在GPU0出现CUDA context，被守卫及时终止；第二次已正确隔离GPU1，但日志UUID截短
导致解析拒绝，Kit异常关闭返回0暴露了缺失结果仍标complete的记录缺陷。
两次启动的16个未开始槽位与原始receipt均保留，第二次另加runtime_failure_audit更正解释。
修复后CUDA只暴露GPU1 UUID（logical cuda0），Vulkan保持physical ordinal1，
同时核对GPU Foundation active行、UUID前缀与唯一NVML PCI总线82映射，才进入物理环境。
异常先持久化再关闭Kit；exit0还必须伴随完整rollout结果才能通过执行门槛。
映射依据与边界见[NVIDIA CUDA文档](https://docs.nvidia.com/deploy/topics/topic_5_2_1.html)和
[Omniverse设备选择说明](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html)。

完成闭环的一次耗时74.76s（含setup/teardown），峰值owned GPU2693MiB、RSS6080688128bytes；
三次worker累计82.22s，未超用户批准的600s/10GiB GPU/20GiB RSS预算。
并发外部作业存在，wall-time不能解释为独占GPU吞吐。仅首episode8/8是策略cohort，
16个startup未开始槽位单列，不丢弃，也不伪作16次policy失败或独立统计重复。
源/eval参数保留上游eval_offline默认：preInsertAndFinal、random_goal_fraction0、
insertion tolerance0.01、retract tolerance0.005、sdf flag false、keep_dr false；
原URDF collision标签仍由converter保留，未修改上游训练/策略算法。

原生资格证据位于
`results/_development/p3_20261014/results/dexterous/tight_insertion/qualification/play2perfect_native/level_isaacsim_uuid_pci_verified/seed_42/native_qualification.json`，
含所有startup引用、真实结果、授权、源码ZIP/SHA、持久Kit日志与费用限制。
此项通过的是published checkpoint的一seed/8环境原生闭环资格，
没有复现play pretraining/fine-tuning训练全链，也不是原论文aggregate score、正式统计测试
或T1–T4改编资格；没有把in-hand任务删成固定夹持T3。

## 4. 环境与实际入口

源码参考 clone 保存在仓库根 `baselines/`，固定来源与许可证 lock；推理移植在 solver。
所有新 profile 与 P2 `.runtime/cross_ebdm/` 分开；结果和必要 checkpoint 在 configs 镜像结果树。

```bash
python3 -m scripts.baselines.sources fetch
python3 -m scripts.baselines.sources verify
python3 -m scripts.baselines.environment create --name baseline_tasks_torch --profile tasks-pytorch --device cuda --gpu 1
python3 -m scripts.baselines.environment doctor --name baseline_tasks_torch --profile tasks-pytorch --device cuda --gpu 1

# run 把 pip/HF/Torch/编译/tmp 缓存归入此 runtime；stdout 写入 environment.log。
python3 -m scripts.baselines.environment run --name baseline_tasks_torch --profile tasks-pytorch --device cuda --gpu 1 -- python -m scripts.baselines.human2sim2robot_tasks --help

python3 -m scripts.baselines.fada_native_environment create --gpu 0
python3 -m scripts.baselines.acp_native_environment create --gpu 1
python3 -m scripts.baselines.play2perfect_native_environment create --gpu 1
python3 -m scripts.baselines.play2perfect_native_environment doctor --gpu 1
python3 -m scripts.baselines.p3
```

每个任务入口的 `--help` 给出 source demonstration、config、seed、device 和训练预算。
`human2sim2robot_tasks --epochs` 支持扩大原生 PPO 训练，`fada_tasks` 支持 source/DAgger/LoRA
预算与完整目标秒数，`acp_tasks` 支持 source updates；续训 checkpoint 含 optimizer/RNG。
同一已计费 seed/attempt 不覆盖。FADA 原生重资源 launcher 带时间/GPU/RSS 上限和日志。

可删除命令不删源码/结果/持久 checkpoint：

```bash
python3 -m scripts.baselines.environment clean --name baseline_tasks_torch --profile tasks-pytorch --device cuda
python3 -m scripts.baselines.fada_native_environment clean
python3 -m scripts.baselines.acp_native_environment clean
python3 -m scripts.baselines.play2perfect_native_environment clean
```

这些命令本轮仅对独立测试环境实际验证删除，保留目前研究运行环境供继续实验。

## 5. 验证、费用和进入下一阶段的边界

源码核心 CPU/CUDA parity 入口为 `scripts.baselines.qualify_core`，最终工件在
`results/_development/p3_20261014/_reports/core/{cpu,cuda}/`。
新增真实接口、训练 continuation、资源 ZIP/ownership、审计 fail-closed 和环境生命周期测试。
最终CPU **99/99**、CUDA(GPU1) **99/99**，零跳过、零失败；物理合同仍使用独立CPU参考。
[CPU证据](../../results/_development/p3_20261014/_reports/core/cpu/qualification.json)、
[CUDA证据](../../results/_development/p3_20261014/_reports/core/cuda/qualification.json)
保留JUnit、日志、完整依赖闭包和源码指纹；当前两个core gate均通过且changed_files为空。
旧76/76证据完整归档至`_reports/core_before_source_refinement/`，旧40/40 parity也是历史版本。
本轮启动前95/95证据保存在`_reports/core_before_native_launch/`；新增守卫回归覆盖真实截短
Vulkan UUID/PCI映射、Kit退出0但缺失rollout、关闭前异常持久化及旧receipt的失败审计解释。
来源指纹现包括根source训练入口及任务合同依赖的真实CPU物理/控制代码，代码变化会使旧gate失效。

源采集、公开目标模型 RL、真实目标 B-log/评估、影子查询、重放和恢复核验分别记账。
report 中 evaluator/logical/lifetime/replay 指向重叠调用，不能无区分相加。
CPU replay 是确定性资格，不是独立统计重复。旧错误 attempt、native_failures 和 superseded 工件保留。

P3 可执行链已大幅补齐，但不把 2/128 PPO epochs、300 diffusion updates 或 reduced DAgger
称作充分训练的强对照。P4 首先建立 source held-out sampling、真实源闭环稳定性、
学习曲线和合理训练预算，再冻结共同权限的开发比较。原生 H2S2R 资产、FADA/ACP源端质量、
ACP原生硬件闭环和 T4 近期基线训练分别维护 open gate；Play2Perfect已通过上述范围的原生闭环。

另一个 P4 前置是 ours 的公共接口迁移：旧 `cpu_mechanism_validation.policy_step` 使用
backend delay 后的 reference carry；新 baseline 只看到 issued reference。旧 P2 成绩
不能直接加入同信息排名。所有方法经相同 public adapter 后重新比较。

## 6. 由本轮诊断提出的理论修订方向

目前没有证据支撑“ours 最强”。首先应在方法描述中明确观测滤波、动作 chart 和控制器
共同组成的闭环算子；相同动作分布在不同控制器/单位下不对应相同物理响应。
行为距离需要固定公共坐标/尺度，避免恒定源通道的经验标准差造成虚假大偏差。
隐藏 controller carry 不属于合法反馈信息；可实现 transport 必须对公共历史可测。

ACP 的控制族/示范不兼容说明 feasibility 和 source support 应在真实执行算子上验证，
不能只用采样 loss 或动作空间匹配判断。正式目标仍先要求成功/力/身体约束，再在
预声明任务容差内比较响应/模式；这些是待 P4/P5 消融检验的方向，本轮未将其写成论文结论。

用户确认暂无原生 H2S2R 资产后，其他路线的源端审计又暴露了更具体的理论边界。
它们先写成以下开发约束，待充分训练的对照结果出来后再决定如何修订论文。

1. **分析状态与策略信息分开。** 理论中的 Markov 增广状态可以包含隐藏controller carry，
   但可部署策略必须对公共历史生成的 filtration 可测，即 `pi(du|H_public)`。
   在 latent joint kernel 上作分析，不意味着可把 delay 后的reference送入actor。
   所有方法的策略类都应受相同信息约束，物理checkpoint和审计观察不扩大该策略类。

2. **混合模式误差不能藏在连续MSE中。** 当前控制器在 `force>0.01 N` 时切换法向位置/力控制。
   源teacher本应卸载时，几十分之一牛的回归误差就可能改变控制分支；总体动作MSE低并不
   保证恢复动作仍有效。这也是需要记录接触/控制模式错误率和边界margin的具体原因。
   用source支持设计显式卸载codec可作改编与连续版本对照，不能将decoder收益归于transport。

   对共同coupling下的真实/预测轨迹，令G表示全部控制/接触模式一致且处于验证支持内。
   若任务成本J在[0,Cmax]，并仅在G内满足 `|J(tau)-J(tau_hat)| <= L_J d(tau,tau_hat)`，则

   ```text
   |E J(tau) - E J(tau_hat)|
     <= L_J E[d(tau,tau_hat) 1_G] + Cmax P(G^c)
     <= L_J E[d(tau,tau_hat) 1_G]
        + Cmax sum_k (p_mode_error,k + p_support_exit,k).
   ```

   第一行直接分解G/G的补集，第二行仅用union bound、不要求各步独立。
   这是有条件的成本误差分解，不是新安全证书；有限验证也不能证明统一常数或零支持外概率。
   现有草稿的局部Lipschitz/事件margin讨论应落实到这些实际控制边界，而非再假设全局平滑。

3. **先限定源质量，再解释迁移。** 当前FADA源student也失败，目标失败不能单独归因于
   embodiment或校准。运输目标的 `2*model_error + optimization_error` 辅助界不自动
   包含源prior近似误差，也不等于目标task-only regret。应单列源held-out完整闭环、
   训练teacher-state误差与闭环occupancy偏移，再研究目标响应差距。
   ACP低/高刚度样本不均衡的诊断同样需要先排除普通源监督拟合不足；这类训练改善
   本身不能作为我们transport方法的贡献。

4. **用联合控制响应核对动作距离。** 对同一观测TCP，记静态impedance项为
   `w = K e`、预测项为 `w_hat = K_hat e_hat`，则有精确恒等式
   `w_hat-w = K_hat(e_hat-e) + (K_hat-K)e`。因此，VT与K的边际误差分别较小，
   并不自动保证联合wrench误差较小；应同时审计二者的配对样本及接触阶段尾部。
   本轮ACP审计保留原生anisotropic K与坐标旋转，测量的是该静态项，
   不能把它当作包含阻尼、限幅和plant dynamics的实际接触力预测。
   真正可用的比较指标仍须结合完整闭环、模式与力约束；若以后引入响应损失，
   应作为明确的改编/消融，而非静默改变原生baseline训练目标。
