# Cross-embodiment CPU 验证记录

日期：2026-10-12。分支：`codex/cross-embodiment-transport`。
范围：当前 CPU 功能链及全部保留的物理结果；正式多种子研究实验与 GPU 资格独立验收。

最新逐项修复以 [CPU机制闭环](cross-embodiment-validation-closure.md) 为准：短/多步状态条件模型、有限联合响应law、独立校准＋优化50/50、ObstacleTransfer16/16、3D导向10/10及Go2+Z1联合操作/支撑10/10。下文旧77/86项回归和旧失败均为相应历史版本；当前NumPy/native、矩阵重验及JAX/Torch环境阻塞另记，不能混合数字。

## 匹配键槽的理论可行性研究

最新协议和六项验收见[理论可行性计划](cross-embodiment-theory-feasibility.md)，工件为`tmp/cross_embodiment/theory_feasibility/`。此协议有0.8mm单边配合键槽、完整工具包络对齐、实际lip stall/卸载和工具TCP world wrench，区别于下文宽开口、aggregate载荷的历史结果。

首轮xArm7独立5reset×2模式：direct0/10，calibrated10/10，same-model task-only search10/10，nominal-response5/10。两种校准优化目标选择完全相同参数，因此不支持response anchoring必要性。actual/model各40条轨迹独立物理验收共80条、0 mismatch；小样本uniform model-cost界未认证。Panda/xArm7两模式在2/1ms时间步共8/8成功。所有新协议数值、失败、小网络unsupported与后续配对统计另见上述计划，不与旧600条混合。

进一步固定25个新配对clusters：direct0/50、calibrated50/50，实际task-cost下降.80203，单侧95%下界.31248。模式采样显示打乱coupling仍12/12成功且target模式6:6，却0/12保留source对应；小幅额外噪声暂无收益。UR5e冻结xArm map0/2、身体条件化后6/6，direct也6/6；contact-shift/readout-noise occupancy通过，.8kg真实载荷/.5kg模型的验证拒绝。完整CPU回归86/86通过，核心研究332条独立验收0 mismatch；其中80条是模型occupancy预测，252条为actual评估。详细边界与artifact索引以六项计划为准。

## 机制修正后的受限研究闭环

本节覆盖新的 `research_contact_config`，与下文 legacy 工程结果分开。代码和详细门槛见 [development 计划](cross-embodiment-transport-plan.md#机制修正与理论验收2026-10-12)。工件根目录为 `tmp/cross_embodiment/mechanism_revision/`。

已完成的修正包括事件条件 source prior、独立可行任务空间初态、实际接触探针、物理支持内 gain/delay 校准、法向参考投影与主动释放、source 静止时仍能纠偏的支持，以及真正检查姿态/侧向力/保持时间的插入判据。新的结构 twin 经 solver 风格 lazy frontend 委托到 NumPy provider，portable prior/affine map 经 JAX/Torch/C++ 实际前向执行。

研究协议固定 16 mm 入座目标、2 mm/2°、.3 s 保持、15 s 总时限、25 N 总力峰值和10 N 侧向峰值；5 N 法向目标允许 ±2.5 N。末段力 RMSE 使用最后15个 valid控制步，排除 terminal padding。Source 使用 nominal Panda；target 使用 xArm7，工具质量已知 .5 kg，仅 gain/delay 隐藏。当前法向反馈是约束接触轴向载荷 `sum(abs(F_contact,z))`，保持阶段为单向 floor 支撑；完整转矩尚未变换到 TCP，不能当作已校准 wrist F/T。所有测量来自同一 MuJoCo 结构族且未加传感噪声，因此这是正确结构模型下的控制研究，不是 sim-to-real 可靠性证书。

### 保留的失败与修正依据

- `chain_final_derated` 旧 46 mm 深度配置的 calibrated 9/10 不能作新协议成绩：当时无2°姿态检查，且有恢复模式改变。
- `strict_xarm_dev` / `strict_approach_dev` 的15s超时暴露已知 native stiction 下的接近积分不足；`strict_admittance_dev` 的位置误差与孔口超力暴露把最终力补偿应用于全部阶段的缺点。所有失败仍在工件中。
- 第一版 `strict_chain_moderate` 被独立验证拒绝：gain 估计 .806863、真实 .8，法向力 RMSE .43356 N 超过 .25 N 阈值。没有放宽阈值或偷偷加入验证数据；新版本加强 free-space excitation，先拟合稳定的 actuator response，再局部拟合接触，并更换全部校准/测试 seeds。
- `strict_chain_derated` 的匹配 nominal-response baseline 也是10/10成功；它的末段力 RMSE约2.42 N，calibrated约.65 N。因此只看该组成功率不能证明 calibration 必要。

### 版本与结果工件

- `strict_source_current`：6条 nominal Panda 源训练轨迹；`strict_phase_prior`：冻结 Gaussian phase/mode reference prior。
- `strict_source_heldout`：独立10条源行为，10/10插入且模式匹配。
- `strict_chain_moderate_v2`（.8 gain /20ms delay）、`strict_chain_derated_v2`（.65/60ms）、`strict_chain_stress_v2`（.6/80ms）：各5个计费探针、3 fit/2 val、48次 calibrated-twin优化；目标任务测试均10/10插入。前两组模式10/10，stress为9/10，保留孔口阶段的真实模式偏移。
- 新版本 gain/delay 估计分别约 .800000055/1、.649999704/3、.600000152/4，独立法向力 RMSE分别约4.10e-7、3.45e-6、1.77e-6 N。极小误差来源于无测量噪声且模型结构正确，不外推实机辨识精度。
- `strict_chain_derated_v2/acceptance_audit.json` 与 moderate 对应文件：独立重算80条保存的真实轨迹，位置、深度、姿态、法向力、总/侧向峰值、碰撞和时限均与终止判定一致，0个 mismatch。
- `frozen_derated`、`frozen_moderate`、`frozen_stress`：每组20个新 reset clusters、各2模式，冻结 source/transport/config/SHA 后只评估；5个方法含 direct、calibrated、force-only、inverse-gain与同预算nominal-response。三组均已完成，`complete=true`，最终统计见下表。
- `qualification_final`：按模块串行完整CPU回归77/77通过；新增6项机制回归另见 `tests_revision_provider.xml`，包括主动卸载、strict orientation、真实接触probe/resume以及JAX/Torch/C++ phase-conditioned map。

### 冻结评估最终结果

三组各20个独立reset clusters，每个含2种模式，每方法40条；各组一次校准、每次5个probe。所有600条最终物理轨迹独立审计，0个验收mismatch。以下数值仅限这个无传感噪声、正确结构族的研究配置。

| gain / delay | Direct | Inverse gain | Force only | Nominal response | Calibrated | Calibrated模式匹配 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| .80 /20ms | 0/40 | 0/40 | 40/40 | 40/40 | 40/40 | 40/40 |
| .65 /60ms | 0/40 | 0/40 | 0/40 | 40/40 | 40/40 | 40/40 |
| .60 /80ms | 0/40 | 0/40 | 0/40 | 36/40 | 40/40 | 40/40 |

| gain / delay | 末段载荷RMSE：Calibrated / Nominal (N) | 平均耗时：Calibrated / Nominal (s) |
| --- | ---: | ---: |
| .80 /20ms | .389 /2.292 | 14.16 /8.91 |
| .65 /60ms | .636 /2.438 | 11.73 /11.14 |
| .60 /80ms | 1.527 /2.437 | 11.96 /13.82 |

Calibrated共120/120完成且保持实际模式，每组“两个任务均成功”的20/20 reset-cluster exact95%区间为[.832,1]。这不是跨校准重复的区间。相对nominal-response的成功率前两组持平，stress的配对sign p=.125，**不支持显著/普遍成功率优势**。三组的末段轴向载荷更接近5N，但前两组耗时更长。Force-only在后两组载荷RMSE甚至更小（.424/.546N），位置未达标仍计0/40；不能只展示力误差。辅助任务/响应成本与bootstrap区间仅作探索性分析，不取代预先定义的单项指标。

最终统计为 `frozen_summary.json`，图为 `frozen_mechanism_results.png` / `.svg`，各组原始轨迹、冻结SHA/配置、失败和计费ledger全部保留。图左侧画的是每reset两次任务均成功的比例（因此stress nominal为16/20），与上表逐任务36/40的分母不同。绘图源码为 `scripts/cross_embodiment/summarize_mechanism_revision.py`。

本轮 LaTeX 已补齐条件性的 calibration→response→transport 推导和 hybrid-event margin。结果目前只检验受限结构模型的机制；neural response ensemble、随机 transport 相对确定性映射的必要性、未知接触参数、移动身体和实机尚无相同强度的证据。逆增益补偿也不等于完整 system-ID+MPC baseline。论文级别的新颖性和普适优势不能从这组原型直接推出。

### 冻结轨迹 GIF 回放

渲染入口为 `scripts/cross_embodiment/render_mechanism_gifs.py`；输出目录 `tmp/cross_embodiment/mechanism_revision/gifs/`，`index.html` 汇总预览。该目录仍为 ignored 本地工件，脚本和文档保留在源码中。

| GIF | 冻结样本 | 画面与结果 |
| --- | --- | --- |
| `panda_recovery_modes.gif` | Panda，seed8100，mode0/1 | 直撤与侧撤两种恢复模式，均成功，1倍速 |
| `xarm7_mechanism_comparison.gif` | xArm7，gain.65/delay60ms，seed12000/mode1 | direct接触丢失；calibrated成功（最终误差1.9mm）；force-only超时（2.8mm），2倍速 |
| `xarm7_stress_example.gif` | xArm7，gain.60/delay80ms，seed12403/mode1 | nominal-response超时、calibrated成功；有意选取的失败对照，不代表显著成功率优势，2倍速 |

原始评估未逐步保存关节角，因此以相同冻结配置、随机流、prior和transport重新执行，在原10步policy clock内记录每个控制步的关节角。7条回放的全部已保存trace、phase、reference通道逐项比较，最大绝对误差均0；显示模型逐状态TCP FK误差也为0。核对信息在 `manifest.json`、各GIF同名JSON及 `replays/**/verification.json`。显示模型不积分；原生link外观网格、示意刚性连接杆、支撑台和近侧孔壁透明不改变原评估物理。载荷标注沿用前述轴向接触aggregate语义。GIF逐帧解码通过，循环时长分别6.1s/8.5s/8.5s，检查记录为 `gif_decode_check.json`。这些回放用于可视化，不计为额外独立试验。

```bash
docker exec -w /workspace cross-ebdm-mechanism-cpu \
  python -m scripts.cross_embodiment.render_mechanism_gifs
```

容器需设置 `MUJOCO_GL=osmesa`、`JAX_PLATFORMS=cpu`、`PYTHONPATH=/workspace`，并保留原有CPU XLA设置。`--only panda|comparison|stress`可单独渲染，`--gallery-only`仅刷新HTML。

## 历史 CPU 功能结果（2026-10-12）

工作分支 `codex/cross-embodiment-transport`。Docker 为 `genedynamics/dev-cpu:local`，Python 3.10.20、JAX 0.6.2、MuJoCo/MJX 3.6.0，JIT 开启，`JAX_PLATFORMS=cpu`，`XLA_FLAGS=--xla_disable_hlo_passes=constant_folding`。GPU/CUDA/DDP 均 **not_run**。

### 软件与接口验收

- 公共 backend-neutral contracts、lazy registry、capability 拒绝、真实 6/7DoF FK/Jacobian/物理、source query、分组 dataset/checksum/masks、partial reset、固定 carry、sticky terminal。
- JAX compiled collector/query/training chunk；完整 numeric MJX 模型的逐世界 gain/delay/mass；执行 microbatch；两遍全局 fractional CVaR/VJP 与一次 Adam。CPU 大图顺序编译，按模块独立进程释放缓存。
- JAX/Torch 同权重 forward/input gradient、context 和 obstacle/contact transport Adam 对照；C++ 实际编译后的整批 forward 与 contact prior/noise replay。
- inference bundle 与独立 resume checkpoint；原子 tensor-generation/metadata 发布中断后保留上一代；实际训练恢复后的下一更新完全相同；实际物理 probe 恢复不重复计费。
- 固定/平面接触的 measured wrench/normal projection/anti-windup/子步峰值/物理时步细化、UR5e query，以及 Go2+Z1 真实浮动惯性/4ms QP/足支撑/显式 QP carry 和联合失败记录。

最终 **71 个独立 testcase 通过，0 failed、0 skipped**。JUnit、日志、版本/flags、actual-code fingerprint 在 `tmp/cross_embodiment/qualification_final/`。最终聚合记录在 `acceptance.json`；测试项以唯一 testcase 去重，单独复跑不重复计数。一次并行大图编译发生 OOM，重跑序列已改为串行大图；不能将退出 137 算作通过。

### 真实物理 learned-loop 回放

每格仅 2 条 CPU 功能轨迹。数字为成功条数，所有失败在分母中。Exploratory target configurations 已用于调试，不是未触碰的最终统计测试集。四种方法共享同一源 prior、reset、decoder 和源随机流。

| 环境 / 目标 | direct | uncalibrated | deterministic | full | 当前工件 |
| --- | --- | --- | --- | --- | --- |
| ObstacleTransfer / xArm7 | 2/2 | 2/2 | 2/2 | 2/2 | `learned_closed_loop_v4` |
| GuidedContactAlignment / xArm7 | 1/2 | 1/2 | 1/2 | 1/2 | `guided_learned_cpu_final` |
| KeyedInsertionRecovery / xArm7 | 1/2 | 2/2 | 2/2 | 2/2 | `keyed_learned_cpu_final` |
| KeyedInsertionRecovery / UR5e | 2/2 | 2/2 | 2/2 | 2/2 | `ur5e_keyed_learned_cpu` |
| MobileKeyedInsertionRecovery / planar xArm7 | 0/2 | 2/2 | 2/2 | 2/2 | `mobile_keyed_learned_cpu_final` |
| MobileKeyedInsertionRecovery / Go2+Z1 | 0/2 | 0/2 | 0/2 | 0/2 | `go2_z1_learned_cpu` |

接触 target diagnostic combination 为 `(gain=.95, delay=1, mass=.55)`；ObstacleTransfer 为 `(.87,3,.7)`。这些组合不在四个公开开发组合中，但已用于 exploratory debugging，不能只据“组合未见”声称 untouched test。UR5e 有公开 response pretraining，属于 seen-body/held-out-mechanism 工程对照。当前小模型不证明校准、随机性或 transport 的因果增益；例如 obstacle direct 同样成功，UR5e B=0 全组成功。

导向任务失败是约束力子步峰值超 25N，full 峰值 35.64N；保留失败并正确终止。Go2 learned full 在 socket lip 附近超时：TCP z≈.5946/.5948m，峰值约 7.27N，姿态偏差导致未入座；direct 还包含关节边界失败。该结果证明完整调用/记录链执行，**不证明四足迁移成功**。Go2 脚本物理 controller 的 clear/jammed 回放分别成功，峰值约 3.44/3.50N（`go2_z1_clear_cpu_v2` / `go2_z1_keyed_cpu_v2`）；脚本回放与 learned 结果分开。

Task 4 的 source prior 逐字节复用 Task 3 artifact，SHA256 为 `56bb867c9ad76a16fd980e6f20b38508ed5daa11436bd5084284e0e6522bc906`。平面底座支持 x/y/yaw，height/roll/pitch 受机械约束；地面 caster priority 已修正，真实 body qpos 保存在 learned bundle。Go2+Z1 为 25qpos/24qvel/18actuator，四足真实 contact，host OSQP stance QP，不含导航/步态或人形结果。其 full-carry feature 另含足锚点/高度/QP warm start/residual，不能只 pad 成 7DoF。

### 数据、训练和机制留出

| 当前数据 | attempted | success | valid controls |
| --- | --- | --- | --- |
| obstacle source | 4 | 4 | 3699 |
| obstacle public response | 32 | 32 | 32141 |
| obstacle excitation | 32 | 6 | 44634 |
| guided source v3 | 4 | 4 | 984 |
| keyed source v3 | 4 | 4 | 3430 |
| guided public response v3 | 48 | 28 | 13066 |
| keyed public response v3 | 48 | 29 | 40379 |
| planar mobile response v4 | 16 | 5 | 11940 |
| Go2+Z1 public response | 16 | 4 | 4228 |

Prior 是源模式条件 Gaussian MLP 加源 pose/reference feedback，不是 diffusion/flow。源先验只用 nominal Panda 成功且 observed/requested mode 匹配的训练窗，失败分母保留。Response 为三成员 MLP，normalization 仅训练行。`learned_cpu_v2` 的 source/response normalized validation MSE 约 .00689/.02677；guided response .31377、keyed response .17403、planar mixed .17978、floating mixed .26129，表明接触泛化和风险精度仍不足。

另保留公开机制组合 `(.9,1,.6)` / `(1.05,3,.9)`，数据角色 `response_validation`，未进入模型更新。冻结 obstacle response 在 1581 个保留窗上的 normalized MSE .01215、TCP delta RMSE .462mm；报告 `heldout_response_report.json`。这不是隐藏目标额外数据或真实安全证书。

预算 B=0/10/30/100 已在 obstacle 链跑通，接触用 B=10 和 UR5e B=0；所有 charged reset/probe/validation 有持久 ledger。短初态探针对接触/质量辨识较弱，某些预算下 validation 变差。ensemble disagreement 与 risk heads 未校准；不报告有效主动探索/单调预算收益。CPU 正式研究仍需冻结新的测试机制/初态、多种子及更有信息的探针，完整统计与 GPU 性能另验收。

### 设备驻留与复现

- Obstacle source→query→prior→transport→MJX K=2/H=10 短窗严格 transfer guard 通过：重跑 warm chunk .0942s、编译/预热约 136.9s（`device_pipeline_transport_retry`）。
- 接触同链 K=1/H=10 通过 strict transfer guard：首轮 warm chunk .0592s、预热约 338.2s（`contact_device_pipeline`）；最终版本复跑 `contact_device_pipeline_final` 同样通过，预热约 134.9s、热窗 .0300s。
- 首次编译、热执行、完整 host-QP 回放各自报告；CPU 时间不外推为 GPU 吞吐。冻结输入/noise/mode/context/初末 qpos 随当前 device fixtures 保存。
- [README](../../genedynamics/learning/cross_embodiment/README.md) 提供 Docker、data/train/loop/native/qualification 命令。LaTeX 的 CPU 工程说明与 Task 4 协议已同步，内置编译器编译成功。

以下内容为历史执行与首批增量，保留当时结果与旧标识。历史中“未实现”等描述仅对应当时阶段，不能覆盖本节状态；原工件 metadata/checksum 不改写。

## 历史执行回放

两个机器人、两种通路、两个 physics provider，各使用 B=2、L=10；2 ms
physics / 20 ms control、名义 0.5 kg 负载、gain=1、delay=0。每个环境
在 5 mm / 3° 内持续保持 0.3 s 后成功，且只产生一次 terminal event。
以下为每个 case 两个环境的最大终点误差。

| Backend / physics | 机器人 | 通路 | 终点位置误差 mm | 姿态误差 ° | 状态 |
| --- | --- | --- | --- | --- | --- |
| NumPy / MuJoCo float64 | Panda | 上方 | 2.9147 | 0.8774 | 2/2 SUCCESS |
| NumPy / MuJoCo float64 | Panda | 侧方 | 3.1979 | 0.8914 | 2/2 SUCCESS |
| NumPy / MuJoCo float64 | xArm7 | 上方 | 4.7389 | 1.6882 | 2/2 SUCCESS |
| NumPy / MuJoCo float64 | xArm7 | 侧方 | 4.7409 | 1.9113 | 2/2 SUCCESS |
| JAX / MJX float32 | Panda | 上方 | 2.9140 | 0.8774 | 2/2 SUCCESS |
| JAX / MJX float32 | Panda | 侧方 | 3.1958 | 0.8913 | 2/2 SUCCESS |
| JAX / MJX float32 | xArm7 | 上方 | 4.7385 | 1.6883 | 2/2 SUCCESS |
| JAX / MJX float32 | xArm7 | 侧方 | 4.7399 | 1.9111 | 2/2 SUCCESS |

这些是固定输入的功能 fixtures：同一 case 的两个环境采用相同零 reset noise，
不是 16 个独立随机实验，更不是方法成功率结论。不同初态/指令的 batch 独立性由
conformance 测试另外覆盖。未对扰动机制拟合、挑选目标数据或验证模式概率。

回放目录：`tmp/cross_embodiment/{numpy,jax}_{panda,xarm7}[_side]/`，含
`metadata.json` 和 `tensors.npz`；已核对 8 份 NPZ 的 SHA256。
Panda 模型 hash 前缀 `efada38b3c8f`，xArm7 为 `ad53bda9fe61`；完整 hash
在 metadata。目录 git ignored；新的 [执行入口](../../scripts/cross_embodiment/cpu_obstacle_transfer_smoke.py)
可生成语义命名的当前回放，但新运行可能有不同的命名/模型 hash，须单独记录。
具体 Docker 命令见 [README](../../genedynamics/learning/cross_embodiment/README.md)。

## 测试覆盖与边界

测试覆盖轻依赖导入、registry 委托与缺失能力、错误设备拒绝、配置校验、
失败 probe 计数、portable 工件 roundtrip/损坏拒绝；SO(3) 零角度正反向梯度、
旋转矩阵保持；真实机器人短段 MuJoCo/MJX 位置与力矩对照、独立 batch、
seed/sample ID、partial reset、跨 chunk 续接、13D mask/速率和 delay memory、
transfer guard；碰撞失败、位置/姿态保持、timeout、nonfinite 输入和 terminal 分母。
额外测试负载实际质量改变时名义补偿不改变，以及其他 solver 开启 JAX x64 后
本 backend 仍为 FP32，并恢复调用方的精度设置。

首批历史验收：**27 个独立测试项通过**（26 项完整回归 + 后新增的 1 项
x64/FP32 共存回归）。完整回归为 `26 passed`，JUnit 记录在
`tmp/cross_embodiment/pytest.xml`；新增测试以
`pytest test/cross_embodiment/test_execution.py -k fp32 --confcutdir=test/cross_embodiment`
独立执行，结果 `1 passed`。最终源码语法编译和 `git diff --check` 通过。
8 份回放另核对 checksum、shape/dtype、ObstacleTransfer force mask、terminal=1、
`valid.sum == final_steps` 及终点位置判据，全部通过。

full-arm geometry 使用保守 primitive 包络及已记录的安装/近邻排除，尚非实机
碰撞几何验收。clearance 是 contact-derived 且在 5 mm 截断，不是精确距离场。
以上历史测试只证明当时的 CPU 执行功能，不覆盖后来新增的异构机制 batch、
microbatch 和源查询。当前增量状态见下节；GPU 吞吐、训练梯度链、Torch/native
和移动身体/WBC 仍须独立验收。

## 本轮语义命名、源查询与数据增量

新增代码包括 `SourceQueryAdapter`、JAX 设备内固定迭代 IK/速度匹配/碰撞检查、
MuJoCo 诊断 provider、同 topology 异构 mechanism batch、执行 microbatch，
以及 [data.py](../../genedynamics/learning/cross_embodiment/data.py) 和
[数据 CLI](../../scripts/cross_embodiment/collect_obstacle_transfer_data.py)。
学习 prior/response/context/transport 尚未实现，GPU 仍为 **not_run**。

最终完整 CPU 回归：**49 passed，108 warnings，180.88 s，exit 0**。
JUnit：`tmp/cross_embodiment/obstacle-transfer-cpu-pytest.xml`。该套覆盖公共接口、
真实机器人执行、异构完整模型、执行/query microbatch、warm-start sample ID/
有限性保护、unsupported virtual state 状态 7 冻结、transfer guard，以及数据
角色/split/mask/失败统计。此前 eager slicing 的 int32 索引 host→device transfer
已修为静态 start/stop 的 JIT tree slicing 和 JIT concat，并包含在完整复测中。
49 是本轮最终 suite 数，不与历史 27 项或中间单项复测相加。

完整 suite 后做了纯协议清理：`SourceQueryBackend` 显式声明 `synchronize`，
frontend 委托 provider 方法，不再依赖内部 `.runner` 属性；数值流程未变。
随后 source-query 模块复测 **4 passed，29 warnings，32.85 s，exit 0**，
JUnit：`tmp/cross_embodiment/source-query-pytest.xml`。这 4 项属于已有 49 项，
不增加独立测试数量。

CP0/CP1/CP2 的本轮 CPU 出口通过。CP2 限固定 ObstacleTransfer workcell、
phase 0 源投影、脚本 source demonstrations 与数据基础；未覆盖任意场景/
contact-history 源查询、学习 prior/response/context/transport 或 GPU。

### 新命名的预期失败 fixture

使用 `cpu_obstacle_transfer_smoke.py --backend numpy --robot xarm7 --batch 1
--chunk 2 --timeout 0.04`，保存
`tmp/cross_embodiment/obstacle_transfer_timeout_numpy/`。实测符合预期：exit 1、
`scenario.task=obstacle_transfer`、final TIMEOUT、terminal=1、valid=2，
checksum/loader 通过。该 fixture 的**预期失败行为验收通过**，工件内
`cpu_functional=failed` 保持原值，不改写为任务成功。与后文完整 source 数据
一起提供 CP0 新名称下的真实成功/失败记录。

### 新命名的完整 Panda 源数据 fixture

已完成一次 NumPy/MuJoCo、nominal Panda、B=4 的 full-length 脚本示范生成：

| 项目 | 实测 |
| --- | --- |
| 环境 / role | `obstacle_transfer` / `source_demonstrations` |
| Episode | 4 attempted、4 SUCCESS、0 failed、0 truncated |
| 路线与 split | upper/side 各在 train、validation 有 1 条 |
| 有效控制步 | 3699；`prior.training_mask` 1859，`prior.validation_mask` 1840 |
| 终止 | 每条 episode 恰好 1 个 terminal，总计 4 |
| 时钟/记录 | physics/control 2/20 ms，H=10，L=10，K=150，30 s 上限 |
| 非有限数据 | `nonfinite_values_sanitized={}`，无 NaN/Inf 清理 |
| GPU 资格 | `not_run` |

工件目录：`tmp/cross_embodiment/obstacle_transfer_source_numpy/`。tensor SHA256：
`9ac583b2d0fd46ba0950338b54aef9e5ef9477c0389cb792489bd180690a4f9e`。
生成时完整模型 SHA256：
`e6b67b18ce5e098c884e3205994b8b7faa37f84792b1538d214eb5af6262d6cb`；
生成时实际 Python 源码 aggregate fingerprint：
`f3f201cc3094195fb418a32e5305ce24e47b9b9b3a2b408a58a4cdc62cba0713`。
这些 hash 来自该次 metadata；后续 source-query/microbatch 修复会改变代码
fingerprint，保留实际生成版本，不声称这份工件由最终修复源码生成。
该源工件最终已通过 strict loader、checksum、split/mask 与有效步核对。

数据保存 initial/chunk body qpos/qvel、实际 TCP semantic、controller carry、
raw/applied command、finite/valid/terminal mask、episode group/split/结果及
独立 observed route mode。`policy_features` 使用白名单；actual mechanism truth
只供 diagnostic/公开响应数据训练标签使用。失败及 truncated episode 会保留，
只有成功、真实 mode 与请求相符、finite 且 valid 的 source 步进入 prior mask。

本 fixture 只有 4 条 reset variation 的 scripted demonstrations，分组 split
验证数据管线，不能证明生成先验已训练、模式泛化或 cross-body 收益。route mode
来自 TCP 通路交叉证据，尚非全臂/负载 homotopy 分类。下一步为开发计划 CP3：
小型 stochastic source prior、显式噪声重放、冻结/加载和两通路行为验收。

### 公开响应数据 plumbing fixture

`tmp/cross_embodiment/obstacle_transfer_response_numpy/` 新生成 16 个公开开发
episodes，覆盖 Panda/xArm7 × upper/side × 2 groups × 2 mechanisms，共 320 个
valid 控制步。每条只运行 20 个控制步，因此 16 条均为 ACTIVE/truncated；
这是管线记录，不能计为任务成功，两份 prior training/validation masks 均为 0。
最终 strict loader/checksum/split/mask 核对通过；tensor SHA256 为
`3be159d9611bfb999a34301888eb380821c114ae61972c1b1f5dc69d06fc43a9`。
重复生成的 `_response_numpy_tiny`
是同类 fixture，不作为额外独立实验计数。

数据生成器在每个 chunk 边界同步并转换 NumPy，以离线保存 semantic、body/
controller carry 与 trace。这条 host 工件路径不作为未来 GPU learner 热路径，
也不以 CPU kernel transfer guard 通过替代 end-to-end GPU pipeline 验收。
执行/query microbatch 使用 compiled slice/join，完整 per-world model 仍随 B
存储，尾块维度与静态分块会产生各自编译/cache 项；实际 GPU 显存和吞吐未测。

JAX CLI 额外回放已成功（exit 0）：`--role response_development --chunk 4 --steps 8
--microbatch-size 2`，输出
`tmp/cross_embodiment/obstacle_transfer_response_jax_tiny/`。
16 个 episodes 覆盖两个 robot × 两条 route × 两组 reset × 两种 mechanism，
全部 ACTIVE/truncated，128 个 valid 控制步，H=10/L=4/K=2、microbatch=2。
strict loader/checksum、两份 prior mask 均为 0 已核对，GPU 仍为 `not_run`。
这是 JAX 短响应采集及数据 CLI 资格，不是完整任务成功。

该 JAX 工件 tensor SHA256：
`a10584b1ed2473223594ced72e321284fc2de797dbab048e81da82f82e10d2c6`；
生成时实际源码 aggregate fingerprint：
`3383f055ff4720ad26c83a054304d3d0d53565cdaa7e914323444910751c9abd`。
保留实际生成版本，不因后续纯协议委托清理而改写 producer fingerprint。
