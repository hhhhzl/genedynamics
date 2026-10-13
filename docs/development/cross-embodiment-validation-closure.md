# CPU 机制闭环：逐关修复与验收

日期：2026-10-12。当前研究开发分支 `codex/cross-embodiment-transport`，保留未提交改动；后续GPU交接继续使用此项目分支。承接 [六项理论验收](cross-embodiment-theory-feasibility.md)。工件根目录 `tmp/cross_embodiment/mechanism_closure_v2/`；旧失败、旧协议和冻结测试保留，不覆盖。

**执行状态：因磁盘耗尽，按用户最新要求停止本机实验、测试及安装，转交GPU机器。整体机制验收尚未完成。** 矩阵断点、代码/工件同步要求、后端能力缺口和接续顺序见 [GPU交接清单](cross-embodiment-gpu-handoff.md)。以下通过数仅指已保存的对应证据，不将中断任务写成通过。

## 验收规则

- 每一关区分实现测试、模型验证和真实物理执行；下一关的成功不能追认上一关失败。
- 初态按 reset group 划分；训练、误差验证、区间校准、封存评估分开。目标最终轨迹不进入拟合。
- 修改控制器、模型或优化目标后，以新协议和新测试种子重验；不降低成功率、几何、力、保持或预测误差阈值。
- 计费 target probe、公开身体开发、校准后的模型模拟、实际目标评估单列。相同策略的别名共享真实轨迹，不重复计入独立样本。
- NumPy/MuJoCo 是 CPU reference；数值前端继续复用 solver/provider 接口与命名张量工件。CPU 并发调度不是 GPU qualification。Torch/C++ 模型推理与物理 backend 的能力分别登记。

## 已完成、未完成与失败

| 关卡 | 本轮证据 | 状态 |
| --- | --- | --- |
| 状态条件短窗 learned response | 480 条校准模型轨迹；12 train、4 validation、4 interval-calibration reset groups，每组 12 candidates × 2 modes。全 carry + 原始命令 → carry delta，3-member 128×128。独立 validation：position RMSE 0.1671 mm，rotation-matrix RMSE 0.000189，force RMSE 0.5941 N；固定阈值 2 mm/0.03/1 N | 通过 .2 s 局部预测；还不能称全局共享 kernel 或自主整段 rollout |
| 条件 episode 均值网络 | 64×64、3 members，显式物理初态和有限候选身份；signature RMSE .03913、task-cost RMSE .20945、failure error 7/96。原阈值 .05/.15/.1 | cost 失败，保留 unsupported，不部署该均值模型 |
| 有限联合响应 law | 训练集中的完整响应向量联合采样，显式 uniform noise，portable affine artifact；冻结后新 8 reset groups × 24 cases = 192 条模型评估。平均 signature marginal W1 .00530、cost W1 .02641、failure-probability error .01910；固定阈值 .05/.15/.1 | 通过有限分布预测；不把边际 W1 当成联合 uniform certificate |
| 联合分布补充诊断 | 28D 完整 response 向量的经验最优运输，L1 mean ground metric，mean joint empirical W1 .008505 | 诊断通过；同一 development holdout，无额外 population certificate |
| law 选出的实际策略 | 冻结 law 选 candidate 10；8 新物理 reset groups × 2 modes。Direct 0/16，learned-law map 16/16，same-budget task-only 16/16，nominal 8/16；mode 全部16/16 | 物理闭环通过；强同类 baseline 持平 |
| 独立校准＋独立优化 | 复用 5 份此前已计费的独立校准，每份新 2 train resets、48 calibrated + 48 nominal model rollouts，5 新 test resets × 2 modes | 首轮 49/50，逐重复门槛未过；不是“已经全通” |
| 正则化独立重复 | 新 8 train resets/replicate，calibrated 与 nominal 各192条；共同平方速度 bias 正则 .1；新 test seeds 43000+。5次独立优化各10/10，合计50/50；task-only相同，Direct0/50、nominal11/50 | 通过；旧49/50保留。配对成本改善 .83640，5个上层cluster的描述性bootstrap95% [.79560,.87619] |
| 闭环反馈噪声＋真实 occupancy | 新 seeds 47000+，TCP位置20µm、姿态0.2mrad、力0.03N；高层反馈0.2s加噪，高率低层反馈精确。Direct0/10、calibrated10/10；状态条件神经response的 .2/.6s 各窗位置RMSE最大 .1254/.2867mm，force最大 .4777/.6469N | 通过固定2/3mm、.03/.05、1/2N阈值；递推使用实际外给命令窗，不是自主完整策略预测；窗口不当成独立试验 |
| 3D GuidedContactAlignment | 实际ridge、12mm tip、16/18mm guide、双轴斜率.1。冻结原Panda导向prior，新 source qualification4/4；5探针3fit2val，calibrated/nominal各48模型episodes；新45200+双模式 | v4校准映射10/10、模式10/10；task-only相同；Direct及nominal4/10。旧v1–v3全部失败保留 |
| 3D ObstacleTransfer | 冻结原Panda Gaussian prior；5自由空间探针3fit2val，gain/delay辨识；两搜索各20模型episodes；新48200+，原始统一30s期限、5mm/3°/.3s/碰撞/速度标准 | 四种方法均16/16、模式16/16；direct/calibrated/nominal的log-stiffness为0/.5/1，task-only与calibrated相同，共48次实际执行。新增20s协议源资格失败保留，不与30s结果混合 |
| MobileKeyedInsertionRecovery | 原Task3冻结Panda prior；真实Go2+Z1浮动身体和足接触，站立时vx .005m/s ×4s。5独立计费探针、12点arm gain/delay/support gain族、3fit2val；每对照24模型episodes；新55100+双模式 | 操作/支撑联合10/10、模式10/10；Direct5/10、模式5/10。task-only、arm-only、nominal同为10/10。arm-only probe force/peak误差1.1806/1.2299N超阈值，耦合model通过；不能宣称耦合模型独有成功率收益 |
| 身体×机制交叉 | Panda/xArm7各matched/derated，加已知UR5e留出身体；冻结同一prior，3份新计费校准、2份旧计费校准具SHA溯源；各同72模型预算，5新reset×2模式 | v1四个cell通过10/10；xArm7 matched7/10、3次侧撤分支峰值超力，矩阵整体失败。v2统一8训练reset、24候选、calibrated/nominal各384模型预算和可行性优先；已中断，保存64条source、1239条calibrated、147条nominal，最终物理评估0条，未生成总report |
| 回归及portable能力 | 合并当前NumPy/native主回归、预算/事件/恢复、异构batch/query与移动恢复无目标reset/step共53/53，JUnit为`cpu_reference_final.xml`；不同子集的重复测试不累加 | Docker虚拟磁盘I/O错误；整套JAX/Torch/MJX重跑待环境恢复，低磁盘安装已被自动审批拒绝，旧86/86不追认新版本 |
| 新环境时间步细化 | 绕障、3D导向和Go2+Z1各用冻结map、新seed、两模式，在2/1ms物理步长执行；Go2 WBC固定4ms，backend拒绝不整除的时钟配置 | 12/12实际任务与模式通过；仅支持三个开发点的定性稳定，不是所有接触分岔的uniform收敛证明 |

## 必须保留的理论区别

随机接触事件的均值预测误差与响应 law 的分布误差是不同对象。新的分布实验不改写均值网络失败；有限经验 joint law 也不等同共享 autoregressive neural dynamics。短窗神经 response、整段分布估计、校准模型及真实部署分别报告。

`2 epsilon_F + epsilon_opt` 仍只对同一声明目标和可行集合成立；它不承诺优于 task-only/MPC。独立重复首轮暴露训练初态覆盖与命令变化正则缺项，因此新协议增加覆盖并补上共同正则，之后使用新冻结测试。不能用旧成功样本宣布新优化算法的通过。

移动身体的公共接口指定笛卡尔阻抗。开发中的acceleration PID和操作空间惯量映射均失败，不能作为正证据。最终Go2在stance QP内优化机械臂的物理力矩阻抗目标，复用[800 N/m]×3+[60 Nm/rad]×3刚度和六个log-ratio通道；足接触/摩擦锥/力矩/加速度/QP残差保持约束。持久积分与QP状态一起显式存入原52-slot WBC carry。接近阶段的lip擦触不等同已建立座底支撑；浮动身体的失去座底事件从实际入座hold witness起算，保持原2mm/2°/力/.3s成功标准。站稳准备50步属于计费reset内部，任务计时从准备结束开始。

`HiddenTargetEnv.probe_policy`现在保留skill和body_twist的完整命令历史；此前只保存skill会使移动探针无法正确重放。`MobileContactMechanismSpec`区分arm servo/delay与support servo，未知值只用于构造目标物理过程，不进入prior/decoder。已知名义质量、足地摩擦和站立能力仍是本协议条件；未知payload继续使用验证拒绝负对照，不声称已辨识。

停止前的CPU reference使用macOS项目隔离环境：MuJoCo3.6.0、NumPy2.2.6、SciPy1.17.0、OSQP1.0.4、Python3.12.3；Linux原SciPy1.15.3 wheel在本机不兼容。跨平台两条完整Panda源轨迹的状态/命令/事件完全一致至数值精度，TCP差≤1.12e−15m、normal force差≤1.04e−11N，见`host_replay/report.json`；不是所有接触分岔的uniform证明。

磁盘不足时只对已有实验NPZ做可逆外层gzip，metadata与解码后的原始SHA不变；`load_bundle`解码后仍核对原SHA，未删除失败证据。新保存使用压缩NPZ。未完成Torch安装隔离在本项目临时env中，未改系统安装。JAX/Torch缺失、安装被拒绝与代码测试失败分别记录，不混成通过。

停止前最近一次独立审计为446条不同物理评估轨迹、0次成功/模式判定mismatch；`acceptance.json`的`passed=false`，原因包括v2矩阵未完成。审计脚本已加入后续12条时间步评估入口，但尚未重新运行，不能把它们算进已审计446条。4ms WBC时钟修改后的9/9回归与53项存在重叠，不相加。最终审计在交接后完成。

## 主张 → 实现 → 证据 → 允许结论

| 理论对象 | 当前实现/验证入口 | 独立证据与允许结论 |
| --- | --- | --- |
| 干预的任务相关可辨识性 | `PhysicsTwinResponse`、`fit_contact_twin`、`ResearchAudit.separation` | free probe对仅接触变化响应相同，loaded probe不同；真实族内5探针独立验证通过。κ与族覆盖仍是定理前提，小残差不证明前提成立 |
| 状态与历史充分性 | q/qdot、原始命令、完整controller carry；模型自行重建，不复制隐藏warm start | 状态条件短窗网络和实际访问状态的3窗递推通过；不是任意历史或未建模迟滞的充分性证明 |
| 联合响应分布 | `FiniteResponseLaw`完整向量联合采样；case是有限candidate×mode，reset被边际化 | 新192模型episodes的分布指标通过，另16/16实际部署；联合经验距离为补充诊断。均值网络cost失败独立保留，不能替代full-law验证 |
| 源—目标coupling | 冻结source mode/noise；实际轨迹重分类 | shuffle仍12/12成功且marginal6:6，却对应0/12；证明只看成功/target marginal会漏掉耦合错误，不证明anchoring总能提高task cost |
| 额外transport随机性 | 确定性map与显式附加noise对照 | 两者12/12，暂无独立收益。源随机性已可产生多峰，不把额外noise列为必要机制 |
| 受限优化误差 | 完整枚举声明候选池；共同bias正则、同预算task-only/nominal | `2ε_F+ε_opt`限同目标/同可行集合；有限枚举不等于连续类全局优化。task-only持平被保留，未建立独有控制优势 |
| 跨校准统计 | 5次独立校准＋新模型优化＋新reset评估 | 修订协议各10/10，合计50/50；旧49/50保留。描述性5-cluster bootstrap不当成总体或uniform模型证书 |
| 身体条件化 | Panda/xArm7 factorial、已知UR5e topology留出，prior不变 | v1 xArm7 matched7/10已否定旧搜索鲁棒性；v2共同新协议在模型搜索阶段中断，待交接后接续。不能把已知URDF称为未知形态发现 |
| 移动身体耦合 | `MobileContactMechanismSpec`、`mobile_response` NumPy provider、真实Go2 stance QP | coupled独立probe通过，arm-only force预测拒绝；同任务操作/支撑10/10，但同预算对照也10/10。只支持有限耦合预测及站立操作适用范围 |
| 模型外处理 | 未知payload负对照、持久probe ledger、validation拒绝 | .8kg真值/.5kg模型验证失败，5探针计费，4请求任务计unsupported失败；不能声称已辨识载荷或绕过拒绝继续部署 |
| 可移植与并行 | solver/provider前端、portable命名张量、native ABI v1、明确时钟/完整carry | 当前53项NumPy/native回归和12条时间步评估通过；GPU热路径、Torch物理、Rust、真实多卡均未验收。CPU进程并发不是GPU并行证明 |

## 复现入口

```bash
python -m scripts.cross_embodiment.collect_conditional_response --study STUDY --output NEW_DATA
python -m scripts.cross_embodiment.qualify_conditional_response --dataset DATA --study STUDY --output NEW_FIT
python -m scripts.cross_embodiment.validate_response_law --dataset DATA --study STUDY --output NEW_LAW
python -m scripts.cross_embodiment.collect_conditional_response --study STUDY --output NEW_TEST --test-only
python -m scripts.cross_embodiment.validate_response_law --dataset DATA --study STUDY --output LAW --test-data TEST
python -m scripts.cross_embodiment.validate_repeated_transport --output NEW_REPEATS --training-clusters 8 --train-seed-start 42000 --test-seed-start 43000 --regularization .1
python -m scripts.cross_embodiment.audit_noisy_response_occupancy --output NEW_NOISE
python -m scripts.cross_embodiment.validate_spatial_guide --output NEW_GUIDE
python -m scripts.cross_embodiment.validate_obstacle_mechanism --output NEW_OBSTACLE
python -m scripts.cross_embodiment.pilot_mobile_phase --output NEW_MOBILE --stiffness 1
```

批量采集使用已显式构建的C++ CPU inference provider，设置 `CROSS_EMBODIMENT_NATIVE_LIBRARY` 指向本机 ABI v1 库。原Docker环境已有磁盘I/O错误，当前不继续使用；Linux GPU机器须重新构建`.so`，不能复制macOS的`.dylib`运行。上列命令为参考入口，不在本机继续执行。所有正式完成状态必须来自各自 `report.json`，不能把“脚本已启动”写成“已通过”。
