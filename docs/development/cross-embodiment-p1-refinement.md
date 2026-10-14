# P1.5–P1.7：可行支持、条件 law 与理论修订

本轮承接 [P0/P1](cross-embodiment-p0-p1.md)，使用本机保存的新工件。
后续 GPU 实现与扩大覆盖见 [P2](cross-embodiment-p2.md)，独立保留本轮工件。
旧 CPU 数字仅保留为历史记录，不进入当前可复核论文结果。
所有阶段均是 development；任务真值不参与搜索、策略训练或模型选择。

## 协议与路径

结果继续对应 `configs/<domain>/<task>/<category>/<method>.yaml`，
内部为 `level_<condition>/seed_<calibration_seed>/`，不用 project/run_id。
本轮 development root 为 `results/_development/p1_refinement_20261014`，
按既有 runner 规则镜像完整 `results/arm/keyed_insertion_recovery/validation/`。

| 配置/输出方法名 | 作用 |
| --- | --- |
| `cross_ebdm_diagnostics` | P1.5 只读审计及四条单独计账的复现 |
| `cross_ebdm_feasibility` | v1 固定候选池及共同映射可行性 |
| `cross_ebdm_distribution` | v1 同模型条件 law 对照 |
| `cross_ebdm_recovery_feasibility` | v2 单列回撤前缀补偿 |
| `cross_ebdm_recovery_distribution` | v2 条件 law 与完整/控制重加权对照 |

两个版本都保留。v2 根据 v1 开发结果设计，不是独立确认性实验，不能把
多个版本中最好的 cell 拼成正式主表。三个校准实例为 7100/8100/9100；
v2 显式引用同六份计费校准，不能再计作六次新的独立校准。

## P1.5 的结果

对已有 240 对 actual/occupancy 轨迹只读分析，终止状态和时间全部一致。
matched calibrated 的 17 次支撑丢失、10 次超时、3 次成功均被 twin 复现。
支撑丢失组的 transported force 约 5 N；超时组约 12.5 N，实际末力
10.10–10.83 N，任务要求仍是 5±2.5 N。derated 唯一失败侧向峰值
17.30 N>10 N；总力约 22.01 N<25 N。

四条开发复现另记四次目标任务访问、零个新 probe，全部重现原终止状态。
增加的事件 side channel 包括 axial support、contact_seen、loss/hold
计数、approach integral、控制步初始 established 和实际法向速度。
JAX/Torch/native 的模型可移植性不等于这些新物理语义的 GPU 资格。

成功保持窗口的最小力裕度约 0.0008–0.0144 N。但首次满足 hold 就终止
会使窗口起点自然靠近阈值；这是描述性裕度，不能直接解释为闭环扰动鲁棒半径。
模型与实际共享控制/物理实现，预测准确也不能排除共享实现问题。

## 两版 P1.6 的共同规则

- 冻结同一个本机 Panda phase/mode reference prior；共享 decoder、controller、
  wrench、几何、15 s 期限、25 N 总力和 10 N 侧向限制，不放宽成功判据。
- 每条件三份独立校准，五个 probe 分为 3 fit/2 validation；六次验证均通过。
  模型结构与身体已知，受控真值接触参数保持名义点；拟合器检查固定 gain/delay
  及摩擦/时间常数族，不证明任意未建模接触或工具惯量都可辨识。
- 每版固定 24 个候选×4 训练 reset×2 模式，calibrated/nominal 各 192
  模型 episodes/实例，两条件三校准共 2304；full/task-only 共享整个池。
- 先做两模式有限训练可行性过滤，再以 task-only 最小任务成本+0.02 为预算带，
  在带内选最小归一化响应差。另用四个模型验证 reset；拒绝后不改挑别的候选。
- 最后每实例五个新 reset、两模式。拒绝计失败；相同冻结映射/随机流的别名
  共享物理执行，逻辑请求分母与实际不同目标访问次数分开。
- train/model-validation/evaluation/law roles 全局分离，同角色可跨条件配对。
  v2 全部 reset 加 50,000,000，和 v1 的开发/评估隔离。

24D 响应特征按阶段包含出现、时间、均值 xy、均值法向力、峰值侧向力。
xy 转为 2 mm 单位，法向力转为 2.5 N 单位，各平方误差截到 [0,1] 后平均。
这只是声明的响应几何，不是完整连续轨迹 law。终端 task cost 沿用原有界定义。

v1 保留旧 12 个候选，增加中间力倍率与有限全阶段阻抗。P1.6 full/task-only
均为 derated 27/30、matched 10/30；matched 两个校准格拒绝，不能只报已部署
一格的 10/10。独立审计 2670 条计划轨迹、30 probes，零 mismatch。
新目标物理评估 120 条、预测 120 条；其余为 source/search/model-validation。

v1 的部分 stage-only 候选在回撤阶段 1.04 s 已发生总力/侧向超限，后缀
对齐/插入参数尚未激活。v2 保留六个代表候选，并加入三种有界回撤阻抗
profile×三种终段力倍率×两种对齐/插入刚度，共 24 个。
回撤补偿、对齐补偿和插入补偿独立，终段力倍率不放大初始加载；所有对照共享。
物理控制器未修改，旧默认参数的执行语义保持；新模型仍是 portable affine。

另外固定 24 条模型配对诊断，不回流搜索、零个新目标访问：在 v1 已观察
失败的 s17100001/mode1 上，六个零回撤补偿控制在 1.04 s 全部超力，
其失败前位置和指令逐元素相同；18 个带回撤阻抗的候选均成功。
这是单 reset 模型内反事实，不是目标真值 oracle 或总体因果结论。

v2 P1.6 full/task-only 均为 matched 30/30、derated 20/30；后者一个
重复在独立模型验证失败后拒绝，已部署的两个重复均10/10。不能把已执行
50/50 改写成总请求60/60。独立审计2866条计划轨迹，零 mismatch；
新增目标物理评估170条，新增probe为零。

## P1.7 与统计范围

逐模式组合相同有限候选，无额外搜索 rollout，组合容量对 full/task-only 相同。
每实例六个新 reset groups，每组 antithetic uniforms 产生两种原始源模式。
full、task-only、nominal、identity reweight、task-only reweight、coupling shuffle、
extra noise、无 source mode 全部保留；v2 额外开放 full reweight。
模型支持过滤、概率权重和验证在真实目标评估前冻结。

v1 full/task-only 六次都选同映射，均 51/72（18 个拒绝）。task-only reweight
70/72，实际成功模式数 41:29；identity reweight 全部拒绝，0/72。
shuffle 51/72、成功模式数 27:24 与 full 相同，但原始源对应为零。
extra noise 52/72，单次开发差异不足以证明额外噪声有用。
独立审计 372 条计划轨迹，零 mismatch；新实际目标 132 条，零个新 probe。

v2 P1.7 full 64/72、task-only 65/72；两者均有6个拒绝。
full reweight 70/72、task-only reweight 71/72，全部请求执行。
按条件分开，full 为 derated30/36、matched34/36，task-only为30/36、35/36。
额外噪声65/72，固定mode0诊断70/72但只保留一个模式，不能据此宣称最佳单模态
控制器的收益。shuffle64/72，与full相同实际模式质量而原始coupling为零。
独立审计540条计划轨迹、186条实际目标，零 mismatch；新增probe为零。

matched/seed8100 的新law reset67100303中，full的mode0在10.06s丢失支撑，
task-only成功；两者共享的mode1在1.0s超力。该重复full相对task-only的实际
样本平均task cost高0.06788，超过训练带0.02，同时响应成本低0.00309。
这不是定理反例：统一成本误差假设没有认证，有限训练均值不等于真实期望。
它明确反对“响应loss下降就说明任务收益”的解释。

重加权改变原始源模式 law，因此“与原始 draw 对应”的下降不是自动错误；
必须同时报告源支持保留、任务成功、模式质量与停止概率。当前经验可行性 mask
不是真实不可行证明，既不能泄漏最终 target oracle，也不能用“未找到”宣称全局无解。

配对 task 审计按 reset 内平均模式、条件于冻结策略和一次校准。
当前 n=5/6 的保守 Hoeffding 下界通常不认证改进；不能把极小 twin 误差
代替这些置信界，更不能把控制步当独立试次。

## 根据结果修订理论

论文新增 `sections/finite_feasibility_theory.tex`，明确四个逻辑层：

1. 校准误差：模型能否预测访问分布；小误差可以准确预测失败。
2. 可行支持：共同映射需要各模式可行集合有交，条件映射需要各集合非空；
   回撤前缀不变的族不能修复该前缀上的不可撤销失败。
3. 目标偏置：相对更宽 task 类的比较必须包含候选表达缺口与 anchoring 偏置；
   同集合的目标次优界不能变成相对任意 task-only 控制器的优势。
4. 分布取舍：模式边缘与源条件 coupling 分开；保留不支持的质量与重加权的
   任务/覆盖代价分开，额外随机性不预设必要。

新增任务预算带命题：在同一非空集合、统一成本误差 ε_C 的条件下，
真实 task cost≤同集合最优+δ+2ε_C；响应比较在缩紧的真实任务带内成立。
这里 δ=.02 是开发选择规则，统一误差假设没有被本实验认证。
重加权若改变 law，需要在同一联合候选集合重新比较，不能继承固定权重下的非劣界。
任务成本误差还要分为模型偏差和reset采样误差；即使枚举对有限训练样本
精确最优，也不能把期望目标中的优化误差当作零。当前完整方法的独立增量
仍未建立，不能用表达族的改进或重加权收益替代同模型强对照。

## 运行、恢复与审计

```bash
.runtime/cross_ebdm/venv/bin/python -m scripts.cross_embodiment.run_refinement \
  configs/arm/keyed_insertion_recovery/validation/cross_ebdm_recovery_feasibility.yaml \
  --development-root results/_development/p1_refinement_20261014

.runtime/cross_ebdm/venv/bin/python -m scripts.cross_embodiment.run_refinement \
  configs/arm/keyed_insertion_recovery/validation/cross_ebdm_recovery_distribution.yaml \
  --development-root results/_development/p1_refinement_20261014
```

重入需 `--resume`，代码/配置/资产/版本必须与原契约一致。
v1 采集源码在其 `_environment/source-overlay.zip`，当前恢复该旧协议必须
使用旧 overlay；当前代码不能绕过 fingerprint 直接续写旧研究。
每条实际/model 尝试先记 pending 再执行，缺少 committed bundle 不免费重放。
每配置输出树 10 GiB、文件系统保留 20 GiB，阶段 4 h、日志 16 MiB。

完成后执行 `scripts.paper.cross_ebdm.summarize_refinement --root <配置输出>`，
检查所有 manifests、tensor checksum、collection/request fingerprint、事件计数、
frozen map、probe ledger 和目标验收；不生成新仿真。
开发报告表格保留在各实验 `_reports` 中。需要额外 LaTeX 导出时使用
`--report-export-dir docs/development/cross-ebdm-experiment-reports/refinement_exports`；
报告不写入 `latex/`。旧 `--paper-dir` 仅为参数兼容别名，同样拒绝论文目录。
论文草稿中的开发报告已于 2026-10-14 迁出，见
[迁移索引](cross-ebdm-experiment-reports/README.md)。

理论草稿用本地 [Tectonic 0.15.0 官方发布](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.15.0)
编译，binary/hash 与 bundle v33、系统 CJK font hash、TeX inputs 留在 build provenance。
编译器和缓存仅位于 `.runtime/cross_ebdm/tex/`，随已有环境清理一起删除；
不修改系统 TeX 或数值 Python 依赖。

## 当前执行状态

P1.5、v1/v2的P1.6与P1.7全部执行完成，四个协议的独立审计均通过。
共30个新probe、612条新实际目标任务（含4条诊断复现）；仍是同六份独立校准。
全部脚本、拒绝、失败和两个版本保留，逐格可行性与transport增量门槛尚未全部通过。
最终汇总位于 `cross_ebdm_recovery_distribution/_reports/program-report.md`；
每协议以 `scientific_report.json` 和 `_reports/independent-audit.json` 为准。
最新实现47项runtime/config/finite-workflow测试、29项受影响物理回归均通过，
0失败/跳过；新回撤映射的JAX/Torch/native前向与保存加载均已验证。
