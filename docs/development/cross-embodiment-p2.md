# P2：匹配接触的 GPU 模型执行与 CPU 确认

**后续研究路线更新（2026-10-14）：** 先确定成熟强 baseline 和四任务比较协议，
再据任务表现确定贡献与方法修正，见
[比较计划](cross-embodiment-baseline-comparison-plan.md)。本记录保留开发实验事实；
当时加入论文的报告、表图与旧 PDF 已迁至
[开发报告归档](cross-ebdm-experiment-reports/README.md)，当前 LaTeX 是论文草稿。

承接 [P1.5–P1.7](cross-embodiment-p1-refinement.md)。本阶段的实现与实验
保持固定基 Panda/xArm7、matched keyway、TCP signed wrench、当前
PhaseGaussianPrior、共享反馈 decoder 和 recovery_v2 的 24 个候选。
不迁移空间导槽、移动身体、OSQP WBC、Torch 物理或多卡训练。
沿用的 25 N 峰值判据取“单接触力范数最大值”和“合成 TCP 力范数”中的
较大者；10 N 侧向判据取合成 TCP xy 力。它们不是同一个读数量。

## 协议与输出

配置与结果继续使用 domain/task/category/method，再接 level/seed：

- `qualification/cross_ebdm_gpu.yaml`：冻结 CPU 案例、GPU 接触语义、batch 与性能。
- `validation/cross_ebdm_gpu_coverage.yaml`：8 个训练 reset、16 个独立 CPU
  模型验证 reset、12 个 CPU 目标评估 reset；每 reset 含两模式。
- `validation/cross_ebdm_gpu_distribution.yaml`：复用同一模型搜索池，
  12 个新 law reset clusters；full/task-only 与各重加权对照同容量。

CPU 参考首次冻结于 `results/_development/p2_20261014`，空张量审计聚合
修复后的完整执行使用 `results/_development/p2_20261014_v2`，显式复用
原 CPU 参考，不重跑目标诊断。内部镜像完整
`results/arm/keyed_insertion_recovery/<category>/<method>/`。
训练/验证/评估/law 的 reset 全部增加 100,000,000，与前两版开发数据隔离。
校准仍引用已有三份×两条件的计费实例，新增 probe 为零；不增加独立校准数。

## P2.0 与 P2.1

冻结 9 个历史开发案例：2 个 Panda 源身体与 7 个 xArm7 目标诊断。
包含成功、早期超力、后期轴向支撑丢失、超时、两模式、gain .65/delay3。
保存每个 0.2 s 边界的 qpos/qvel/warmstart/ctrl/time，以及完整控制 carry，
保留 raw commands、事件计数和已审计的 prior/map hashes。目标诊断单独计账。

JAX 后端增加工具侧力符号、接触点到 TCP 的力矩平移、轴向支撑 witness，
reset 复用 NumPy 每个 `(seed, sample_id)` 的 task-space IK，避免广播单个初态
或重复施加噪声。新 phase 融合执行器使用明确的 per-episode 随机流、
共享 reference decoder 和 portable affine 参数；时间 scan、世界 vmap。

legacy 接触保持 FP32；匹配接触采用 FP64 物理与控制 carry、FP32 冻结模型
和原始命令，精度作用域不修改其他调用者的全局设置。

## 两个不同的资格出口

1. GPU 模型搜索：符号/坐标/TCP 语义、轴向 witness、事件 carry、稳定终止
   掩码、partial reset、reset 重排与 microbatch 一致性通过，稳态执行禁止
   隐式主机传输。每个候选共享同一随机流、几何、控制与判据。
2. 替代 CPU 目标物理：额外要求冻结长闭环 status、终止时间、末端位置与
   峰值力满足预声明容差。不能用单步一致或 GPU 成功率更高代替这一出口。

开发诊断已发现 CPU/MJX 接触解算差异可累积并改变超力/支撑丢失事件；
FP64 仍不保证长时等价。因此 GPU 只进入模型训练搜索；所有选中映射
继续通过独立 CPU calibrated-model 验证，最终目标任务继续在冻结 CPU
环境执行。GPU twin 与 CPU 目标的误差应单列，不借 CPU occupancy 的
高准确率声称 GPU twin 也准确。该误差已加入论文理论分解。

## 运行与资源

默认物理 GPU 1，通过 `CUDA_VISIBLE_DEVICES=1` 映射为 `gpu:0`。
GPU worker 使用既有 `.runtime/cross_ebdm/` 环境与缓存，没有新增系统依赖。
主配置的 `backend: numpy/device: cpu` 描述编排与目标参考；模型搜索通过
`model_collection_backend: jax` 和 `gpu_device` 显式选择 GPU。
GPU 0 先逐元素复现 GPU 1 的五个冻结 xArm7 案例，随后两张卡分别执行
独立校准实例；这是独立单卡实验调度，不是分布式 learner 验收。
两个条件中除输出路径外完全相同的 nominal 请求可复用已提交结果，
逐条保存输入相等与 tensor hash 证明；逻辑模型预算与物理调用数分别报告。
稳态与首次编译分开计时，报告实际设备、batch、有效控制步和显存；
CPU 串行对照不等于多进程 CPU 吞吐。CPU timer 排除构造、reset、加载与
工件 IO；原始 benchmark scope 中的 IO 描述不准确，汇总报告单独更正。
GPU 固定时域包含终止后的 masked padding，另报有效控制步，避免混淆。

每配置 10 GiB 上限、20 GiB 空闲保留，阶段 timeout 与 16 MiB 日志上限
沿用研究 runner。每模型/目标尝试先写 pending，再原子发布 tensor bundle
与 complete；已计费但未完成的尝试不自动免费重放。GPU padding 也记模型调用。

```bash
.runtime/cross_ebdm/venv/bin/python -m scripts.cross_embodiment.run_p2 \
  configs/arm/keyed_insertion_recovery/qualification/cross_ebdm_gpu_replay.yaml \
  --development-root results/_development/p2_20261014_v2 --resume

.runtime/cross_ebdm/venv/bin/python -m scripts.cross_embodiment.run_refinement \
  configs/arm/keyed_insertion_recovery/validation/cross_ebdm_gpu_coverage.yaml \
  --development-root results/_development/p2_20261014_v2 --resume

.runtime/cross_ebdm/venv/bin/python -m scripts.cross_embodiment.run_refinement \
  configs/arm/keyed_insertion_recovery/validation/cross_ebdm_gpu_distribution.yaml \
  --development-root results/_development/p2_20261014_v2 --resume
```

这些命令用于已完整工件的重入；运行中的阶段不重复启动。
原始 GPU 物理验收未通过并保留；`model_search_admission.json` 是后续
27 个保留状态上的受限用途审计，零新 episode/probe，并明确要求 CPU 验证。
后两项需要该审计明确允许模型搜索；不允许静默 CPU fallback。
当前阶段的最终通过状态与科学结论以保存的 qualification/scientific reports
和独立审计为准；方法优越性不由工程验收自动继承。

## 最终结果与验收

P2 的实现、两轮实验、独立审计和真实 CLI 重入均已执行完。验收部分通过：
单卡 phase/affine/MJX 批量链可以执行，原 CPU 目标物理等价门槛未通过，
完整方法相对强 task-only 的独立任务收益仍未建立。不能据此启动正式 GPU
目标主表或宣称端到端 GPU 加速已验收。

| 设置 | Full | Task-only | Full/Task-only 拒绝 |
| --- | --- | --- | --- |
| 共同映射 | 46/144 | 70/144 | 96 / 72 |
| 条件 law | 106/144 | 119/144 | 36 / 24 |
| 各自重加权 | 142/144 | 143/144 | 0 / 0 |

共同映射中两方法共同部署的 48 个请求全部复用同一物理轨迹，响应差为零。
完整方法额外拒绝 24 个请求，全部请求的有界任务成本差为 +0.147560，
对应停止质量上界为 1/6。条件 law 共同部署 108 个请求，其中 84 个轨迹相同；
共同部署子集的平均响应差为 −0.002471，但全部请求的任务成本差为 +0.074480，
并有 12 个额外拒绝。不同部署集合上的响应均值不能直接比较。

独立 CPU 模型验证还观察到：共同映射 matched/7100 的 full-task 均值差
为 +0.047454，条件 law matched/9100 为 +0.023225，均超过训练带 0.02。
这不反驳条件定理，统一模型/采样误差并未认证；也不把小样本均值当总体界。
相同冻结映射与接受标记的相等属于结构事实，不构成独立改进。

重加权提高任务成功但改变模式质量：full 成功模式数 110:32，task-only
为 102:41。固定 mode0 诊断为 144/144，成功模式数 144:0；它只覆盖一支，
不是训练过的最佳单模态 baseline。shuffle 为 106/144、原始源对应为零；
固定额外噪声为 104/144，没有支持其必要性。

两轮新增 **612 次 CPU 目标确认**、零新 probes；另有 7 条完整 CPU 开发
诊断与 1 条短窗逐步对照，均不回流选择。校准仍为原六份已计费实例。
GPU 搜索保留 4,608 条逻辑请求，实际执行 3,520 次模型调用（含 42 padding），
1,130 条 nominal 请求复用已提交结果。缓存交接保护触发时未覆盖已尝试请求；
剩余 22 条由原进程执行，原失败与最终计账都保留。

独立审计覆盖共同映射 5,784 条、条件 law 1,248 条计划轨迹，零 mismatch；
GPU 搜索的全部 4,608 条请求另核对物理验收与别名输入/哈希，均通过。
资格另审计 18 条 CPU/GPU 配对轨迹及 27 个 witness 检查点。
79 项相关测试通过；`pip check` 无冲突。三个真实完成态 CLI 重入前后
19,573 份 metadata/attempt/ledger 的路径与 SHA 完全相同，零新增模拟或 probes。

最终汇总为
`results/_development/p2_20261014_v2/results/arm/keyed_insertion_recovery/qualification/cross_ebdm_gpu/_reports/program-report.md`。
全量数值见同目录 JSON、两个 validation 方法的 `scientific_report.json`、
`_reports/common-execution-audit.json` 与 `_reports/cpu-model-task-band.json`。

GPU 资源由 1 Hz 外部采样补充，两个生产进程显存采样峰值约 390 MiB；
这不是 allocator 高水位保证。64 环境 kernel 稳态约 70–77 s，首次编译单列；
完整 worker 还包含 reset、输入准备、IO、派发及预算计账，不能只报告 kernel
时间。不同 CPU/GPU 终止结果下的串行计时比不证明等价物理加速。

论文新增后端误差项、停止成本分解与不同模式权重的 TV 项，P2 表格和
图由独立审计报告生成；没有覆盖 P1 结果。新版 GPU 入口是
`JaxPhaseContactPolicy`；旧 `ContactSourceQuery/JaxContactPolicy` 的 learned
GaussianSkillPrior 链不继承本次资格，仍需独立迁移/验收。

下一步优先解决 GPU 模型偏差与稀疏验证下的可行性/停止策略，再检验同容量
强对照下的任务--响应取舍。当前证据不支持直接扩大正式训练或主表。
