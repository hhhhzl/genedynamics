# Cross-embodiment P0/P1：从空目录重建

本轮不使用旧 CPU 工件。`docs/development` 中的旧数值只作为开发记录，
新的模型、目标探针、搜索及实际评估均重新生成。P1 是有限阶段 affine
transport 的机制预实验；新版匹配键槽的 GPU physics 属于 P2。

后续 P1.5–P1.7 已执行两个保留版本，当前结果与理论修订见
[可行支持、条件 law 与理论修订](cross-embodiment-p1-refinement.md)。
它们使用本轮重新生成的 prior，并显式记录校准复用及额外目标访问。

## 配置与结果位置

沿用现有任务目录，不新增 project/run_id 结果体系：

```text
configs/arm/keyed_insertion_recovery/
  _base.yaml
  validation/cross_ebdm.yaml

results/arm/keyed_insertion_recovery/validation/cross_ebdm/
  workflow_contract.json
  artifact_policy.json
  _qualification/
  _source/teacher/
  _source/prior/
  level_xarm7_derated/seed_4100/
  level_xarm7_matched/seed_4100/
  ...
  acceptance.json
  scientific_report.json
  workflow_report.json
```

后续其他方法在同一任务下增加 `main/`、`baseline/`、`ablation/` 或
`validation/<method>.yaml`；其他操作任务按 `configs/arm/<task>/` 扩展。
移动身体按现有机器人类别组织对应配置，共享 prior 用显式工件引用。

开发结果复用 `ExperimentConfig.use_development_output_root`。例如
`--development-root results/_development/p0p1_example` 会把完整项目相对路径
镜像到 `results/_development/p0p1_example/results/arm/...`。
这里额外的 `results/` 是既有重定位规则，保留该规则以兼容其他实验。

## 可删除环境

从仓库根目录执行：

```bash
python3 -m scripts.research.environment create --torch-cpu --gpu 1
bash scripts/setup/setup_mujoco_menagerie.sh
```

环境位于 `.runtime/cross_ebdm/`，包括 venv、pip/JAX 缓存、临时目录、
native `.so`、安装 wheel/hash 报告及实际版本锁。使用 Python 3.12、
JAX/jaxlib 0.6.2、CUDA 12.6、MuJoCo/MJX 3.6.0；Torch 使用 CPU 参考版本。
不安装系统 CUDA、不修改全局 Python，不需要 Docker daemon。
完整传递依赖约束保存在 `requirements/cross-embodiment.lock.txt`，CPU Torch
从官方 CPU wheel index 单独安装，避免重复安装另一套 CUDA 库。

`create` 监控环境大小和剩余空间，默认限制 30GiB、保留 20GiB。
CUDA 数值 smoke 在指定卡执行并保存 `doctor.json`；它只验证设备可用，
不代表研究接触协议通过 GPU 资格。

```bash
python3 -m scripts.research.environment doctor --gpu 1
```

实验完成后，用 `scripts.research.snapshot --output <实际结果目录>/_environment
--policy <实际结果目录>/artifact_policy.json` 保存完整源码 overlay、环境锁、
wheel/hash 和 native 构建记录。该记录在环境删除后仍可重建研究代码。

停止相关实验进程后，可删除此环境，研究结果保持原位：

```bash
python3 -m scripts.research.environment clean
```

清理要求 repository ownership marker 匹配，不执行 Docker 全局 prune。
机器人模型是可重建资产，单独位于 `third_party/mujoco_menagerie`。
setup 使用固定公开基础 revision 和仓库 patch，避免依赖历史本地 commit。

## 运行 P1

先解析配置和实际输出位置：

```bash
JAX_PLATFORMS=cpu .runtime/cross_ebdm/venv/bin/genedynamics-run \
  configs/arm/keyed_insertion_recovery/validation/cross_ebdm.yaml \
  --development-root results/_development/p0p1_example --dry-run
```

移除 `--dry-run` 开始运行。已有相同目录必须显式 `--resume`；源码、
配置、环境版本、模型资产改变后拒绝复用，应使用新的 development root。
也可以用 `python -m scripts.cross_embodiment.run_pilot` 调用相同 workflow。

顺序为资源与配置回归 → 当前全部 cross-embodiment CPU 模块串行回归
→ 新 nominal Panda teacher → 两模式物理资格 → 冻结 phase prior
→ 每个条件每个 seed 独立五探针校准 → 独立校准验证
→ 同候选池 calibrated/task-only 与 nominal 搜索 → 冻结
→ 配对目标评估与 occupancy audit → 独立物理判定 → 科学报告。

配置默认 xArm7 matched/derated、三个独立校准 seed、每次五个测试 reset
且每 reset 两种模式。每个搜索有 12 候选 × 2 训练 reset × 2 模式，
calibrated 和 nominal 各 48 模型 episodes。transport 和 task-only
消费同一 calibrated 候选池，不额外访问目标任务训练数据。

`acceptance.json` 只回答原始轨迹与报告是否一致。
`scientific_report.json` 单独回答源支持、校准支持及 transport 的增量
证据；同参数、持平、校准拒绝和实际失败都保留。三个重复的小样本不能
代替正式统计，也不自动证明一般 cross-embodiment 或随机性优势。

完成后运行 `python -m scripts.paper.cross_ebdm.summarize_pilot --root
<实际结果目录>`，生成 `_reports/pilot-summary.json` 和 Markdown 主表，
分别列出轨迹一致性、校准响应支持、每格实际任务可行性与 transport 增量。

如果完整研究已经冻结完成、仅需重建审计，可运行
`python -m scripts.paper.cross_ebdm.audit_pilot --root <实际结果目录>`。
它核验原始 collection fingerprint、prior SHA、冻结契约、probe ledger
及全部物理轨迹，不执行新探针、仿真或优化。分析代码的 fingerprint 单独
记录；原 workflow 契约和失败历史保持不变。正常部分恢复仍必须满足原
源码/环境/配置契约，不能用只读审计绕过未完成采集。

## 资源与恢复

`ArtifactPolicy` 在原子 NPZ/JSON 发布之前检查配额，跨进程用文件锁
串行 writer；预留新临时文件、metadata 和旧 checkpoint 并存的空间。
P1 默认每个配置输出树限额 10GiB、文件系统保留 20GiB；阶段日志限制
16MiB，并设置超时。空间不足时停止当前阶段并发布
`resource_exhausted` 状态；已经计费的 probes 保留，不免费重放。

进程监控同时覆盖未通过 bundle writer 的日志、JUnit 和控制记录。
直接调用底层脚本时，需要显式设置 `GENEDYNAMICS_ARTIFACT_POLICY` 才会
启用这套配额；旧脚本不会自动继承不存在的 policy。

未来 GPU 训练入口已经能够显式传入 `--device`；physics provider 的
支持范围保持独立检查，匹配键槽/导槽/Go2 的未支持 GPU 组合仍拒绝。

## 2026-10-13 本轮执行结果

完整结果位置：
`results/_development/p0p1_20261013_v2/results/arm/keyed_insertion_recovery/validation/cross_ebdm/`。
共 15 个 cross-embodiment 模块、96 项 CPU 回归通过，0 失败/跳过；
另有 32 项资源、配置与恢复测试通过。GPU 数值 smoke 与 regressor 两次
参数更新通过；接触物理 GPU 资格仍未执行。新 Panda teacher 六条均成功
且两模式匹配，prior 已冻结。

六次独立校准各收费五探针，共 30；全部独立响应验证通过。
每次 calibrated/nominal 搜索各 48 模型 episodes，共 576。
最终四方法共 240 条实际目标评估，经只读原始轨迹审计为 0 mismatch。

| xArm7 条件 | direct | calibrated transport | 同模型 task-only | nominal response |
| --- | ---: | ---: | ---: | ---: |
| derated | 0/30 | 29/30 | 29/30 | 8/30 |
| matched | 0/30 | 3/30 | 3/30 | 3/30 |

每格来自三次独立新校准/优化。六次 transport 和 task-only 均选择相同
参数。本轮显示 derated 开发条件中校准有用，尚未建立 transport 增量；
matched 有 17 条轴向支撑丢失和 10 条超时，当前有限阶段映射族未通过任务
可行性门槛。derated 的一个失败为力超限。不能用聚合成功率隐藏逐格失败。
后续先扩展/验证可行映射族并检验分布贡献，正式评估使用新冻结协议。

恢复汇总时发现 tuple 参数经 JSON 变成 list 后被误判为契约变化，现已
规范化比较。实际 study 主流程的廉价计数回归确认：第二次恢复不增加
probe/rollout，改变机制配置则拒绝。已完成研究通过只读审计恢复汇总，
没有重采集任何目标交互；原错误在 `_logs/workflow-before-final-audit.json`
保留。collection fingerprint 为
`3af810b4af788347767d0d76e7ec51c720c063276f0977df742d1795bfcf9cba`，
原源码 overlay 在 `_environment/source-overlay.zip`；修复后的当前源码
另存 `_environment/current_source/`，分析 fingerprint 与采集 fingerprint
分别记录。
