# H1 Push-to-Line：P1–P3 正式实验运行手册

本文档只描述当前冻结状态和从干净 CPU 机器运行 P1、P2、P3 全部正式实验的方法。旧的 A–F 修复记录、开发 seed 结果、失败尝试和中间参数不再属于正式运行说明。

当前结论很明确：

- P1、P2、P3 已进入正式实验阶段；
- 当前唯一未完成的任务是 P4 `p4_walk_push`；
- P4 尚未同时满足稳定行走、持续推动到目标线和全过程力安全，因此不能混入 P1–P3 的正式结果；
- 运行 P1–P3 时必须设置 `MGA_HUMANOID_SCOPE=p123`，不要使用默认的 `all`。

## 1. 冻结的 P1–P3 实验协议

### 1.1 三个任务

| 阶段 | canonical suite | 实验内容 | 主要回答的问题 |
|---|---|---|---|
| P1 | `p1_force_15n` | 固定箱体，15 N 力阶跃 | 在无任务位移的条件下，控制器能否稳定跟踪接触力并限制冲击峰值？ |
| P1 | `p1_force_30n` | 固定箱体，30 N 力阶跃 | 更高目标力下，力跟踪和尾部风险是否仍然稳定？ |
| P2 | `p2_push_nominal` | 固定站姿，将箱体推 10 cm | 控制器能否在保持站立与力安全的同时完成直线推动？ |
| P2 | `p2_push_ood` | 10 cm 推动，mass/friction OOD | 在未见质量/摩擦条件下，安全性和任务成功能否保持？ |
| P3 | `p3_unjam` | 平面自由箱体，unjamming 与 yaw correction | 接触几何、纠偏和安全筛选能否处理旋转/卡滞接触？ |

所有 suite 使用：

- seeds：`0,1,2,3,4,5,6,7,8,9`；
- `n_steps=100`；
- `Nsample=64`；
- `Hsample=16`；
- `Hnode=4`；
- backend：JAX CPU；
- 同一环境、同一低层执行接口和同一正式 metrics extractor。

不要为了缩短正式运行时间修改 YAML 中的 steps、samples、horizon、seed 或 suite 参数。

### 1.2 正式算法

P1–P3 均运行以下 8 个算法：

1. MGA；
2. Model-based Only；
3. standalone RL；
4. DIAL；
5. MPPI；
6. PegasusFlow；
7. ISSA；
8. ATACOM。

对应配置位于：

```text
configs/humanoid/push_to_line/main/mga.yaml
configs/humanoid/push_to_line/baseline/model_based_only.yaml
configs/humanoid/push_to_line/baseline/standalone_rl.yaml
configs/humanoid/push_to_line/baseline/dial.yaml
configs/humanoid/push_to_line/baseline/mppi.yaml
configs/humanoid/push_to_line/baseline/pegasusflow.yaml
configs/humanoid/push_to_line/baseline/issa.yaml
configs/humanoid/push_to_line/baseline/atacom.yaml
```

### 1.3 正式消融

四个跨任务消融均运行 P1、P2、P3 的全部 5 个 suites：

```text
no_rl_prior
no_learned_reliability
no_controllability_geometry
no_retraction
```

P1 额外保留一个任务专属消融：

```text
no_stiffness
```

`no_stiffness` 只运行 `p1_force_15n` 和 `p1_force_30n`，不能把它扩展到 P2/P3，也不能把旧的 `no_tangent` 结果改名后混入正式结果。

### 1.4 正式运行数量

| 组别 | 计算方式 | runs |
|---|---:|---:|
| MGA | 1 × 5 suites × 10 seeds | 50 |
| 7 个 baseline | 7 × 5 × 10 | 350 |
| 4 个跨任务消融 | 4 × 5 × 10 | 200 |
| P1 `no_stiffness` | 1 × 2 × 10 | 20 |
| **合计** |  | **620** |

按 suite 检查时，P1 每个 suite 应有 130 runs，P2 每个 suite 应有 120 runs，P3 应有 120 runs。

## 2. 干净机器需要准备什么

推荐使用项目已经提供的 CPU Docker 环境。这样不需要在宿主机手工组合 JAX、Brax、MuJoCo 和渲染依赖。

### 2.1 宿主机要求

- 当前 `docker/compose.cpu.yml` 面向 Apple Silicon / `linux/arm64`；
- 安装 Git、Docker Desktop 和 Docker Compose v2；
- Docker Desktop 建议分配至少 12 GB 内存；16 GB 更稳妥；
- 可以给 Docker 配置 4–8 GB swap，但 swap 只用于避免瞬时 OOM，不会加速 JAX；
- 至少预留 25 GB 磁盘空间给镜像、620 组轨迹、GIF/PNG 和报告；
- 正式运行只启动一个 H1 容器，不要并行启动多个 JAX 实验容器。

如果新机器是 Linux x86_64，不要在 QEMU 下模拟 `linux/arm64` 跑正式计时；应先建立与 `docker/install/install_cpu_dev.sh` 相同依赖的原生 x86_64 CPU 镜像，再做 smoke test。不要为此修改实验 YAML。

### 2.2 代码必须是冻结且干净的版本

在原机器上先把 P1–P3 的代码、配置、入口和本文档做 scoped commit。新机器应 checkout 该 commit，而不是复制一个无法追踪的混合工作区。

```bash
git clone <repository-url> enerdynamics
cd enerdynamics
git checkout <p1-p3-frozen-commit>
git submodule update --init --recursive
git status --short
git rev-parse HEAD
```

验收要求：`git status --short` 无输出，并记录 `git rev-parse HEAD` 的值。正式实验期间不要修改以下路径：

```text
configs/humanoid/push_to_line/
genedynamics/envs/domains/humanoid/
genedynamics/experiments/
scripts/paper/mga/run_humanoid_push.sh
```

### 2.3 必须单独迁移的学习产物

这些文件位于 `results/`，未由 Git 跟踪；只 clone 代码无法运行 MGA、RL、ISSA 和 ATACOM。P1–P3 需要迁移以下 4 个文件，并保持相对路径不变：

| 文件 | SHA-256 |
|---|---|
| `results/humanoid/box_push/_policies/fixed_ppo_seed0.pkl` | `1fe928bd143d7c03df8fcfe381aad4fc27a81ee50825187270f76643d394cf7d` |
| `results/humanoid/box_push/_policies/atacom_p12_ppo_seed0.pkl` | `242212e7604799a178414ff2028386757b4825a12ce017ce2f3323bc2ded81d0` |
| `results/humanoid/box_push/_policies/atacom_p3_ppo_seed101.pkl` | `a6adb58f617cca4c1f560d9d4d54d8efc6b928fb222cfcf4a06efd6c63aa6388` |
| `results/humanoid/box_push/_policies/reliability_q95.json` | `fd62fc3c251f57efa7578c8548f30a943511004c3e568869795d24dfc99dbbd9` |

例如，从原机器向新机器传输：

```bash
rsync -av --relative \
  results/humanoid/box_push/_policies/fixed_ppo_seed0.pkl \
  results/humanoid/box_push/_policies/atacom_p12_ppo_seed0.pkl \
  results/humanoid/box_push/_policies/atacom_p3_ppo_seed101.pkl \
  results/humanoid/box_push/_policies/reliability_q95.json \
  <user>@<new-machine>:/path/to/enerdynamics/
```

在 Linux 上验证：

```bash
sha256sum \
  results/humanoid/box_push/_policies/fixed_ppo_seed0.pkl \
  results/humanoid/box_push/_policies/atacom_p12_ppo_seed0.pkl \
  results/humanoid/box_push/_policies/atacom_p3_ppo_seed101.pkl \
  results/humanoid/box_push/_policies/reliability_q95.json
```

在 macOS 上把 `sha256sum` 换成逐文件执行 `shasum -a 256 <file>`。四个 hash 必须与上表完全一致。

P1–P3 不需要迁移 `walk_ppo_seed0.pkl` 或 `atacom_p4_ppo_seed0.pkl`；它们只属于尚未冻结的 P4。

## 3. 构建 CPU 环境

以下命令均从仓库根目录执行。

```bash
cp docker/env/dev-cpu.env.example docker/env/dev-cpu.env
cp docker/.env.example docker/.env
docker compose -f docker/compose.cpu.yml build genedynamics-dev-cpu
```

确认容器内的后端和核心依赖：

```bash
docker compose -f docker/compose.cpu.yml run --rm \
  genedynamics-dev-cpu \
  python -c "import jax, brax, mujoco; print('jax=', jax.__version__); print('backend=', jax.default_backend()); print('brax=', brax.__version__); print('mujoco=', mujoco.__version__)"
```

必须看到 `backend= cpu`。如果这里失败，不要启动正式矩阵。

## 4. 正式运行前检查

### 4.1 检查冻结配置

```bash
docker compose -f docker/compose.cpu.yml run --rm \
  genedynamics-dev-cpu \
  python -m genedynamics.experiments.utils.metrics audit \
  configs/humanoid/push_to_line
```

命令应返回 `"ok": true`，且没有 `errors`。

### 4.2 运行相关单元测试

```bash
docker compose -f docker/compose.cpu.yml run --rm \
  genedynamics-dev-cpu \
  pytest -q \
  test/unit/test_experiment_config.py \
  test/unit/test_mga_report.py
```

### 4.3 对完整入口做 dry-run

先只覆盖两个 seed，但保留完整的 8 算法、5 suites 和消融调度：

```bash
docker compose -f docker/compose.cpu.yml run --rm \
  -e MGA_HUMANOID_SCOPE=p123 \
  -e "MGA_SEEDS=0 1" \
  -e MGA_DRY_RUN=1 \
  genedynamics-dev-cpu \
  ./scripts/paper/mga/run_humanoid_push.sh
```

dry-run 只能用来检查解析后的运行计划，不能作为实验结果。

## 5. 启动 620-run 正式矩阵

唯一的论文入口是：

```text
scripts/paper/mga/run_humanoid_push.sh
```

它内部调用统一的 `genedynamics.experiments.runner`，不要直接调用 solver 下的旧实验入口，也不要逐个手写临时 YAML。

建议用有名字的后台容器启动；不要加 `--rm`，这样退出状态和日志仍可检查：

```bash
docker compose -f docker/compose.cpu.yml run -d \
  --name mga_humanoid_p123_formal \
  -e MGA_HUMANOID_SCOPE=p123 \
  -e "MGA_SEEDS=0 1 2 3 4 5 6 7 8 9" \
  genedynamics-dev-cpu \
  ./scripts/paper/mga/run_humanoid_push.sh
```

入口按以下顺序串行运行：

1. MGA；
2. Model-based Only；
3. standalone RL；
4. DIAL；
5. MPPI；
6. PegasusFlow；
7. ISSA；
8. ATACOM；
9. 四个跨任务消融；
10. P1 专属 `no_stiffness`。

不要为了“充分利用 CPU”再启动第二个相同容器。JAX 编译和 Brax rollout 并行竞争内存时，通常只会增加 swap、触发 kill，并破坏对运行时间的判断。

## 6. 监控、判断是否跑完和断点续跑

查看日志：

```bash
docker logs -f --tail 100 mga_humanoid_p123_formal
```

查看资源：

```bash
docker stats mga_humanoid_p123_formal
```

查看容器是否已经退出以及退出码：

```bash
docker ps -a --filter name=mga_humanoid_p123_formal
docker inspect mga_humanoid_p123_formal --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}'
```

正式结果通过 bind mount 直接写到宿主机：

```text
results/humanoid/push_to_line/
```

随时可统计已经产生的 seed 结果：

```bash
find results/humanoid/push_to_line \
  -path '*/seed_*/results.json' -type f | wc -l
```

干净结果目录最终应为 `620`。

如果容器因重启、断电或 OOM 中断，不要删除已经完成的结果，也不要改 seed 范围。入口已经固定使用 `--resume`；换一个容器名执行完全相同的命令即可：

```bash
docker compose -f docker/compose.cpu.yml run -d \
  --name mga_humanoid_p123_resume_01 \
  -e MGA_HUMANOID_SCOPE=p123 \
  -e "MGA_SEEDS=0 1 2 3 4 5 6 7 8 9" \
  genedynamics-dev-cpu \
  ./scripts/paper/mga/run_humanoid_push.sh
```

`--resume` 会跳过已经具有完整正式产物的 seed，并从缺失处继续。若发生 OOM，应先确认旧容器已经停止，再启动 resume 容器；不要让两个入口同时写同一结果目录。

## 7. 数值结果验收

运行结束后，先做不依赖 GIF 的结构与协议验收。

### 7.1 MGA 与 7 个 baseline

```bash
docker compose -f docker/compose.cpu.yml run --rm \
  genedynamics-dev-cpu \
  python -m genedynamics.experiments.utils.metrics verify \
  configs/humanoid/push_to_line/main \
  configs/humanoid/push_to_line/baseline \
  --suites \
  p1_force_15n p1_force_30n \
  p2_push_nominal p2_push_ood p3_unjam
```

### 7.2 四个跨任务消融

```bash
docker compose -f docker/compose.cpu.yml run --rm \
  genedynamics-dev-cpu \
  python -m genedynamics.experiments.utils.metrics verify \
  configs/humanoid/push_to_line/ablation/no_rl_prior.yaml \
  configs/humanoid/push_to_line/ablation/no_learned_reliability.yaml \
  configs/humanoid/push_to_line/ablation/no_controllability_geometry.yaml \
  configs/humanoid/push_to_line/ablation/no_retraction.yaml \
  --suites \
  p1_force_15n p1_force_30n \
  p2_push_nominal p2_push_ood p3_unjam
```

### 7.3 P1 专属 stiffness 消融

```bash
docker compose -f docker/compose.cpu.yml run --rm \
  genedynamics-dev-cpu \
  python -m genedynamics.experiments.utils.metrics verify \
  configs/humanoid/push_to_line/ablation/no_stiffness.yaml \
  --suites p1_force_15n p1_force_30n
```

三条命令都必须返回 `"ok": true`。验收会检查：

- protocol manifest；
- seed 和 suite 身份；
- `n_steps`、`Nsample`、`Hsample`、`Hnode` 等预算；
- `results.json`；
- 执行轨迹 `trajectory/trajectory.json`；
- suite summary 与 overall summary；
- MGA 的 certified rejection 是否具有完整、可核查的失败证书。

## 8. 生成 GIF、PNG、汇总数据和论文图

### 8.1 从已执行轨迹渲染，不重新运行控制器

```bash
docker compose -f docker/compose.cpu.yml run --rm \
  genedynamics-dev-cpu \
  python -m genedynamics.experiments.utils.vis \
  results/humanoid/push_to_line \
  --stride 2 --fps 15 --width 640 --height 448
```

每个成功渲染的 seed 会在原轨迹目录产生：

```text
trajectory/trajectory_best.gif
trajectory/trajectory_best.png
```

渲染只读取已经保存的 q/qd，不会重新执行算法，因此不会改变数值结果。

### 8.2 在要求视觉产物的条件下再次验收

把第 7 节的三条 `verify` 命令分别再运行一次，并在末尾加：

```text
--require-visuals
```

只有这一步也通过，才能认定每个正式 seed 的数据、轨迹和视觉产物均完整。

### 8.3 生成 H1 独立汇总

不要在只迁移了 H1 结果的干净机器上直接调用全任务的 `scripts/paper/mga/summarize_results.sh`，因为那个入口同时期望 Surface 和 PegInsert 结果。只汇总 H1 应执行：

```bash
docker compose -f docker/compose.cpu.yml run --rm \
  genedynamics-dev-cpu \
  python -m genedynamics.experiments.utils.metrics summarize \
  results/humanoid/push_to_line \
  --output reports/mga/humanoid_push
```

主要输出为：

```text
reports/mga/humanoid_push/summary.csv
reports/mga/humanoid_push/summary.json
reports/mga/humanoid_push/paired_deltas.csv
reports/mga/humanoid_push/paired_deltas.json
reports/mga/humanoid_push/representative_seeds.json
reports/mga/humanoid_push/safe_success.{png,pdf}
reports/mga/humanoid_push/physics_force_peak.{png,pdf}
reports/mga/humanoid_push/physics_force_normalized_cvar95.{png,pdf}
```

论文统计必须从 `summary.csv/json` 和原始 seed-level `results.json` 生成；不能从 GIF 目测填表。

## 9. 结果目录合同

最终正式目录只使用：

```text
results/humanoid/push_to_line/
├── main/mga/
├── baseline/model_based_only/
├── baseline/standalone_rl/
├── baseline/dial/
├── baseline/mppi/
├── baseline/pegasusflow/
├── baseline/issa/
├── baseline/atacom/
└── ablation/
    ├── no_rl_prior/
    ├── no_learned_reliability/
    ├── no_controllability_geometry/
    ├── no_retraction/
    └── no_stiffness/
```

单个 seed 的权威产物位于：

```text
<algorithm-root>/level_<suite>/seed_<seed>/results.json
<algorithm-root>/level_<suite>/seed_<seed>/trajectory/trajectory.json
<algorithm-root>/level_<suite>/seed_<seed>/trajectory/trajectory_best.gif
<algorithm-root>/level_<suite>/seed_<seed>/trajectory/trajectory_best.png
```

`results/_development/`、临时日志、旧命名目录和手工挑选的 GIF 都不是正式统计来源。

## 10. 时间预估

在此前同等级 Apple Silicon CPU、单容器串行运行的吞吐下，620-run 数值矩阵预计约 38–46 小时。新机器首次运行还会有镜像构建和 JAX 编译开销；全部轨迹渲染会额外占用时间与磁盘。

更可靠的现场估计方法是：完成前 20–30 runs 后，用实际平均时间计算

```text
剩余时间 ≈ (620 - 已完成 runs) × 已完成 runs 的平均耗时
```

不要通过减少 seeds、samples 或 steps 来追赶时间；如果时间不足，应先完整跑完 MGA 与 7 个 baseline，再运行消融，但最终协议仍必须补齐到 620 runs。

## 11. P4：当前问题、已有修复和后续路径

P4 是当前唯一未完成的 humanoid 子任务。它要求 30–50 cm 的 walk-and-push，而不是 P1 式原地站立，也不是只让箱体短暂移动。P4 的目标是在**同一条 full-MGA 轨迹**中同时实现：

- H1 产生真实的左右支撑交换并持续向前移动；
- 与箱体建立接触后持续推动，最终进入目标带；
- 到线后保持至少 0.20 s，箱体速度不高于 0.08 m/s；
- pelvis/body progress 不低于 0.20 m；
- support progress 不低于 0.15 m；
- 左右脚各至少完成一次有效 lift–land，单步前进不低于 0.04 m；
- 全过程不跌倒、不发生非法非手接触，4 ms 物理子步手力不超过 60 N；
- episode 完整完成，不以 `aborted_unrecoverable` 结束。

在这些条件满足前，P4 必须与本文的 P1–P3 620-run 正式矩阵隔离。

### 11.1 当前最好证据及其边界

下表中的运行都是 P4 开发诊断，不是公平的论文横向比较。它们的预算、prior 和控制组件不同，不能拼接成一次成功，也不能用于正式均值。

| 开发结果 | 实际行为 | 物理安全 | 最终结论 |
|---|---|---|---|
| `p4_synchronized_closed_loop_dial_seed110_concrete_fix` | 左右脚各 1 步；body progress 0.235 m；箱体距目标约 14.98 cm | 峰值力 70.632 N；存在 force、fall 和 balance violation | DIAL 设定能够产生行走/推动趋势，但不是安全成功 |
| `p4_viability3` | 左右脚各 1 步；body/support progress 0.326/0.267 m | 峰值力 49.819 N；前缀无物理安全违规 | 证明双脚安全步态在物理上可达，但使用旧 prior、H8/node2；第 137 步不可恢复中止，未到线 |
| `p4_support_sweep_recovery_no_prior` | body/support progress 0.203/0.136 m；左脚 1 步、右脚 0 步；目标误差约 6.38 cm | 峰值力 32.284 N；前缀无物理安全违规 | 当前 no-prior 可以安全推进，但第 131 步候选集合耗尽 |
| `p4_phase_lag_rescue_provenance_no_prior` | body/support progress 0.204/0.136 m；左脚 1 步、右脚 0 步；目标误差约 6.10 cm | 峰值力 32.284 N；前缀无物理安全违规 | provenance 修复有效，但 phase-lag rescue 只将中止从第 131 步推迟到第 132 步 |

对应原始结果位于：

```text
results/_development/humanoid_mga_repair/
```

这里最重要的结论不是“P4 已经接近成功”，而是两个能力尚未同时出现在当前 full-MGA 中：

1. 旧 learned-prior 路径展示了左右支撑交换，但没有到线并最终中止；
2. 当前 no-prior 路径展示了低力、安全推进，但缺少第二次支撑交换并最终中止。

### 11.2 已确认的根因

当前问题不是“箱子推不动”，也不再是简单的接触力过大。已确认的根因有四项。

1. **当前候选集合缺少有效的第二次支撑交换。** Gaussian/no-prior 路径可以推进到约 0.20 m，但在右脚有效落地前耗尽安全候选。旧 learned prior 说明 gait proposal 可能提供该方向，但旧 checkpoint 和旧 H8/node2 协议不能直接作为当前证据。
2. **有限 horizon 安全不等于递归可行。** 当前候选可以在本 horizon 内满足 force、fall/invalid-contact 和 balance gate，却可能把系统带到下一次规划时没有任何安全后继的状态。
3. **emergency 安全不等于 emergency 后可恢复。** UNLOAD/retract 本身可能安全，但从晚期边界状态卸载后，机器人不一定还能回到具有下一步 gait candidate 的双支撑可行集。
4. **shift/provenance bug 已修复，但不是唯一根因。** 修复后 incumbent 的预测风险下降，真实执行与候选 provenance 更一致；然而 step 132 仍出现 balance-only terminal infeasibility，说明剩余问题属于 gait candidate coverage 和 terminal viability。

因此，以下做法已经被反例否定，不能再当作主修复路线：

- 不断增加即时 phase-lag pulse；
- 只检查单帧 retract 是否安全；
- 只扩大 horizon，却不定义 terminal backup；
- 通过放宽 60 N 力限制、减少有效步数或裁掉失败尾段来制造成功；
- 把旧 `p4_viability3` 的好看片段与当前低力前缀拼成 full-MGA 成功。

### 11.3 已完成并应保留的修复

下面的工作已经解决了真实实现问题，应作为后续 P4 的起点，而不是重新推倒：

- 已接入并核对 DIAL push-walk reference、任务时钟和 loaded-contact 行为；
- 腿部动作已经使用真实 direct joint-target realization，而不是只在 planner 空间里移动；
- startup、gait reference、measured support/phase 与控制时钟已经同步；
- node 表示、dense spline 和真正执行的低层动作已经完成一致性修复；
- shifted incumbent 与 task-owned recovery 保存并传播正确 provenance；
- P4 使用 `certified_terminal_hold`，不再用未经验证的任意尾部填充；
- emergency 已扩展为 minimum UNLOAD dwell，再由同一个动力学模型验证 NORMAL recovery；
- 已提供 roll/pitch capture 和 task-owned recovery candidate bank；
- 真实 4 ms 子步手力、非手接触、fall/balance、安全拒绝和实际执行长度会完整落盘；
- 当 normal、incumbent 和 emergency 都不安全时，会记录真实的 certified rejection，不会执行一个已知不安全动作。

当前代码中的 `normal_rescue_plans`/phase-lag candidates 仍可用于反例诊断，但其目前只多维持一步，不能被视为已经解决递归安全，也不能单独晋级到 P4 正式机制。

### 11.4 对齐 DIAL 时必须保留的任务条件

原始参考配置是：

```text
baselines/dial-mpc/dial_mpc/examples/unitree_h1_push_crate.yaml
```

原配置使用 300 steps、2048 samples、H24、Hnode6、Ndiffuse4、20 ms 控制周期、slow-walk gait 和 2 s ramp。CPU 开发可以使用较小 sample budget，但必须明确它不是原始 DIAL 的计算等价复现。

后续对齐必须区分两层：

1. 先验证原 DIAL 任务定义和闭环确实产生 walk-and-push，包括 reset、joint mapping、gait phase、速度/力 ramp、box mass/friction、接触点和 episode 时间；
2. 再把相同任务物理、低层动作接口、安全阈值和成功判据提供给 DIAL、MGA 及其他 baseline。

记录轨迹加 residual 是本项目中的实现选择，不等于原始 DIAL planner。若继续使用 reference trajectory，必须冻结生成协议、reference seed、内容 hash、时间索引和 terminal behavior，并避免把生成 reference 的 rollout 再当作独立验证 seed。

### 11.5 后续修复路径

P4 按 R0–R5 顺序推进。前一门不通过，不启动后一阶段的大矩阵。

#### P4-R0：清理诊断分支并建立可回退基线

1. 保留 node/dense 一致性、task-owned provenance、measured phase、UNLOAD dwell、NORMAL recovery 和完整 abort 记录。
2. 将只多维持一步的 phase-lag 实验路径从 canonical 默认行为中隔离；保留失败结果用于回归，不覆盖原目录。
3. 对 archived step 111、124、128、131、132 状态做定点测试，保证清理后不会恢复旧 shift/provenance bug。
4. 重跑 Surface Scan、PegInsert 和 H1 P1–P3 的受影响回归。P4 修复不能改变另外两个环境的 reward、physics、checkpoint 或已有结果。
5. 形成一个 scoped commit，后续每个机制阶段使用独立 development result root。

**R0 验收：** 已修复的边界状态不回退；P1–P3 合约测试通过；canonical 路径中没有一个已知无效但默认开启的单步 rescue。

#### P4-R1：证明 RL prior 提供 Gaussian 缺少的 gait proposal

在相同 archived pre-failure states、相同物理模型、H16/node4 和相同安全评分下比较：

- Gaussian/no-prior candidate bank；
- 当前 canonical walk PPO；
- task-owned gait reference/incumbent。

比较重点不是 reward，而是候选是否包含右脚 lift–land、support exchange 和更好 terminal support 所需的关节方向，并且这些方向能经过同一 model-based rollout 安全精化。

若当前 walk PPO 在现行 observation/action schema 下无法产生该候选，应使用现有 `scripts/tasks/robot/humanoid/train_box_push_rl.py` 重训，不建立第二套训练框架，也不把旧 smoke checkpoint 改名后升级。

**R1 验收：** learned-prior bank 至少包含一个 Gaussian bank 缺失或显著更差的、安全且改善下一次支撑交换的候选。若二者无差异，应如实判定 P4 上的 RL-prior 贡献尚未验证。

#### P4-R2：将短时安全升级为 terminally viable certificate

1. 每个候选先完成当前 horizon 内的 force、invalid-contact/fall、balance 和 force-tracking 检查。
2. 在候选尾部连接确定性的 task-owned backup；backup 共享 gait phase/reference，并包含 measured roll/pitch capture。
3. 候选只有在“当前 horizon 安全 + terminal backup 安全 + 下一周期仍有可用 incumbent/backup”时才可执行。
4. emergency 必须验证完整的 `UNLOAD dwell → NORMAL gait recovery → terminal viability`，而不是只验证即时 retract。
5. 对 archived step 111/124/128/131/132 做离线反例回放，保证安全判定与真正执行使用相同 mode、task memory、force integrator、reference clock 和 realization。

**R2 验收：** 边界状态要么找到完整的可恢复链，要么在进入该状态之前被 terminal gate 拒绝并选择另一条保留恢复余量的候选。把 abort 从 132 推到 133 不算通过。

#### P4-R3：单 seed 拉通 full-MGA

先使用 seed 110、200 steps、N16/H16/node4 做开发闭环，按固定顺序运行：

1. 有效 walk prior + model-based refinement/terminal gate，暂时关闭 learned reliability；
2. 相同配置的 no-RL-prior，验证 prior 是否真的改善 candidate coverage；
3. 加载与当前 feature/risk/horizon 匹配的 reliability，形成真正 full-MGA。

该 seed 必须同时满足第 11 节开头列出的全部 locomotion、到线、保持、60 N 力安全和无 abort 条件。任何失败都保留完整结果，不只汇报安全前缀。

#### P4-R4：第二开发 seed、预算确认和 reliability 校准

seed 110 通过后，使用完全相同的源码、配置和 checkpoint 运行 seed 111。两者均通过后，再用拟冻结预算复验：

- MGA：N64/H16/node4；
- P4 episode：计划采用 300 steps，以覆盖与 DIAL 对齐的完整 walk-and-push 时间；
- 所有算法共享同一任务时长、物理、安全阈值和指标；
- baseline 保留各自预注册的算法级搜索参数。

P4 reliability 必须按当前 feature、risk heads、realization 和 horizon 重新收集与校准。至少报告 coverage、false-safe、false-reject、校准误差，以及 full-MGA 相对 `no_learned_reliability` 的真实影响。全部 abstain 到 model-based fallback 不能被解释为 learned reliability 有效。

**R4 验收：** seed 110/111 均达到完整 strict-safe success；full-MGA 相对 no-prior 至少在成功率、可恢复性或 progress–safety frontier 上产生可重复且可解释的改善；reliability 不引入新的 false-safe 执行。

#### P4-R5：冻结、最小比较矩阵和正式运行

1. 将验证通过的设置写回现有 `configs/humanoid/push_to_line/`，不新增带 CPU/GPU 或版本号的正式 YAML。
2. 记录 code、reference、policy、reliability、镜像和 metrics schema hash。
3. 先运行 MGA、no-RL-prior、Model-based Only、DIAL、ATACOM × P4 × seeds 110/111。
4. MGA 未达到 2/2 strict-safe success 时，不扩大到八算法正式矩阵。
5. 通过 completeness/provenance verification 后，再运行正式 seeds 0–9 和剩余算法/消融。
6. 生成真实成功 GIF、全身 motion strip、足底承载/落脚时间条、box/body/support progress 和手部接触力曲线；失败 baseline 也保留完整中止或违规段。

P4 最终达到论文级完成的定义不是“一条视频看起来走起来”，而是冻结协议下 full-MGA 的多 seed strict-safe success 可以报告，并且相对 no-prior、Model-based Only 和安全 RL baseline 的差异能够对应 MGA 的 RL proposal、model-based refinement、realization-aware geometry 和 receding safety 机制。
