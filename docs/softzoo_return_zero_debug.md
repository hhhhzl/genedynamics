# SoftZoo return_=0 排查报告

## 问题
Phase A/B 实验 `return_` 全为 0，`mean_reward` 全为 0。

## 排查过程

### 1. Reward 计算链路
- `SoftZooRolloutEvaluator._run_single_rollout` → `env.step(act)` → `ep_return += float(reward)`
- `BaseEnv.get_reward()` → `self.objective.get_reward(s)`
- `MoveForward.get_reward()`: `reward_mode=per_step_velocity` 时，`rew = (v_avg * forward_dir).sum().item()`（速度在前进方向上的投影）

### 2. 已修复项
- **YAML ConstructorError**: CfgNode dump 改为 `_cfg_to_plain_dict` + `yaml.safe_dump`
- **Taichi 未初始化**: 在 `make_softzoo_env` 开头调用 `ti.init()`
- **attrdict collections.Mapping**: 在 evaluator 中 patch `collections.Mapping/MutableMapping/Sequence`
- **max_episode_steps 断言**: 低 fidelity 时 `max_steps = max_substeps // n_substeps` 较小，需设置 `objective_config.max_episode_steps` 与之匹配

### 3. 当前阻塞
- **Segmentation fault (exit 139)**：首次创建 SoftZoo env 并运行 rollout 时崩溃
- 可能与 Taichi + Open3D 在 macOS ARM64 上的兼容性有关（GLFW 类冲突警告）

### 4. 可能原因（return_=0）
1. **仿真未真正跑完**：若在 segfault 前就失败，evaluator 会返回 0
2. **设计/控制器参数**：随机初始化可能无法产生有效运动
3. **PCD vs base_shape**：ground.yaml 使用 `base_shape: Primitive.Box`，而 task 指定 `pcd_name: Caterpillar`，需确认实际加载的是 PCD 还是 Box
4. **CPU 仿真精度**：macOS 无 CUDA，Taichi 回退到 CPU，物理精度可能不同

### 5. 建议
1. 在 Linux + CUDA 环境重跑 Phase A/B
2. 用 `scripts/debug/debug_softzoo_reward.py` 单步打印 reward（需先解决 segfault）
3. 检查 `design_space` 是否加载 Caterpillar.pcd
4. 尝试更高 fidelity（如 level 2）以增加 max_steps
