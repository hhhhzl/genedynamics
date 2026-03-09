# 实验配置与脚本清单 (Experiment Config & Script Checklist)

---

## 1. 实验配置 (Configs)

### 1.1 目录结构

```
configs/
├── README.md                    # 配置说明
├── mrmfmbd/
│   ├── README.md
│   └── softzoo_flat.yaml        # MRMFMBD SoftZoo flat
├── mbd3d/
│   └── mujoco_scene_mapping.yaml
├── single_2d/
│   ├── mbd.yaml, mdoc.yaml, mdcoas.yaml, ...
│   └── ebmbd.yaml
├── quadruped/
│   ├── flat/
│   │   ├── mbd_plan.yaml
│   │   ├── mbd_deploy.yaml
│   │   ├── mdcoas_plan.yaml
│   │   └── mbd_plan_go2.yaml
│   ├── rough_terrain/
│   ├── push_recovery/
│   └── obstacle_avoid/
├── d3il_avoiding/
├── uav3d/
├── humanoid/
└── ...
```

### 1.2 配置分类

| 类别 | 路径模式 | 用途 |
|------|----------|------|
| 实验 (plan) | `configs/<env>/<task>/*.yaml` | 纯规划，输出 trajectory |
| 部署 (deploy) | `configs/<env>/*_deploy*.yaml` | 仿真/实机执行 |
| 方法专用 | `configs/<method>/` | 如 mbd3d, mrmfmbd |

### 1.3 MRMFMBD 配置

| 文件 | env_name | method | 说明 |
|------|----------|--------|------|
| `configs/mrmfmbd/softzoo_flat.yaml` | softzoo | mrmfmbd | Caterpillar flat, stub 可用 |

**关键字段**:
- `env_params.use_stub_when_unavailable: true` — 无 SoftZoo 时用 stub
- `method_params.fidelity_num_levels`, `fidelity_step_ratio` — 保真度阶梯

### 1.4 常用配置速查

| 场景 | 配置文件 |
|------|----------|
| MRMFMBD SoftZoo | `configs/mrmfmbd/softzoo_flat.yaml` |
| MBD3D 场景映射 | `configs/mbd3d/mujoco_scene_mapping.yaml` |
| **3DGS Lego iid** | `configs/3dgs/lego_mbd_iid.yaml` |
| **3DGS Lego corr** | `configs/3dgs/lego_mbd_corr.yaml` |
| **3DGS Chair iid** | `configs/3dgs/chair_mbd_iid.yaml` |
| Quadruped MBD | `configs/quadruped/flat/mbd_plan.yaml` |
| Quadruped MD-COAS | `configs/quadruped/flat/mdcoas_plan.yaml` |
| Single 2D MDOC | `configs/single_2d/mdoc.yaml` |

### 1.5 3DGS 实验配置 (configs/3dgs/)

| 文件 | 说明 |
|------|------|
| `_template.yaml` | 完整注释模板，复制后修改 |
| `_base_nerf_synth.yaml` | 基础配置 |
| `lego_mbd_iid.yaml` | Lego + Ours-MBD (iid) |
| `lego_mbd_corr.yaml` | Lego + Ours-MBD (corr) |
| `lego_3dgs_map.yaml` | 3DGS-MAP 协议 |
| `chair_mbd_iid.yaml` | Chair + Ours-MBD (iid) |
| `chair_mbd_corr.yaml` | Chair + Ours-MBD (corr) |

---

## 2. 运行脚本 (Scripts)

### 2.1 入口命令

```bash
python -m genedynamics.experiments.runner <config.yaml> [--level N] [--seed S] [--dry-run]
```

### 2.2 脚本清单

| 脚本 | 用途 | 调用的配置 |
|------|------|-------------|
| `scripts/run_mrmfmbd.sh` | MRMFMBD SoftZoo 实验 | `configs/mrmfmbd/softzoo_flat.yaml` |
| `scripts/run_quadruped_plan.sh` | Quadruped MBD + MD-COAS + Go2 | `configs/quadruped/flat/mbd_plan.yaml` 等 |
| `scripts/experiments/run_mdcoas.sh` | MD-COAS 实验 | 见脚本内容 |
| `scripts/run_phase4_validation.sh` | Phase 4 验证 | - |
| `scripts/run_acceptance.sh` | 验收测试 | - |
| **scripts/3dgs/run_all.sh** | 批量运行 lego+chair iid/corr | - |
| **scripts/3dgs/run_3dgs_map.sh** | 3DGS-MAP baseline | 官方 3DGS 训练 |
| **scripts/3dgs/run_mbd3d_iid.sh** | Ours-MBD (iid) | `configs/3dgs/<object>_mbd_iid.yaml` |
| **scripts/3dgs/run_mbd3d_corr.sh** | Ours-MBD (corr) | `configs/3dgs/<object>_mbd_corr.yaml` |
| **scripts/3dgs/eval_3dgs_metrics.py** | PSNR/LPIPS/NLL 评估 | - |
| **scripts/3dgs/export_3dgs_figures.py** | 对比图、不确定性热图 | - |

### 2.2b 3DGS 脚本详情 (scripts/3dgs/)

详见 `scripts/3dgs/README.md`。

### 2.3 run_mrmfmbd.sh

```bash
#!/bin/bash
# Run MRMFMBD (Soft-robot S1+S3) experiments
./scripts/run_mrmfmbd.sh
```

输出目录: `results/mrmfmbd/softzoo_flat/`

### 2.4 环境变量

| 变量 | 说明 |
|------|------|
| `VENV` | Python 解释器路径，默认 `.venv_arm64/bin/python` |
| `GENEDYNAMICS_BACKEND` | `jax` 或 `numpy` |
| `MUJOCO_GL` | Linux: `egl`; Darwin: `glfw` |

---

## 3. 输出结构

```
results/
└── mrmfmbd/
    └── softzoo_flat/
        └── level_0/
            └── seed_0/
                ├── results.json
                ├── trajectory_*.png
                └── ...
```

---

## 4. 新增配置检查清单

1. 在 `configs/` 下创建 YAML
2. 确保 `env_name` 对应已注册环境插件
3. 确保 `method` 对应已注册方法插件
4. 可选：在 `configs/method_name/README.md` 写说明
5. 可选：在 `scripts/` 下创建 `run_xxx.sh`
