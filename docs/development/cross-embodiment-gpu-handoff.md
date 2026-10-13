# Cross-embodiment：CPU停止点与GPU交接

日期：2026-10-12。按用户要求停止本机实验、测试和依赖安装，后续转到GPU机器；本文件是交接计划，不是GPU运行报告。当前工作区位于项目分支`codex/cross-embodiment-transport`，未提交改动保留，GPU交接继续使用此分支。完整科学证据见[机制闭环记录](cross-embodiment-validation-closure.md)，架构及并行契约见[development主计划](cross-embodiment-transport-plan.md)。

## 已有证据与未完成项

工件根目录为`tmp/cross_embodiment/mechanism_closure_v2/`，路径均相对仓库根目录。

| 项目 | 已保存结果 | 接续边界 |
| --- | --- | --- |
| 当前代码回归 | `cpu_reference_final.xml`：53/53 NumPy/native；时钟修改后`clock_regression.xml`：9/9，部分重叠 | 当前完整JAX/Torch/MJX回归未完成；旧71/77/86项不追认 |
| learned response | `conditional_response_fit/report.json`短窗资格通过；`noisy_response_occupancy/report.json`实际访问状态下.2/.6s递推通过 | 外给命令窗；整段均值cost网络失败，仍unsupported |
| 有限响应law | `finite_response_law/report.json`：新192模型episodes分布验证，实际16/16 | 有限candidate×mode经验联合law，不是通用神经kernel |
| 多校准、多优化 | `regularized_transport_repeats/report.json`：5次各10/10、共50/50 | task-only持平；保留旧49/50失败，不声明独有优势 |
| ObstacleTransfer | `obstacle_mechanism_chain_v2/report.json`：四方法各16/16，共48次不同实际执行 | 原统一30s协议；尚无成功率优势 |
| GuidedContactAlignment | `spatial_guide_chain_v4/report.json`：calibrated10/10，direct/nominal4/10 | 真实3D导槽；task-only持平，旧v1–v3失败保留 |
| MobileKeyedInsertionRecovery | `mobile_coupled_chain/report.json`：Go2+Z1操作/支撑10/10、direct5/10；coupled预测通过、arm-only预测失败 | task-only/arm-only/nominal同样10/10；站立操作，未验收步态/导航/人形 |
| 时间步检查 | `environment_timesteps.json`：三个场景2/1ms共12/12，WBC固定4ms | 三个开发点的定性稳定，不是uniform接触收敛 |
| 身体×机制 | `body_mechanism_matrix/report.json`：v1整体失败，xArm7 matched7/10；其余四格10/10 | v2模型搜索中断，尚无新物理测试或总report |
| 独立验收 | `acceptance.json`最近审计446条、0 mismatch、`passed=false` | 后加12条时间步轨迹尚未纳入重跑；矩阵完成后重新审计 |

GPU运行、真实多卡、完整Torch物理backend、native CUDA及Rust均未验收。当前证据支持声明范围内的机制链，尚未证明一般cross-embodiment或相对强task-only控制的独有收益。

## 矩阵停止点与恢复

目录：`tmp/cross_embodiment/mechanism_closure_v2/body_mechanism_matrix_v2/`。下表统计已提交`metadata.json`的episode，不含未提交临时文件；停止时没有根`report.json`。

| cell | source | calibrated model | nominal model | 最终物理评估 |
| --- | ---: | ---: | ---: | ---: |
| panda_matched | 16 | 384 | 41 | 0 |
| panda_derated | 16 | 384 | 106 | 0 |
| xarm7_matched | 16 | 228 | 0 | 0 |
| xarm7_derated | 16 | 243 | 0 | 0 |
| ur5e_derated | 0 | 0 | 0 | 0 |
| 合计 | 64 | 1239 | 147 | 0 |

v2固定24候选、每格8训练reset groups×两模式，calibrated/nominal各384模型episodes；优先满足模型成功及模式可行性，再比较共同目标和.1 bias正则。测试采用64000+各格seed的+100…+104、每reset两模式；成功率≥.95且mode全部匹配。5份校准复用v1计费记录，不能免费重放丢失探针。

**先完成下述存储限制和Linux小样本一致性检查，才恢复长任务。** 现有脚本仍为每个搜索episode保存完整carry，以下入口本身没有磁盘限额。它在GPU机器上仍执行NumPy/MuJoCo CPU reference，不会使用CUDA加速。

```bash
# 在Linux目标机器重新构建native CPU推理库。
python -m scripts.cross_embodiment.build_native \
  --output tmp/cross_embodiment/libcross_embodiment.so

# 原有工件须先按原相对路径迁移；此处复用已提交记录。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
CROSS_EMBODIMENT_NATIVE_LIBRARY="$PWD/tmp/cross_embodiment/libcross_embodiment.so" \
python -m scripts.cross_embodiment.validate_body_mechanism_matrix \
  --output tmp/cross_embodiment/mechanism_closure_v2/body_mechanism_matrix_v2 \
  --refined \
  --calibration-root tmp/cross_embodiment/mechanism_closure_v2/body_mechanism_matrix \
  --workers 4

# 全部相关研究完成后才生成新的最终审计。
python -m scripts.cross_embodiment.audit_validation_closure
```

恢复要求contract、source SHA、校准来源及原始tensor SHA一致。Linux先在冻结输入上验证任务/模式/力判定，记录平台和数值库版本；少数Panda源轨迹跨平台一致不能保证所有接触分岔一致。若运行语义改变，使用新目录和新协议，不覆盖旧记录或将不兼容记录混为同次研究。

## 要迁移的代码和工件

1. **先保存当前未提交工作。** 此项目大量源码、文档、LaTeX和测试仍未跟踪；仅`git push`不会传走它们。在GPU开始前提交到项目分支`codex/cross-embodiment-transport`，或同步当前工作区；此次交接未代为commit、push。`third_party/mujoco_menagerie`也有本地改动，须连同所需资产保留，不能只传父仓库submodule指针。
2. 迁移`genedynamics/learning/cross_embodiment/`、`genedynamics/envs/domains/manipulation/cross_embodiment/`、`genedynamics/robots/ur5e/`、`scripts/cross_embodiment/`、`test/cross_embodiment/`及已修改的公共入口、development和LaTeX。保持现有solver/provider接口，不另起主流程。
3. `tmp/cross_embodiment/`被git忽略，必须单独同步。优先保留`mechanism_revision/strict_phase_prior`（Task3/Task4和矩阵）、`learned_cpu_v2/source_prior`（绕障）、`mechanism_closure_v2/spatial_guide_chain_v3/source_prior`（3D导槽）、`theory_feasibility/`校准来源，以及`mechanism_closure_v2/`的contracts、probes/ledger、datasets、模型、搜索断点、冻结map、实际评估、失败和reports。旧设备链资格另需README列出的`learned_*`、`guided_*`等对应工件。迁移前按实际依赖列清单和大小，不复制整个虚拟环境或Docker磁盘镜像。
4. 保存`.npz`或已压缩的`.npz.gz`及其metadata；loader对解码后的原始SHA核验。不能只传JSON遗漏tensor。当前native macOS`.dylib`不适用于Linux，须重新编译`.so`；不复制失败Torch安装。
5. 目标环境记录Python、JAX/jaxlib、Torch、MuJoCo/MJX、NumPy/SciPy、OSQP、GPU/driver、flags和代码fingerprint。CPU旧Docker参考为Python3.10、JAX0.6.2、Torch2.5.1、MuJoCo3.6；停止前macOS reference为Python3.12.3、MuJoCo3.6、NumPy2.2.6、SciPy1.17、OSQP1.0.4。版本差异必须显式记录。

GPU容器和依赖起点见[Docker说明](../getting-started/docker.md)、`docker/compose.gpu.yml`与`requirements/gpu-jax.txt`。这些已有入口不意味着本研究新版物理协议已支持GPU。

## GPU开发顺序与验收

| 顺序 | 实施内容 | 完成标准 |
| --- | --- | --- |
| G0 存储与迁移 | 保存工作区/必需资产、SHA清单；实现下节限额后再开长任务 | 全部冻结依赖可加载；空间估算及停止恢复行为有效 |
| G1 当前CPU参考回归 | 在目标机器按模块串行跑`qualify_cpu`；先不启用新采集/learned loops | 当前JAX/Torch/MJX与NumPy/native结果独立记录；失败不被历史通过数覆盖 |
| G2 新协议provider | 按现有frontend→capability→provider接入matched keyway、TCP signed wrench、真实3D导槽和响应模型的JAX/MJX实现 | 冻结scalar CPU输入与新设备结果在声明容差内；事件、完整carry、子步力峰值及失败语义一致。未支持组合显式拒绝 |
| G3 单卡batch | 同拓扑`vmap`世界/机制/候选轴，时间与更新轴`scan`；不同Panda/xArm7/UR5e拓扑分桶；完整chunk和microbatch | 世界独立、partial reset无串扰；预热后热路径无NumPy/SciPy/Python逐步回传，通过严格transfer guard |
| G4 学习与风险 | 同权重/输入/noise的response与transport前向、梯度、单步更新；全局CVaR两遍microbatch | loss/gradient/mask与全量参考一致；不能平均各microbatch CVaR；相同noise重放 |
| G5 性能 | 分开测compile、first call、同步steady state、数据传输；batch递增、长运行及恢复 | 保存实际GPU/driver、signature、compile count、吞吐、峰值显存和失败；达到主计划对应设备门槛 |
| G6 科学补齐 | 完成v2矩阵、独立审计和当前回归；再冻结GPU正式规模协议、预算曲线/强baseline | 每项以独立report及原阈值判定；新的GPU物理/优化协议单独记结果 |
| G7 后端扩展 | 单卡稳定后验收多卡全局归约；Torch复用portable权重、明确batch/compile能力；native/Rust沿ABI逐项扩展 | 同全局batch/noise下归约与梯度一致；inference-only不冒充训练/GPU能力 |

`qualify_cpu`当前明确要求`JAX_PLATFORMS=cpu`；它可以在GPU机器上做CPU参考回归，不能改环境变量就当GPU资格脚本。通用旧JAX设备链可以提供fixture/并行基础，但不能静默回退到legacy接触语义。

Go2+Z1现有WBC为**host OSQP**，固定4ms更新，物理步长必须整除4ms；它不是GPU驻留控制器。Task4先保留CPU参考，再通过controller/provider接口接入并验收设备内批量QP或明确登记host hybrid成本。仅把neural response移到GPU不能宣称全身physics/collector已并行。

## 防止再次耗尽磁盘：下一阶段必须实现

以下是**待实现要求**，目前仅有压缩NPZ、可逆外层gzip和逐episode原子工件，不具备全局磁盘限额；压缩也不能代替限额。

- 统一`ArtifactPolicy`/writer，由公共runner传给各backend；区分`response_training`、`search_summary`、`frozen_evaluation`及诊断输出。训练所需完整carry、计费probes、冻结最终轨迹和失败证据保留；候选搜索默认只保留重算目标/选择所需的signature、cost、status、seed和SHA，抽样诊断，避免保存每个候选的完整carry。恢复协议须显式版本化并验证summary足够，不能直接删旧tensor。
- 每次运行预估episode数×单条大小及模型/cache临时空间；配置`max_artifact_bytes`、`min_free_bytes`、最大并发writer及日志/cache上限。建议初始每run工件限额10GiB、所在文件系统预留20GiB，按实测和机器容量调整；GPU显存限额与磁盘限额分别管理。
- 调度器为并发任务预留磁盘配额，写前复查；不足时停止提交新job、发布断点和`resource_exhausted`状态，保留pending预算，不自动重试大型安装、不继续产生轨迹。用小型临时目录/模拟配额验证，不实际填满磁盘。
- 同步时直接传原工件，不额外制作全量tar副本；先验证目标端SHA和恢复，再按用户指定范围清理源端临时env/cache。此次停止未删除研究证据，也未进行Docker全局prune。

本地依赖安装曾因空间不足失败，重复大型JAX/Torch/MJX安装被自动审批拒绝，理由是可能再次耗尽主机磁盘并损坏工件。该安装未继续；后续在目标机器完成环境准备和资格测试。
