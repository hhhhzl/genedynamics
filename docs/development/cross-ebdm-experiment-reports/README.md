# Cross EBDM 开发实验报告

论文草稿位于 `latex/cross_ebdm`。本目录保存开发报告与论文清理前的历史材料，
不参与论文编译，也不充当正式论文结果。

2026-10-14 按用户要求，将论文中的开发实验报告迁出：

- [迁移清单及 SHA](20261014_pre_cleanup/manifest.json)：13 个报告附件、旧 PDF，
  以及修改前主稿与理论文件的两份快照。
- [原开发报告 PDF](20261014_pre_cleanup/intervention_calibrated_sim2real_blueprint.pdf)：
  仅用于历史查阅。
- `20261014_pre_cleanup/sections/`：P1、P1.5–P1.7、P2 的实验记录正文。
- `20261014_pre_cleanup/tables/`、`figures/`：原实验结果与失败诊断附件。
- `20261014_pre_cleanup/archive/`：更早且原工件不保留的 CPU 数值说明。
- `20261014_pre_cleanup/draft_snapshot/`：清理前原稿，保留主稿内嵌的开发记录。

原始实验工件和独立审计继续位于 `results/_development/`，P0–P2 解释记录
继续位于 `docs/development/cross-embodiment-p*.md`。本轮没有改写历史结果。

实验报告生成默认写入各实验 `_reports`。需要额外 LaTeX 导出时可指定
`--report-export-dir docs/development/cross-ebdm-experiment-reports/<export_name>`；
`--paper-dir` 为兼容别名，两个参数都拒绝论文 `latex/` 目录及指向它的链接。
草稿编译日志与构建证明同样保存在论文源目录以外。
