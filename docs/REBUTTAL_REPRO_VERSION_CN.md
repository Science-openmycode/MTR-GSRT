# Rebuttal 实验复现版本记录

本文件按实验记录公开代码变化。`rebuttal1.md` 与 `rebuttal2.md` 保持原样；两份回复共用的实验只保存一套计算结果。

## Q1：输入容量与输出规模分离（2026-09-30）

- 生成入口：`generation/mtr_gsrt/generation/mtr/generate.py`。
- 增加 `--public-input-capacity N`，将私有测量的公开输入容量与 `--public-slot-count m` 分离。旧命令省略前者时仍取 `N=m`，不改变等规模运行的测量参数。
- 文件输入完整读取。若有效轨迹数超过公开容量 `N`，直接拒绝，而不是静默只测量前 `m` 条。输出仍恰为 `m` 条轨迹及 `m` 个道路 witness。
- 公开协议写入 `N,m`，不写入实际有效输入条数和运行耗时。注册数据集的完整文件 SHA-256 比对改为显式 `--verify-frozen-input` 本地预检；该预检不属于相邻数据集上的 DP 发布接口。
- 相邻输入的正式隐私命题针对单条已预处理轨迹的 add/remove 变化，要求公共图、预处理规则、`N,m` 与预算预先固定。此仓库的公开噪声种子运行用于研究复算，仍带 `RESEARCH_ONLY_REPRODUCIBLE_NOISE_DO_NOT_RELEASE` 标记；不将确定性种子扫描称为正式 DP 发布。

复现实验命令见 `README_CN.md` 第 3 节和 `docs/COMMAND_PARAMETERS_CN.md`。本地功能检查覆盖 `N=5`、同一份 5 条私有输入、`m=2/3/5/7`；输出分别有相应条数的轨迹与 witness，12 个 DP transcript 数组逐项相同。相同 `m=5` 的旧提交和本版三件生成物（轨迹、witness、DP transcript）SHA-256 完全相同。另以 4 条及空输入执行了 `N=5,m=3`，公开输出仍为 3 条。`m=2` 使用 Python 3.11.15 锁定环境通过统一 `commands/reproduce.py generate-main` 入口。功能测试使用 `python -m pytest -q tests/test_public_output_cardinality.py`；完整测试结果为 49 passed、119 subtests passed。这些小规模检查证明公开 CLI 的数量分离行为，不代替生产随机源认证或大规模完整实验。

## 后续实验

### 共用 M×R：rebuttal1-Q2 / rebuttal2-Q4（2026-09-30）

- 新增 `evaluation/materialize_direct_road_rows.py`：五方法 Nearest 真实执行，四个坐标基线的 Original 按无补路的直接有向相邻关系执行；MTR 原生 witness 路线保留原文件。Nearest 使用每条最多 32 个观测点、200 m 公共道路搜索半径。
- 新增 `evaluation/materialize_family_routers.py`：从**同一方法**的保存 STMatch 载体生成四种路径族重采样。历史方法索引和随机种子原样保留；五方法 20 份路线各 17,123 条，与隔离实验逐条相同。
- 补充原有 MTR-GSRT 完整 FMM/STMatch 合成缓存及其 SHA-256 manifest；不上传真实道路参考。
- 新增 `evaluation/evaluate_rebuttal_mr.py`：唯一 40 格结果 `experiment_results/rebuttal_shared_mr/metrics/`，同一输入哈希和五项指标供两个回复引用。旧图源中已有真实缓存的 31 格全部数值保持一致；原先的九个空占位改为实测。Python 3.11.15 锁定环境与当前工作环境重新评分的 40×5 数值相同（最大差异小于 (10^{-12})）。
- 具体复现顺序、全部参数、预期文件和图见 `docs/REBUTTAL_SHARED_MR_CN.md`。旧回复 PDF 仍含空占位版图像；用户文档按要求保持原样，公开结果不被冒充为旧 PDF 的逐像素重现。

严格 TSTR、Q5 和 Q6 的版本与结果将在逐项完成公开重跑后追加，不预先将保存图表登记为已复现。

### 严格 TSTR 路径族候选视图（阶段性，2026-09-30）

- `materialize_family_routers.py` 增加 `--seed-schedule tstr-v1`；原完整发布矩阵仍默认使用 `full-road-v1`，原先路线与哈希不变。
- 新增 `materialize_family_tstr_views.py` 和统一 CLI 的 `materialize-family-routes`、`materialize-family-tstr-views` 两阶段入口。输出同时保存下游通用坐标视图、道路任务坐标视图及完整来源哈希；无效路线退回同槽合成坐标。
- 四种 train-only 基线 × 四种路径族 × 两种坐标视图共 32 个文件已从公开代码重新生成，逐文件 SHA-256 与 2026-09-28 隔离批次一致。命令及参数在 `docs/REBUTTAL_TSTR_FAMILY_VIEWS_CN.md`。
- `run_tstr_experiment.py` 识别路径族视图的来源链，并在路由文件被篡改时拒绝评分。当前仅证明候选视图能复现；28 候选验证选择、最终真实测试评分、旧表批次差异仍未完成。
