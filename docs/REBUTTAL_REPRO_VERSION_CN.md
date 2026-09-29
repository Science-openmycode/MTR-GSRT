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

M×R、严格 TSTR、Q5 和 Q6 的版本与结果将在逐项完成公开重跑后追加，不预先将保存图表登记为已复现。
