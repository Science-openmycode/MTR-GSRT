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

Q5 和 Q6 的版本与结果将在逐项完成公开重跑后追加，不预先将保存图表登记为已复现。

### 严格 TSTR 七路由比较（阶段性，2026-09-30）

- `materialize_family_routers.py` 增加 `--seed-schedule tstr-v1`；原完整发布矩阵仍默认使用 `full-road-v1`，原先路线与哈希不变。
- 新增 `materialize_family_tstr_views.py` 和统一 CLI 的 `materialize-family-routes`、`materialize-family-tstr-views` 两阶段入口。输出同时保存下游通用坐标视图、道路任务坐标视图及完整来源哈希；无效路线退回同槽合成坐标。
- 四种 train-only 基线 × 四种路径族 × 两种坐标视图共 32 个文件已从公开代码重新生成，逐文件 SHA-256 与 2026-09-28 隔离批次一致。命令及参数在 `docs/REBUTTAL_TSTR_FAMILY_VIEWS_CN.md`。
- `run_tstr_experiment.py` 识别路径族视图的来源链，并在路由文件被篡改时拒绝评分；同时兼容已有 STMatch manifest 的大写 `STMATCH` 标签，仍核查基础发布哈希。
- 新增 `run_rebuttal_tstr_selection.py` 与统一 CLI 的 `run-rebuttal-tstr`，对四基线各七个实际候选完成验证集评分，然后在独立真实测试划分上同批重评原生、最佳路由和 MTR-GSRT 共九行。验证阶段选出的四个路由与旧回复一致；最佳路由四行的 16 个指标数值与历史表一致；五个原生行的 20 个数值中有 15 个不同，因为旧表拼接了保存的历史原生批次。公开入口保留两阶段哈希和原始结果，不悄悄替换旧表。
- 本次选择使用真实训练划分内的私有验证标签，是研究诊断，不属于无额外预算的 DP 后处理。完整输入生成链认证和独立克隆端到端重跑仍未完成。代码测试结果：58 passed、119 subtests passed。

### Q3：长度条件路由数学核对（2026-09-30）

- 新增 `audit-rebuttal-q3` 公共 CLI，在无私有输入的两路图上调用实际 `RSPBridge` 和 `portal_information_projection`；逐项断言 Doob 转移与 Gibbs 分配函数、期望代价与解析值、长度目标下的 beta 选择、portal 信息投影与显式指数倾斜相同。
- Python 3.11.15 锁定环境与当前测试环境生成的 JSON SHA-256 同为 `0489E99AF2CE3EE7E4C7C5FAD7AE6170AA941EAA3F5B5C0486C5B109130339A8`。具体数值、命令和主配置两阶段调用链见 `docs/REBUTTAL_Q3_ROUTING_CN.md`。
- 该核对也发现旧回复把区域桥接与跨界 portal 投影写成一次统一核投影，并把多因子操作写成先组合后截断，均不是主配置逐操作规格。原回复按用户要求未改；公开文档精确标出差异。公开测试现为 59 passed、119 subtests passed。

### Q5：区域粒度诊断与 Portal-Fiber 组件对照（2026-09-30）

- 新增 `run-rebuttal-q5-partition`，显式传入本地完整真实输入、公开 OSM、bbox 和容量，重算 22 个嵌套与 8 个非嵌套划分，每个使用五个固定噪声种子。17,123 条北京冻结输入的 150 个种子行和 30 个汇总行按字段与旧补充材料完全一致；汇总 CSV SHA-256 同为 `527444A0105475E3034AFD4122911CC70CEF13089C6FAE194CECF43FA68F74F2`。详细 CSV 列顺序不同，不能要求字节哈希一致，逐字段差异为 0。
- 新增 `plot-rebuttal-q5-q6`，从新汇总结果生成的 Q5 分区 PNG 与从旧汇总结果经同一代码生成的 PNG SHA-256 相同；结果含输入/图像哈希 manifest。这里复算的是研究诊断，未公开原始轨迹或 exact q5 向量。
- 四臂重解码入口 `run-rebuttal-q5-ablation` 复用一份 DP transcript。三条新解码臂的轨迹和 witness SHA-256 均与历史协议逐项相同；四臂共同评分的 68 个指标字段与历史结果逐项相同（容差 `1e-12`）。由新结果绘出的消融 PNG 与旧结果经同一绘图入口生成的 PNG SHA-256 均为 `EE133DE8B0B15F099E7D9CDBEAF612B9259568B0636B9D522BFBFAC37674E286`。去跨界流与去局部投影两臂彼此也完全相同，不能当作两份独立效果证据。具体输入、命令和产物见 `docs/REBUTTAL_Q5_PARTITION_CN.md`。整体 Portal 单元的严格 TSTR 消融是另一项实验，尚未在本版本完成来源核验。

### Q6：北京七预算×五种子的查询与发布层诊断（2026-09-30）

- 新增 `run-rebuttal-q6-low-epsilon`：显式接收完整真实输入、公共 OSM、bbox、容量、35 份 DP transcript/协议、35 份统一评估和路线选择 CSV；对缺失矩阵、错误输入哈希和既有输出目录拒绝运行。原始 exact 查询仅在内存中计算，输出清单绑定所有输入文件和脚本 SHA-256。
- 在 Python 3.11.15 锁定环境，用已有 35 份发布重新计算后，查询块详细 CSV SHA-256 为 `D66043928B0A369F8AEE74A3141BA219F157ABAF156BB2316C971A9659142857`，完整发布逐种子详细 CSV 为 `156210F90705D29EFB27B02C1C87C6207765BB921F480F03830D7171EE0B0F1A`，均与旧补充材料逐字节一致。两份均值/95% 区间汇总只有约 `1e-13` 的浮点末位差异。新旧汇总经同一公开作图入口生成的 PNG SHA-256 均为 `95C75ADDEDF0759C78EAC2534E9210F044B69F6A132F60BC5B6E11714A1BB797`；旧保存 PNG 因绘图环境不同不逐字节相同。代码测试为 65 passed、119 subtests passed。
- 命令与目录结构见 `docs/REBUTTAL_Q6_LOW_EPSILON_CN.md`。这一核对复算的是已存在的 35 份发布上的测量和效用表，**没有**在独立克隆重新生成 35 份发布，也没有完成跨城市和攻击部分。
- 新增 `aggregate-rebuttal-q6-multicity`，要求 Porto/SF 各五个生成协议及评分文件全齐，并记录二十份输入 SHA-256。读取已有十份发布重新聚合，详细 CSV SHA-256 为 `0D00A327A9A04EE36265151C7B2470BCD7CCFC8BD1521AFD95EEFB568373AE31`，汇总 CSV 为 `A323BE108FA44B902A39284F933FBA8B969765CDF75F6D3A0213AA8BF438BFCA`，均与旧补充材料逐字节一致。这认证十份已保存结果的聚合，尚未从未公开的两城真实输入重新生成或评估十份轨迹。
