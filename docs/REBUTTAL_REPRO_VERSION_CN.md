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

### Q2：三城主运行协议核验（2026-09-30）

- 新增 `evaluation/audit_rebuttal_q2_protocols.py`，从三份主运行的保存协议、带噪 transcript、公开 OSM 和输出文件重新核验 Reviewer 2 Q2 的输出数、回退数、道路选择维度、预算及输出哈希；命令和预期字段见 `docs/REBUTTAL_Q2_CONFIG_CN.md`。
- 历史三城主发布的九件主要输出文件 SHA 均与各自协议相符；带噪向量维度分别为北京 `438/1879/6733`、波尔图 `446/1837/6071`、旧金山 `390/1537/2825`。历史耗时约 `726/1031/520` 秒，脚本只报告耗时而不把机器速度当算法正确性断言。
- 公开 Porto/SF OSM cache 的 pickle 字节哈希与旧协议不同；对固定 bbox 与道路类别过滤后的有序公共道路对象，经规范 JSON 哈希与历史对象相同。脚本同时报告字节不一致和内容一致，绝不把两者混为一谈。旧协议没有记录私有输入哈希；公开发布仍保持这一隐私边界。此项是保存主运行的协议复核，不冒充从三个原始数据集重新生成全部主发布。
- 单元测试覆盖内容相同但 pickle 哈希不同的公开图，以及输出文件被改动时拒绝通过。
- 当前 GitHub 发布不在 DP 协议写入输入相关耗时。Q2 审计入口现接受三个城市各自的 `--CITY-performance-log`，逐个验证外部日志绑定的协议 SHA，历史 `elapsed_sec` 协议仍可直接读取。历史 fallback 数严格校验；新版解码器的 fallback 与历史值并列报告，不拿旧值替代新运行。以新北京 + 旧 Porto/SF 完整协议实际测试了混合版本审计；全部单元测试仍为 75 passed、119 subtests passed。
- 独立 GitHub 克隆现已用三城本地输入全量生成一次：北京 17,123 条、波尔图和旧金山各 20,000 条，三城 DP transcript SHA 与各自历史发布逐字节相同；三城轨迹/witness 均非旧哈希。新回退数为 `0/11/1`，历史为 `0/14/1`；新生成耗时约 `497/1042/409` 秒，和历史机器时间不应相等。统一 `audit_rebuttal_q2_protocols.py` 已对三份新发布、外部计时日志、公共图、维度与输出哈希通过，结果在本地 `check/rebuttal_public_repro_20260930/q2_three_city_current_clean_clone_audit_v2.json`。字节 OSM 字段现将“当前发布协议匹配”与“历史协议匹配”严格区分；新版历史字段为 `null`，跨版本以所选公共道路对象哈希比对。上述新轨迹是当前源码版本的复现，不是历史解码器的逐字节重建。
- 三城新旧合成输出又在**同一当前统一评估器**上分别评分。统计需求侧 Trip、OD JSD 三城逐项相同；Grid、Len、RoadSeg 及任务指标有小幅差异，和路由版本不同一致。旧保存 `metrics.json` 中 `directed_road_validity` 曾表示坐标投影有效率（北京约 `0.324`），当前评估器在传入 witness 时把该旧口径保留为 `coordinate_projection_directed_road_validity`，而 `directed_road_validity` / `witness_valid` 表示已验证的有向道路对象，均为 `1.000`。旧值和新值同名不同义，不能直接相减；当前统一评估器对旧北京输出重算，也得到 witness-based `directed_road_validity=1.000`。三城重算结果分别保存在本地 `check/rebuttal_public_repro_20260930/{beijing,porto,sf}_{current_eval_all,historical_eval_current_code}`。

### 共用 M×R：rebuttal1-Q2 / rebuttal2-Q4（2026-09-30）

- 新增 `evaluation/materialize_direct_road_rows.py`：五方法 Nearest 真实执行，四个坐标基线的 Original 按无补路的直接有向相邻关系执行；MTR 原生 witness 路线保留原文件。Nearest 使用每条最多 32 个观测点、200 m 公共道路搜索半径。
- 新增 `evaluation/materialize_family_routers.py`：从**同一方法**的保存 STMatch 载体生成四种路径族重采样。历史方法索引和随机种子原样保留；五方法 20 份路线各 17,123 条，与隔离实验逐条相同。
- 补充原有 MTR-GSRT 完整 FMM/STMatch 合成缓存及其 SHA-256 manifest；不上传真实道路参考。
- 新增 `evaluation/evaluate_rebuttal_mr.py`：唯一 40 格结果 `experiment_results/rebuttal_shared_mr/metrics/`，同一输入哈希和五项指标供两个回复引用。旧图源中已有真实缓存的 31 格全部数值保持一致；原先的九个空占位改为实测。Python 3.11.15 锁定环境与当前工作环境重新评分的 40×5 数值相同（最大差异小于 (10^{-12})）。
- 具体复现顺序、全部参数、预期文件和图见 `docs/REBUTTAL_SHARED_MR_CN.md`。旧回复 PDF 仍含空占位版图像；用户文档按要求保持原样，公开结果不被冒充为旧 PDF 的逐像素重现。

Q5 和 Q6 的版本与结果将在逐项完成公开重跑后追加，不预先将保存图表登记为已复现。

### 北京主运行的干净克隆核验（2026-09-30）

- 在独立 GitHub 克隆、Python 3.11.15 锁定环境中，对本地冻结的北京 17,123 条完整输入执行 `generate-main`；先取回公共 OSM 的 Git LFS 实体文件，随后全量完成。输出恰有 17,123 条轨迹和 17,123 个 witness，公共路由回退为 0。
- 新 DP transcript SHA-256 为 `5343F9BE4F7C0E9EEDC42FC76231931F317C83E76788C5A7B50D40C41D4E4AAD`，与旧保存结果逐字节相同。新轨迹/witness SHA-256 分别为 `34DE44012537557783A0ECF375C33145C13B2A75CD06B733FA2FDADBB26CDAC7` / `EE87AEF1C919244BC47FEFCBB376BB9652A18B0231C1BEB12498BD8BB9AEA001`，与旧保存的 `BBDB4093366A8E6CE8B9018BE0E19AFA63DB1F294938B407D0ED4337D781FBEA` / `AF920F6FF71150AE66BAC7A80F594C1598D2514BC7D8A4D488764520497A23FD` 不同。旧协议记载的解码器源码绑定与当前 GitHub `SOURCE_MANIFEST.json` 不同；此结果是**当前版本**从头生成成功，不能称旧论文生成物逐字节复现。后续效用表需要分别绑定所用版本。
- 同一真实道路缓存下作回顾式路线选择复评，当前 GitHub/旧保存输出的 RC-CPC 为 `0.264935/0.263674`、RC-NDCG 为 `0.676456/0.671910`、NextRoadAcc 为 `0.237699/0.221148`、NextRoadNLL 为 `1.953299/1.960210`。数值接近但不相同，这不能替代前述哈希差异，也不是严格 train-only 结果。

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

### Q5：整体 Portal 单元的独立 train-only 研究臂（2026-09-30）

- `generate-main --portal-unit-mode ablated` 在 q5 私有测量中删去 portal-fiber 块，按固定质量重分配其余四块；公共 crossing-edge prior 由单独的 `portal_unit_research_decoder.py` 读取。正式 SOURCE_MANIFEST 中原解码器、默认 `full` 输出和生产账本不变。此开关不同于只在带噪 transcript 后中和信号的 `--component-mode no-portal-fiber`。
- 锁定 Python 3.11.15 环境以真实训练划分生成 13,698 条 ablated 轨迹和 witness，fallback 0。base/compact-flow 七块与已有 full 臂逐项相同，五块 q5 按定义改变，portal-fiber 块为零。研究解码器哈希与协议绑定一致。
- 相同未见测试分割的 Full 与 Portal-removed 的 RC-CPC 为 `0.253970/0.296508`，RC-NDCG 为 `0.672019/0.620615`，NextRoadAcc 为 `0.227050/0.153624`，NextRoadNLL 为 `1.961308/1.939319`。两臂呈现排序/判别与分布重合/对数损失之间的权衡，不得写成全指标提升。命令见 `docs/REBUTTAL_Q5_TRAIN_ONLY_DIAGNOSTIC_CN.md`；它不追认旧 PDF 中以全量输入合成、却称严格 TSTR 的历史表。公开测试为 75 passed、119 subtests passed。
- GitHub 独立克隆取回公共 OSM 的 LFS 实体后，以相同 5 条 fixture、`N=5,m=2` 和研究种子运行 ablated；轨迹/witness/transcript 三件输出的 SHA 与工作克隆逐件相同，分别为 `58C956984CF39E18F4460DAF36B50D9C025DEE432A16B8332CA9A93A1B3FFC00`、`0A9311D4ECC100046BFB53C60220C8344C654BDEB257165D2A815DB5EC6E1CCA`、`FB63FBB2AB1ACAAFEC3C798A8A930D848EE816843E87484866239F97C9A6CE81`。独立克隆完整测试亦为 75 passed、119 subtests passed。

### Q6：北京七预算×五种子的查询与发布层诊断（2026-09-30）

- 新增 `run-rebuttal-q6-low-epsilon`：显式接收完整真实输入、公共 OSM、bbox、容量、35 份 DP transcript/协议、35 份统一评估和路线选择 CSV；对缺失矩阵、错误输入哈希和既有输出目录拒绝运行。原始 exact 查询仅在内存中计算，输出清单绑定所有输入文件和脚本 SHA-256。
- 在 Python 3.11.15 锁定环境，用已有 35 份发布重新计算后，查询块详细 CSV SHA-256 为 `D66043928B0A369F8AEE74A3141BA219F157ABAF156BB2316C971A9659142857`，完整发布逐种子详细 CSV 为 `156210F90705D29EFB27B02C1C87C6207765BB921F480F03830D7171EE0B0F1A`，均与旧补充材料逐字节一致。两份均值/95% 区间汇总只有约 `1e-13` 的浮点末位差异。新旧汇总经同一公开作图入口生成的 PNG SHA-256 均为 `95C75ADDEDF0759C78EAC2534E9210F044B69F6A132F60BC5B6E11714A1BB797`；旧保存 PNG 因绘图环境不同不逐字节相同。代码测试为 65 passed、119 subtests passed。
- 命令与目录结构见 `docs/REBUTTAL_Q6_LOW_EPSILON_CN.md`。这一核对复算的是已存在的 35 份发布上的测量和效用表，**没有**在独立克隆重新生成 35 份发布，也没有完成跨城市和攻击部分。
- 新增 `aggregate-rebuttal-q6-multicity`，要求 Porto/SF 各五个生成协议及评分文件全齐，并记录二十份输入 SHA-256。读取已有十份发布重新聚合，详细 CSV SHA-256 为 `0D00A327A9A04EE36265151C7B2470BCD7CCFC8BD1521AFD95EEFB568373AE31`，汇总 CSV 为 `A323BE108FA44B902A39284F933FBA8B969765CDF75F6D3A0213AA8BF438BFCA`，均与旧补充材料逐字节一致。这认证十份已保存结果的聚合，尚未从未公开的两城真实输入重新生成或评估十份轨迹。
- 新增 `assemble-rebuttal-q6-threat-model`，显式读取四组实际攻击结果及冻结候选划分，并拒绝缺失文件。重新组装的攻击 CSV SHA-256 `6487C6EA874F6BDB389B006E266BF5C3EF36F365684AB831656AD64D78CF1E9C`，威胁模型 Markdown SHA-256 `0AFEC5887043C0F55FFCB184C89D6466B5DFC2293C59B2F53AFE5CA81022392A`，均与旧补充材料一致。这只认证旧攻击文件的组装口径，不等于从私有候选集重新运行四组攻击。
- 新增可选 `generate-main --local-performance-log`：在 DP 发布目录外写入准备、私有测量、公共路由三段耗时与 OS 进程峰值 RSS，并用协议 SHA-256 绑定；不在公开发布协议中增加输入相关计时。Q6 聚合器接收 Porto/SF 各自性能日志根目录；对新协议缺失计时、日志错绑或缺少种子一律拒绝，对旧协议 `elapsed_sec` 保持兼容。五条输入的工程小样本实跑证明三件 DP 生成物与旧同种子输出逐字节同哈希；旧十份跨城聚合的两张 CSV SHA 仍为 `0D00A327A9A04EE36265151C7B2470BCD7CCFC8BD1521AFD95EEFB568373AE31` / `A323BE108FA44B902A39284F933FBA8B969765CDF75F6D3A0213AA8BF438BFCA`。新日志模式的十种子 fixture CLI 测试通过。命令见 `docs/REBUTTAL_Q6_LOW_EPSILON_CN.md`。

- 2026-09-30：`generate_sweep.py` 已修正为共享查询、冻结输入和当前配置路径。协议不再写入 `input_record_count` 或 `elapsed_sec`；阶段耗时由发布目录外的本地性能日志承载。北京 `epsilon=7/5, seed=20260719` 单元格已从当前公开入口完成，输出 17,123 条、fallback 0，transcript SHA 与主运行一致。该单元格位于 `check/rebuttal_public_repro_20260930/shared_query_sweep_one`，仅作为本地复核记录；完整 7×5 扫描和跨城市十种子从零重跑仍未完成。
- 2026-09-30：`commands/reproduce.py generate-sweep` 已加入统一复现入口，提交 `e2ff4b9b11aea9093539382f9f9e064df01ff579` 已推送；`python -m pytest -q tests` 为 `78 passed, 119 subtests passed`。该入口只转发显式预算/种子参数，不改变共享查询或发布协议。
- 2026-09-30：公开文件清单已随审计后的源码、文档与结果更新；`python commands/reproduce.py verify` 返回 `VERIFY OK: MTR-GSRT (337 files)`，提交 `9b616a8ac3e358e96d52e479a4e015fa9662f663` 已推送。
- 2026-09-30：统一入口新增 `audit-rebuttal-q2`，可从 GitHub 命令行复核三城协议、维度、预算、回退和外部性能日志；它只读取已保存发布与公共 OSM，不读取私有轨迹。
- 2026-09-30：从当前公开入口完成北京 7×5 隐私预算生成扫描，共 35 个单元，目录为本地隔离记录 `check/rebuttal_public_repro_20260930/q6_current_clean_clone_full`。35 个协议均为 17,123 条固定槽位、`input_record_count=not_released`、无 `elapsed_sec`，无残留 staging 目录；该批次完成了生成复现，尚未替代已有 Q6 的统一指标和道路选择评分结果。
- 2026-09-30：从该批次的 `epsilon=7/5, seed=20260719` 产物接入 `commands/reproduce.py evaluate` 完成一次端到端指标 smoke，统一评估器写出 `metrics.json/csv/manifest`；完整 35 格评分仍按 Q6 文档命令执行，旧 Q6 聚合结果不被新 smoke 覆盖。
