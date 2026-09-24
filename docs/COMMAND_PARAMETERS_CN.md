# MTR-GSRT 命令行参数与跨任务迁移

## 1. 路径和独立运行规则

- `commands/reproduce.py` 以自身位置寻找代码和资产，不依赖启动时的当前目录。从任意目录调用时，可把脚本写成绝对路径。
- 传给子命令的绝对路径保持不变；相对路径统一相对于本复现文件夹根目录，而不是调用者的当前目录。
- 所有真实数据只作为显式参数输入。代码不会从仓库外的固定盘符读取数据。
- 迁移到新城市时，必须成组替换真实轨迹、`bbox`、OSM/道路网络、匹配路线和 edge cache；这些文件必须共享同一坐标系和道路 ID 空间。
- `--` 用于分隔总入口参数和被调用实验程序的参数。参数值含空格时必须加引号。

## 2. MTR-GSRT 生成参数

| 参数 | 示例值 | 含义与迁移规则 |
|---|---:|---|
| `--data` | `C:\data\real.pkl` | 私有坐标轨迹。迁移任务时替换为新数据；支持命名数据集或 `.pkl/.npy/.npz/.json`。坐标顺序必须为纬度、经度。 |
| `--dataset-config` | `geolife` | `configs/datasets.json` 中的数据集键；显式 `--bbox/--osm-cache` 可覆盖相应公共配置。 |
| `--epsilon-total` | `7/5` | 总隐私预算，支持精确分数或小数。不同预算实验只改变此项并固定其余设置。 |
| `--noise-seed` | `20260719` | DP 噪声种子；独立重复实验必须更换并记录。 |
| `--decoder-seed` | `30260719` | 公共路由随机种子，不消耗隐私预算。 |
| `--request-seed` | `decoder_seed + 483729` | 固定槽位请求采样种子；省略时按左式确定，比较实验应保持一致。 |
| `--decoder-betas` | `0.03,0.06,0.12,0.24,0.48,0.96` | QRSP/Doob 路由温度候选，逗号分隔；只有调参实验才覆盖。 |
| `--length-proposals` | `1` | 每个固定槽位尝试的公共长度候选数。 |
| `--length-log-penalty` | `8.0` | 路由长度偏离 DP 长度需求的对数惩罚权重。 |
| `--od-likelihood-ratio-cap` | `100.0` | OD 后处理的公共似然比上限。 |
| `--occupancy-strength` | `0.25` | 层次占用测量进入公共路由的权重。 |
| `--dwell-strength` | `0.10` | 驻留测量进入公共路由的权重。 |
| `--hierarchy-likelihood-ratio-cap` | `100.0` | 层次测量后处理的公共似然比上限。 |
| `--component-mode` | `full` | `full` 使用全部组件；`no-portal-fiber` 关闭 Portal-Fiber 路线信号；`no-graph-flow` 关闭 compact graph-flow；`demand-only` 同时关闭两类路线测量。关闭操作发生在已写出的 DP transcript 之后，以等质量公共均匀场替代该组件，其余测量、预算、随机种子和路由设置保持不变。 |
| `--public-slot-count` | `17123` | 预先声明的公开输出条数；不得从待保护数据运行时推断。严格 TSTR 中使用预先声明的训练发布规模。 |
| `--bbox` | `39.75 40.15 116.10 116.65` | `lat_min lat_max lon_min lon_max`；迁移城市时必须替换。 |
| `--osm-cache` | `osm_cache_beijing.pkl` | 与 bbox/城市一致的公共有向道路缓存。 |
| `--out-dir` | `C:\runs\mtr_gsrt` | 独立输出目录。相对路径按复现文件夹根目录解释。 |
| `--log-file` | 可选 | 将运行日志另存到指定文件。 |
| `--limit` | 不用于正式实验 | 仅 smoke test；正式实验必须与 `--public-slot-count` 相等。 |

## 3. Baseline 输入

本文件夹不含外部 baseline 源码，实验命令直接读取 `datasets/synthetic/baselines/` 和 `datasets/synthetic/train_only_baselines/` 中的冻结合成数据。需要从原始数据重新生成 baseline 时使用 03 或 04 文件夹及其同名参数表。

## 4. 数据划分与攻击参数

| 入口与参数 | 含义 |
|---|---|
| `prepare-split --data` | 待划分真实轨迹。 |
| `prepare-split --train-fraction` | 训练比例，默认 `0.8`；其余记录形成互斥测试集。 |
| `prepare-split --seed` | 固定划分顺序的随机种子，默认 `20260713`。 |
| `prepare-split --out-dir` | 写出 `train.pkl`、`test.pkl`、`split_manifest.json` 的目录。 |
| `prepare-attack-split --split-dir` | 上一步的严格划分目录。 |
| `prepare-attack-split --attack-candidates` | 每组攻击候选数量，默认 `1000`。 |
| `prepare-attack-split --seed` | 攻击候选抽样种子，默认 `20260715`。 |
| `prepare-attack-split --out-dir` | member、nonmember、reference 文件输出目录。 |
| `prepare-road-reference --dataset-config` | 数据集注册名，用于读取公开槽位数和城市配置。 |
| `prepare-road-reference --real` | 用户提供的真实坐标轨迹文件。 |
| `prepare-road-reference --network` | 与轨迹同城的公共有向道路图。 |
| `prepare-road-reference --stmatch-bin/--runtime-dir` | STMatch 可执行文件及其运行库目录。 |
| `prepare-road-reference --max-points/--radius-m/--gps-error-m/--candidates` | 每条轨迹的采样上限与公共地图匹配参数。 |
| `prepare-road-reference --out-dir` | 写出 `matched_paths/Real.pkl.gz` 与匹配 manifest 的目录。 |
| `run-privacy --method` | 图表中的方法名。 |
| `run-privacy --members/--nonmembers/--reference` | 三个互斥攻击集合。 |
| `run-privacy --release` | 被攻击的合成发布。 |
| `run-privacy --bbox` | 四项顺序为 `lat_min lat_max lon_min lon_max`。 |
| `run-privacy --seed` | 攻击模型随机种子，默认 `20260724`。 |
| `run-privacy --out-dir` | 新攻击结果目录。 |

## 5. 统一评估与实验参数

| 入口与参数 | 含义与迁移规则 |
|---|---|
| `evaluate --real/--synthetic` | 真实参考与一份合成轨迹。两者必须为同一城市和坐标系。 |
| `evaluate --witness` | 可选道路 witness；只有发布 witness 的方法传入。 |
| `evaluate --dataset-config` | 可选数据集配置键；注册数据集用它加载冻结公共设置。任意新数据集可省略并显式提供后述三项。 |
| `evaluate --bbox/--osm-cache/--public-slot-count` | 任意新数据集必须显式给出城市范围、公共道路缓存和预声明评估条数。 |
| `evaluate --task-mode` | 当前为 `retrospective`；严格泛化任务使用独立的 `run-tstr`。 |
| `evaluate --evaluation-seed` | 评估抽样随机种子，默认 `20260713`。 |
| `evaluate --cdtw-samples` | cDTW 真实样本数，默认 `400`。 |
| `evaluate --cdtw-candidates` | 每条轨迹比较的候选数，默认 `24`。 |
| `evaluate --cdtw-od-grid` | cDTW 的 OD 分层网格分辨率，默认 `12`。 |
| `evaluate --out-dir/--log-file` | 指标目录和可选日志文件。 |
| `run-profile --real` | 统一真实参考。 |
| `run-profile --synthetic` | `NAME=PATH`，每个方法重复一次。 |
| `run-profile --witness` | `NAME=PATH`，只为有 witness 的同名方法提供。 |
| `run-profile --dataset-config/--bbox/--osm-cache/--public-slot-count` | 注册数据集可用配置键；任意新数据集显式给出范围、同城道路缓存和评估条数。 |
| `run-profile --out-dir` | 逐方法 raw 指标与汇总 CSV 目录。 |
| `run-framework --real-routes` | 真实匹配道路路线参考。 |
| `run-framework --edge-cache` | 可选公共 edge cache；省略时使用包内北京资产。迁移城市必须显式提供。 |
| `run-framework --routed` | `NAME=ROUTE_PICKLE`，对每个已路由方法重复。 |
| `run-framework --out-dir` | 框架提升结果目录。 |
| `run-mr --real-routes` | 真实匹配道路路线。 |
| `run-mr --edge-cache` | 与所有路线对象共享边 ID 的公共缓存。 |
| `run-mr --route` | `M::R=PATH`，显式加入一个测量—路由组合，可重复。 |
| `run-mr --route-dir` | 批量扫描 `<M>/<R>.pkl[.gz]` 的根目录；缺省为包内冻结路线目录。 |
| `run-mr --out-dir` | M×R 原始值和矩阵目录。 |
| `run-ablation --data/--dataset-config/--bbox/--osm-cache` | GSRT 消融的显式真实坐标数据和同城公共配置。 |
| `run-ablation --epsilon-total/--noise-seed/--decoder-seed/--request-seed/--public-slot-count` | GSRT 各臂共用的隐私预算、随机种子和固定输出规模。 |
| `run-ablation --limit` | 仅用于小规模功能检查，且必须等于 `--public-slot-count`；正式实验不使用。 |
| `run-ablation --component-modes` | GSRT 实际重新合成的算法臂，可选 `full no-portal-fiber no-graph-flow demand-only`。 |
| `run-ablation --real-routes/--edge-cache/--network` | DFR 消融的真实匹配路线、公共缓存和同一有向道路图。 |
| `run-ablation --seed/--length-aware/--public-slot-count` | DFR 各臂共用的噪声种子、长度组件开关和固定输出规模。 |
| `run-ablation --variants` | DFR 实际重新合成的算法臂，可选 `no-endpoint-measurement endpoint-measurement-only family-measured-unmatched-router full`。 |
| `run-ablation --out-dir` | 各臂新合成数据、逐臂指标、聚合 CSV 和 manifest 的目录。 |
| `run-tstr --train-real/--test-real` | 严格互斥的真实训练/测试轨迹；合成器只能读取 train。 |
| `run-tstr --synthetic` | 合成训练语料路径；每个方法重复一次。 |
| `run-tstr --names` | 与 `--synthetic` 完全同序、同数量的方法名。 |
| `run-tstr --osm-cache/--bbox` | 测试城市的公共道路缓存和范围。 |
| `run-tstr --seed` | 下游模型训练与抽样种子，默认 `20260713`。 |
| `run-tstr --out-dir` | 逐任务模型输出和 TSTR 汇总目录。 |
| `run-structure --dataset` | `NAME=PATH`，Real 与各合成语料各重复一次。 |
| `run-structure --out-dir` | 逐轨迹结构诊断和汇总目录。 |

## 6. 作图参数

| 参数 | 含义 |
|---|---|
| `plot --figure` | `evidence/framework/privacy/profile/mr/ablation/tstr/structure/all` 之一。 |
| `plot --data-root` | 本轮真实运行结果根目录；指定后若缺失对应结果会直接报错，不回退到冻结数值。 |

## 7. 最小跨城市替换清单

1. 用新城市真实轨迹替换 `--data`、`--real`、`--train-real`、`--test-real`。
2. 用新城市范围替换所有 `--bbox`，顺序固定为纬度最小、纬度最大、经度最小、经度最大。
3. 用新城市公共图替换 `--osm-cache` 或 `--network`，并重建同图的 `--edge-cache`。
4. 所有 `--real-routes`、`--real-matched` 和 `--routed/--route` 文件必须由同一图产生。
5. 预先声明 `--public-slot-count`，记录全部随机种子、隐私预算和输出目录。
6. 先运行 `--check-only`（DFR）或小规模 smoke test，再执行完整生成；随后依次运行评估、实验聚合与作图。
