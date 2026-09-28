# 数据批次、严格任务评估与城市迁移

## 保存的数据和重新生成的数据

`datasets/synthetic/` 保存已经用于报告的合成数据。`experiment_results/frozen/` 和 `experiment_results/published_figures/` 是这些历史批次的结果、图片。新生成的文件保存在使用者指定的独立目录；评估后使用 `plot -- --data-root <新结果根目录>`，不沿用旧表格。

运行：

```powershell
python commands/audit_release_batches.py --out-dir experiment_results/release_inventory
```

输出：

```text
experiment_results/release_inventory/
├─ releases.json
└─ RELEASES_CN.md
```

逐个检查保存数据的哈希，列出方法、源路径、批次和有依据的预算。保存的 PrivTrace 批次来自 official-stageA+p30，预算为 1；新的 privtrace-native 是另一实现配置。本轮保留旧 PrivTrace 数据。其他保存发布没有随包、绑定到该哈希的生成预算协议时，报告写“未绑定生成协议”，不从文件名推断预算。

`all-precomputed` 校验保存文件并重画保存结果，**不执行合成、攻击或下游训练**。从头运行实验请依次执行 README 每节的生成命令、评估命令、作图命令。

## 全量拟合与严格 TSTR

`evaluate` 中的任务代理是全量拟合语料上的回顾性任务评估。`run-tstr` 读取独立的 `--train-real`、`--test-real`，实际训练和评分，并在本地研究 manifest 中绑定两个输入文件、公共图、bbox、seed 和合成数据哈希。这些私有输入哈希属于本地复现记录，不是具有 DP 保证的公开发布对象。

严格 TSTR 必须先用训练划分重新生成合成数据；预测器只在这些训练侧发布上训练。把全量合成数据传给 `run-tstr` 不会自动成为严格 TSTR。生成器命令与训练输入对应关系仍需保留：任务入口仅能证实评分使用了哪两个文件，不能凭文件名证明生成器没有读取测试集。因此 manifest 单独记录 `synthesis_lineage_verified=false`，而不是给未核验的生成流程签发“严格”结论。

GSRT 的三种路由表使用以下命令合并（DFR 也提供相同独立脚本）：

```powershell
python evaluation/combine_tstr_mr.py --native experiment_results/recomputed/tstr_native/results.csv --fmm experiment_results/recomputed/tstr_fmm/results.csv --stmatch experiment_results/recomputed/tstr_stmatch/results.csv --out-dir experiment_results/recomputed/tstr_mr
```

每个输入旁必须有新 `run-tstr` 生成的 `manifest.json`。聚合器核对训练/测试哈希、公共图、bbox、seed；拒绝重复名称、重复路由单元、缺方法、NaN/Inf、超范围比例、非单位 Real-train 以及回顾性输入。旧结果缺少输入绑定时，重新运行任务评分（复用已有训练侧合成数据），不要手工添加一个未经核验的哈希。

同一方法的 Native、FMM、STMatch 还必须来自同一个路由前合成发布。入口从坐标视图 sidecar 读取源发布哈希，聚合器比较 `base_releases`；不同批次即使方法名相同也不能拼成矩阵。坐标视图与道路视图必须对应同一份源发布。

新 FMM 缓存会保存逐方法的输入、网络、输出哈希和匹配参数。GSRT 的路线转换命令核验这三个哈希，并要求缓存条数与源数据条数相等，不再默默截短输入。历史无来源缓存只能通过显式 `--allow-unbound-cache` 用于诊断，其未绑定状态写入 manifest；正式重跑使用新缓存。

输出 `results.csv`（六面板）、`results_all.csv`（七指标）和绑定三份输入表哈希的 `manifest.json`。这些命令的相对路径以独立目录为根，不依赖当前工作目录。

## 城市迁移

四目录目前随包提供北京道路缓存和合成数据。Porto/SF 注册项给出公共配置，**不等于对应输入数据和公共缓存已随包提供**。跨城市执行时显式提供同城真实轨迹 pickle、公共 OSM 缓存及槽位；不能使用北京缓存代替。

|城市|bbox（纬度最小、最大、经度最小、最大）|公开槽位|
|---|---|---|
|北京|39.75 40.15 116.10 116.65|17123|
|Porto|41.10 41.20 -8.70 -8.55|20000|
|San Francisco|37.60 37.85 -122.55 -122.30|20000|

以下是已存在 CLI 的跨城市用法，真实文件和公共图由使用者提供；本轮不将北京测试记作这些城市的从零复现：

```powershell
python commands/reproduce.py evaluate -- --real "C:\data\porto_real.pkl" --synthetic "C:\runs\porto\trajectories.pkl" --bbox 41.10 41.20 -8.70 -8.55 --osm-cache "C:\data\osm_cache_porto.pkl" --public-slot-count 20000 --task-mode retrospective --out-dir experiment_results/recomputed/porto_metrics
```

输出 `metrics.json`、`metrics.csv`、`manifest.json`。本例不带 `--dataset-config`，避免把其它输入误绑到冻结北京哈希；`--real`/`--synthetic` 为坐标轨迹文件，`--bbox`/`--osm-cache` 为公共地图配置，`--public-slot-count` 为实验前确定的数量，`--task-mode retrospective` 声明回顾性任务口径。道路 witness 可追加 `--witness <文件>`；不提供时，坐标匹配成功不能冒充算法原生发布了道路 witness。
