# Q6：隐私预算与完整发布的分层诊断

北京实验使用七个总预算 `1/5, 2/5, 7/10, 1, 7/5, 2, 3`，每个预算五个研究噪声种子 `20260719`—`20260723`。公开生成入口 `generate-main` 按同一公共图、槽位和预算比例生成 35 份独立合成语料；随后对每份运行统一指标评估和道路选择评估。它们是研究扫描，不能把 35 次发布合称一次 ε-DP 发布。

从仓库根目录也可以使用统一转发入口：`python commands/reproduce.py generate-sweep -- --dataset geolife --data <real_full_frozen.pkl> --verify-frozen-input --out-root <generation-root> --resume`。需要只复核一个单元格时追加 `--epsilon 7/5 --noise-seed 20260719`；不加筛选参数则依次运行全部 35 个单元格。

从头执行时，先按 README 生成 `C:\data\real_full_frozen.pkl` 与 `C:\runs\real_road_reference\matched_paths\Real.pkl.gz`。下面的命令为 35 个单次生成分别记录完整耗时，而不是把共享查询的初始化时间分摊到各次发布。它在本地空目录执行；已经存在的发布目录会由生成器拒绝覆盖。

```powershell
$work = "C:\runs\q6"
$genRoot = "$work\synthetic\geolife"
$metricsRoot = "$work\metrics\geolife"
$timingRoot = "$work\timing\geolife"
$choiceRoot = "$work\road_choice_runs"
$slugs = @{'1/5'='1_5'; '2/5'='2_5'; '7/10'='7_10'; '1'='1_1'; '7/5'='7_5'; '2'='2_1'; '3'='3_1'}
foreach ($epsilon in @('1/5','2/5','7/10','1','7/5','2','3')) {
  foreach ($seed in 20260719..20260723) {
    $epsDir = "eps_$($slugs[$epsilon])"
    $name = "${epsDir}_seed_${seed}"
    $release = "$genRoot\$epsDir\seed_$seed"
    $decoder = $seed + 10000000
    $request = $decoder + 483729
    python commands/reproduce.py generate-main -- `
      --data C:\data\real_full_frozen.pkl --dataset-config geolife `
      --verify-frozen-input --epsilon-total $epsilon `
      --noise-seed $seed --decoder-seed $decoder --request-seed $request `
      --public-input-capacity 17123 --public-slot-count 17123 `
      --out-dir $release --local-performance-log "$timingRoot\$epsDir\seed_$seed.json"
    if ($LASTEXITCODE -ne 0) { throw "generation failed: $name" }
    python commands/reproduce.py evaluate -- `
      --real C:\data\real_full_frozen.pkl --synthetic "$release\trajectories.pkl" `
      --witness "$release\road_witnesses.pkl" --dataset-config geolife `
      --out-dir "$metricsRoot\$epsDir\seed_$seed\metrics"
    if ($LASTEXITCODE -ne 0) { throw "evaluation failed: $name" }
    python evaluation/evaluation/evaluate_road_choice.py `
      --dataset-config geolife `
      --match-dir C:\runs\real_road_reference\matched_paths `
      --network public_assets/beijing_network/network.shp `
      --witness "$name=$release\road_witnesses.pkl" `
      --expected-count 17123 --synthetic-expected-count 17123 `
      --real-test-start-index 13698 --out-dir "$choiceRoot\$name"
    if ($LASTEXITCODE -ne 0) { throw "road-choice evaluation failed: $name" }
  }
}
python evaluation/evaluation/aggregate_privacy_budget_road_choice.py `
  --input-dir $choiceRoot --out-dir "$work\road_choice_aggregate"
```

每个 `release` 目录包含 17,123 条 `trajectories.pkl`、同数 `road_witnesses.pkl`、`dp_transcript.npz`、`protocol.json`；对应指标在 `metrics/.../metrics/metrics.json`，道路选择结果在 `road_choice_runs/<名称>/metrics.json`。最后一步要求恰好 7×5 格，写出 `road_choice_aggregate/metrics_long.csv` 与汇总表。`--real-test-start-index` 只固定评价参考的后 20% 轨迹；35 份合成语料仍读取全量真实输入，所以这是回顾式效用，不是严格 train-only TSTR。

若已有 35 份完整发布及其指标，可以从这些文件和本地原始轨迹重算回复中查询层与发布层的统计表。原始轨迹只在本地内存中用于构造未加噪查询对照；脚本不保存 exact 向量。相对路径按本仓库根目录解析。

```powershell
python commands/reproduce.py run-rebuttal-q6-low-epsilon -- `
  --real C:\data\real_full_frozen.pkl `
  --input-sha256 6a160ca557fbd7bab97af489b56c931e73532c498c390e31965c0d61e53361fd `
  --osm-cache generation/mtr_gsrt/data/osm/osm_cache_beijing.pkl `
  --osm-sha256 7159e12ab8712338b4ff5a2859c68b158e1374504f999dc44080c2fe2d53a5c5 `
  --bbox 39.75 40.15 116.10 116.65 `
  --public-capacity 17123 `
  --generation-root C:\runs\q6\synthetic\geolife `
  --generation-performance-root C:\runs\q6\timing\geolife `
  --metrics-root C:\runs\q6\metrics\geolife `
  --route-choice-csv C:\runs\q6\road_choice_aggregate\metrics_long.csv `
  --out-dir C:\runs\rebuttal_q6_low_epsilon
```

`--generation-root` 的目录结构为 `<eps_分子_分母>/seed_<噪声种子>/dp_transcript.npz` 和 `protocol.json`；`--metrics-root` 具有相同两层目录，其叶目录内有 `metrics/metrics.json`。`--route-choice-csv` 包含每个预算/种子的 `road_choice_ndcg`、`next_road_accuracy`、`next_road_nll`。脚本要求全部 35 组文件齐全，并在输出清单中记录每份文件的 SHA-256。`--public-capacity` 是预声明的完整输入条数；两个可选哈希参数用于本地冻结批次预检。

若按上面的从零命令运行，把聚合命令中的 `--generation-root`、`--metrics-root`、`--route-choice-csv` 分别设为 `$genRoot`、`$metricsRoot`、`$work\road_choice_aggregate\metrics_long.csv`。`--generation-performance-root` 指向发布目录外的 35 份日志；旧保存协议本身含有 `elapsed_sec` 时可省略该参数。旧指标文件中的 `DirectedRoadValidity`/`RouteCompatibleYield` 属坐标投影口径；新版指标文件同时提供该口径与 witness 道路有效率，聚合器选择前者以保持列含义不变，WitnessValid 单列报告。

得到 `q6_all_query_blocks_detailed.csv`、`q6_query_family_summary.csv`、`q6_full_release_utility_detailed.csv`、`q6_full_release_utility_summary.csv` 和 `q6_complete_low_epsilon_manifest.json`。前两表展示 Demand、Graph-flow 和 Portal-Fiber 查询块的噪声传播；后两表展示统计、道路、路线选择、任务、回退和运行时间。作图：

```powershell
python commands/reproduce.py plot-rebuttal-q5-q6 -- `
  --stage q6 `
  --input-dir C:\runs\rebuttal_q6_low_epsilon `
  --out-dir C:\runs\rebuttal_q6_figures
```

输出 `q6_complete_low_epsilon_audit.png`、`.pdf` 和图像哈希清单。该入口重算已有 35 份发布的诊断，不替代 35 次合成、跨城市五种子生成或攻击实验。若需从原始轨迹重做，先按 `README_CN.md` 的生成、统一评估与道路选择评估命令准备上述三类输入。

## Porto 与旧金山的五种子结果

两城各有五份 ε=7/5 的独立发布与统一指标。聚合入口要求十份协议和十份指标全部存在，不会静默跳过缺失种子；`--sf-seed-suffix` 对应已保存的 `_arterial_v2` 生成批次。使用自己的城市运行时，分别把四个根目录指向相应生成和评分结果。

从零生成时，给每一次 `generate-main` 增加一个位于发布目录**之外**的本地性能文件。例如 Porto 第一种子：

```powershell
python commands/reproduce.py generate-main -- `
  --data C:\runs\porto\real.pkl --dataset-config porto `
  --epsilon-total 7/5 --noise-seed 20260719 --decoder-seed 30260719 `
  --public-input-capacity 20000 --public-slot-count 20000 `
  --out-dir C:\runs\mtr\porto\eps_7_5\seed_20260719 `
  --local-performance-log C:\runs\mtr\porto\timing\seed_20260719.json
```

其余种子将两个种子参数同步递增；旧金山相应使用 `--dataset-config sf`、同一公开容量，并将生成目录设为 `seed_20260719_arterial_v2`、性能日志设为 `seed_20260719_arterial_v2.json`。性能 JSON 包括准备、私有测量、公共路由三段耗时及进程峰值 RSS，并以发布协议 SHA-256 绑定到本次运行。它是本地非 DP 性能诊断，不属于合成数据发布物，也不应与 DP transcript 一起公开。耗时因机器而异。生成入口不向公开 `protocol.json` 写入输入相关运行时间。

```powershell
python commands/reproduce.py aggregate-rebuttal-q6-multicity -- `
  --porto-generation-root C:\runs\mtr\porto\eps_7_5 `
  --porto-metrics-root C:\runs\rebuttal_q6_multicity\porto `
  --sf-generation-root C:\runs\mtr\sf_trip20k\eps_7_5 `
  --sf-seed-suffix _arterial_v2 `
  --sf-metrics-root C:\runs\rebuttal_q6_multicity\sf `
  --out-dir C:\runs\rebuttal_q6_multicity_summary
```

上面的命令兼容旧保存协议中的 `elapsed_sec`，重新聚合旧十份结果不会改变历史详细/汇总 CSV。对于新的从零运行，在相同命令中另外加入：

```powershell
  --porto-performance-root C:\runs\mtr\porto\timing `
  --sf-performance-root C:\runs\mtr\sf_trip20k\timing `
```

两项必须一起提供。聚合器核验十份性能日志各自绑定的发布协议哈希；输出详细/汇总 CSV 会额外包含 `peak_rss_bytes`、`preparation_sec`、`private_measurement_sec`、`public_routing_sec`，并将性能日志纳入结果 manifest 的输入哈希。缺少性能日志时，新协议不会退回猜测耗时，而是报错。

每个生成根目录下应有 `seed_<seed><suffix>/protocol.json`，每个指标根目录下有 `seed_<seed>/metrics.json`，五个 seed 固定为 `20260719`—`20260723`。得到 `q6_multicity_five_seed_detailed.csv`、`q6_multicity_five_seed_summary.csv` 与绑定二十份输入哈希的 `manifest.json`。这一步从已有完整发布重算均值和 95% 区间；从零生成两城十份轨迹仍需先按跨城市生成命令完成。

## 经验攻击的威胁模型表

四组已保存的原始攻击结果与冻结候选划分可以重新组装为回复中的统一威胁模型表：

```powershell
python commands/reproduce.py assemble-rebuttal-q6-threat-model -- `
  --attack-root C:\runs\active_hierarchy\privacy_attacks `
  --split-manifest C:\runs\attack_split\manifest.json `
  --out-dir C:\runs\rebuttal_q6_threat_model
```

`--attack-root` 下需有 `domias_v3`、`gda_v3`、`population_linkage_v3`、`ordered_v3` 四个子目录，各含 `results.json`；`--split-manifest` 记录成员、非成员和参考轨迹的冻结划分。输出为 `privacy_attack_results.csv`、`threat_model.md` 和输入/输出 SHA 清单。此命令只汇总已实际完成的攻击，不重新训练攻击器；要从头复现攻击数值，仍需先运行各攻击入口并核对候选划分与受攻击的合成发布。
