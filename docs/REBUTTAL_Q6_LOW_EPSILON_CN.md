# Q6：隐私预算与完整发布的分层诊断

北京实验使用七个总预算 `1/5, 2/5, 7/10, 1, 7/5, 2, 3`，每个预算五个研究噪声种子 `20260719`—`20260723`。公开生成入口 `generate-main` 按同一公共图、槽位和预算比例生成 35 份独立合成语料；随后对每份运行统一指标评估和道路选择评估。它们是研究扫描，不能把 35 次发布合称一次 ε-DP 发布。

若已有 35 份完整发布及其指标，可以从这些文件和本地原始轨迹重算回复中查询层与发布层的统计表。原始轨迹只在本地内存中用于构造未加噪查询对照；脚本不保存 exact 向量。相对路径按本仓库根目录解析。

```powershell
python commands/reproduce.py run-rebuttal-q6-low-epsilon -- `
  --real C:\data\real_full_frozen.pkl `
  --input-sha256 6a160ca557fbd7bab97af489b56c931e73532c498c390e31965c0d61e53361fd `
  --osm-cache generation/mtr_gsrt/data/osm/osm_cache_beijing.pkl `
  --osm-sha256 7159e12ab8712338b4ff5a2859c68b158e1374504f999dc44080c2fe2d53a5c5 `
  --bbox 39.75 40.15 116.10 116.65 `
  --public-capacity 17123 `
  --generation-root C:\runs\synthetic_datasets\mtr\geolife `
  --metrics-root C:\runs\privacy_budget_experiment\geolife `
  --route-choice-csv C:\runs\road_choice_metrics\aggregate\metrics_long.csv `
  --out-dir C:\runs\rebuttal_q6_low_epsilon
```

`--generation-root` 的目录结构为 `<eps_分子_分母>/seed_<噪声种子>/dp_transcript.npz` 和 `protocol.json`；`--metrics-root` 具有相同两层目录，其叶目录内有 `metrics/metrics.json`。`--route-choice-csv` 包含每个预算/种子的 `road_choice_ndcg`、`next_road_accuracy`、`next_road_nll`。脚本要求全部 35 组文件齐全，并在输出清单中记录每份文件的 SHA-256。`--public-capacity` 是预声明的完整输入条数；两个可选哈希参数用于本地冻结批次预检。

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

```powershell
python commands/reproduce.py aggregate-rebuttal-q6-multicity -- `
  --porto-generation-root C:\runs\mtr\porto\eps_7_5 `
  --porto-metrics-root C:\runs\rebuttal_q6_multicity\porto `
  --sf-generation-root C:\runs\mtr\sf_trip20k\eps_7_5 `
  --sf-seed-suffix _arterial_v2 `
  --sf-metrics-root C:\runs\rebuttal_q6_multicity\sf `
  --out-dir C:\runs\rebuttal_q6_multicity_summary
```

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
