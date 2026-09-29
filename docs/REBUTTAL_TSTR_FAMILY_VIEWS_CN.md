# 严格 TSTR：四种公共路径族路由的训练侧视图

本页补齐两份回复所需的七路由 TSTR 比较。四种路径族分别从**同一个基线的 train-only 合成坐标**经公共 STMatch 得到的道路载体出发，不读取真实测试轨迹，也不借用其它合成方法的路线。Native、FMM、STMatch 三种视图的生成命令见根目录 `README_CN.md` 第 6 节；下面先补充四种路径族，再运行 28 候选验证与最终测试。

以下命令在仓库根目录运行。首先按照 `README_CN.md` 第 6 节生成训练划分、四种基线的 train-only 发布和 `C:\runs\tstr_mr\stmatch\matched_paths\<方法>.pkl.gz`。四个保存的基线训练发布也在 `datasets/synthetic/train_only_baselines/`。`--method-index` 仅确定固定研究随机流，顺序为 SPRT 0、PrivTrace 1、DPTraj-PM 2、DPStd 3。

```powershell
$items = @(
  @('SPRT', 0, 'sprt_train.pkl'),
  @('PrivTrace', 1, 'privtrace_train.pkl'),
  @('DPTraj-PM', 2, 'dptrajpm_train.pkl'),
  @('DPStd', 3, 'dpstd_train.pkl')
)
foreach ($item in $items) {
  $method = $item[0]; $index = $item[1]; $source = $item[2]
  python commands/reproduce.py materialize-family-routes -- `
    --stmatch-route "C:\runs\tstr_mr\stmatch\matched_paths\$method.pkl.gz" `
    --method $method --method-index $index `
    --edge-cache public_assets/ordered_portal_route_cache.pkl.gz `
    --head-threshold 36 --seed-schedule tstr-v1 `
    --out-dir "C:\runs\tstr_mr\family_routes\$method"
  python commands/reproduce.py materialize-family-tstr-views -- `
    --source "datasets/synthetic/train_only_baselines/$source" `
    --stmatch-route "C:\runs\tstr_mr\stmatch\matched_paths\$method.pkl.gz" `
    --family-dir "C:\runs\tstr_mr\family_routes\$method" `
    --network public_assets/beijing_network/network.shp `
    --edge-cache public_assets/ordered_portal_route_cache.pkl.gz `
    --method $method --out-dir "C:\runs\tstr_mr\family_views\$method"
}
```

`--head-threshold 36` 是公共 Region96 路线族的最小载体计数；`--seed-schedule tstr-v1` 固定本次严格 TSTR 的四个独立随机流，不使用完整语料 M×R 图的随机流。第一条命令为每个方法写四个 `*.pkl.gz` 道路路线及对应 manifest。第二条命令写 `family_additive_generic.pkl`、`family_additive_road.pkl` 等四组 generic/road 坐标视图及各自的来源哈希。generic 视图按原合成记录点数等距采样公共路线，road 视图保留路线顶点；路径无效时两种视图均退回该槽的**合成坐标**，不退回真实轨迹。每个输出始终有 13,698 个槽。

这两条命令只依赖传入的基线训练发布、该基线的 STMatch 结果及公共道路资源，可以替换为别的城市数据、公共图和输出目录；对应边编号必须与传入的 edge cache 一致。来源验证会拒绝方法名、路线文件哈希或公共边缓存哈希不匹配的输入。

在本地保存的 2026-09-28 隔离批次上，用上述公开代码为四基线重新生成的 4×4×2=32 个 generic/road 文件与历史候选视图逐文件 SHA-256 相同。这是视图生成的回归证据；后续还需公开执行 28 个候选的验证集任务、冻结路由选择，并在真实测试集复评，才能认证回复中的表 1。

## 七路由验证与最终测试入口

`run-rebuttal-tstr` 读取四个方法各七个训练侧视图。`--views-root` 指向第 6 节 `materialize-tstr-mr` 得到的 `generic/fmm`、`generic/stmatch`、`road/fmm`、`road/stmatch` 目录；`--family-root` 指向上例 `family_views/<方法>/`。`--split-audit` 由 `python evaluation/audit_tstr_split.py --full-real <完整真实输入> --split-manifest <split_manifest.json> --train-real <train.pkl> --test-real <test.pkl> --out <split_audit.json>` 创建，验证记录索引互斥与保存内容一致。下面的输出目录必须是新的，避免覆盖历史批次。

```powershell
$common = @(
  '--train-real', 'C:\runs\strict_split\train.pkl',
  '--test-real', 'C:\runs\strict_split\test.pkl',
  '--split-audit', 'C:\runs\strict_split\split_audit.json',
  '--source', 'SPRT=datasets/synthetic/train_only_baselines/sprt_train.pkl',
  '--source', 'PrivTrace=datasets/synthetic/train_only_baselines/privtrace_train.pkl',
  '--source', 'DPTraj-PM=datasets/synthetic/train_only_baselines/dptrajpm_train.pkl',
  '--source', 'DPStd=datasets/synthetic/train_only_baselines/dpstd_train.pkl',
  '--mtr-source', 'C:\runs\tstr_mr\mtr_gsrt\trajectories.pkl',
  '--views-root', 'C:\runs\tstr_mr\views',
  '--family-root', 'C:\runs\tstr_mr\family_views',
  '--osm-cache', 'generation/mtr_gsrt/data/osm/osm_cache_beijing.pkl',
  '--bbox', '39.75', '40.15', '116.10', '116.65',
  '--seed', '20260713', '--validation-fraction', '0.2',
  '--out-dir', 'C:\runs\tstr_mr\seven_router_scores',
  '--diagnostic-private-validation'
)
python commands/reproduce.py run-rebuttal-tstr -- --stage validate @common
python commands/reproduce.py run-rebuttal-tstr -- --stage final @common
```

第一条命令把真实训练部分按固定随机种子再分为选择拟合集与验证集，并对 28 个实际候选运行通用移动任务和道路任务，写出 `seven_router_scores/validation/results.csv`、`validation_all_router_scores.csv` 和 `selected_routers.json`。选择依据是四项相对 Real-train 保留率的等权平均；最终真实测试集的任务标签不参与选择。第二条命令核对选择文件、验证结果、五种训练发布、原训练/测试哈希，并在未参与合成的真实测试集上**重新评价**四个基线原生行、四个被选路由行及 MTR-GSRT 原生行，写出 `final_test/results.csv`、九行 `final_native_best_mtr.csv` 和 `manifest.json`。若某方法选中的就是 Native，只计算一次模型，两行引用同一原生结果。

`--diagnostic-private-validation` 是必须显式填写的解释性标记：验证集来自真实训练数据，这种基于私有标签的路由选择不是无额外预算的 DP 后处理；该实验仅作比较诊断。输入划分独立于最终真实测试，和完整 DP 发布保证是两个不同的命题。保存的历史 baseline 来源记录若没有可执行生成链，结果 manifest 会将 `synthesis_lineage_verified` 保持为 false，不借文件名自动认证 train-only 来源。

## 本地隔离复算与历史表的关系

Python 3.11.15 锁定环境下，上述四种路径族的 32 个坐标视图与历史文件逐一同哈希。28 个候选的验证评分重选出 SPRT/Family additive、PrivTrace/Self-carrier reweight、DPTraj-PM/Family residual、DPStd/Self-carrier reweight，与旧回复保存的选择一致。最终测试在**同一轮**重新评分九行：四个最佳公共路由行的 16 个数值与历史表一致；五个原生行的 20 个数值中 15 个不同。最大差异是 DPTraj-PM 原生 Next-cell Hit@1 从旧表 0.095001 降到同批重算 0.020656。旧回复的原生行来自另一份保存的 `native_results.csv`，而最佳公共路由行来自后续新计算，因此不可把旧表说成九行同批复算。

隔离复算产物保存在工作区 `check/rebuttal_public_repro_20260930/tstr_seven_router_batch/`：`validation_all_router_scores.csv` 是 28 候选，`selected_routers.json` 记录选择与哈希，`final_native_best_mtr.csv` 是九行同批结果，`manifest.json` 绑定 train/test、OSM、五种合成发布及两阶段结果。此目录使用本地真实轨迹划分，不发布到 GitHub。公开入口可在提供相同合法输入时重新生成这些结果；旧表的历史原生行仍需原始评估环境证据才能精确认证。
