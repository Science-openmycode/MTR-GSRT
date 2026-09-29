# 严格 TSTR：四种公共路径族路由的训练侧视图

本阶段补齐两份回复所需的七路由候选中的四种路径族。它们分别从**同一个基线的 train-only 合成坐标**经公共 STMatch 得到的道路载体出发，不读取真实测试轨迹，也不借用其它合成方法的路线。Native、FMM、STMatch 三种视图的生成命令见根目录 `README_CN.md` 第 6 节；本页只补充四种路径族。七路由验证选择和最终测试表仍是后续阶段，不能把这里的 16 组生成视图当作已完成的 TSTR 表。

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
