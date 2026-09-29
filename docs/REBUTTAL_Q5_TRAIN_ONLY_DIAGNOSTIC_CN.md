# Train-only 的完整 q5 对照诊断

此实验从同一份北京真实训练划分生成两份各 13,698 条的研究发布，再用从未进入合成器的 3,425 条真实测试轨迹评价道路选择。`full` 保留全部 q5；`no-portal-fiber` 将整个 q5 对路由的输入中和。它回答“完整 q5 在严格未见测试上有何作用”，**不是**旧补充材料中只删除 crossing flow、局部投影与 crossing-edge sampling 的“整体 Portal 单元”消融，两组定义不能合并为同一张表。

以下命令在仓库根目录执行。真实 `train.pkl`、`test.pkl` 及完整真实道路缓存须由用户按根目录 README 的严格划分与道路匹配章节生成。`--public-input-capacity` 与 `--public-slot-count` 均为公开指定的 13,698；北京公共 bbox 和 OSM cache 在两臂相同。噪声与路由种子相同，以便逐臂比较。

```powershell
$root = "C:\runs\q5_train_only"
foreach ($mode in @("full", "no-portal-fiber")) {
  python commands/reproduce.py generate-main -- `
    --data "C:\runs\strict_split\train.pkl" --dataset-config geolife `
    --epsilon-total 7/5 --noise-seed 20260719 --decoder-seed 30260719 `
    --public-input-capacity 13698 --public-slot-count 13698 `
    --bbox 39.75 40.15 116.10 116.65 `
    --osm-cache generation/mtr_gsrt/data/osm/osm_cache_beijing.pkl `
    --component-mode $mode --out-dir "$root\$mode"
}
python evaluation/evaluation/evaluate_road_choice.py `
  --dataset-config geolife `
  --match-dir "C:\runs\real_full_match\matched_paths" `
  --network public_assets/beijing_network/network.shp `
  --witness "Full=$root\full\road_witnesses.pkl" `
  --witness "NoQ5=$root\no-portal-fiber\road_witnesses.pkl" `
  --expected-count 17123 --synthetic-expected-count 13698 `
  --real-test-start-index 13698 `
  --out-dir "$root\road_choice_strict"
```

生成命令各输出 `trajectories.pkl`、`road_witnesses.pkl`、`dp_transcript.npz` 和 `protocol.json`；评价命令输出 `metrics.json`、`metrics.csv`、`manifest.json`。评价器先按完整真实缓存的**记录索引**在 13,698 处切分，之后才过滤不产生有效道路选择事件的轨迹。省略 `--real-test-start-index` 时保留历史回顾式评分口径，不代表严格未见测试。

在保存的本地实验中，两臂 DP transcript 的 SHA-256 相同。完整 q5 的 RC-CPC/RC-NDCG/NextRoadAcc/NextRoadNLL 分别为 `0.253970/0.672019/0.227050/1.961308`；整个 q5 中和后的相应值为 `0.265156/0.644898/0.183906/1.927479`。完整 q5 提高排序和主导道路准确率，却没有同时提高分布重合与对数损失，因此这些数值不支持“四项指标全面胜出”的表述。用户自己的真实训练和公共图配置变化后应以新运行的 `metrics.json` 为准。

## 单独的整体 Portal 单元消融

`--portal-unit-mode ablated` 是与上面的 `--component-mode no-portal-fiber` 不同的研究实验。它在私有测量阶段删除 `portal_fiber_flow` 查询，将固定的 q5 逐记录质量重新分配给其余四块；公共路由采用仅由公共图决定的 crossing-edge prior。`full` 与 `ablated` 必须从同一 train-only 输入分别执行私有测量，不能共用一份 transcript。整体单元同时涉及查询和路由，所以不能只把某个路由开关改为零。该模式使用独立的 `portal_unit_research_decoder.py`，不改变经过 SOURCE_MANIFEST 审计的正式解码器。研究种子和多次研究发布不得冒充单次正式 DP 发布。

四块逐记录量化质量分别为 `153846/153846/230769/461539`，总和恰为 `10^6`。每条有效轨迹贡献一个非负、总质量为 `10^6` 的向量；无效轨迹贡献零向量。因此 add/remove 相邻输入的 q5 查询 L1 敏感度不超过 `10^6`，沿用 q5 的 `1/5` 预算与精确离散 Laplace 机制。其它测量块和预算不变，研究解码只读取带噪 transcript 与公共图，故单次随机机制的组合证明仍为纯 `7/5`-DP。代码中指定公开固定种子是可复算研究执行，不能据此把实际多次公开研究运行称为正式 `7/5`-DP 发布。

```powershell
$root = "C:\runs\q5_portal_unit"
foreach ($mode in @("full", "ablated")) {
  python commands/reproduce.py generate-main -- `
    --data "C:\runs\strict_split\train.pkl" --dataset-config geolife `
    --epsilon-total 7/5 --noise-seed 20260719 --decoder-seed 30260719 `
    --request-seed 30744448 --public-input-capacity 13698 --public-slot-count 13698 `
    --bbox 39.75 40.15 116.10 116.65 `
    --osm-cache generation/mtr_gsrt/data/osm/osm_cache_beijing.pkl `
    --portal-unit-mode $mode --out-dir "$root\$mode"
}
python evaluation/evaluation/evaluate_road_choice.py `
  --dataset-config geolife --match-dir "C:\runs\real_full_match\matched_paths" `
  --network public_assets/beijing_network/network.shp `
  --witness "PortalFull=$root\full\road_witnesses.pkl" `
  --witness "PortalRemoved=$root\ablated\road_witnesses.pkl" `
  --expected-count 17123 --synthetic-expected-count 13698 `
  --real-test-start-index 13698 --out-dir "$root\road_choice_strict"
```

两臂各输出 13,698 条合成轨迹和 witness，并在 `road_choice_strict/metrics.json` 得到同一未见真实测试集上的评分。此命令不覆盖前面的完整 q5 / 整个 q5 中和实验，也不表示旧补充材料中的全语料消融已被严格 train-only 结果逐字节复现。

本地锁定环境的整体单元结果为：`full` 的 RC-CPC/RC-NDCG/NextRoadAcc/NextRoadNLL 是 `0.253970/0.672019/0.227050/1.961308`；`ablated` 为 `0.296508/0.620615/0.153624/1.939319`。完整单元改善主导选择排序与准确率，但 CPC 和 NLL 不占优。这是组件取舍结果，不能只摘取两项优势写成全面改进。

这一诊断使用固定研究种子，不是额外的正式 DP 发布。多个研究发布若同时对外释放，必须对隐私预算顺序组合；公开仓库不附带私有真实训练轨迹或真实道路匹配缓存。
