# Q5：区域粒度与 Portal-Fiber 测量诊断

这个实验在相同北京完整轨迹输入上改变公共道路区域划分，重新构造 q5 工作负载、加离散 Laplace 噪声并投影，比较 Portal 分布的恢复质量。每个配置都重新计算全部真实轨迹的 q5 贡献；真实轨迹只用于本地诊断，`exact` 向量不落盘，也不属于对外的 DP 合成发布。

准备已经按论文协议预处理的真实轨迹文件和公共 OSM 缓存。本文北京冻结输入应为 17,123 条，真实文件 SHA-256 `6a160ca557fbd7bab97af489b56c931e73532c498c390e31965c0d61e53361fd`；GitHub 提供公共 OSM 缓存，SHA-256 `7159e12ab8712338b4ff5a2859c68b158e1374504f999dc44080c2fe2d53a5c5`。从仓库根目录运行：

```powershell
python commands/reproduce.py run-rebuttal-q5-partition -- `
  --real C:\data\real_full_frozen.pkl `
  --input-sha256 6a160ca557fbd7bab97af489b56c931e73532c498c390e31965c0d61e53361fd `
  --osm-cache generation/mtr_gsrt/data/osm/osm_cache_beijing.pkl `
  --osm-sha256 7159e12ab8712338b4ff5a2859c68b158e1374504f999dc44080c2fe2d53a5c5 `
  --bbox 39.75 40.15 116.10 116.65 `
  --public-capacity 17123 `
  --out-dir C:\runs\rebuttal_q5_partition
```

`--real` 是本地完整预处理轨迹，`--osm-cache` 是公开地图缓存；`--bbox` 是固定城市范围；`--public-capacity` 要与完整输入有效轨迹数相同；两个可选 SHA 参数拒绝错批次；`--out-dir` 必须是新目录。省略 `--spec` 时运行 22 个嵌套分辨率与 8 个非嵌套对照，使用固定的五个噪声种子。可用 `--spec nested:48` 限定单个配置做功能核对；`--seed` 可重复五次指定另一组研究随机流，区间仍按五种子 Student-t 计算。完整 30 配置的输出包括：

- `q5_partition_sensitivity_detailed.csv`：30×5 个种子级结果；
- `q5_partition_sensitivity_summary.csv`：每配置均值、标准差、95% 区间、Portal 上下文覆盖及信噪比；
- `q5_partition_sensitivity_manifest.json`：输入/OSM/代码 SHA、容量、种子与配置清单。

完整结果生成后作图：

```powershell
python commands/reproduce.py plot-rebuttal-q5-q6 -- `
  --stage q5-partition `
  --input-dir C:\runs\rebuttal_q5_partition `
  --out-dir C:\runs\rebuttal_q5_partition_figures
```

得到 `q5_partition_granularity_sensitivity.png`、`.pdf` 和记录输入/输出哈希的 `figure_manifest.json`。

这一步是测量层的查询保真诊断，不能直接解释为 30 次完整合成机制的下游性能，也不能把确定性随机种子形成的多次结果称作一份正式 DP 发布。若改变城市、道路图或输入文件，必须同步提供对应 bbox、公开容量和分区协议；北京冻结数值不可直接外推。

## 同一 DP 转录本上的 Portal-Fiber 组件对照

另一个 Q5 实验固定一份 ε=7/5 的 Portal-Fiber 发布及其 DP transcript、请求随机种子、道路图和输出槽数，只改变公共解码阶段。四臂分别为完整方案、将跨界流换成同质量均匀分布、移除局部信息投影、将跨界抽样改为确定性最大概率选择。真实道路参考缓存由本地预处理得到，不上传原始轨迹。

```powershell
python commands/reproduce.py run-rebuttal-q5-ablation -- `
  --source C:\runs\mtr_geolife_eps_7_5_seed_20260719 `
  --osm-cache generation/mtr_gsrt/data/osm/osm_cache_beijing.pkl `
  --network public_assets/beijing_network/network.shp `
  --real-match-cache C:\runs\real_reference\Real.pkl.gz `
  --dataset-config geolife `
  --out-dir C:\runs\rebuttal_q5_ablation
```

`--source` 必须含 `protocol.json`、`dp_transcript.npz`、`trajectories.pkl` 和 `road_witnesses.pkl`，并与协议内三个输出 SHA 一致；公共 OSM 也必须同哈希。`--network` 是评估侧道路图，`--real-match-cache` 是同一真实参考匹配缓存；`--dataset-config` 指定已登记的城市 bbox。输出目录必须不存在。完成后四臂各有合成坐标、道路 witness 和协议，`evaluation/native_road_choice_metrics.json` 保存共同评分，`manifest.json` 保存输入哈希和消融定义。此对照仅重解码同一份带噪测量，不重新读取原始轨迹或重新花隐私预算。

```powershell
python commands/reproduce.py plot-rebuttal-q5-q6 -- `
  --stage q5-ablation `
  --input-dir C:\runs\rebuttal_q5_partition `
  --ablation-dir C:\runs\rebuttal_q5_ablation `
  --out-dir C:\runs\rebuttal_q5_ablation_figures
```

这一步得到 `q5_portal_component_ablation.png`、`.pdf` 和图像输入/输出哈希清单。`--input-dir` 在此模式只需是一个已存在的研究结果目录，实际消融数值从 `--ablation-dir/manifest.json` 读取。

解释四臂时须注意：在当前投影公式里，将 q5 的 portal 质量设为均匀会使信息奖励为零，因此“去跨界流”和“去局部投影”都退回同一个公共先验。这两臂的已保存合成轨迹与 witness **完全同哈希**，应视为同一有效对照的两种实现表述，而不是两条独立的性能证据。第三臂把该先验/投影后的抽样改成最大概率选择，用来单独检查随机道路选择的贡献。
