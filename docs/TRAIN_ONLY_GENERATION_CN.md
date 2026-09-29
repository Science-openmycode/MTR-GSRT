# 从训练侧生成到下游评分

`commands/generate_train_only.py` 调用已有生成器，不改变算法或预算；同时写一份独立的本地生成记录。相对路径以本文件夹为根，绝对路径可位于任意数据盘。

## 1. 生成 MTR-GSRT

先按 README 第 6 节生成 `C:\runs\strict_split\train.pkl` 和 `test.pkl`，再运行：

```powershell
python commands/reproduce.py generate-train-only -- --train-real C:\runs\strict_split\train.pkl --test-real C:\runs\strict_split\test.pkl --record C:\runs\tstr_new\lineage\gsrt.json --out-dir C:\runs\tstr_new\gsrt -- --epsilon-total 7/5 --noise-seed 20260719 --decoder-seed 30260719 --public-slot-count 13698 --bbox 39.75 40.15 116.10 116.65 --osm-cache generation/mtr_gsrt/data/osm/osm_cache_beijing.pkl --component-mode full
```

得到：

```text
C:\runs\tstr_new\
├─ gsrt\
│  ├─ trajectories.pkl
│  ├─ road_witnesses.pkl
│  ├─ dp_transcript.npz
│  ├─ protocol.json
│  └─ manifest.json
└─ lineage\
   ├─ gsrt.json
   └─ gsrt.log
```

## 2. 生成 MTR-DFR（DFR 文件夹）

DFR 使用训练轨迹的道路匹配输入。先按 README 的训练侧匹配命令生成 `matched_paths/Real.pkl.gz`；其 `Real.manifest.json` 必须绑定 `train.pkl` 与匹配输出的 SHA-256。

```powershell
python commands/reproduce.py generate-train-only -- --train-real C:\runs\strict_split\train.pkl --test-real C:\runs\strict_split\test.pkl --train-routes C:\runs\train_matching\matched_paths\Real.pkl.gz --record C:\runs\tstr_new\lineage\dfr.json --out-dir C:\runs\tstr_new\dfr -- --edge-cache public_assets/ordered_portal_route_cache.pkl.gz --network public_assets/beijing_network/network.shp --seed 20260912 --length-aware --public-slot-count 13698
```

得到 DFR 生成器原有的道路路径输出，以及目录外的 `dfr.json`/`dfr.log`。坐标由下一步的转换命令生成。若匹配输入来自全量数据或测试数据，训练哈希不符，生成器启动前报错。

DFR 生成的是道路路径；接着生成下游任务的坐标视图：

```powershell
python commands/reproduce.py routes-to-coordinates -- --routes C:\runs\tstr_new\dfr\seed_20260912_length\road_routes.pkl.gz --network public_assets/beijing_network/network.shp --edge-cache public_assets/ordered_portal_route_cache.pkl.gz --out C:\runs\tstr_new\dfr_coordinates.pkl
```

得到 `dfr_coordinates.pkl` 和 `dfr_coordinates.pkl.manifest.json`。第 4 节评分命令的 `--synthetic` 改为这个坐标文件，名称及来源记录改为 `MTR-DFR` 和 `MTR-DFR=C:\runs\tstr_new\lineage\dfr.json`；评分器沿坐标视图的来源链核验实际生成的道路文件。

## 3. 生成统计基线（含基线源码的 03/04 文件夹）

```powershell
python commands/reproduce.py generate-train-only -- --generator baseline --train-real C:\runs\strict_split\train.pkl --test-real C:\runs\strict_split\test.pkl --record C:\runs\tstr_new\lineage\sprt.json --out-dir C:\runs\tstr_new\sprt -- --method sprt-native --epsilon 1.4 --seed 20260713 --public-slot-count 13698 --bbox 39.75 40.15 116.10 116.65 --osm-cache generation/baselines/data/osm/osm_cache_beijing.pkl
```

得到基线合成文件、生成协议、运行日志及目录外的 `sprt.json`/`sprt.log`。其它方法更换 `--method` 并使用新的记录/输出目录。只有保存数据的 01/02 文件夹不执行基线生成。

## 4. 执行评分

先验证划分文件与完整源数据的逐记录对应关系及索引不重叠：

```powershell
python evaluation/audit_tstr_split.py --full-real C:\runs\real_full_frozen.pkl --split-manifest C:\runs\strict_split\split_manifest.json --train-real C:\runs\strict_split\train.pkl --test-real C:\runs\strict_split\test.pkl --out C:\runs\tstr_new\lineage\split.json
```

生成 `split.json`，记录两份划分文件哈希及源记录索引核验。评分命令还需追加 `--split-audit C:\runs\tstr_new\lineage\split.json`。迁移数据时 `--full-real` 指向自己的完整数据；划分必须覆盖其全部记录且索引各使用一次。

MTR-GSRT 示例：

```powershell
python commands/reproduce.py run-tstr -- --train-real C:\runs\strict_split\train.pkl --test-real C:\runs\strict_split\test.pkl --synthetic C:\runs\tstr_new\gsrt\trajectories.pkl --names MTR-GSRT --osm-cache evaluation/data/osm/osm_cache_beijing.pkl --bbox 39.75 40.15 116.10 116.65 --synthesis-lineage MTR-GSRT=C:\runs\tstr_new\lineage\gsrt.json --split-audit C:\runs\tstr_new\lineage\split.json --require-synthesis-lineage --out-dir C:\runs\tstr_new\scores\native
```

生成 `raw_generic/`、`raw_road/`、`results.csv`、`manifest.json`。多方法实验重复 `--synthetic` 和 `--synthesis-lineage`，并按相同顺序列出 `--names`。FMM/STMatch 视图按 README 第 6 节生成，仍传入原测量方法的同一份生成记录，例如 `SPRT=.../sprt.json`，不是 `SPRT x FMM=...`。

三路由评分完毕，按 README 聚合命令追加 `--require-synthesis-lineage`。输出 6 个主图任务指标及第 7 个完整任务指标，并记录全部生成来源。没有记录的历史结果可以作输入已绑定的评分研究，但不能变成已核验生成来源的新结果。

## 参数

| 参数 | 含义 |
|---|---|
| `--train-real` | 已准备好的真实训练划分；是生成器唯一的私有坐标输入 |
| `--test-real` | 真实测试划分；只计算本地划分哈希，不传给生成器 |
| `--generator main/baseline` | 本文件夹主算法或包含源码的统计基线；默认 main |
| `--train-routes` | 仅 DFR：由训练输入得到并绑定哈希的匹配路径 |
| `--record` | 新的本地 JSON 记录路径；必须在合成数据目录外 |
| `--out-dir` | 新的合成数据目录；已有目录不覆盖 |
| `--` 后面的参数 | 原生成器参数，含预算、种子、槽位、公共 bbox/地图等；禁止覆盖 `--data`、`--real-matched`、`--out-dir` |
| `--synthesis-lineage NAME=JSON` | 评分时提供每个测量方法的实际生成记录 |
| `--require-synthesis-lineage` | 评分/聚合时要求所有测量方法具有通过核验的生成记录 |
| `--split-audit` | 评分时提供逐记录核验过的不重叠划分记录；严格模式必需 |

`13698` 是冻结北京训练侧的公开实验输出规模。迁移数据时先划分自己的真实文件，再显式设置公开地图、bbox、预算和输出规模。生成记录包含私有输入哈希、模型审计包含私有特征/预测哈希，均留在本地研究目录；它们不是额外的正式 DP 发布对象。记录证明本次受审计入口调用的输入来源，不替代算法 DP 证明，也不根据文件名认证旧数据的来源。
