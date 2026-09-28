# 跨城市输入准备与运行

相对路径均以当前独立文件夹为根目录；也可以使用绝对路径。从其它工作目录调用时，给脚本一个绝对路径即可，参数的相对路径含义不变。道路匹配依赖按 README 安装，并先运行 verify-matcher。

## 1. 准备真实轨迹

Porto 使用公开 train.csv；POLYLINE 是经度、纬度顺序，转换后为纬度、经度。转换保留 CSV 顺序，在公开 bbox 内保留至少两个坐标，每条最多 100 点，选前 20,000 条。

```powershell
python commands/reproduce.py prepare-porto -- --source-csv "C:\data\porto\train.csv" --count 20000 --out "C:\runs\porto\real.pkl"
```

生成 real.pkl 和 real.manifest.json。--count 是输入选择数量；--bbox LAT_MIN LAT_MAX LON_MIN LON_MAX 是公开研究区域；--expected-sha256 可指定预期输出哈希，不一致则在写文件前退出。默认 bbox 为 41.10 41.20 -8.70 -8.55。

San Francisco 使用解压后的 Cabspotting 原始 taxi 文本目录，而不是下载页附件中的分析脚本 ZIP。每行格式是 lat lon occupied timestamp。按时间排序后切分载客行程，输出单位是一段载客行程。

原始数据入口为 [CRAWDAD epfl/mobility（IEEE DataPort）](https://ieee-dataport.org/open-access/crawdad-epflmobility)，下载文件名为 cabspottingdata.tar.gz；需要按数据站点要求登录取得下载链接。解压后目录应包含 new_*.txt 等出租车记录文件。以下命令增加 --expected-sha256 544cc51b459554d52bebe0f881c58f4aa6896ae2a319f5b23c048d7ec415511e 可要求输出与本文冻结输入一致；不一致时不会写文件。新数据集不使用此冻结哈希。

```powershell
python commands/reproduce.py prepare-sf -- --source-dir "C:\data\cabspotting" --count 20000 --bbox 37.60 37.85 -122.55 -122.30 --max-gap-seconds 300 --max-speed-kmh 160 --min-points 5 --min-duration-seconds 120 --min-length-km 0.5 --out "C:\runs\sf\real.pkl"
```

生成 real.pkl 和 real.manifest.json；后者记录源文件树哈希、选取规则、输出哈希和行程统计。--source-dir 是实际包含 taxi .txt 的目录；--max-gap-seconds、--max-speed-kmh 决定切分条件；三个 --min-* 决定保留条件。使用与原冻结语料相同的原始文件和参数才能得到同一输入。

## 2. 构建公共道路图

随目录提供 Porto 和 SF 的公开 OSM cache。SF cache 预先执行与现有算法相同的 bbox 和道路类别过滤，未使用轨迹。每个 cache 旁的 manifest 记录原 OSM 来源哈希和选取条件。

```powershell
python evaluation/evaluation/prepare_road_evaluation_network.py --dataset-config porto --osm-cache evaluation/data/osm/osm_cache_porto.pkl --skip-ubodt --out-dir "C:\runs\porto\network"
python evaluation/evaluation/prepare_road_evaluation_network.py --dataset-config sf --osm-cache evaluation/data/osm/osm_cache_sf_bay.pkl --skip-ubodt --out-dir "C:\runs\sf\network"
```

输出 network.shp、dbf、shx、prj 和 manifest.json。STMatch 使用此图无需 UBODT。FMM 需要把 --skip-ubodt 替换为 --ubodt-format csv --ubodt-delta-m 1000 --ubodt-bin public_assets/matcher/ubodt_gen.exe --fmm-runtime-dir $runtime，输出另含 ubodt.txt。相对二进制参数建议使用绝对路径；$runtime 取 README 的兼容依赖目录。

公共路线评估支撑也可独立构建：

```powershell
python commands/prepare_public_route_support.py --dataset-config porto --network "C:\runs\porto\network\network.shp" --out "C:\runs\porto\route_support.pkl.gz"
```

输出只有公共图字段，没有真实路线 rows：edge_nodes、outdegree、labels24、labels96、labels384。--dataset-config 选择 bbox 和道路类别；--osm-cache 可覆盖同格式公共 OSM；--network 必须是刚构建的同城图；--out 输出 gzip pickle，旁边 manifest 绑定图和输出哈希。SF 将 porto 换为 sf。北京已有冻结支撑用于已有结果复现，不需要重建。

## 3. 生成 MTR-GSRT 数据

```powershell
python commands/reproduce.py generate-main -- --data "C:\runs\porto\real.pkl" --dataset-config porto --epsilon-total 7/5 --noise-seed 20260719 --decoder-seed 30260719 --public-slot-count 20000 --out-dir "C:\runs\porto\synthetic"
```

输出 trajectories.pkl、road_witnesses.pkl、DP transcript 和 protocol.json。--data 是准备后的输入，--epsilon-total 是精确有理数预算，两个 seed 分别控制研究噪声和公共路由，--public-slot-count 是预声明输出规模，不从私有输入推断。SF 使用 --dataset-config sf；该注册项核验原冻结输入 SHA，若是新的 SF 语料，省略 --dataset-config 并显式给 --bbox 37.60 37.85 -122.55 -122.30 --osm-cache evaluation/data/osm/osm_cache_sf_bay.pkl --public-slot-count 20000。小样本工程检查可用 --limit 20 --public-slot-count 20，不是全量实验。

## 4. 重新评分与作图

```powershell
python commands/reproduce.py evaluate -- --real "C:\runs\porto\real.pkl" --synthetic "C:\runs\porto\synthetic\trajectories.pkl" --dataset-config porto --public-slot-count 20000 --out-dir "C:\runs\porto\metrics"
```

输出 metrics.json、metrics.csv、manifest.json。GSRT 可加 --witness C:\runs\porto\synthetic\road_witnesses.pkl；DFR 坐标文件取上一步的转换输出，并按 README 传 --road-routes、--edge-cache。道路选择指标另外由 run-mr 对同城路线对象评分，--real-routes 指定真实匹配结果，--route 方法::路由=路径 指定每份合成路线，--network 和 --edge-cache 指定同城公共图与支撑。输出 results.csv 和输入绑定 manifest。作图时使用 README 的 plot --data-root 指向这次的完整实验结果目录，--out-dir 指定新图片目录；不是把单份 metrics.csv 自动当成所有论文图的输入。

本目录不含统计基线源码。北京保存数据可以直接重新评分；新增城市的 baseline 比较需要提供该城市的合成数据，或在 full_baseline_code 目录先运行生成命令。
