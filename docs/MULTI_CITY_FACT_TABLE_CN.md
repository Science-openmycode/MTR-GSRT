# 多城市复现事实表

下表按当前 GitHub 提交 `ca6a659` 的实际文件和配置整理。\“可运行\”表示仓库代码具备入口且在补齐表中列出的外部输入后可以运行；\“已发布结果\”只表示仓库中确实存在对应结果文件。

| 数据集 | 数据语义与坐标 | 公共道路图 | MTR-GSRT 生成 | 统计 baseline | 道路/统一评估 | GitHub 已发布结果 | 当前结论 |
|---|---|---|---|---|---|---|---|
| Beijing / GeoLife | 北京 GeoLife，WGS84；论文冻结输入哈希记录在配置中 | 已提供 Beijing OSM cache | 可运行；默认 `public_slot_count=17123` | 四种 baseline 合成结果已提供；baseline 源码不在该仓库 | 道路匹配、M×R、消融、隐私、TSTR 和作图入口均提供 | Beijing 合成文件、路线文件和论文图已提供 | **论文主城市，可复现**；真实 GeoLife 原始文件需按 README 从公开来源准备 |
| Porto | Porto taxi 轨迹，WGS84 | 已提供 Porto OSM cache | 可运行；配置为研究推荐，固定槽位 20,000 | baseline 统一入口可用，但本仓库不含 baseline 源码 | 统一评估和作图入口可迁移运行 | 当前仓库没有 Porto 全量生成结果 | **代码可迁移运行**；需要外部准备 Porto 原始轨迹，不能声称已有冻结数值 |
| San Francisco / Cabspotting trips | 预处理后的 occupied-taxi trip，WGS84；不是用户级轨迹 | 已提供 SF Bay OSM cache，并限定机动车道路类别 | 可运行；配置为 trip-level research recommended，固定槽位 20,000 | baseline 统一入口可用，但本仓库不含 baseline 源码 | 统一评估和作图入口可迁移运行 | 当前仓库没有 SF 全量生成结果；配置中的冻结输入路径需外部准备 | **代码可迁移运行**；必须先准备 20k trip-level 输入，不能把 legacy taxi history 当作 trip 数据 |
| Oldenburg | 模拟道路网络轨迹，局部投影坐标，不是真实 GPS | 没有公开 OSM/Portal-Fiber graph；使用原始轨迹覆盖构造诊断图 | **正式 MTR-GSRT 不支持** | baseline/模拟诊断可用 | Oldenburg 输入审计和诊断性比较可运行 | 原始数据（LFS）、统计评估 JSON/MD、比较图和脚本已提供 | **诊断基准可复现**；不能写成正式多城市 MTR-GSRT 发布 |
| SF taxi history (legacy) | 多日 taxi history，一条记录不是一次 trip | 有 cache，但对象语义不匹配 | 禁止用于 trip synthesis | 仅历史兼容 | 不进入论文主比较 | 不作为有效结果 | **明确排除** |

## 已实际核对的文件

- 数据集状态与槽位：[generation/mtr_gsrt/configs/datasets.json](../generation/mtr_gsrt/configs/datasets.json)
- 评估状态：[evaluation/configs/datasets.json](../evaluation/configs/datasets.json)
- Oldenburg 数据与结果：[datasets/raw/oldenburg.dat](../datasets/raw/oldenburg.dat)、[experiment_results/oldenburg/](../experiment_results/oldenburg/)
- Oldenburg 输入复核：[commands/oldenburg_diagnostics.py](../commands/oldenburg_diagnostics.py)

## 一句话结论

当前仓库已经把 Beijing 的论文主实验和 Oldenburg 的诊断实验固定下来；Porto 与 San Francisco 已具备公共道路图、配置和统一运行入口，但真实输入和全量结果没有随仓库发布，因此它们属于“可迁移运行”，不是“已发布的同数值复现”。
