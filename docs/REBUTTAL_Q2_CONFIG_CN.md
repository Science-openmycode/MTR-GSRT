# 三城主发布配置与协议核验

本实验核对 Reviewer 2 Q2 的北京、波尔图和旧金山主运行。先按 [跨城市输入准备](CROSS_CITY_COMMANDS_CN.md)生成真实输入与主发布；北京主发布也可按 README 的 `generate-main` 命令生成。每个发布目录应包含 `protocol.json`、`dp_transcript.npz`、`trajectories.pkl` 和 `road_witnesses.pkl`。

在本文件夹根目录运行（把三个路径换成实际生成目录）：

```powershell
python evaluation/audit_rebuttal_q2_protocols.py `
  --geolife-release C:\runs\beijing\synthetic `
  --porto-release C:\runs\porto\synthetic `
  --sf-release C:\runs\sf\synthetic `
  --out C:\runs\q2_protocol_audit.json
```

`--geolife-release`、`--porto-release`、`--sf-release` 分别指定三份主运行的完整发布目录；`--out` 指定尚不存在的 JSON 文件。相对路径以本文件夹为根。脚本核验预算、公共 bbox、固定输出数量、回退数、三份发布文件的 SHA-256、公共 OSM 选择结果，以及三个带噪查询向量维度；不读取真实轨迹或未加噪查询。缺文件、输出哈希不符或维度不符立即报错。生成耗时受机器影响，输出实际耗时及其是否恰好等于历史表的标志，但不把耗时差异判为算法错误。

历史主运行应得到：

| 城市 | 输出数 | 最后一级回退 | 96 区域流维度 | 384 区域流维度 | Portal-Fiber 维度 | 历史生成时间（约） |
|---|---:|---:|---:|---:|---:|---:|
| 北京 | 17,123 | 0 | 438 | 1,879 | 6,733 | 726 s |
| 波尔图 | 20,000 | 14 | 446 | 1,837 | 6,071 | 1,031 s |
| 旧金山 | 20,000 | 1 | 390 | 1,537 | 2,825 | 520 s |

`public_osm_byte_hash_matches_historical_protocol` 另行记录公共 OSM pickle 的**字节**哈希是否与旧协议相同。当前公开目录中，波尔图的 pickle 对象逐项等于历史对象，但 pickle 字节哈希不同；旧金山公开 cache 已预先裁成主干机动车道路，历史 cache 则保存完整 OSM 列表。两者按同一固定 bbox 与道路类别过滤后，各自与历史运行的**有序公共道路对象**一致。脚本用 `selected_public_graph_content_sha256` 核验该事实，并保留字节哈希差异，不伪称旧 `protocol.json` 可由现有 cache 逐字节重建。

旧协议为保护 add/remove 邻接下的输入接口，明确设置 `private_input_hash_persisted: false`；公开发布不能通过未加噪的私有输入哈希认证。北京与旧金山注册的冻结数据可在运行前用生成器的 `--verify-frozen-input` 做**本地**文件预检；波尔图按原始 CSV、公开解析顺序及准备阶段 manifest 核验。预检不属于 DP 发布机制，也不把输入哈希写回公开发布对象。
