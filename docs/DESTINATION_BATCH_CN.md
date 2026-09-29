# Destination 数值批次

旧 `experiment_results/executed_reference/profile/raw/MTR-GSRT` 和消融 raw 目录保留原数值。原运行环境没有备份，不再把这些旧数值作为当前依赖下逐位复现的参考。

现行参考为 [mtr_gsrt.json](../experiment_results/current_verified/destination/mtr_gsrt.json)，来自 2026-09-29 锁定环境下的全量重新评分。这里是全量拟合、真实语料后 20% 评分的回顾性代理，不是 train-only TSTR；严格下游实验使用 README 第 6 节及生成来源核验流程。

| 指标 | 旧保存值 | 现行锁定环境值 |
|---|---:|---:|
| Destination 8 Top-1 | 0.179270073 | 0.178102190 |
| Destination 8 Top-5 | 0.507153285 | 0.503941606 |
| Destination 8 Macro-F1 | 0.045988385 | 0.046007451 |
| Destination 16 Top-1 | 0.030364964 | 0.031532847 |
| Destination 16 Top-5 | 0.192700730 | 0.199124088 |
| Destination 16 Macro-F1 | 0.007065837 | 0.007615965 |

两种线程数、三种 NumPy CPU 指令设置、两种 OpenBLAS 核心设置和新建环境均得到相同的现行六值。新的 `evaluate` 输出包含 `destination_model_audit.json`：依赖版本、模型参数及训练/测试特征、系数、预测哈希。该审计文件保存在本地研究输出，不作为正式 DP 发布对象。

重新评分仍按 README 的 `evaluate` 命令，使用新的 `--out-dir`；不覆盖旧 raw 目录，也不把两批数值拼成同一张比较表。只有依赖、输入与任务口径一致的新批次互相比对。
