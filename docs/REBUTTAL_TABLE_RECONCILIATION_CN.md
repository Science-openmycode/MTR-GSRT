# Rebuttal 表格与公开结果的口径说明

当前 GitHub 版本保留严格 TSTR 的原始结果、冻结评估器结果和道路任务明细。它们不能合并成一张表：三者使用的任务列和评价器不同。

`experiment_results/executed_reference/tstr/results.csv` 中四个统计基线与 MTR-GSRT 的前三项严格 TSTR 结果为：

```text
SPRT       0.038244  0.067089  0.051151  0.000000
PrivTrace  0.088715  0.147434  0.057334  0.000000
DPTraj-PM  0.095001  0.125373  0.101063  0.000000
DPStd      0.107061  0.151859  0.204627  0.000000
MTR-GSRT   0.609228  0.752157  0.306164  0.843002
```

PDF 回复表中的前三个 MTR-GSRT 数值 `0.609/0.752/0.306` 与该公开结果一致。PDF 中的 `0.716` 不对应该 CSV 的 `Road continuation Hit@1`：当前公开结果的 Hit@1 是 `0.843002`；道路任务明细中的 `0.716074` 字段名为 `continuation_mrr`。另一份冻结评估器文件给出 `RoadContinuationHit1=0.715506`，属于不同评价批次。

因此，复现时必须根据命令选择一个完整结果目录，并同时使用该目录的指标定义、输入划分和评价器。公开代码不会把 `0.716` 重新标注为 Hit@1，也不会把历史冻结批次与当前 TSTR 批次拼接。Portal 单元消融的 `NextRoadAcc=0.468/0.391` 则位于 `experiment_results/published_figures/portal_fiber_ablation.csv`，属于独立消融总体。
