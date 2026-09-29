# 带噪测量与长度条件路由：实现核对

本页对应 rebuttal1-Q3 / rebuttal2-Q3。运行下列命令，用当前解码器在一个完全公开的两路图上重算 Gibbs/Doob 桥、期望代价、固定候选集中的 `beta` 选择和 portal 信息投影：

```powershell
python commands/reproduce.py audit-rebuttal-q3 -- `
  --out C:\runs\rebuttal_q3\routing_math.json
```

输出 `routing_math.json` 记录解码器源码 SHA-256 和逐项数值。图中从 `s` 到 `d` 有两条路径：经中间点的代价为 2，直达代价为 3，参考概率各为 1/2。在 `beta=0.1,0.5,1.0` 下，程序计算的期望代价分别约为 2.475021、2.377541、2.268941，并逐项核对解析分配函数；目标代价 2.4 选择 `beta=0.5`。这说明长度以**期望代价最近的固定 beta**进入单候选区域路径分布，不要求每条路径长度等于目标。多候选模式会轮换固定 beta 并以真实提升后道路长度的软权重选候选，另有步数上限及公共回退，因此最后完整输出分布不等于未经截断的理想 Gibbs 分布。

当前主配置由公开生成入口传入 `--reference-conditioning portal-fiber`。Graph-flow 构成区域级 `reference_kernel`，`RSPBridge` 从该核生成区域路径。Portal-Fiber 的已加噪 `portal_fiber_flow` 在把区域跨越提升为实际道路边时，由 `lift_quotient_path` 调用 `portal_information_projection`。其一条跨界选择的具体形式是

\[
q_j\propto p^{\mathrm{public}}_j\exp\!\left(\eta\,\operatorname{clip}\left[
\log\frac{p^{\mathrm{DP}}_j}{1/k},-\log\Lambda,\log\Lambda\right]\right),
\]

其中 `p_public` 来自公共 portal 几何距离，`p_DP` 是已发布的 portal 质量经非负化、正支撑平滑与归一化所得，`k` 为该公共区域边上的 portal 数。示例的公共先验为 `(0.8,0.2)`，已发布质量示例为 `(6,4)`，`eta=0.5`、`Lambda=2` 时输出约 `(0.830479,0.169521)`。只要真实原始轨迹不进入此阶段，它属于测量发布后的路由计算。

这也明确了文稿公式的适用范围：把 Graph-flow 和 Portal-Fiber 写成一次作用在同一个区域转移核上的逐行投影，可以概述思想，但不是主配置的逐操作规格。区域桥接核和跨界 portal 选择核需要分开描述。其他条件分支对多因子 log-likelihood 的顺序是**每项先截断，再加权求和**；先求和再截断通常不等价。这里的公开两路图复算检验路由数学，不替代上游差分隐私测量与正式输出 witness 的证明。
