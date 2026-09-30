# 四个 baseline 的固定结果

本代码目录只发布 MTR-GSRT 的生成实现。四个统计式 baseline 没有在本目录重新实现；论文实验直接使用已经生成好的合成数据。这样做是有意的，避免把外部算法结果误写成当前仓库重新生成的结果。

## 已发布文件

| 方法 | 文件 | 轨迹数 | SHA-256 |
|---|---|---:|---|
| SPRT | `datasets/synthetic/baselines/sprt_native.pkl` | 17,123 | `bb95796a5d4340b49912f71c90df18a90dbab665ea976c0a9ba1e21a2900f01b` |
| PrivTrace | `datasets/synthetic/baselines/privtrace_native.pkl` | 17,123 | `76538f38dbd78420ed0594b391f0a26bc0958d905d1bf0d98144873b684d6c73` |
| DPTraj-PM | `datasets/synthetic/baselines/dptrajpm_native.pkl` | 17,123 | `3b468623db12ec4f963d1b7774016842d289ee948c3c170d2641144560d863d1` |
| DPStd | `datasets/synthetic/baselines/dpstd_native.pkl` | 17,123 | `4c881f309c0d14ea6b8222c9aaa4793101d77694b81be8159ad7c22d04eab4cd` |

严格 train-only TSTR 使用对应的 `datasets/synthetic/train_only_baselines/` 文件，README 中的 TSTR 命令直接读取这些文件。

## 复现方式

以下命令只检查冻结 baseline 文件，不重新生成：

```powershell
python commands/reproduce.py verify
python commands/reproduce.py smoke
```

完成 MTR-GSRT 生成后，将 `--synthetic` 或 `--corpus` 指向上述四个文件，即可重新运行统一评估、道路适配、TSTR、M×R 和论文作图。

```powershell
python commands/reproduce.py run-profile -- --real <真实轨迹> `
  --synthetic "SPRT=datasets\synthetic\baselines\sprt_native.pkl" `
  --synthetic "PrivTrace=datasets\synthetic\baselines\privtrace_native.pkl" `
  --synthetic "DPTraj-PM=datasets\synthetic\baselines\dptrajpm_native.pkl" `
  --synthetic "DPStd=datasets\synthetic\baselines\dpstd_native.pkl" `
  --synthetic "MTR-GSRT=<MTR-GSRT trajectories.pkl>" `
  --witness "MTR-GSRT=<MTR-GSRT road_witnesses.pkl>" `
  --out-dir <输出目录>
```

如果运行：

```powershell
python commands/reproduce.py generate-baselines
```

程序会返回拒绝信息，因为本目录不包含 baseline 源码。这是正确行为，不是运行错误。
