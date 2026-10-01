# Oldenburg diagnostic benchmark

The repository now includes the Oldenburg input used by the published local
diagnostics at `datasets/raw/oldenburg.dat` (Git LFS).  The corpus contains
simulated road-network trajectories in local projected coordinates.  It is a
road-structure stress test, not a human GPS dataset and not a formal
MTR-GSRT public-OSM release.

## Recompute the input audit

From the repository root:

```powershell
python commands/oldenburg_diagnostics.py `
  --data datasets/raw/oldenburg.dat `
  --limit 20000 `
  --out-dir experiment_results/oldenburg/recomputed
```

This writes `input_audit.json`, including the input SHA-256, record count,
coordinate range, and trajectory-length statistics.

## Existing measured results

The previously measured diagnostic outputs are included under
`experiment_results/oldenburg/`:

- `oldenburg_dataset_assessment.json/.md`: 20,000-record corpus assessment;
- `cross_dataset_ara_mode_benchmark.json`: 1,400-record cross-dataset run;
- `fair_published_vs_ara_oldenburg.json`: equal-count visual diagnostic;
- `oldenburg_real_vs_dppgm_on_basemap.png` and
  `oldenburg_published_vs_ara_map.{png,pdf}`: rendered comparisons.

These files are evidence for the diagnostic benchmark.  They must not be
described as a formal Oldenburg MTR-GSRT release because the repository does
not ship an Oldenburg public OSM/Portal-Fiber graph.
