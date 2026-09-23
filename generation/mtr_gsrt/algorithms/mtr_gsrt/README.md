# MTR-GSRT (paper release)

This is the sole MTR instantiation published by the current paper release.

- Source: `../../src/mtr/DP_GSRT/`
- Curated source snapshot: `source_snapshot/DP_GSRT/`
- Unified generator: `../../generation/mtr/generate.py`
- Curated entry-point snapshot: `entrypoints/`
- Active Beijing release:
  `../../outputs/synthetic_releases/current_best/active_hierarchy_release_v1_20260726/`
- Current metrics: `../../results/active_hierarchy_release_v1_20260726/`
- Privacy contract: trajectory-record `(7/5, 0)` DP with fixed public slots.

Historical and exploratory code under `src/mtr/` is not an additional paper
algorithm. The paths above define the active MTR-GSRT release.

The curated snapshots make the algorithm boundary visible in one directory.
Existing CLI and manifest paths continue to use the authoritative source paths;
`ALGORITHM_INDEX.json` verifies source-snapshot parity.
