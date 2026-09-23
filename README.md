# MTR-GSRT

Public code and reproducibility materials for **MTR-GSRT**, an instantiation of
the Measure-Then-Route framework for differentially private road-constrained
trajectory synthesis.

The repository contains the MTR-GSRT generator, unified evaluation and plotting
code, precomputed synthetic releases for the reported statistical baselines,
and the inputs required to reproduce the paper figures. It contains no raw or
private trajectory dataset.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python commands/reproduce.py verify
python commands/reproduce.py smoke
```

Generate a new release from an explicit coordinate-trajectory dataset and a
public road graph:

```powershell
python commands/reproduce.py generate-main -- `
  --data "C:\data\real.pkl" `
  --epsilon-total 7/5 `
  --noise-seed 20260719 `
  --decoder-seed 30260719 `
  --public-slot-count 17123 `
  --bbox 39.75 40.15 116.10 116.65 `
  --osm-cache "C:\data\osm_cache.pkl" `
  --component-mode full `
  --out-dir "C:\runs\mtr_gsrt"
```

`--component-mode` also provides executable matched ablations:
`no-portal-fiber`, `no-graph-flow`, and `demand-only`. Each arm reruns synthesis
and writes its own trajectories, witnesses, transcript, metrics, and manifest.

The complete experiment-by-experiment guide, expected output trees, all command
parameters, and cross-dataset migration rules are documented in
[README_CN.md](README_CN.md) and
[docs/COMMAND_PARAMETERS_CN.md](docs/COMMAND_PARAMETERS_CN.md).

## Repository layout

- `generation/mtr_gsrt/`: MTR-GSRT generation code and public assets.
- `evaluation/`: unified metrics, attacks, TSTR tasks, and ablation runners.
- `plotting/`: scripts for regenerating the paper figures.
- `datasets/synthetic/`: synthetic releases only.
- `experiment_results/`: reported results and figures.
- `commands/reproduce.py`: the single command-line entry point.

The fixed output size is a public parameter and must be declared with
`--public-slot-count`; it is never inferred from a private input at runtime.
