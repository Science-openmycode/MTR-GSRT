# Run the Final Portal-Fiber DP-GSRT

The active mechanism is `dp-gsrt-portal-fiber-nested-qrsp384-edge-witness-v1` with promotion status `APPROVED_ACTIVE_FINAL_EDGE_WITNESS_V1`. It is intentionally not routed through the retired single-process `--method dp-gsrt` adapter. A production release has two DP measurements and one authenticated post-processing stage.

Run from `public_release` in PowerShell. The private input must be a pickle containing at most 17,123 raw trajectory slots; the public OSM cache must have the committed SHA-256 recorded in `CURRENT_METHOD_STATUS.json`.

```powershell
$env:PYTHONPATH = "$((Resolve-Path .).Path);$((Resolve-Path src\mtr\DP_GSRT\final).Path)"
$REAL = "data\geolife\geolife.pkl"
$OSM = "data\osm\osm_cache_beijing.pkl"
$BASE = "outputs\dp_gsrt_final\base"
$Q5 = "outputs\dp_gsrt_final\q5"
$OUT = "outputs\dp_gsrt_final\synthetic"
```

## 1. Base measurement: pure 6/5-DP

```powershell
python src\mtr\DP_GSRT\final\graph_cycle_dp_production_sanitizer.py `
  --private-source $REAL `
  --out-dir $BASE `
  --production-release
```

## 2. Portal-Fiber query Q5: pure 1/5-DP

```powershell
python src\mtr\DP_GSRT\final\portal_fiber_route_production_sanitizer.py `
  --real $REAL `
  --osm $OSM `
  --out-dir $Q5 `
  --capacity 17123 `
  --epsilon-route 1/5 `
  --coarse-occupancy-regions 24 `
  --coarse-transition-regions 96 `
  --fine-regions 384 `
  --macro-regions 4 `
  --phase-count 3 `
  --portals-per-fine-edge 6 `
  --production-release
```

## 3. Authenticated QRSP post-processing: zero additional privacy loss

```powershell
python src\mtr\DP_GSRT\final\portal_fiber_qrsp_production.py `
  --portal-release-dir $Q5 `
  --out-dir $OUT `
  --limit 17123 `
  --route-release-schema portal-fiber-nested384 `
  --regions 384 `
  --reference-source dp-flow `
  --reference-conditioning portal-fiber `
  --od-family-source released `
  --od-likelihood-ratio-cap 100 `
  --portal-count 6 `
  --max-region-steps 192 `
  --length-proposals 9 `
  --length-log-penalty 8 `
  --betas 0.003,0.006,0.012,0.024,0.048,0.096,0.192,0.384,0.768 `
  --production-postprocess
```

The production decoder resolves the base transcript exclusively from its
committed system ledger; passing an external production-request directory is
forbidden.  Omit `--decoder-seed` and `--request-seed`: formal post-processing
uses fresh `SystemRandom`-derived streams, whereas fixed seeds are only for
matched research controls.

The decoder emits the coordinate artifact, `road_witnesses.pkl`, the
same-request public shortest-path control, and `protocol.json`.  Package the
consumer-visible witness and evaluate it before promotion with:

```powershell
powershell -ExecutionPolicy Bypass `
  -File commands\run_formal_edge_witness_release.ps1 `
  -Stage all -Resume
```

The promoted 2026-07-20 release is
`outputs/synthetic_releases/current_best/edge_witness_release_v1_20260720`.
It contains 17,123 coordinate records and 17,123 consumer-visible witnesses,
has no fallback record, and evaluates to `WitnessValid=1.0`.  Its sealed hashes
are recorded in `CURRENT_METHOD_STATUS.json`: coordinate artifact
`33efa30a8161722af81e19a4848458a4b2a363b665cffef3f7460f7d54a7db55`,
consumer witness
`64e3ad167e4b42f9b98cc638160115916cf5d604e37f7fc29e00d96bb3196930`,
and release protocol
`cd70edd0191a2680caf7bc4e6dadf0c3315666d84696ea4231a536e0ef9546b3`.
Re-running the command must create a new production event; it must not reuse
or overwrite this sealed release.

The two measurements compose to pure `(7/5,0)` trajectory-record DP. The third stage reads only authenticated DP transcripts and public graph data. Production event reservations are fail-closed and cannot be silently reused; use separate research-only code and a separate ledger for repeated development experiments.
