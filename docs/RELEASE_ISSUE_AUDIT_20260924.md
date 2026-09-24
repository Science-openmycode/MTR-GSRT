# Public reproduction issue audit

- [x] Read the external report against the cited commit and current `main`.
- [x] Rebuild the file list and audited source bindings without bypasses.
- [x] Verify LF and CRLF checkouts from clean Git worktrees.
- [x] Exercise precomputed figures, artifact loading, generation, and evaluation.
- [x] Verify every non-LFS committed blob against the release manifest (274 blobs, zero mismatches before this update).
- [x] Exercise the real-road-reference entry point and downstream consumer. Initially blocked by missing `gdal204.dll` and `boost_serialization.dll`; a compatible installed conda environment was found. `verify-matcher` started both executables, a 50-trajectory STMatch run wrote `matched_paths/Real.pkl.gz`, and `run-framework` read that output. This is a smoke test, not a recomputation of the full paper table.
- [x] Resolve documented matcher, real-road-reference, and bundled OSM-cache paths; update the release manifest and pass `verify`.
- [x] Push the repaired release to `origin/main`; run `verify`, `smoke`, and `verify-matcher` from a clean `core.autocrlf=true` checkout.

Paper numerics requiring the original real trajectories remain a separate data-availability boundary. The repository must state this plainly and provide executable instructions when users have that input.
The frozen Beijing input is not in the public repository. Its SHA-256 is `6a160ca557fbd7bab97af489b56c931e73532c498c390e31965c0d61e53361fd`. Exact paper numerics cannot be independently recomputed without that input. The bundled matcher requires externally supplied binary-compatible DLLs. On this development machine, `C:\Users\ASUS\.conda\envs\LSTM_TrajGAN\Library\bin` supplies working versions. The 50-trajectory test output is `C:\Users\ASUS\AppData\Local\Temp\mtr_gsrt_road_ref_smoke_20260924`; it must not be described as full raw-to-figure reproduction.

## Follow-up metric-contract audit

An additional discrepancy was found when executing `run-framework` and `run-mr` on the available frozen road reference. The previous `route_metric_core.py` looked up node-indexed Region96 labels using edge tuples, making `FamilyCPC` spuriously equal to 1 for almost every nonempty route set. It also normalized BTF by event occurrences instead of assigning bounded mass per fixed output slot, and retained unmatched real slots in the conditional real reference. The corrected evaluator conditions the real reference on connected matched routes, keeps all synthetic slots, computes bounded physical-turn overlap, and multiplies conditional family overlap by road-object yield.

Re-executed framework results match all eight rows of `frozen/shared_mtr_framework_lift.csv` for RoadYield, BTF, and FamilyCPC to within `1e-12`; re-executed M×R results match all nine packaged cells of `frozen/mr_matrix.csv` on the same three metrics. For SPRT × STMatch the corrected values are RoadYield `0.9284003971266718`, BTF `0.28648998611353416`, and FamilyCPC `0.0764959667258665`. The earlier executable reference incorrectly reported BTF `0.5821873384288432` and FamilyCPC `1.0`. Unit and frozen-row regression tests are in `tests/`.

## Full Beijing road-reference rerun

Using the locally available frozen input (not published), the public `prepare-road-reference` entry point ran STMatch on all 17,123 trajectories with the bundled graph, radius 200 m, GPS error 50 m, eight candidates, 32 points per trajectory, batch size 20,000, and eight CPU threads. It produced `check/release_repro_audit_fast_20260924/matched_paths/Real.pkl.gz` outside the public repository. The new and historical decoded pickle objects are exactly equal on all 17,123 records (`unequal_rows=0`), with 10,040 accepted matches. Their compressed file hashes differ because gzip serialization metadata differs; semantic equality is the relevant check. The new manifest binds the expected real-input SHA-256 `6a160ca5...53361fd` and public network SHA-256 `f9ab3c39...e46f473c0`.

Using this freshly created reference, `run-framework` reproduced all eight frozen rows and `run-mr` all nine packaged cells on RoadYield, BTF, and FamilyCPC within `1e-12`. The tests do not make the private input publicly available; they establish that the documented commands reproduce the frozen road results when that input is supplied.
