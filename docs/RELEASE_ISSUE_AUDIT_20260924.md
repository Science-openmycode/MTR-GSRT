# Public reproduction issue audit

- [x] Read the external report against the cited commit and current `main`.
- [x] Rebuild the file list and audited source bindings without bypasses.
- [x] Verify LF and CRLF checkouts from clean Git worktrees.
- [x] Exercise precomputed figures, artifact loading, generation, and evaluation.
- [x] Verify every non-LFS committed blob against the release manifest (274 blobs, zero mismatches before this update).
- [x] Exercise the real-road-reference entry point and downstream consumer. Initially blocked by missing `gdal204.dll` and `boost_serialization.dll`; a compatible installed conda environment was found. `verify-matcher` started both executables, a 50-trajectory STMatch run wrote `matched_paths/Real.pkl.gz`, and `run-framework` read that output. This is a smoke test, not a recomputation of the full paper table.
- [x] Resolve documented matcher, real-road-reference, and bundled OSM-cache paths; update the release manifest and pass `verify` (299 files).
- [ ] Verify the final remote commit from a clean checkout.

Paper numerics requiring the original real trajectories remain a separate data-availability boundary. The repository must state this plainly and provide executable instructions when users have that input.
The frozen Beijing input is not in the public repository. Its SHA-256 is `6a160ca557fbd7bab97af489b56c931e73532c498c390e31965c0d61e53361fd`. Exact paper numerics cannot be independently recomputed without that input. The bundled matcher requires externally supplied binary-compatible DLLs. On this development machine, `C:\Users\ASUS\.conda\envs\LSTM_TrajGAN\Library\bin` supplies working versions. The 50-trajectory test output is `C:\Users\ASUS\AppData\Local\Temp\mtr_gsrt_road_ref_smoke_20260924`; it must not be described as full raw-to-figure reproduction.
