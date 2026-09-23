# DP-GSRT Active Final: Portal-Fiber Nested QRSP-384

The only active final algorithm is `dp-gsrt-portal-fiber-nested-qrsp384-v1`.

It releases a sensitivity-bounded nested quotient/portal-fiber graph query and performs bounded information projection, randomized-shortest-path bridging, portal lifting, and serialization as authenticated post-processing. The production transcript composes a `6/5` base event with a `1/5` Q5 event, giving pure `(7/5,0)` trajectory-record DP.

The final 17,123-record release has SHA-256:

```text
b770a924c1c491719e36355d9c9bd3fe66e60ca54f11913e98b70b73cfe62668
```

See `CURRENT_METHOD_STATUS.json` for event IDs, proof hashes, and the exact privacy contract. The rejected predecessor remains only in the central archive and must not be used by a run or paper entry point.

The frozen 38-module implementation closure is under `final/`, with per-file hashes in `final/SOURCE_MANIFEST.json`. Follow `RUN_FINAL.md` for the two DP measurements and authenticated QRSP post-processing command sequence. The old `synthesize_dataset.py --method dp-gsrt` single-process adapter is intentionally disabled because it cannot represent the two-event privacy boundary.
