import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "evaluation" / "audit_rebuttal_q2_protocols.py"
SPEC = importlib.util.spec_from_file_location("audit_rebuttal_q2_protocols", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_public_graph_semantics_can_match_when_pickle_bytes_differ(tmp_path, monkeypatch):
    bbox = [39.75, 40.15, 116.10, 116.65]
    way = {"type": "way", "id": 1, "tags": {"highway": "primary"},
           "geometry": [{"lat": 39.9, "lon": 116.3}]}
    cache = tmp_path / "osm.pkl"
    cache.write_bytes(pickle.dumps([way], protocol=4))
    content_hash = hashlib.sha256(json.dumps(
        [way], sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    monkeypatch.setitem(audit.EXPECTED, "geolife", {
        "count": 2, "fallback": 0, "seconds": 1, "ways": 1,
        "flow": 1, "fine": 2, "portal": 3,
        "public_graph_content_sha256": content_hash,
    })
    release = tmp_path / "release"
    release.mkdir()
    np.savez(release / "dp_transcript.npz", flow_fine_flow=np.zeros(1),
             q5_fine384_flow=np.zeros(2), q5_portal_fiber_flow=np.zeros(3))
    (release / "road_witnesses.pkl").write_bytes(b"witness")
    (release / "trajectories.pkl").write_bytes(b"trajectory")
    hashes = {name: audit.sha256_file(release / name) for name in (
        "dp_transcript.npz", "road_witnesses.pkl", "trajectories.pkl")}
    protocol = {
        "dataset": "geolife", "bbox": bbox,
        "privacy": {"epsilon_total_rational": "7/5", "delta": 0},
        "public_slot_count": 2, "output_count": 2,
        "decoder": {"fallback_count": 0}, "elapsed_sec": 1.1,
        "outputs": hashes,
        "public_osm": {"sha256": "historical-different-pickle-bytes"},
        "private_input_hash_persisted": False,
    }
    (release / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    config = {"bbox": bbox, "osm_cache": str(cache), "osm_max_ways": 0}
    row = audit.audit_one("geolife", release, config)
    assert row["public_osm_byte_hash_matches_historical_protocol"] is False
    assert row["selected_public_graph_content_sha256"] == content_hash
    assert row["fallback_matches_historical"] is True

    # Current releases keep timing outside the DP object and may use a
    # different audited decoder version from the historical table.
    del protocol["elapsed_sec"]
    protocol["decoder"]["fallback_count"] = 1
    (release / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    timing = tmp_path / "local_timing.json"
    timing.write_text(json.dumps({
        "classification": "LOCAL_PERFORMANCE_DIAGNOSTIC_NOT_DP_RELEASE",
        "release_protocol_sha256": audit.sha256_file(release / "protocol.json"),
        "generation_elapsed_sec": 2.25,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="performance-log"):
        audit.audit_one("geolife", release, config)
    current = audit.audit_one("geolife", release, config, timing)
    assert current["generation_sec"] == 2.25
    assert current["fallback_matches_historical"] is False
    assert current["generation_cost_source"] == "external_local_performance_log"
    timing.write_text(json.dumps({
        "classification": "LOCAL_PERFORMANCE_DIAGNOSTIC_NOT_DP_RELEASE",
        "release_protocol_sha256": "bad",
        "generation_elapsed_sec": 2.25,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="performance protocol SHA-256"):
        audit.audit_one("geolife", release, config, timing)

    (release / "trajectories.pkl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="trajectories.pkl SHA-256"):
        audit.audit_one("geolife", release, config)
