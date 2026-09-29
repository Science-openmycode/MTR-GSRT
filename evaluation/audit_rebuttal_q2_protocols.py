"""Verify the three main MTR-GSRT release protocols against the Q2 table.

This reads only saved DP releases and public OSM caches. It never opens the
private input trajectories or materializes exact (unnoised) query vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
GENERATION_ROOT = ROOT / "generation" / "mtr_gsrt"
sys.path.insert(0, str(GENERATION_ROOT))
from public_utils import filter_osm_ways_by_bbox, filter_osm_ways_by_highway, thin_osm_ways_public  # noqa: E402


EXPECTED = {
    "geolife": {"count": 17123, "fallback": 0, "seconds": 726,
                "ways": 29872, "flow": 438, "fine": 1879, "portal": 6733,
                "public_graph_content_sha256": "8cda1458396a97d67769b30648ee60e0b67f8ef1c3b4cea8c465c14ee5119726"},
    "porto": {"count": 20000, "fallback": 14, "seconds": 1031,
              "ways": 44582, "flow": 446, "fine": 1837, "portal": 6071,
              "public_graph_content_sha256": "7a7f2fc8466281e902d36667b8c0fce6149c8fedc98126ec1b6a0af253bd5198"},
    "sf": {"count": 20000, "fallback": 1, "seconds": 520,
           "ways": 14114, "flow": 390, "fine": 1537, "portal": 2825,
           "public_graph_content_sha256": "c39a84e8f3fd1de0efd7276398c34e303c840e8e9a6a1c16f184fe9b6239f803"},
}
QUERY_KEYS = {
    "flow": "flow_fine_flow",
    "fine": "q5_fine384_flow",
    "portal": "q5_portal_fiber_flow",
}


def rooted(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_equal(label: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise ValueError(f"{label}: observed {observed!r}, expected {expected!r}")


def generation_cost(protocol_path: Path, protocol: dict, performance_path: Path | None) -> float:
    if performance_path is None:
        if "elapsed_sec" not in protocol:
            raise ValueError(f"{protocol_path}: supply an external --CITY-performance-log for a new release")
        return float(protocol["elapsed_sec"])
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    require_equal("performance classification", performance.get("classification"),
                  "LOCAL_PERFORMANCE_DIAGNOSTIC_NOT_DP_RELEASE")
    require_equal("performance protocol SHA-256", performance.get("release_protocol_sha256"),
                  sha256_file(protocol_path))
    elapsed = float(performance["generation_elapsed_sec"])
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError(f"{performance_path}: invalid generation time")
    return elapsed


def audit_one(city: str, release_dir: Path, config: dict,
              performance_path: Path | None = None) -> dict:
    protocol_path = release_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require_equal(f"{city} dataset", protocol["dataset"], city)
    require_equal(f"{city} bbox", protocol["bbox"], config["bbox"])
    require_equal(f"{city} epsilon", protocol["privacy"]["epsilon_total_rational"], "7/5")
    require_equal(f"{city} output slots", protocol["public_slot_count"], EXPECTED[city]["count"])
    require_equal(f"{city} output count", protocol["output_count"], EXPECTED[city]["count"])
    fallback_matches = protocol["decoder"]["fallback_count"] == EXPECTED[city]["fallback"]
    if "elapsed_sec" in protocol:
        require_equal(f"{city} historical fallback", fallback_matches, True)
    require_equal(f"{city} delta", protocol["privacy"]["delta"], 0)

    output_hashes = {}
    for name in ("dp_transcript.npz", "road_witnesses.pkl", "trajectories.pkl"):
        path = release_dir / name
        digest = sha256_file(path)
        require_equal(f"{city} {name} SHA-256", digest, protocol["outputs"][name])
        output_hashes[name] = digest

    cache_path = rooted(config["osm_cache"])
    cache_hash = sha256_file(cache_path)
    release_cache_hash = protocol["public_osm"]["sha256"]
    historical_protocol = "elapsed_sec" in protocol
    with cache_path.open("rb") as stream:
        ways = pickle.load(stream)
    bbox_ways = filter_osm_ways_by_bbox(ways, tuple(config["bbox"]))
    highway_ways = filter_osm_ways_by_highway(bbox_ways, config.get("osm_highway_classes"))
    selected = thin_osm_ways_public(highway_ways, tuple(config["bbox"]), config.get("osm_max_ways", 0))
    require_equal(f"{city} selected OSM ways", len(selected), EXPECTED[city]["ways"])
    content_hash = hashlib.sha256(json.dumps(
        selected, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    require_equal(f"{city} selected public graph content SHA-256",
                  content_hash, EXPECTED[city]["public_graph_content_sha256"])
    if "selected_way_count" in protocol["public_osm"]:
        require_equal(f"{city} protocol OSM ways", protocol["public_osm"]["selected_way_count"], len(selected))
    del ways, bbox_ways, highway_ways, selected

    with np.load(release_dir / "dp_transcript.npz", allow_pickle=False) as transcript:
        dimensions = {label: int(transcript[key].size) for label, key in QUERY_KEYS.items()}
    for label, actual in dimensions.items():
        require_equal(f"{city} {label} dimension", actual, EXPECTED[city][label])
    elapsed = generation_cost(protocol_path, protocol, performance_path)
    return {
        "city": city,
        "release_dir": str(release_dir),
        "protocol_sha256": sha256_file(protocol_path),
        "output_count": protocol["output_count"],
        "fallback_count": protocol["decoder"]["fallback_count"],
        "historical_fallback_count": EXPECTED[city]["fallback"],
        "fallback_matches_historical": fallback_matches,
        "generation_sec": elapsed,
        "generation_cost_source": ("external_local_performance_log" if performance_path else
                                   "historical_protocol_elapsed_sec"),
        "historical_rounded_generation_sec": EXPECTED[city]["seconds"],
        "historical_generation_time_matches": round(elapsed) == EXPECTED[city]["seconds"],
        "selected_public_osm_ways": EXPECTED[city]["ways"],
        "public_osm_sha256": cache_hash,
        "release_protocol_osm_sha256": release_cache_hash,
        "public_osm_byte_hash_matches_release_protocol": cache_hash == release_cache_hash,
        "historical_protocol_osm_sha256": release_cache_hash if historical_protocol else None,
        "public_osm_byte_hash_matches_historical_protocol": (
            cache_hash == release_cache_hash if historical_protocol else None
        ),
        "selected_public_graph_content_sha256": content_hash,
        "query_dimensions": dimensions,
        "output_sha256": output_hashes,
        "private_input_hash_in_public_protocol": bool(protocol.get("private_input_hash_persisted", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for city in EXPECTED:
        parser.add_argument(f"--{city}-release", required=True,
                            help=f"{city} main-run directory containing protocol.json and three release files")
        parser.add_argument(f"--{city}-performance-log",
                            help=f"{city} separate local timing log for a current release")
    parser.add_argument("--out", required=True, help="new JSON audit path; relative to this repository")
    args = parser.parse_args()
    out = rooted(args.out)
    if out.exists():
        parser.error(f"output already exists: {out}")
    configs = json.loads((GENERATION_ROOT / "configs" / "datasets.json").read_text(encoding="utf-8"))["datasets"]
    rows = [audit_one(
        city, rooted(getattr(args, f"{city}_release")), configs[city],
        rooted(getattr(args, f"{city}_performance_log"))
        if getattr(args, f"{city}_performance_log") else None,
    ) for city in EXPECTED]
    result = {
        "classification": "SAVED_DP_RELEASE_AND_PUBLIC_GRAPH_AUDIT_NO_PRIVATE_INPUT",
        "rebuttal": "reviewer_2/Q2",
        "cities": rows,
        "note": "Input hashes are intentionally not disclosed in public DP release protocols; supply and verify raw inputs locally.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "cities": [r["city"] for r in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
