"""Full-release, one-factor Portal-Fiber decoder ablations for rebuttal Q5.

All arms reuse one saved DP transcript and identical DP-generated requests.
Only post-processing is changed; no arm rereads private trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PUBLIC = ROOT / "generation" / "mtr_gsrt"
EVAL_PUBLIC = ROOT / "evaluation"
EVALUATION = EVAL_PUBLIC / "evaluation"
OUT: Path
SOURCE: Path
OSM_CACHE: Path
NETWORK: Path
REAL_MATCH_CACHE: Path
DATASET_CONFIG: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GreedyChoice:
    """Retain projected probabilities but replace sampling with their MAP choice."""

    def choice(self, count, p=None):
        if p is None:
            return 0
        return int(np.argmax(np.asarray(p, dtype=float)))


def load_inputs():
    sys.path.insert(0, str(PUBLIC))
    from generation.common.runtime import add_runtime_paths
    add_runtime_paths()
    from public_utils import filter_osm_ways_by_bbox
    protocol = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    if (protocol.get("privacy", {}).get("epsilon_total_rational") != "7/5"
            or protocol.get("privacy", {}).get("budget_allocation", {}).get("compact_graph_flow") != "1/5"
            or protocol.get("privacy", {}).get("budget_allocation", {}).get("portal_fiber_q5") != "1/5"
            or protocol.get("decoder", {}).get("route_release_schema") != "portal-fiber-nested384"):
        raise ValueError("Ablation requires the frozen 7/5 Portal-Fiber transcript schema")
    for name in ("dp_transcript.npz", "trajectories.pkl", "road_witnesses.pkl"):
        if sha256_file(SOURCE / name) != protocol.get("outputs", {}).get(name):
            raise ValueError(f"Source release hash mismatch: {name}")
    with np.load(SOURCE / "dp_transcript.npz", allow_pickle=False) as archive:
        base = {
            "endpoint": np.asarray(archive["base_endpoint"]),
            "od": np.asarray(archive["base_od"]),
            "geometry": np.asarray(archive["base_geometry"]),
            "length": np.asarray(archive["base_length"]),
        }
        flow = {
            "fine_occupancy": np.asarray(archive["flow_fine_occupancy"]),
            "fine_flow": np.asarray(archive["flow_fine_flow"]),
            "dwell": np.asarray(archive["flow_dwell"]),
        }
        q5 = {
            name: np.asarray(archive[f"q5_{name}"])
            for name in (
                "coarse24_occupancy", "fine384_occupancy", "fine96_flow",
                "fine384_flow", "portal_fiber_flow",
            )
        }
    if sha256_file(OSM_CACHE) != protocol["public_osm"]["sha256"]:
        raise ValueError("Supplied OSM cache does not match source DP transcript protocol")
    with OSM_CACHE.open("rb") as handle:
        osm = pickle.load(handle)
    osm = filter_osm_ways_by_bbox(osm, protocol["bbox"])
    return protocol, base, flow, q5, osm


def decode_arm(name, protocol, base, flow, q5, osm):
    from generation.mtr.generate import _decode
    import quotient_rsp_bridge_development as decoder

    arm = OUT / name
    staging = OUT / f".{name}.staging"
    if staging.exists() or arm.exists():
        raise FileExistsError(f"Ablation arm already exists: {name}")
    staging.mkdir(parents=True)
    q5_arm = {key: np.asarray(value, dtype=float).copy() for key, value in q5.items()}
    original_projection = decoder.portal_information_projection
    original_lift = decoder.lift_quotient_path
    if name == "no-crossing-flow-measurement":
        values = q5_arm["portal_fiber_flow"]
        q5_arm["portal_fiber_flow"] = np.full(values.shape, float(values.sum()) / values.size)
    elif name == "no-local-information-projection":
        def public_prior(public_probability, released_portal_mass, conditioned_weight, likelihood_ratio_cap):
            values = np.asarray(public_probability, dtype=float).ravel()
            return values / float(values.sum())
        decoder.portal_information_projection = public_prior
    elif name == "no-crossing-road-sampling":
        def greedy_lift(
            regions, source, destination, graph, coords, labels, portals, router, rng,
            portal_probabilities=None, portal_weight=7.0 / 13.0,
            portal_likelihood_ratio_cap=100.0,
        ):
            return original_lift(
                regions, source, destination, graph, coords, labels, portals, router,
                GreedyChoice(), portal_probabilities=portal_probabilities,
                portal_weight=portal_weight,
                portal_likelihood_ratio_cap=portal_likelihood_ratio_cap,
            )
        decoder.lift_quotient_path = greedy_lift
    elif name != "full":
        raise ValueError(name)
    try:
        result = _decode(
            staging, base, flow, q5_arm, osm, protocol["public_osm"]["sha256"],
            protocol["public_slot_count"], Fraction(7, 5), Fraction(1, 5), Fraction(1, 5),
            protocol["postprocessing"]["decoder_seed"], protocol["postprocessing"]["request_seed"],
            tuple(protocol["bbox"]), occupancy_strength=0.25, dwell_strength=0.10,
            hierarchy_likelihood_ratio_cap=100.0,
        )
        arm.mkdir(parents=True)
        shutil.move(str(result / "dp_gsrt_portal_qrsp.pkl"), arm / "trajectories.pkl")
        shutil.move(str(result / "road_witnesses.pkl"), arm / "road_witnesses.pkl")
        shutil.move(str(result / "protocol.json"), arm / "protocol.json")
    finally:
        decoder.portal_information_projection = original_projection
        decoder.lift_quotient_path = original_lift
        if staging.exists():
            shutil.rmtree(staging)
    print(f"[component] completed {name}", flush=True)
    return arm


def evaluate(arms):
    output = OUT / "evaluation"
    if output.exists():
        raise FileExistsError(output)
    cmd = [
        sys.executable, str(EVALUATION / "evaluate_native_road_choice.py"),
        "--dataset-config", DATASET_CONFIG,
        "--network", str(NETWORK),
        "--real-match-cache", str(REAL_MATCH_CACHE),
        "--out-dir", str(output),
    ]
    for name, arm in arms.items():
        cmd += ["--witness", f"{name}={arm / 'road_witnesses.pkl'}"]
    print("[component] evaluate all arms", flush=True)
    subprocess.run(cmd, cwd=EVAL_PUBLIC, check=True)
    return json.loads((output / "native_road_choice_metrics.json").read_text(encoding="utf-8"))


def main():
    global OUT, SOURCE, OSM_CACHE, NETWORK, REAL_MATCH_CACHE, DATASET_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Saved full MTR release with dp_transcript.npz")
    parser.add_argument("--osm-cache", required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--real-match-cache", required=True)
    parser.add_argument("--dataset-config", default="geolife")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    def rooted(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else ROOT / path).resolve()
    SOURCE, OSM_CACHE = rooted(args.source), rooted(args.osm_cache)
    NETWORK, REAL_MATCH_CACHE = rooted(args.network), rooted(args.real_match_cache)
    DATASET_CONFIG = args.dataset_config
    OUT = rooted(args.out_dir)
    if OUT.exists():
        parser.error(f"Output directory must be new: {OUT}")
    for path in (SOURCE / "protocol.json", SOURCE / "dp_transcript.npz",
                 SOURCE / "trajectories.pkl", SOURCE / "road_witnesses.pkl",
                 OSM_CACHE, NETWORK, REAL_MATCH_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True)
    protocol, base, flow, q5, osm = load_inputs()
    arms = {}
    full = OUT / "full"
    if not full.exists():
        full.mkdir()
        shutil.copy2(SOURCE / "trajectories.pkl", full / "trajectories.pkl")
        shutil.copy2(SOURCE / "road_witnesses.pkl", full / "road_witnesses.pkl")
        shutil.copy2(SOURCE / "protocol.json", full / "protocol.json")
    arms["Full"] = full
    for name in (
        "no-crossing-flow-measurement", "no-local-information-projection",
        "no-crossing-road-sampling",
    ):
        arms[name] = decode_arm(name, protocol, base, flow, q5, osm)
    payload = evaluate(arms)
    (OUT / "manifest.json").write_text(json.dumps({
        "classification": "POSTPROCESSING_ABLATION_REUSING_ONE_DP_TRANSCRIPT",
        "source_dp_transcript": str((SOURCE / "dp_transcript.npz").resolve()),
        "source_dp_transcript_sha256": sha256_file(SOURCE / "dp_transcript.npz"),
        "source_protocol_sha256": sha256_file(SOURCE / "protocol.json"),
        "osm_sha256": sha256_file(OSM_CACHE),
        "network_sha256": sha256_file(NETWORK),
        "real_match_cache_sha256": sha256_file(REAL_MATCH_CACHE),
        "private_trajectory_access": False,
        "arms": {
            "Full": "all three components",
            "no-crossing-flow-measurement": "portal_fiber_flow replaced by equal mass; local projection and sampling retained",
            "no-local-information-projection": "public distance prior used directly; measured flow and stochastic sampling retained",
            "no-crossing-road-sampling": "projected distribution retained; stochastic draw replaced by deterministic MAP crossing edge",
        },
        "results": payload["results"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["results"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
