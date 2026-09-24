from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
import pickle
import random
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

if __package__ in {None, ""}:
    for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if ((_candidate / "configs" / "datasets.json").is_file()
                and (_candidate / "generation" / "common" / "runtime.py").is_file()):
            sys.path.insert(0, str(_candidate))
            break
    else:
        raise RuntimeError("cannot locate public_release root")

from generation.common.runtime import (  # noqa: E402
    PUBLIC_RELEASE,
    add_runtime_paths,
    configure_logging,
    dataset_config,
    parse_fraction,
    public_path,
    require_file,
    require_new_directory,
    sha256_file,
    write_json,
)
from generation.mtr.budget import allocate, demand_allocation  # noqa: E402
from public_utils import thin_osm_ways_public  # noqa: E402


CLASSIFICATION = "RESEARCH_ONLY_REPRODUCIBLE_NOISE_DO_NOT_RELEASE"
ALGORITHM_SCHEMA = "mtr-gsrt-active-hierarchy-v1"
Q5_BLOCK_NAMES = (
    "coarse24_occupancy",
    "fine384_occupancy",
    "fine96_flow",
    "fine384_flow",
    "portal_fiber_flow",
)


def _canonical_source_sha256(path: Path) -> str:
    """Hash audited text independently of Git's LF/CRLF checkout policy."""
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _source_binding() -> dict:
    source_dir = PUBLIC_RELEASE / "src" / "mtr" / "DP_GSRT" / "final"
    manifest_path = source_dir / "SOURCE_MANIFEST.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    mismatches = []
    for entry in entries:
        current = _canonical_source_sha256(source_dir / entry["file"])
        if current != entry["sha256"]:
            mismatches.append({
                "file": entry["file"],
                "manifest_sha256": entry["sha256"],
                "current_sha256": current,
            })
    if mismatches:
        raise RuntimeError(f"audited-source manifest mismatch: {mismatches}")
    return {
        "manifest": "src/mtr/DP_GSRT/final/SOURCE_MANIFEST.json",
        "manifest_sha256": _canonical_source_sha256(manifest_path),
        "manifest_matches_current_source": True,
        "algorithm_schema": ALGORITHM_SCHEMA,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one research-only Portal-Fiber MTR release at a chosen epsilon."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--epsilon-total", required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--decoder-seed", type=int, required=True)
    parser.add_argument("--request-seed", type=int, default=None)
    parser.add_argument("--decoder-betas", default="0.03,0.06,0.12,0.24,0.48,0.96")
    parser.add_argument("--length-proposals", type=int, default=1)
    parser.add_argument("--length-log-penalty", type=float, default=8.0)
    parser.add_argument("--od-likelihood-ratio-cap", type=float, default=100.0)
    parser.add_argument("--occupancy-strength", type=float, default=0.25)
    parser.add_argument("--dwell-strength", type=float, default=0.10)
    parser.add_argument("--hierarchy-likelihood-ratio-cap", type=float, default=100.0)
    parser.add_argument(
        "--component-mode",
        choices=("full", "no-portal-fiber", "no-graph-flow", "demand-only"),
        default="full",
        help=("Algorithm-level ablation applied after the DP transcript is written. "
              "Disabled route blocks are replaced by mass-preserving public uniform fields."),
    )
    parser.add_argument("--public-slot-count", type=int)
    parser.add_argument("--bbox", nargs=4, type=float)
    parser.add_argument("--osm-cache")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--log-file")
    parser.add_argument("--limit", type=int, help="Smoke-test only; must equal public slot count for full experiments")
    return parser


def _resolve_configuration(args: argparse.Namespace) -> dict:
    registered = None
    try:
        registered = dataset_config(args.dataset_config or args.data)
    except KeyError:
        if args.dataset_config:
            raise
    bbox = args.bbox or (registered or {}).get("bbox")
    osm = args.osm_cache or (registered or {}).get("osm_cache")
    capacity = args.public_slot_count or (registered or {}).get("public_slot_count")
    status = (registered or {}).get("mtr_status", "custom_research")
    if status == "unsupported_missing_portal_fiber_public_graph":
        raise RuntimeError((registered or {}).get("note", "dataset is unsupported by Portal-Fiber MTR"))
    if bbox is None or osm is None or capacity is None:
        raise ValueError(
            "unknown/custom datasets require explicit --bbox, --osm-cache and --public-slot-count"
        )
    if int(capacity) <= 0:
        raise ValueError("public slot count must be positive")
    data_spec = (
        (registered or {}).get("data", args.data)
        if args.data == (args.dataset_config or args.data)
        else args.data
    )
    expected_data_sha256 = (registered or {}).get("data_sha256")
    if expected_data_sha256 is not None:
        data_path = require_file(data_spec, "frozen trajectory input")
        actual_data_sha256 = sha256_file(data_path)
        if actual_data_sha256 != expected_data_sha256:
            raise RuntimeError(
                f"frozen trajectory input hash mismatch: {actual_data_sha256}"
            )
        data_spec = data_path
    return {
        "name": (registered or {}).get("name", args.dataset_config or "custom"),
        "data": data_spec,
        "bbox": tuple(float(value) for value in bbox),
        "osm_cache": require_file(osm, "OSM cache"),
        "capacity": int(capacity),
        "status": status,
        "osm_max_ways": int((registered or {}).get("osm_max_ways", 0)),
        "osm_highway_classes": list((registered or {}).get("osm_highway_classes", [])),
    }


def _portal_release(real, coords, graph, capacity: int, epsilon: Fraction, seed: int):
    import audit_two_level_semimarkov_dp as audit
    import certified_discrete_dp as certified
    import nested_quotient_graph as nested
    import portal_fiber_route_release_development as portal

    if tuple(portal.BLOCK_BUDGETS) != Q5_BLOCK_NAMES:
        raise RuntimeError("frozen Portal-Fiber q5 block schema changed")

    context96, _ = audit.build_context(coords, graph, 24, 96, 4, 3)
    context384, _, _ = nested.build_nested_context(coords, graph, 24, 384, 4, 3)
    portals = portal.diverse_portals(graph, coords, context384.node_fine, 6)
    offsets, portal_atoms = portal.portal_layout(portals)
    templates = {
        "coarse24_occupancy": np.zeros(24),
        "fine384_occupancy": np.zeros(384),
        "fine96_flow": np.zeros(len(context96.fine_edge_index) + 1),
        "fine384_flow": np.zeros(len(context384.fine_edge_index) + 1),
        "portal_fiber_flow": np.zeros(portal_atoms + 1),
    }
    totals = {name: np.zeros_like(value, dtype=np.int64) for name, value in templates.items()}
    for trajectory in real:
        blocks = portal.trajectory_blocks(
            trajectory, context96, context384, portals, offsets, coords
        )
        quantized = portal.quantize_trajectory_blocks(blocks)
        if quantized is None:
            continue
        for name in portal.BLOCK_BUDGETS:
            totals[name] += quantized[name]
    exact = np.concatenate([totals[name].ravel() for name in portal.BLOCK_BUDGETS])
    noisy, sampler = certified.add_exact_discrete_laplace(
        exact,
        epsilon_numerator=epsilon.numerator,
        epsilon_denominator=epsilon.denominator,
        sensitivity=portal.ROUTE_LATTICE,
        rng=random.Random(int(seed) + 770_027),
    )
    released, offset = {}, 0
    for name, mass in portal.BLOCK_BUDGETS.items():
        size = int(totals[name].size)
        block = noisy[offset : offset + size].reshape(totals[name].shape)
        released[name] = (
            portal.project_simplex(block.astype(float), float(capacity) * int(mass))
            / float(portal.ROUTE_LATTICE)
        )
        offset += size
    return released, sampler


def _decode(
    work_dir: Path,
    measurements: dict[str, np.ndarray],
    graph_flow: dict[str, np.ndarray],
    q5: dict[str, np.ndarray],
    osm,
    osm_sha256: str,
    capacity: int,
    total_epsilon: Fraction,
    graph_flow_epsilon: Fraction,
    q5_epsilon: Fraction,
    decoder_seed: int,
    request_seed: int,
    bbox: tuple[float, float, float, float],
    *,
    betas: str = "0.03,0.06,0.12,0.24,0.48,0.96",
    length_proposals: int = 1,
    length_log_penalty: float = 8.0,
    od_likelihood_ratio_cap: float = 100.0,
    occupancy_strength: float = 1.0,
    dwell_strength: float = 1.0,
    hierarchy_likelihood_ratio_cap: float = 100.0,
    route_alignment: str = "correct-dp",
) -> Path:
    import graph_cycle_dp_production_sanitizer as lineage
    import graph_voronoi_doptimal_support_probe as support
    import portal_fiber_qrsp_production_gate as gate
    import portal_fiber_route_production_sanitizer as portal_sanitizer
    import prepare_graph_cycle_production_requests as requests
    import quotient_rsp_bridge_development as decoder

    support.BBOX = bbox
    lineage.PUBLIC_CAPACITY = int(capacity)
    gate.base_sanitizer.PUBLIC_CAPACITY = int(capacity)
    request_rng = np.random.default_rng(int(request_seed))
    request_arrays = requests.sample_request_arrays_from_objects(measurements, osm, request_rng)
    request_sha = gate._request_digest(request_arrays)
    request_manifest = {
        "schema_version": 1,
        "request_count": int(capacity),
        "sanitizer_manifest_sha256": "research-only-no-production-ledger",
        "public_osm_sha256": osm_sha256,
        "randomness": f"fixed research request seed {request_seed}",
    }
    q5_digest = hashlib.sha256()
    for name in portal_sanitizer.BLOCK_NAMES:
        q5_digest.update(np.ascontiguousarray(q5[name]).tobytes())
    q5_sha = q5_digest.hexdigest()
    graph_flow_snapshot = {
        name: np.asarray(values, dtype=float).copy()
        for name, values in graph_flow.items()
    }
    decoder.GRAPH_FLOW_RELEASE = graph_flow_snapshot

    def sample_research_requests(_osm_path, *, request_seed=None):
        return request_manifest, request_arrays, request_sha, osm

    def verify_research_portal(_release_dir, public_osm_sha256, **kwargs):
        if public_osm_sha256 != osm_sha256 or int(kwargs["capacity"]) != int(capacity):
            raise RuntimeError("research decoder input binding mismatch")
        for name, shape in kwargs["expected_block_shapes"].items():
            if tuple(q5[name].shape) != tuple(shape):
                raise RuntimeError(f"q5 block shape mismatch: {name}")
        manifest = {
            "classification": CLASSIFICATION,
            "privacy": {"epsilon_route_rational": str(q5_epsilon)},
        }
        return manifest, q5, q5_sha, "research-manifest", q5_epsilon

    original_sample = gate.sample_committed_base_requests
    original_verify = gate.verify_portal_release
    original_ledger = lineage.read_privacy_ledger
    original_argv = sys.argv[:]
    gate.sample_committed_base_requests = sample_research_requests
    gate.verify_portal_release = verify_research_portal
    lineage.read_privacy_ledger = lambda: {
        "epsilon_rational": str(total_epsilon - q5_epsilon), "status": "research"
    }
    decoder_out = work_dir / "decoder"
    try:
        sys.argv = [
            "portal_fiber_research_decoder",
            "--portal-release-dir", str(work_dir),
            "--out-dir", str(decoder_out),
            "--limit", str(capacity),
            "--route-release-schema", "portal-fiber-nested384",
            "--regions", "384",
            "--reference-source", "dp-flow",
            "--reference-conditioning", "portal-fiber",
            "--od-family-source", "released",
            "--od-likelihood-ratio-cap", str(od_likelihood_ratio_cap),
            "--active-hierarchy",
            "--epsilon-graph-flow", str(float(graph_flow_epsilon)),
            "--occupancy-strength", str(occupancy_strength),
            "--dwell-strength", str(dwell_strength),
            "--hierarchy-likelihood-ratio-cap", str(hierarchy_likelihood_ratio_cap),
            "--route-alignment", str(route_alignment),
            "--portal-count", "6",
            "--betas", str(betas),
            "--length-proposals", str(length_proposals),
            "--length-log-penalty", str(length_log_penalty),
            "--decoder-seed", str(decoder_seed),
            "--request-seed", str(request_seed),
            "--production-postprocess",
        ]
        internal_stdout = io.StringIO()
        with contextlib.redirect_stdout(internal_stdout):
            decoder.main()
    finally:
        sys.argv = original_argv
        gate.sample_committed_base_requests = original_sample
        gate.verify_portal_release = original_verify
        lineage.read_privacy_ledger = original_ledger
        decoder.GRAPH_FLOW_RELEASE = None
    return decoder_out


def main() -> None:
    args = build_parser().parse_args()
    add_runtime_paths()
    config = _resolve_configuration(args)
    source_binding = _source_binding()
    if args.limit is not None and int(args.limit) != int(config["capacity"]):
        raise ValueError(
            "--limit may not change the release cardinality; it must equal --public-slot-count"
        )
    total_epsilon = parse_fraction(args.epsilon_total)
    budgets = allocate(total_epsilon)
    out_dir = require_new_directory(args.out_dir)
    log_file = public_path(args.log_file) if args.log_file else out_dir.parent / f"{out_dir.name}.log"
    logger = configure_logging(log_file, "mtr-research-generator")
    work_dir = out_dir.parent / f".{out_dir.name}.research-staging"
    if work_dir.exists():
        raise FileExistsError(f"stale staging directory exists: {work_dir}")
    work_dir.mkdir(parents=True)
    started = time.time()
    try:
        import audit_two_level_semimarkov_dp as audit
        import compact_graph_flow_experiment as compact
        import dp_graph_voronoi_release as base
        import geometric_base_requests as geometric
        import graph_voronoi_doptimal_support_probe as support
        import route_structure_potential_experiment as route
        from public_utils import (
            filter_osm_ways_by_bbox,
            filter_osm_ways_by_highway,
            load_osm_ways,
            load_trajectories,
        )

        logger.info("loading trajectory data: %s", config["data"])
        real = load_trajectories(str(config["data"]), limit=config["capacity"])
        if len(real) != config["capacity"]:
            raise RuntimeError(
                "fixed-slot MTR requires input count to equal the public slot count; "
                f"received {len(real)} records for {config['capacity']} public slots"
            )
        with config["osm_cache"].open("rb") as handle:
            osm = pickle.load(handle)
        osm = filter_osm_ways_by_bbox(osm, config["bbox"])
        osm_bbox_count = len(osm)
        osm = filter_osm_ways_by_highway(osm, config["osm_highway_classes"])
        osm_highway_count = len(osm)
        if config["osm_max_ways"] > 0:
            osm = thin_osm_ways_public(osm, config["bbox"], config["osm_max_ways"])
        if not osm:
            raise RuntimeError("public OSM cache has no ways inside the configured bbox")
        support.BBOX = config["bbox"]
        coords, graph = route.prepare_graph([], bbox=config["bbox"], osm_ways=osm, raw_graph=False)
        fine_nodes = support.farthest_point_landmarks(coords, 256)
        fine_coords = coords[np.asarray(fine_nodes, dtype=int)]
        fine_tree = cKDTree(fine_coords)
        coarse_nodes = support.farthest_point_landmarks(coords, 24)
        coarse_coords = coords[np.asarray(coarse_nodes, dtype=int)]
        coarse_tree = cKDTree(coarse_coords)
        fine_to_coarse = np.asarray(coarse_tree.query(fine_coords)[1], dtype=int)

        original_budgets = geometric.BASE_BUDGETS
        geometric.BASE_BUDGETS = demand_allocation(total_epsilon)
        try:
            base_release = geometric.fit_geometric_measurements(
                real, fine_tree, coarse_tree, support.standardized_xy(coarse_coords),
                fine_count=256, coarse_count=24, capacity=config["capacity"],
                exact_rng=random.Random(int(args.noise_seed) + 101),
            )
        finally:
            geometric.BASE_BUDGETS = original_budgets
        base_release["od"], sinkhorn = base.sinkhorn_od_projection(
            base_release["od"], base_release["endpoint"], fine_to_coarse, 24
        )
        base_samplers = base_release.pop("samplers")
        context96, _ = audit.build_context(coords, graph, 24, 96, 4, 3)
        flow_totals = compact.aggregate(real, context96)
        flow_release, flow_sampler = compact.release(
            flow_totals, config["capacity"], random.Random(int(args.noise_seed) + 202),
            budgets["compact_graph_flow"],
        )
        q5_release, q5_sampler = _portal_release(
            real, coords, graph, config["capacity"], budgets["portal_fiber_q5"], args.noise_seed
        )
        transcript = {}
        transcript.update({f"base_{k}": v for k, v in base_release.items()})
        transcript.update({f"flow_{k}": v for k, v in flow_release.items()})
        transcript.update({f"q5_{k}": v for k, v in q5_release.items()})
        transcript_path = work_dir / "dp_transcript.npz"
        np.savez_compressed(transcript_path, **transcript)

        # The decoder rebuilds its public graph from ``osm`` and consumes only
        # the released measurements.  Drop raw trajectories and aggregation
        # contexts before allocating 17,123 decoded paths.
        del real, coords, graph, fine_tree, coarse_tree, context96, flow_totals
        gc.collect()

        decode_flow = {name: np.asarray(value, dtype=float).copy()
                       for name, value in flow_release.items()}
        decode_q5 = {name: np.asarray(value, dtype=float).copy()
                     for name, value in q5_release.items()}

        def neutralize(blocks):
            for name, value in blocks.items():
                array = np.asarray(value, dtype=float)
                blocks[name] = np.full(array.shape, float(array.sum()) / max(array.size, 1))

        if args.component_mode in {"no-graph-flow", "demand-only"}:
            neutralize(decode_flow)
        if args.component_mode in {"no-portal-fiber", "demand-only"}:
            neutralize(decode_q5)

        request_seed = int(args.request_seed if args.request_seed is not None else args.decoder_seed + 483729)
        decoder_out = _decode(
            work_dir, base_release, decode_flow, decode_q5, osm, sha256_file(config["osm_cache"]),
            config["capacity"], total_epsilon, budgets["compact_graph_flow"],
            budgets["portal_fiber_q5"],
            args.decoder_seed, request_seed, config["bbox"],
            betas=args.decoder_betas,
            length_proposals=args.length_proposals,
            length_log_penalty=args.length_log_penalty,
            od_likelihood_ratio_cap=args.od_likelihood_ratio_cap,
            occupancy_strength=args.occupancy_strength,
            dwell_strength=args.dwell_strength,
            hierarchy_likelihood_ratio_cap=args.hierarchy_likelihood_ratio_cap,
        )
        source_syn = decoder_out / "dp_gsrt_portal_qrsp.pkl"
        source_witness = decoder_out / "road_witnesses.pkl"
        with source_syn.open("rb") as handle:
            decoded_records = pickle.load(handle)
        if not isinstance(decoded_records, list) or len(decoded_records) != config["capacity"]:
            raise RuntimeError("decoder output does not fill the fixed public slot domain")
        del decoded_records
        with source_witness.open("rb") as handle:
            witness_records = pickle.load(handle)
        if not isinstance(witness_records, list) or len(witness_records) != config["capacity"]:
            raise RuntimeError("witness sidecar is not one-to-one with the fixed public slot domain")
        for slot, witness in enumerate(witness_records):
            if not isinstance(witness, dict) or int(witness.get("slot", -1)) != slot:
                raise RuntimeError(f"witness slot order mismatch at slot {slot}")
            if not {"node_sequence", "directed_edges", "fallback"}.issubset(witness):
                raise RuntimeError(f"witness schema is incomplete at slot {slot}")
            nodes = [int(node) for node in witness["node_sequence"]]
            expected_edges = list(zip(nodes[:-1], nodes[1:]))
            actual_edges = [(int(u), int(v)) for u, v in witness["directed_edges"]]
            if actual_edges != expected_edges:
                raise RuntimeError(f"witness edge serialization mismatch at slot {slot}")
        shutil.copy2(source_syn, work_dir / "trajectories.pkl")
        shutil.copy2(source_witness, work_dir / "road_witnesses.pkl")
        decoder_report = json.loads((decoder_out / "protocol.json").read_text(encoding="utf-8"))
        shutil.rmtree(decoder_out)
        (work_dir / "DO_NOT_RELEASE.txt").write_text(
            "Research-only reproducible-noise epsilon sweep artifact. Do not publish as a certified DP release.\n",
            encoding="utf-8",
        )
        outputs = {
            name: sha256_file(work_dir / name)
            for name in ("trajectories.pkl", "road_witnesses.pkl", "dp_transcript.npz")
        }
        protocol = {
            "schema_version": 1,
            "classification": CLASSIFICATION,
            "algorithm_schema": ALGORITHM_SCHEMA,
            "algorithm_id": ALGORITHM_SCHEMA,
            "source_binding": source_binding,
            "dataset": config["name"],
            "dataset_status": config["status"],
            "data_spec": str(config["data"]),
            "input_record_count": config["capacity"],
            "private_input_hash_persisted": False,
            "public_slot_count": config["capacity"],
            "output_count": config["capacity"],
            "output_yield": 1.0,
            "bbox": config["bbox"],
            "public_osm": {
                "path": str(config["osm_cache"]),
                "sha256": sha256_file(config["osm_cache"]),
                "selection": "bbox_filter_then_highway_class_filter_then_optional_deterministic_spatial_thinning",
                "bbox_way_count": osm_bbox_count,
                "highway_filtered_way_count": osm_highway_count,
                "selected_way_count": len(osm),
                "max_ways": config["osm_max_ways"],
                "highway_classes": config["osm_highway_classes"],
            },
            "privacy": {
                "epsilon_total_rational": str(total_epsilon),
                "delta": 0,
                "budget_allocation": {k: str(v) for k, v in budgets.items()},
                "noise_seed": int(args.noise_seed),
                "warning": "Deterministic research randomness is not a certifiable production release.",
            },
            "postprocessing": {
                "decoder_seed": args.decoder_seed,
                "request_seed": request_seed,
                "betas": args.decoder_betas,
                "length_proposals": args.length_proposals,
                "length_log_penalty": args.length_log_penalty,
                "od_likelihood_ratio_cap": args.od_likelihood_ratio_cap,
                "occupancy_strength": args.occupancy_strength,
                "dwell_strength": args.dwell_strength,
                "hierarchy_likelihood_ratio_cap": args.hierarchy_likelihood_ratio_cap,
                "component_mode": args.component_mode,
            },
            "query_reports": {
                "base": base_samplers, "flow": flow_sampler, "q5": q5_sampler, "sinkhorn": sinkhorn
            },
            "schema_compatibility": {
                "base_blocks": ["endpoint", "od", "geometry", "point_length"],
                "flow_blocks": sorted(flow_release),
                "q5_blocks": list(Q5_BLOCK_NAMES),
                "nested_fine_regions": 384,
                "portals_per_fine_edge": 6,
                "witness_sidecar": "road_witnesses.pkl",
            },
            "decoder": {
                "algorithm": decoder_report.get("algorithm"),
                "route_release_schema": decoder_report.get("route_release_schema"),
                "release_object": decoder_report.get("release_object"),
                "fallback_count": decoder_report.get("fallback_count"),
                "length_conditioning": decoder_report.get("length_conditioning"),
                "beta_counts": decoder_report.get("beta_counts"),
                "active_hierarchy": decoder_report.get("active_hierarchy"),
            },
            "outputs": outputs,
            "elapsed_sec": time.time() - started,
        }
        write_json(work_dir / "protocol.json", protocol)
        write_json(work_dir / "manifest.json", {
            "schema_version": 1,
            "classification": CLASSIFICATION,
            "private_exact_query_persisted": False,
            "private_input_hash_persisted": False,
            "outputs": {
                name: sha256_file(work_dir / name)
                for name in (
                    "trajectories.pkl", "road_witnesses.pkl", "dp_transcript.npz",
                    "protocol.json", "DO_NOT_RELEASE.txt",
                )
            },
        })
        os.replace(work_dir, out_dir)
        logger.info("wrote research MTR release: %s", out_dir)
    except Exception:
        logger.exception("MTR research generation failed; staging retained at %s", work_dir)
        raise


if __name__ == "__main__":
    main()
