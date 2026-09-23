"""Memory-only shared-query MTR epsilon sweep.

The private exact query totals are computed once and retained only in this
process.  Every epsilon/seed pair still receives an independent exact-rational
noise draw and an independent decoder stream.  No exact total is serialized.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
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
    epsilon_slug,
    parse_fraction,
    public_path,
    sha256_file,
    write_json,
)
from generation.mtr.budget import allocate  # noqa: E402
from public_utils import thin_osm_ways_public  # noqa: E402
from generation.mtr.generate import (  # noqa: E402
    CLASSIFICATION,
    Q5_BLOCK_NAMES,
    _decode,
    _resolve_configuration,
    _source_binding,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="geolife")
    parser.add_argument(
        "--sweep-config", default="configs/mtr_epsilon_sweep_beijing.json"
    )
    parser.add_argument("--out-root", default="results/synthetic_datasets/mtr")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--epsilon",
        dest="epsilon_filters",
        action="append",
        help="Run only this exact rational epsilon; repeat for a subset.",
    )
    parser.add_argument(
        "--noise-seed",
        dest="seed_filters",
        action="append",
        type=int,
        help="Run only this registered noise seed; repeat for a subset.",
    )
    return parser


def _base_totals(real, endpoint_tree, coarse_tree, coarse_coords_norm):
    import dp_graph_voronoi_release as base
    import geometric_base_requests as geometric

    endpoint = np.zeros((2, 256), dtype=float)
    od = np.zeros((24, 24), dtype=float)
    geometry = np.zeros(
        (
            len(base.OD_DISTANCE_EDGES) - 1,
            len(geometric.CHORD_EDGES_KM) - 1,
            len(geometric.PATH_EDGES_KM) - 1,
        ),
        dtype=float,
    )
    point_length = np.zeros(
        (len(geometric.PATH_EDGES_KM) - 1, len(geometric.POINT_COUNT_EDGES) - 1),
        dtype=float,
    )
    for trajectory in real:
        array = np.asarray(trajectory, dtype=float)
        fine = endpoint_tree.query(array[[0, -1], :2])[1]
        coarse = coarse_tree.query(array[[0, -1], :2])[1]
        endpoint[0, int(fine[0])] += 0.5
        endpoint[1, int(fine[1])] += 0.5
        origin, destination = int(coarse[0]), int(coarse[1])
        od[origin, destination] += 1.0
        coarse_bucket = base.od_distance_bucket(
            origin, destination, coarse_coords_norm
        )
        chord_bucket = geometric.bucket(
            geometric.metric_distance_km(array[0, :2], array[-1, :2]),
            geometric.CHORD_EDGES_KM,
        )
        path_bucket = geometric.bucket(
            geometric.trajectory_length_km(array), geometric.PATH_EDGES_KM
        )
        geometry[coarse_bucket, chord_bucket, path_bucket] += 1.0
        point_bucket = min(
            max(
                int(
                    np.searchsorted(
                        geometric.POINT_COUNT_EDGES, len(array), side="right"
                    )
                    - 1
                ),
                0,
            ),
            len(geometric.POINT_COUNT_EDGES) - 2,
        )
        point_length[path_bucket, point_bucket] += 1.0
    return {
        "endpoint": endpoint,
        "od": od,
        "geometry": geometry,
        "point_length": point_length,
    }


def _release_base(totals, capacity: int, budgets: dict, seed: int):
    import geometric_base_requests as geometric

    rng = random.Random(int(seed) + 101)
    endpoint, endpoint_report = geometric.exact_probability(
        totals["endpoint"], budgets["endpoint"], 2, 2, capacity, rng
    )
    od, od_report = geometric.exact_probability(
        totals["od"], budgets["od"], 1, 1, capacity, rng
    )
    geometry, geometry_report = geometric.exact_probability(
        totals["geometry"], budgets["geometry"], 1, 1, capacity, rng
    )
    length, length_report = geometric.exact_probability(
        totals["point_length"], budgets["point_length"], 1, 1, capacity, rng
    )
    return {
        "endpoint": endpoint,
        "od": od,
        "geometry": geometry,
        "length": length,
        "samplers": {
            "endpoint": endpoint_report,
            "od": od_report,
            "geometry": geometry_report,
            "point_length": length_report,
        },
    }


def _portal_totals(real, coords, graph):
    import audit_two_level_semimarkov_dp as audit
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
        if quantized is not None:
            for name in portal.BLOCK_BUDGETS:
                totals[name] += quantized[name]
    return totals


def _release_portal(totals, capacity: int, epsilon: Fraction, seed: int):
    import certified_discrete_dp as certified
    import portal_fiber_route_release_development as portal

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


def _verified_existing(directory: Path, epsilon: str, seed: int, capacity: int) -> bool:
    manifest_path = directory / "manifest.json"
    protocol_path = directory / "protocol.json"
    if not manifest_path.is_file() or not protocol_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        for name, expected in manifest["outputs"].items():
            path = directory / name
            if not path.is_file() or sha256_file(path) != expected:
                return False
        return (
            protocol["classification"] == CLASSIFICATION
            and protocol["privacy"]["epsilon_total_rational"] == epsilon
            and int(protocol["privacy"]["noise_seed"]) == seed
            and int(protocol["output_count"]) == capacity
        )
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def _validate_decoder_outputs(source_syn: Path, source_witness: Path, capacity: int):
    with source_syn.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or len(records) != capacity:
        raise RuntimeError("decoder output does not fill the fixed public slot domain")
    del records
    with source_witness.open("rb") as handle:
        witnesses = pickle.load(handle)
    if not isinstance(witnesses, list) or len(witnesses) != capacity:
        raise RuntimeError("witness sidecar is not one-to-one with output slots")
    for slot, witness in enumerate(witnesses):
        if not isinstance(witness, dict) or int(witness.get("slot", -1)) != slot:
            raise RuntimeError(f"witness slot mismatch at {slot}")
        nodes = witness.get("node_sequence")
        edges = witness.get("directed_edges")
        if nodes is not None:
            expected = list(zip(nodes[:-1], nodes[1:]))
            if edges != expected:
                raise RuntimeError(f"witness serialization mismatch at {slot}")


def _resolve_filters(
    sweep: dict,
    epsilon_values: list[str] | None,
    seed_values: list[int] | None,
) -> tuple[set[str], set[int]]:
    """Validate an optional shard against the frozen sweep matrix."""
    registered_epsilons = [str(parse_fraction(value)) for value in sweep["epsilons"]]
    registered_seeds = [int(value) for value in sweep["noise_seeds"]]
    epsilon_filters = (
        {str(parse_fraction(value)) for value in epsilon_values}
        if epsilon_values
        else set(registered_epsilons)
    )
    seed_filters = (
        {int(value) for value in seed_values}
        if seed_values
        else set(registered_seeds)
    )
    unknown_epsilons = epsilon_filters.difference(registered_epsilons)
    unknown_seeds = seed_filters.difference(registered_seeds)
    if unknown_epsilons or unknown_seeds:
        raise ValueError(
            "filters must be members of the frozen sweep matrix: "
            f"unknown epsilons={sorted(unknown_epsilons)}, "
            f"unknown seeds={sorted(unknown_seeds)}"
        )
    return epsilon_filters, seed_filters


def main() -> None:
    args = build_parser().parse_args()
    add_runtime_paths()
    sweep = json.loads(public_path(args.sweep_config).read_text(encoding="utf-8"))
    epsilon_filters, seed_filters = _resolve_filters(
        sweep, args.epsilon_filters, args.seed_filters
    )
    config_args = argparse.Namespace(
        data=args.dataset,
        dataset_config=args.dataset,
        bbox=None,
        osm_cache=None,
        public_slot_count=None,
    )
    config = _resolve_configuration(config_args)
    source_binding = _source_binding()
    logger = configure_logging(
        PUBLIC_RELEASE / "logs" / "generation" / f"{args.dataset}_shared_query.log",
        "mtr-shared-query-sweep",
    )
    started_shared = time.time()

    import audit_two_level_semimarkov_dp as audit
    import compact_graph_flow_experiment as compact
    import dp_graph_voronoi_release as base
    import graph_voronoi_doptimal_support_probe as support
    import route_structure_potential_experiment as route
    from public_utils import filter_osm_ways_by_bbox, filter_osm_ways_by_highway, load_trajectories

    real = load_trajectories(str(config["data"]), limit=config["capacity"])
    if len(real) != config["capacity"]:
        raise RuntimeError("frozen input does not fill the public slot domain")
    with config["osm_cache"].open("rb") as handle:
        osm = pickle.load(handle)
    osm = filter_osm_ways_by_bbox(osm, config["bbox"])
    osm_bbox_count = len(osm)
    osm = filter_osm_ways_by_highway(osm, config["osm_highway_classes"])
    osm_highway_count = len(osm)
    if config["osm_max_ways"] > 0:
        osm = thin_osm_ways_public(osm, config["bbox"], config["osm_max_ways"])
    support.BBOX = config["bbox"]
    coords, graph = route.prepare_graph([], bbox=config["bbox"], osm_ways=osm, raw_graph=False)
    fine_nodes = support.farthest_point_landmarks(coords, 256)
    fine_coords = coords[np.asarray(fine_nodes, dtype=int)]
    fine_tree = cKDTree(fine_coords)
    coarse_nodes = support.farthest_point_landmarks(coords, 24)
    coarse_coords = coords[np.asarray(coarse_nodes, dtype=int)]
    coarse_tree = cKDTree(coarse_coords)
    fine_to_coarse = np.asarray(coarse_tree.query(fine_coords)[1], dtype=int)
    logger.info("aggregating private exact workloads in memory; nothing exact is serialized")
    base_totals = _base_totals(real, fine_tree, coarse_tree, support.standardized_xy(coarse_coords))
    context96, _ = audit.build_context(coords, graph, 24, 96, 4, 3)
    flow_totals = compact.aggregate(real, context96)
    portal_totals = _portal_totals(real, coords, graph)
    del (
        real,
        coords,
        graph,
        fine_tree,
        coarse_tree,
        context96,
        fine_nodes,
        fine_coords,
        coarse_nodes,
        coarse_coords,
    )
    gc.collect()
    logger.info("shared aggregation ready in %.1f sec", time.time() - started_shared)

    out_root = public_path(args.out_root) / args.dataset
    for epsilon_text in sweep["epsilons"]:
        epsilon = parse_fraction(epsilon_text)
        if str(epsilon) not in epsilon_filters:
            continue
        budgets = allocate(epsilon)
        for seed_value in sweep["noise_seeds"]:
            seed = int(seed_value)
            if seed not in seed_filters:
                continue
            decoder_seed = seed + int(sweep["decoder_seed_offset"])
            out_dir = out_root / f"eps_{epsilon_slug(epsilon)}" / f"seed_{seed}"
            if out_dir.exists():
                if args.resume and _verified_existing(
                    out_dir, str(epsilon), seed, config["capacity"]
                ):
                    logger.info("skip verified %s", out_dir)
                    continue
                raise FileExistsError(f"unverified output directory exists: {out_dir}")
            work_dir = out_dir.parent / f".{out_dir.name}.research-staging"
            if work_dir.exists():
                raise FileExistsError(f"stale staging directory exists: {work_dir}")
            work_dir.mkdir(parents=True)
            run_started = time.time()
            logger.info("release epsilon=%s seed=%d", epsilon, seed)
            try:
                base_release = _release_base(base_totals, config["capacity"], budgets, seed)
                base_release["od"], sinkhorn = base.sinkhorn_od_projection(
                    base_release["od"], base_release["endpoint"], fine_to_coarse, 24
                )
                base_samplers = base_release.pop("samplers")
                flow_release, flow_sampler = compact.release(
                    flow_totals,
                    config["capacity"],
                    random.Random(seed + 202),
                    budgets["compact_graph_flow"],
                )
                q5_release, q5_sampler = _release_portal(
                    portal_totals, config["capacity"], budgets["portal_fiber_q5"], seed
                )
                transcript = {f"base_{k}": v for k, v in base_release.items()}
                transcript.update({f"flow_{k}": v for k, v in flow_release.items()})
                transcript.update({f"q5_{k}": v for k, v in q5_release.items()})
                np.savez_compressed(work_dir / "dp_transcript.npz", **transcript)
                request_seed = decoder_seed + 483729
                decoder_out = _decode(
                    work_dir,
                    base_release,
                    flow_release,
                    q5_release,
                    osm,
                    sha256_file(config["osm_cache"]),
                    config["capacity"],
                    epsilon,
                    budgets["compact_graph_flow"],
                    budgets["portal_fiber_q5"],
                    decoder_seed,
                    request_seed,
                    config["bbox"],
                    occupancy_strength=float(
                        sweep["postprocessing"]["occupancy_strength"]
                    ),
                    dwell_strength=float(
                        sweep["postprocessing"]["dwell_strength"]
                    ),
                    hierarchy_likelihood_ratio_cap=float(
                        sweep["postprocessing"]["hierarchy_likelihood_ratio_cap"]
                    ),
                )
                source_syn = decoder_out / "dp_gsrt_portal_qrsp.pkl"
                source_witness = decoder_out / "road_witnesses.pkl"
                _validate_decoder_outputs(source_syn, source_witness, config["capacity"])
                shutil.copy2(source_syn, work_dir / "trajectories.pkl")
                shutil.copy2(source_witness, work_dir / "road_witnesses.pkl")
                decoder_report = json.loads(
                    (decoder_out / "protocol.json").read_text(encoding="utf-8")
                )
                shutil.rmtree(decoder_out)
                (work_dir / "DO_NOT_RELEASE.txt").write_text(
                    "Research-only reproducible-noise epsilon sweep artifact. Do not publish as a certified DP release.\n",
                    encoding="utf-8",
                )
                output_hashes = {
                    name: sha256_file(work_dir / name)
                    for name in ("trajectories.pkl", "road_witnesses.pkl", "dp_transcript.npz")
                }
                protocol = {
                    "schema_version": 1,
                    "classification": CLASSIFICATION,
                    "algorithm_id": "mtr-gsrt-active-hierarchy-v1",
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
                        "epsilon_total_rational": str(epsilon),
                        "delta": 0,
                        "budget_allocation": {k: str(v) for k, v in budgets.items()},
                        "noise_seed": seed,
                        "warning": "Deterministic research randomness is not a certifiable production release.",
                    },
                    "postprocessing": {
                        "decoder_seed": decoder_seed,
                        "request_seed": request_seed,
                        "occupancy_strength": float(
                            sweep["postprocessing"]["occupancy_strength"]
                        ),
                        "dwell_strength": float(
                            sweep["postprocessing"]["dwell_strength"]
                        ),
                        "hierarchy_likelihood_ratio_cap": float(
                            sweep["postprocessing"]["hierarchy_likelihood_ratio_cap"]
                        ),
                    },
                    "query_reports": {
                        "base": base_samplers,
                        "flow": flow_sampler,
                        "q5": q5_sampler,
                        "sinkhorn": sinkhorn,
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
                    },
                    "shared_query_execution": {
                        "exact_totals_reused_in_memory": True,
                        "exact_totals_serialized": False,
                    },
                    "outputs": output_hashes,
                    "elapsed_sec": time.time() - run_started,
                }
                write_json(work_dir / "protocol.json", protocol)
                write_json(
                    work_dir / "manifest.json",
                    {
                        "schema_version": 1,
                        "classification": CLASSIFICATION,
                        "private_exact_query_persisted": False,
                        "private_input_hash_persisted": False,
                        "outputs": {
                            name: sha256_file(work_dir / name)
                            for name in (
                                "trajectories.pkl",
                                "road_witnesses.pkl",
                                "dp_transcript.npz",
                                "protocol.json",
                                "DO_NOT_RELEASE.txt",
                            )
                        },
                    },
                )
                out_dir.parent.mkdir(parents=True, exist_ok=True)
                os.replace(work_dir, out_dir)
                logger.info("wrote %s in %.1f sec", out_dir, time.time() - run_started)
            except Exception:
                logger.exception("run failed; staging retained at %s", work_dir)
                raise


if __name__ == "__main__":
    main()
