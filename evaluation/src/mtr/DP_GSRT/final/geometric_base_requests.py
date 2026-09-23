"""Exact-DP geometric request model for short OD, loops, and target path length."""
from __future__ import annotations

import math
import random
from fractions import Fraction

import numpy as np

import cross_dataset_ara_mode_benchmark as cdb
import dp_graph_voronoi_release as base
from certified_discrete_dp import add_exact_discrete_laplace


CHORD_EDGES_KM = np.asarray(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 10.5, 13.0, 17.5, 22.0, math.inf]
)
PATH_EDGES_KM = np.asarray([0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, math.inf])
POINT_COUNT_EDGES = np.asarray(
    [2, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65537],
    dtype=int,
)
BASE_BUDGETS = {
    "endpoint": Fraction(8, 25),
    "od": Fraction(33, 100),
    "geometry": Fraction(4, 25),
    "point_length": Fraction(19, 100),
}
if sum(BASE_BUDGETS.values(), Fraction(0, 1)) != 1:
    raise RuntimeError("Geometric base budgets must sum exactly to one")


def metric_distance_km(a: np.ndarray, b: np.ndarray) -> float:
    latitude = float(0.5 * (a[0] + b[0]))
    scale = np.asarray([111.32, 111.32 * np.cos(np.deg2rad(latitude))])
    return float(np.linalg.norm((np.asarray(b) - np.asarray(a)) * scale))


def trajectory_length_km(trajectory: np.ndarray) -> float:
    array = np.asarray(trajectory, dtype=float)[:, :2]
    if len(array) < 2:
        return 0.0
    latitude = float(np.mean(array[:, 0]))
    scale = np.asarray([111.32, 111.32 * np.cos(np.deg2rad(latitude))])
    return float(np.linalg.norm(np.diff(array, axis=0) * scale, axis=1).sum())


def bucket(value: float, edges: np.ndarray) -> int:
    return min(max(int(np.searchsorted(edges, float(value), side="right") - 1), 0), len(edges) - 2)


def exact_probability(counts, epsilon: Fraction, sensitivity: int, scale: int, capacity: int, rng):
    scaled = np.rint(np.asarray(counts, dtype=float) * int(scale)).astype(np.int64)
    noisy, report = add_exact_discrete_laplace(
        scaled,
        epsilon_numerator=epsilon.numerator,
        epsilon_denominator=epsilon.denominator,
        sensitivity=int(sensitivity),
        rng=rng,
    )
    projected = base.project_simplex(noisy.astype(float), float(capacity) * int(scale))
    probability = projected / max(float(projected.sum()), 1e-15)
    return probability, report


def normalized_probability(values: np.ndarray) -> np.ndarray:
    """Return a finite nonnegative vector whose float64 sum is exactly one."""
    probability = np.asarray(values, dtype=np.float64).ravel().copy()
    probability[~np.isfinite(probability)] = 0.0
    np.maximum(probability, 0.0, out=probability)
    total = float(probability.sum(dtype=np.float64))
    if total <= 0.0:
        probability.fill(1.0 / max(len(probability), 1))
    else:
        probability /= total
    residual = 1.0 - float(probability[:-1].sum(dtype=np.float64))
    if residual < 0.0:
        scale = np.nextafter(1.0, 0.0) / float(
            probability[:-1].sum(dtype=np.float64)
        )
        probability[:-1] *= scale
        residual = 1.0 - float(probability[:-1].sum(dtype=np.float64))
    probability[-1] = residual
    return probability


def fit_geometric_measurements(
    real,
    endpoint_tree,
    coarse_tree,
    coarse_coords_norm,
    *,
    fine_count: int,
    coarse_count: int,
    capacity: int,
    exact_rng,
):
    endpoint = np.zeros((2, int(fine_count)), dtype=float)
    od = np.zeros((int(coarse_count), int(coarse_count)), dtype=float)
    geometry = np.zeros(
        (len(base.OD_DISTANCE_EDGES) - 1, len(CHORD_EDGES_KM) - 1, len(PATH_EDGES_KM) - 1),
        dtype=float,
    )
    point_length = np.zeros(
        (len(PATH_EDGES_KM) - 1, len(POINT_COUNT_EDGES) - 1), dtype=float
    )
    for trajectory in real:
        array = np.asarray(trajectory, dtype=float)
        fine = endpoint_tree.query(array[[0, -1], :2])[1]
        coarse = coarse_tree.query(array[[0, -1], :2])[1]
        endpoint[0, int(fine[0])] += 0.5
        endpoint[1, int(fine[1])] += 0.5
        origin, destination = int(coarse[0]), int(coarse[1])
        od[origin, destination] += 1.0
        coarse_bucket = base.od_distance_bucket(origin, destination, coarse_coords_norm)
        chord_bucket = bucket(metric_distance_km(array[0, :2], array[-1, :2]), CHORD_EDGES_KM)
        path_bucket = bucket(trajectory_length_km(array), PATH_EDGES_KM)
        geometry[coarse_bucket, chord_bucket, path_bucket] += 1.0
        point_bucket = min(
            max(int(np.searchsorted(POINT_COUNT_EDGES, len(array), side="right") - 1), 0),
            len(POINT_COUNT_EDGES) - 2,
        )
        point_length[path_bucket, point_bucket] += 1.0
    endpoint_release, endpoint_report = exact_probability(
        endpoint, BASE_BUDGETS["endpoint"], 2, 2, capacity, exact_rng
    )
    od_release, od_report = exact_probability(od, BASE_BUDGETS["od"], 1, 1, capacity, exact_rng)
    geometry_release, geometry_report = exact_probability(
        geometry, BASE_BUDGETS["geometry"], 1, 1, capacity, exact_rng
    )
    length_release, length_report = exact_probability(
        point_length, BASE_BUDGETS["point_length"], 1, 1, capacity, exact_rng
    )
    return {
        "endpoint": endpoint_release,
        "od": od_release,
        "geometry": geometry_release,
        "length": length_release,
        "samplers": {
            "endpoint": endpoint_report,
            "od": od_report,
            "geometry": geometry_report,
            "point_length": length_report,
        },
    }


def sample_weighted(indices: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> int:
    probability = np.asarray(weights, dtype=float)
    total = float(probability.sum())
    if total <= 0.0:
        probability = np.full(len(indices), 1.0 / max(len(indices), 1), dtype=float)
    else:
        probability /= total
        probability /= probability.sum()
    return int(rng.choice(indices, p=probability))


def sample_geometric_requests(
    measurements,
    fine_to_coarse,
    coarse_coords_norm,
    fine_nodes,
    endpoint_coords,
    graph_nodes,
    graph_coords,
    graph_to_fine,
    graph_to_coarse,
    graph_metric_tree,
    graph_metric_coords,
    *,
    count: int,
    rng: np.random.Generator,
):
    od = np.asarray(measurements["od"], dtype=float)
    od_draws = rng.choice(od.size, size=int(count), p=od.ravel())
    fine_sizes = np.maximum(
        np.bincount(np.asarray(graph_to_fine, dtype=int), minlength=len(fine_nodes)), 1
    )
    coarse_sizes = np.maximum(
        np.bincount(np.asarray(graph_to_coarse, dtype=int), minlength=od.shape[0]), 1
    )
    graph_nodes = np.asarray(graph_nodes, dtype=int)
    graph_coords = np.asarray(graph_coords, dtype=float)
    graph_to_fine = np.asarray(graph_to_fine, dtype=int)
    graph_to_coarse = np.asarray(graph_to_coarse, dtype=int)

    def distances_from(point, indices, chunk_size=4096):
        """Evaluate the existing equirectangular metric without Python lists."""
        indices = np.asarray(indices, dtype=int)
        output = np.empty(len(indices), dtype=float)
        point = np.asarray(point, dtype=float)
        for start in range(0, len(indices), int(chunk_size)):
            stop = min(start + int(chunk_size), len(indices))
            values = graph_coords[indices[start:stop]]
            latitude = 0.5 * (point[0] + values[:, 0])
            north = (values[:, 0] - point[0]) * 111.32
            east = (
                (values[:, 1] - point[1])
                * 111.32
                * np.cos(np.deg2rad(latitude))
            )
            output[start:stop] = np.hypot(north, east)
        return output

    requests = []
    for draw in od_draws:
        origin, destination = divmod(int(draw), od.shape[1])
        origin_indices = base.conditional_indices(fine_to_coarse, origin)
        destination_indices = base.conditional_indices(fine_to_coarse, destination)
        fine_origin = sample_weighted(origin_indices, measurements["endpoint"][0, origin_indices], rng)
        coarse_bucket = base.od_distance_bucket(origin, destination, coarse_coords_norm)
        joint = np.asarray(measurements["geometry"][coarse_bucket], dtype=float)
        choice = int(rng.choice(joint.size, p=normalized_probability(joint)))
        chord_bucket, path_bucket = divmod(choice, joint.shape[1])
        path_low, path_high = PATH_EDGES_KM[path_bucket], PATH_EDGES_KM[path_bucket + 1]
        if np.isfinite(path_high):
            target_path_km = float(rng.uniform(path_low, path_high))
        else:
            target_path_km = float(path_low + rng.exponential(max(path_low * 0.5, 10.0)))
        low, high = CHORD_EDGES_KM[chord_bucket], CHORD_EDGES_KM[chord_bucket + 1]
        # A short chord with a much longer path is a return-like trip. Mapping it
        # to one public node lets the candidate stage construct a genuine loop.
        return_like = (
            origin == destination
            and chord_bucket == 0
            and target_path_km >= max(2.0, 3.0 * high)
        )
        coarse_nodes = graph_nodes[graph_to_coarse == destination]
        distances = distances_from(endpoint_coords[fine_origin], coarse_nodes)
        eligible = (distances >= low) & (distances < high)
        if not np.any(eligible):
            origin_metric = np.asarray(graph_metric_coords)[int(fine_nodes[fine_origin])]
            if np.isfinite(high):
                global_nodes = np.asarray(
                    graph_metric_tree.query_ball_point(origin_metric, float(high) * 1.02),
                    dtype=int,
                )
            else:
                global_nodes = graph_nodes
            global_distances = distances_from(
                endpoint_coords[fine_origin], global_nodes
            )
            global_eligible = (global_distances >= low) & (global_distances < high)
            if np.any(global_eligible):
                allowed_nodes = global_nodes[global_eligible]
                allowed_distances = global_distances[global_eligible]
            else:
                target = 0.5 * (
                    low + (high if np.isfinite(high) else max(low + 10.0, distances.max()))
                )
                nearest = int(np.argmin(np.abs(distances - target)))
                allowed_nodes = coarse_nodes[[nearest]]
                allowed_distances = distances[[nearest]]
        else:
            allowed_nodes = coarse_nodes[eligible]
            allowed_distances = distances[eligible]
        if np.isfinite(high):
            target_chord_km = float(rng.uniform(low, high))
        else:
            target_chord_km = float(low + rng.exponential(max(0.25 * low, 2.0)))
        # Sampling uniformly over road nodes biases toward the outer radius of
        # an annulus. Restrict to a public nearest-radius shortlist first.
        shortlist_size = min(48, len(allowed_nodes))
        shortlist = np.argpartition(
            np.abs(allowed_distances - target_chord_km), shortlist_size - 1
        )[:shortlist_size]
        allowed_nodes = allowed_nodes[shortlist]
        allowed_fine = graph_to_fine[allowed_nodes]
        allowed_coarse = graph_to_coarse[allowed_nodes]
        if return_like:
            destination_node = int(fine_nodes[fine_origin])
        else:
            weights = (
                measurements["endpoint"][1, allowed_fine]
                * od[origin, allowed_coarse]
                / fine_sizes[allowed_fine]
                / coarse_sizes[allowed_coarse]
            )
            destination_node = sample_weighted(
                allowed_nodes, weights, rng
            )
        point_probability = np.asarray(measurements["length"][path_bucket], dtype=float)
        if float(point_probability.sum()) <= 0.0:
            point_probability = np.asarray(measurements["length"], dtype=float).sum(axis=0)
        point_probability /= max(float(point_probability.sum()), 1e-15)
        point_bucket = int(rng.choice(len(point_probability), p=point_probability))
        point_low = int(POINT_COUNT_EDGES[point_bucket])
        point_high = int(POINT_COUNT_EDGES[point_bucket + 1])
        point_count = int(
            np.clip(
                np.floor(np.exp(rng.uniform(np.log(point_low), np.log(point_high)))),
                point_low,
                point_high - 1,
            )
        )
        requests.append(
            (int(fine_nodes[fine_origin]), destination_node, point_count, target_path_km)
        )
    return requests
