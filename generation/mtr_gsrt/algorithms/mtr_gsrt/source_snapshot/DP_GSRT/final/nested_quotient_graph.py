"""Public nested graph-Voronoi quotients for conservative coarse-to-fine lifts."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

import graph_voronoi_doptimal_support_probe as support
import dp_graph_flow_routing_probe as flow
from audit_two_level_semimarkov_dp import build_context
from two_level_semimarkov_query import QueryContext


def allocate_fine_regions(node_coarse: np.ndarray, coarse_regions: int, fine_regions: int) -> np.ndarray:
    counts = np.bincount(np.asarray(node_coarse, dtype=int), minlength=int(coarse_regions))
    if np.any(counts <= 0) or int(fine_regions) < int(coarse_regions):
        raise ValueError("Nested quotient requires nonempty coarse cells and at least one fine cell each")
    quota = counts.astype(float) / float(counts.sum()) * int(fine_regions)
    allocation = np.maximum(np.floor(quota).astype(int), 1)
    while int(allocation.sum()) < int(fine_regions):
        score = quota - allocation
        allocation[int(np.argmax(score))] += 1
    while int(allocation.sum()) > int(fine_regions):
        removable = np.flatnonzero(allocation > 1)
        if not len(removable):
            raise RuntimeError("Unable to satisfy nested fine-region allocation")
        score = allocation[removable] - quota[removable]
        allocation[int(removable[int(np.argmax(score))])] -= 1
    if np.any(allocation > counts):
        raise ValueError("A coarse cell has fewer road nodes than allocated fine regions")
    return allocation


def nested_graph_voronoi_labels(
    graph: dict,
    coords: np.ndarray,
    node_coarse: np.ndarray,
    allocation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_coarse = np.asarray(node_coarse, dtype=int)
    labels = np.full(len(coords), -1, dtype=int)
    centers, parents = [], []
    next_label = 0
    for coarse, count in enumerate(np.asarray(allocation, dtype=int)):
        nodes = np.flatnonzero(node_coarse == int(coarse))
        local_centers = support.farthest_point_landmarks(coords[nodes], int(count))
        global_centers = nodes[np.asarray(local_centers, dtype=int)]
        queue = []
        distance = {}
        for offset, node in enumerate(global_centers):
            label = next_label + int(offset)
            distance[int(node)] = (0.0, label)
            heapq.heappush(queue, (0.0, label, int(node)))
            centers.append(int(node))
            parents.append(int(coarse))
        while queue:
            current_distance, label, node = heapq.heappop(queue)
            if distance.get(int(node)) != (current_distance, label):
                continue
            labels[int(node)] = int(label)
            for neighbor, weight in graph[int(node)]:
                neighbor = int(neighbor)
                if node_coarse[neighbor] != int(coarse):
                    continue
                candidate = (current_distance + float(weight), int(label))
                if candidate < distance.get(neighbor, (float("inf"), 2**31 - 1)):
                    distance[neighbor] = candidate
                    heapq.heappush(queue, (candidate[0], candidate[1], neighbor))
        if np.any(labels[nodes] < 0):
            raise RuntimeError(f"Coarse region {coarse} is disconnected under the public partition")
        next_label += int(count)
    if np.any(labels < 0) or next_label != int(allocation.sum()):
        raise RuntimeError("Nested graph partition did not label every road node")
    return labels, np.asarray(centers, dtype=int), np.asarray(parents, dtype=int)


def build_nested_context(
    coords: np.ndarray,
    graph: dict,
    coarse_regions: int = 24,
    fine_regions: int = 384,
    macros: int = 4,
    phases: int = 3,
) -> tuple[QueryContext, dict, np.ndarray]:
    coarse_context, _ = build_context(
        coords, graph, int(coarse_regions), 96, int(macros), int(phases)
    )
    allocation = allocate_fine_regions(
        coarse_context.node_coarse, int(coarse_regions), int(fine_regions)
    )
    node_fine, fine_nodes, fine_to_coarse = nested_graph_voronoi_labels(
        graph, coords, coarse_context.node_coarse, allocation
    )
    fine_coords = coords[fine_nodes]
    fine_edges, fine_edge_index, fine_public_mass, _, fine_adjacency = flow.region_graph_and_prior(
        graph, coords, node_fine, int(fine_regions)
    )
    fine_importance = 1.0 / np.sqrt(np.maximum(np.asarray(fine_public_mass, dtype=float), 1e-15))
    fine_importance /= max(float(np.median(fine_importance)), 1e-15)
    fine_importance = np.clip(fine_importance, 0.25, 4.0)
    context = QueryContext(
        road_tree=coarse_context.road_tree,
        road_edge_tree=coarse_context.road_edge_tree,
        node_coarse=coarse_context.node_coarse,
        node_fine=node_fine,
        coarse_tree=coarse_context.coarse_tree,
        fine_tree=cKDTree(fine_coords),
        coarse_adjacency=coarse_context.coarse_adjacency,
        fine_adjacency=fine_adjacency,
        coarse_edge_index=coarse_context.coarse_edge_index,
        fine_edge_index=fine_edge_index,
        coarse_to_macro=coarse_context.coarse_to_macro,
        fine_importance=fine_importance,
        phases=int(phases),
        coarse_regions=int(coarse_regions),
        fine_regions=int(fine_regions),
        macro_regions=int(macros),
    )
    observed_parent = np.full(int(fine_regions), -1, dtype=int)
    for fine in range(int(fine_regions)):
        coarse_values = np.unique(context.node_coarse[context.node_fine == fine])
        if len(coarse_values) != 1:
            raise RuntimeError("Nested fine region crosses a coarse parent")
        observed_parent[fine] = int(coarse_values[0])
    if not np.array_equal(observed_parent, fine_to_coarse):
        raise RuntimeError("Nested parent labels disagree with the construction manifest")
    diagnostics = {
        "road_nodes": int(len(coords)),
        "coarse_regions": int(coarse_regions),
        "fine_regions": int(fine_regions),
        "coarse_directed_edges": int(len(context.coarse_edge_index)),
        "fine_directed_edges": int(len(fine_edges)),
        "macro_regions": int(macros),
        "phases": int(phases),
        "fine_regions_per_coarse_min": int(allocation.min()),
        "fine_regions_per_coarse_max": int(allocation.max()),
        "parent_mismatch_count": 0,
    }
    return context, diagnostics, fine_to_coarse
