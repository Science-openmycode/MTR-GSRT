"""Direct and nearest-edge road rows are executed, not empty placeholders."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import shapely
from scipy.spatial import cKDTree


SOURCE = Path(__file__).resolve().parents[1] / "evaluation/materialize_direct_road_rows.py"
SPEC = importlib.util.spec_from_file_location("direct_road_rows_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_direct_original_requires_observed_directed_adjacency():
    xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    tree = cKDTree(xy)
    ids = np.asarray([1, 2, 3])
    edges = {(1, 2): 10, (2, 3): 11}
    assert module.original_route(xy, tree, ids, edges) == (10, 11)
    assert module.original_route(xy, tree, ids, {(1, 2): 10}) == ()


def test_nearest_edge_executes_and_retains_disconnected_sequence():
    segments = [shapely.LineString([(0, 0), (1, 0)]),
                shapely.LineString([(5, 0), (6, 0)])]
    tree = shapely.STRtree(segments)
    ids = np.asarray([10, 20])
    points = np.asarray([[0.1, 0.0], [0.2, 0.0], [5.1, 0.0]])
    assert module.nearest_route(points, tree, ids, 1.0) == (10, 20)
    assert module.nearest_route(np.asarray([[0.1, 2.0], [5.1, 0.0]]), tree, ids, 1.0) == ()
