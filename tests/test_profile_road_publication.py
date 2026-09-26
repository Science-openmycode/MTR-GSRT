import hashlib
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from run_profile_experiment import ROOT, publication_road_metrics, resolve_input


class PublicationRoadMetricTests(unittest.TestCase):
    def test_relative_input_is_package_relative(self):
        self.assertEqual(resolve_input("public_assets/example.osm"),
                         (ROOT / "public_assets" / "example.osm").resolve())

    def test_coordinate_only_retains_projection_metrics(self):
        metrics = {"route_compatible_yield": 0.25, "directed_road_validity": 0.75}
        self.assertEqual(publication_road_metrics(metrics, Path("synthetic.pkl"), None),
                         (0.25, 0.75, "coordinate_projection"))

    def test_hash_bound_witness_is_publication_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic = root / "trajectories.pkl"
            witness = root / "road_witnesses.pkl"
            synthetic.write_bytes(b"coordinate release")
            witness.write_bytes(b"directed witness")
            outputs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (synthetic, witness)}
            (root / "manifest.json").write_text(json.dumps({"outputs": outputs}))
            metrics = {"route_compatible_yield": 0.01,
                       "directed_road_validity": 0.33, "witness_valid": 1.0}
            self.assertEqual(publication_road_metrics(metrics, synthetic, witness),
                             (1.0, 1.0, "directed_witness"))
            witness.write_bytes(b"tampered witness")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                publication_road_metrics(metrics, synthetic, witness)

    def test_derived_route_checks_public_edges_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic = root / "coordinates.pkl"
            routes = root / "routes.pkl"
            cache = root / "edge_cache.pkl"
            synthetic.write_bytes(b"derived coordinates")
            routes.write_bytes(pickle.dumps([[(1, 2), (2, 3)], [(1, 9)]]))
            cache.write_bytes(pickle.dumps({"edge_nodes": {1: (1, 2), 2: (2, 3)}}))
            payload = {
                "schema": "road-route-coordinate-derivation-v1",
                "record_count": 2,
                "coordinates": {"sha256": hashlib.sha256(synthetic.read_bytes()).hexdigest()},
                "route_source": {"sha256": hashlib.sha256(routes.read_bytes()).hexdigest()},
                "edge_cache": {"sha256": hashlib.sha256(cache.read_bytes()).hexdigest()},
            }
            (root / "coordinates.pkl.manifest.json").write_text(json.dumps(payload))
            self.assertEqual(publication_road_metrics({}, synthetic, None, routes, cache, 2),
                             (0.5, 0.5, "derived_directed_routes"))
            with self.assertRaisesRegex(ValueError, "slot count"):
                publication_road_metrics({}, synthetic, None, routes, cache, 3)


if __name__ == "__main__":
    unittest.main()
