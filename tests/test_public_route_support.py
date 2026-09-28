from pathlib import Path
import gzip
import importlib.util
import pickle
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"evaluation"))
from route_metric_core import normalize_routes

class PublicSupportTest(unittest.TestCase):
    def test_public_asset_contains_no_trajectory_rows(self):
        with gzip.open(ROOT/"public_assets/ordered_portal_route_cache.pkl.gz","rb") as handle:
            cache = pickle.load(handle)
        self.assertNotIn("rows", cache)
        self.assertTrue(cache["edge_nodes"])
        self.assertTrue(cache["labels96"])

    def test_unknown_edge_is_rejected_not_silently_dropped(self):
        cache = {"edge_nodes":{1:(1,2),2:(2,3)}}
        self.assertEqual(normalize_routes([{"accepted":True,"cpath":(1,2)}],cache),[((1,2),(2,3))])
        with self.assertRaises(ValueError):
            normalize_routes([{"accepted":True,"cpath":(1,999,2)}],cache)
        with self.assertRaises(ValueError):
            normalize_routes([(1,999,2)],cache)
        self.assertEqual(normalize_routes([{"accepted":False,"cpath":(999,)}],cache),[()])

if __name__ == "__main__":
    unittest.main()
