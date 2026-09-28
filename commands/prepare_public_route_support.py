"""Construct public road/region support; never load a trajectory corpus."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"evaluation"))
sys.path.insert(0, str(ROOT/"evaluation/evaluation"))
from generation.common.runtime import dataset_config, public_path
from evaluate_road_choice import _edge_endpoint_regions

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--osm-cache")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = dataset_config(args.dataset_config)
    if args.osm_cache:
        config["osm_cache"] = args.osm_cache
    network, out = public_path(args.network), public_path(args.out)
    side = out.with_suffix(".manifest.json")
    if out.exists() or side.exists():
        raise FileExistsError(out)
    payload, diagnostics = {}, {}
    canonical = None
    for resolution in (24,96,384):
        regions, edge_nodes, carrier, diagnostic = _edge_endpoint_regions(network,config,resolution)
        if canonical is not None and canonical != edge_nodes:
            raise ValueError("Canonical road identities changed across public resolutions")
        canonical = edge_nodes
        votes = defaultdict(Counter)
        for edge, (source,target) in edge_nodes.items():
            first, second = regions[edge]
            votes[source][first] += 1
            votes[target][second] += 1
        payload[f"labels{resolution}"] = {
            node: min(counts, key=lambda label:(-counts[label],label))
            for node,counts in votes.items()}
        diagnostics[str(resolution)] = diagnostic
    successors = defaultdict(set)
    for source,target in canonical.values():
        successors[source].add(target)
    payload.update(classification="PUBLIC_GRAPH_DERIVED_SUPPORT_NO_TRAJECTORY_ROWS",
                   edge_nodes=canonical, outdegree={node:len(values) for node,values in successors.items()},
                   portals={})
    serialized = gzip.compress(pickle.dumps(payload,protocol=pickle.HIGHEST_PROTOCOL),mtime=0)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_bytes(serialized)
    side.write_text(json.dumps(dict(
        classification=payload["classification"], network_sha256=hashlib.sha256(network.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(serialized).hexdigest(), diagnostics=diagnostics),ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"out":str(out),"edges":len(canonical),"sha256":hashlib.sha256(serialized).hexdigest()}))

if __name__ == "__main__":
    main()
