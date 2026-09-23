"""Verify one hash-bound actual-road route evaluation chunk."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_road_route_chunks import _load_chunk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--dir", required=True, type=Path)
    args = parser.parse_args()
    directory = args.dir.resolve()
    _load_chunk(directory.parent, args.method)
    print(f"verified {args.method}: {directory}")


if __name__ == "__main__":
    main()
