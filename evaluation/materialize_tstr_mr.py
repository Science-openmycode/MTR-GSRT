from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path).resolve()


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def run(command: list[str]) -> None:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize equal-point and full-road views for an executed FMM/STMatch matrix."
    )
    parser.add_argument("--source", action="append", type=named_path, required=True)
    parser.add_argument("--fmm-dir", type=Path, required=True)
    parser.add_argument("--stmatch-dir", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    for router, match_root in (("FMM", args.fmm_dir), ("STMatch", args.stmatch_dir)):
        for name, source in args.source:
            matches = match_root.resolve() / "matched_paths" / f"{name}.pkl.gz"
            if not matches.is_file():
                raise FileNotFoundError(matches)
            for view, sampling in (("generic", "source-count"), ("road", "edge-vertices")):
                output = args.out_dir.resolve() / view / router.lower() / f"{slug(name)}.pkl"
                run([
                    sys.executable, str(HERE / "materialize_public_route.py"),
                    "--source", str(source), "--matches", str(matches),
                    "--network", str(args.network.resolve()), "--router", router,
                    "--sampling", sampling, "--out", str(output),
                ])


if __name__ == "__main__":
    main()
