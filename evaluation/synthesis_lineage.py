"""Local research provenance for executed train-only generation, not a DP artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "executed_train_only_generation_v1"


def bind_split(path: Path, train_sha: str, test_sha: str) -> dict:
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    if (record.get("schema") != "verified_disjoint_split_v1" or record.get("status") != "passed"
            or not record.get("all_source_indices_used_once")
            or record.get("train_sha256") != train_sha or record.get("test_sha256") != test_sha):
        raise ValueError("Missing or mismatched disjoint split audit")
    return {"record_sha256": sha(path), "full_sha256": record["full_sha256"],
            "train_count": record["train_count"], "test_count": record["test_count"]}


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_digest(root: Path) -> str:
    """Bind the generator dependency closure, independent of checkout line endings."""
    digest = hashlib.sha256()
    paths = sorted(p for p in (root / "generation").rglob("*")
                   if p.is_file() and p.suffix in {".py", ".json"})
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def validate_record(path: Path, train_sha: str, test_sha: str, release_sha: str) -> dict:
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    if record.get("schema") != SCHEMA or record.get("status") != "passed":
        raise ValueError("Lineage requires a successfully executed generator record")
    if record.get("train_sha256") != train_sha or record.get("test_sha256") != test_sha:
        raise ValueError("Generator lineage uses different train/test inputs")
    if train_sha == test_sha:
        raise ValueError("Identical training and test inputs")
    if release_sha not in record.get("outputs", {}).values():
        raise ValueError("Generator lineage does not bind this base release")
    if record.get("returncode") != 0 or not record.get("source_sha256"):
        raise ValueError("Incomplete execution/source evidence")
    if record.get("input_policy") != "train_only_explicit_generator_input":
        raise ValueError("Unknown generator input policy")
    return {"record_sha256": sha(path), "source_sha256": record["source_sha256"],
            "release_sha256": release_sha, "generator": record["generator"],
            "evidence_scope": "executed audited CLI input binding; not a formal noninterference proof"}


def bind_records(values: list[str], root: Path, binding: dict, require: bool = False) -> dict:
    records = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--synthesis-lineage expects MEASUREMENT=RECORD.json")
        name, raw_path = value.split("=", 1)
        if name in records or name not in binding["base_releases"]:
            raise ValueError(f"Duplicate or unknown lineage method: {name}")
        path = Path(raw_path)
        path = path if path.is_absolute() else root / path
        records[name] = validate_record(path, binding["train_sha256"], binding["test_sha256"],
                                        binding["base_releases"][name])
    if require and set(records) != set(binding["base_releases"]):
        raise ValueError("Strict TSTR requires executed synthesis lineage for every measurement")
    return records
