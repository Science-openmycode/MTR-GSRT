from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


def _find_public_release(start: Path) -> Path:
    """Find the release root by content, without assuming a fixed depth."""
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (
            (candidate / "configs" / "datasets.json").is_file()
            and (candidate / "generation" / "common" / "runtime.py").is_file()
        ):
            return candidate
    raise RuntimeError(f"cannot locate public_release above {resolved}")


PUBLIC_RELEASE = _find_public_release(Path(__file__).parent)
CONFIG_DIR = PUBLIC_RELEASE / "configs"


PACKAGE_ROOT = next(
    (parent for parent in (PUBLIC_RELEASE, *PUBLIC_RELEASE.parents)
     if (parent / "config" / "package.json").is_file()),
    PUBLIC_RELEASE,
)


def public_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PACKAGE_ROOT / path).resolve()


def require_file(value: str | Path, label: str) -> Path:
    path = public_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_new_directory(value: str | Path) -> Path:
    path = public_path(value)
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_fraction(value: str | int | float) -> Fraction:
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"invalid rational epsilon: {value}") from error
    if result <= 0:
        raise ValueError("epsilon must be positive")
    return result


def epsilon_slug(value: Fraction) -> str:
    return f"{value.numerator}_{value.denominator}"


def add_runtime_paths() -> None:
    locations = [
        PUBLIC_RELEASE,
        PUBLIC_RELEASE / "pipeline",
        PUBLIC_RELEASE / "analysis_scripts",
        PUBLIC_RELEASE / "src" / "mtr" / "DP_GSRT" / "final",
        PUBLIC_RELEASE / "src" / "reproduced_papers" / "common",
        PUBLIC_RELEASE / "src" / "reproduced_papers" / "SPRT",
        PUBLIC_RELEASE / "src" / "reproduced_papers" / "PrivTrace",
        PUBLIC_RELEASE / "src" / "reproduced_papers" / "DPTrajPM",
        PUBLIC_RELEASE / "src" / "reproduced_papers" / "DPStd",
    ]
    for location in locations:
        text = str(location)
        if location.exists() and text not in sys.path:
            sys.path.insert(0, text)


def load_dataset_registry(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    source = public_path(path or CONFIG_DIR / "datasets.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("datasets.json must contain a datasets object")
    return datasets


def dataset_config(name: str, path: str | Path | None = None) -> dict[str, Any]:
    registry = load_dataset_registry(path)
    key = name.lower()
    if key not in registry:
        raise KeyError(
            f"unknown dataset config {name!r}; provide --bbox, --osm-cache and "
            "--public-slot-count explicitly"
        )
    config = dict(registry[key])
    config["name"] = key
    return config


def configure_logging(log_file: Path | None, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
