"""Shared conditional-trajectory-completion probe.

Each method supplies only the training corpus.  Tokenization, architecture,
optimization, seeds, and the frozen real test set are shared.  The probe is a
compact model-based transfer test, not a foundation-model benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PUBLIC_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLIC_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLIC_ROOT))

from public_utils import load_trajectories, parse_bbox


PAD = 0
BOS = 1
CELL_OFFSET = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_sequence(trajectory: np.ndarray, bbox: tuple[float, ...], grid: int, max_len: int) -> list[int]:
    arr = np.asarray(trajectory, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return []
    take = np.linspace(0, arr.shape[0] - 1, min(max_len * 4, arr.shape[0])).astype(int)
    points = arr[take, :2]
    lat0, lat1, lon0, lon1 = bbox
    row = np.floor((points[:, 0] - lat0) / max(lat1 - lat0, 1e-12) * grid).astype(int)
    col = np.floor((points[:, 1] - lon0) / max(lon1 - lon0, 1e-12) * grid).astype(int)
    row = np.clip(row, 0, grid - 1)
    col = np.clip(col, 0, grid - 1)
    cells = (row * grid + col + CELL_OFFSET).tolist()
    collapsed = [cells[0]]
    for cell in cells[1:]:
        if cell != collapsed[-1]:
            collapsed.append(cell)
    return collapsed[:max_len] if len(collapsed) >= 4 else []


def tokenize(
    trajectories: list[np.ndarray],
    bbox: tuple[float, ...],
    grid: int,
    max_len: int,
    cap: int,
    seed: int,
) -> list[list[int]]:
    sequences = [grid_sequence(t, bbox, grid, max_len) for t in trajectories]
    sequences = [seq for seq in sequences if seq]
    rng = np.random.default_rng(seed)
    if len(sequences) > cap:
        keep = rng.choice(len(sequences), size=cap, replace=False)
        sequences = [sequences[int(i)] for i in sorted(keep)]
    return sequences


def tensors(sequences: list[list[int]], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.full((len(sequences), max_len), PAD, dtype=torch.long)
    y = torch.full((len(sequences), max_len), PAD, dtype=torch.long)
    for i, seq in enumerate(sequences):
        source = [BOS] + seq[:-1]
        x[i, : len(source)] = torch.tensor(source)
        y[i, : len(seq)] = torch.tensor(seq)
    return x, y


class CausalProbe(nn.Module):
    def __init__(self, vocabulary: int, max_len: int, width: int, heads: int, layers: int) -> None:
        super().__init__()
        self.token = nn.Embedding(vocabulary, width, padding_idx=PAD)
        self.position = nn.Embedding(max_len, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 3,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Linear(width, vocabulary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        hidden = self.token(x) + self.position(positions)
        mask = torch.triu(
            torch.ones((x.shape[1], x.shape[1]), dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        hidden = self.encoder(hidden, mask=mask, src_key_padding_mask=x.eq(PAD))
        return self.output(hidden)


def train_model(
    sequences: list[list[int]],
    vocabulary: int,
    max_len: int,
    seed: int,
    epochs: int,
    batch_size: int,
    width: int,
    heads: int,
    layers: int,
    learning_rate: float,
    device: torch.device,
) -> CausalProbe:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = CausalProbe(vocabulary, max_len, width, heads, layers).to(device)
    x, y = tensors(sequences, max_len)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits.reshape(-1, vocabulary), yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def transition_f1(reference: list[int], prediction: list[int]) -> float:
    ref = set(zip(reference[:-1], reference[1:]))
    pred = set(zip(prediction[:-1], prediction[1:]))
    if not ref or not pred:
        return 0.0
    overlap = len(ref & pred)
    return 2.0 * overlap / (len(ref) + len(pred))


@torch.inference_mode()
def evaluate_model(
    model: CausalProbe,
    sequences: list[list[int]],
    max_len: int,
    prefix_fraction: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    token_hits = token_total = endpoint_hits = 0
    f1_values: list[float] = []
    exact_hits = 0
    for reference in sequences:
        prefix_len = max(2, min(len(reference) - 1, math.ceil(len(reference) * prefix_fraction)))
        generated = list(reference[:prefix_len])
        while len(generated) < len(reference):
            source = [BOS] + generated
            x = torch.full((1, max_len), PAD, dtype=torch.long, device=device)
            x[0, : len(source)] = torch.tensor(source, dtype=torch.long, device=device)
            logits = model(x)
            next_token = int(logits[0, len(source) - 1].argmax())
            if next_token < CELL_OFFSET:
                next_token = generated[-1]
            generated.append(next_token)
        truth_suffix = reference[prefix_len:]
        pred_suffix = generated[prefix_len:]
        token_hits += sum(int(a == b) for a, b in zip(truth_suffix, pred_suffix))
        token_total += len(truth_suffix)
        endpoint_hits += int(generated[-1] == reference[-1])
        exact_hits += int(pred_suffix == truth_suffix)
        f1_values.append(transition_f1(reference[prefix_len - 1 :], generated[prefix_len - 1 :]))
    count = max(len(sequences), 1)
    return {
        "suffix_token_accuracy": token_hits / max(token_total, 1),
        "suffix_exact_match": exact_hits / count,
        "endpoint_accuracy": endpoint_hits / count,
        "suffix_transition_f1": float(np.mean(f1_values)) if f1_values else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-train", required=True)
    parser.add_argument("--real-test", required=True)
    parser.add_argument("--synthetic", action="append", default=[])
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--bbox", nargs=4, type=float, default=parse_bbox(None))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260719, 20260720, 20260721])
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--train-cap", type=int, default=6000)
    parser.add_argument("--test-cap", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--prefix-fraction", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    names = args.names or [Path(path).stem for path in args.synthetic]
    if len(names) != len(args.synthetic):
        raise ValueError("--names must contain one name per --synthetic")
    if len(set(names)) != len(names):
        raise ValueError("--names must be unique")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        args.device if args.device != "auto" else "cpu"
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch build has no CUDA support")
    device = torch.device(device_name)
    bbox = tuple(args.bbox)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    real_train_path = Path(args.real_train).resolve()
    real_test_path = Path(args.real_test).resolve()
    sources = [("Real-train", real_train_path)] + [
        (name, Path(path).resolve()) for name, path in zip(names, args.synthetic)
    ]
    real_test = load_trajectories(str(real_test_path), None)
    test_sequences = tokenize(
        real_test, bbox, args.grid, args.max_length, args.test_cap, seed=99173
    )
    if not test_sequences:
        raise RuntimeError("real test set produced no valid token sequences")

    started = time.time()
    rows: list[dict[str, object]] = []
    vocabulary = args.grid * args.grid + CELL_OFFSET
    for name, path in sources:
        trajectories = load_trajectories(str(path), None)
        train_sequences = tokenize(
            trajectories, bbox, args.grid, args.max_length, args.train_cap, seed=88103
        )
        if not train_sequences:
            raise RuntimeError(f"{name} produced no valid training sequences")
        for seed in args.seeds:
            tick = time.time()
            model = train_model(
                train_sequences,
                vocabulary,
                args.max_length,
                seed,
                args.epochs,
                args.batch_size,
                args.width,
                args.heads,
                args.layers,
                args.learning_rate,
                device,
            )
            scores = evaluate_model(
                model, test_sequences, args.max_length, args.prefix_fraction, device
            )
            rows.append(
                {
                    "method": name,
                    "seed": seed,
                    "train_sequences": len(train_sequences),
                    "test_sequences": len(test_sequences),
                    **scores,
                    "elapsed_sec": time.time() - tick,
                }
            )
            print(f"[completion] {name} seed={seed} {scores}", flush=True)

    fields = list(rows[0])
    csv_path = out_dir / "completion_by_seed.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for name, _ in sources:
        selected = [row for row in rows if row["method"] == name]
        entry: dict[str, object] = {"method": name, "seeds": len(selected)}
        for metric in (
            "suffix_token_accuracy",
            "suffix_exact_match",
            "endpoint_accuracy",
            "suffix_transition_f1",
        ):
            values = np.asarray([float(row[metric]) for row in selected])
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(entry)
    summary_path = out_dir / "completion_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    manifest = {
        "schema": "conditional-trajectory-completion-probe-v1",
        "claim_boundary": "shared compact Transformer transfer probe; not a foundation-model benchmark",
        "task": "given the first half of a real grid trajectory, autoregressively complete the suffix",
        "device": {
            "selected": str(device),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "platform": platform.platform(),
        },
        "parameters": {
            key: getattr(args, key)
            for key in (
                "seeds", "grid", "max_length", "train_cap", "test_cap", "epochs",
                "batch_size", "width", "heads", "layers", "learning_rate",
                "prefix_fraction",
            )
        },
        "inputs": {
            "real_test": {"path": str(real_test_path), "sha256": sha256_file(real_test_path)},
            "training": [
                {"method": name, "path": str(path), "sha256": sha256_file(path)}
                for name, path in sources
            ],
        },
        "outputs": {
            csv_path.name: sha256_file(csv_path),
            summary_path.name: sha256_file(summary_path),
        },
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[completion] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
