"""Build an auditable threat-model matrix from existing attack artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    def rooted(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else ROOT / path).resolve()
    attack_root, split_path, out = (rooted(args.attack_root),
                                    rooted(args.split_manifest), rooted(args.out_dir))
    if out.exists():
        parser.error(f"Output directory already exists: {out}")
    inputs = {
        "split_manifest": split_path,
        **{name: attack_root / name / "results.json" for name in
           ("domias_v3", "gda_v3", "population_linkage_v3", "ordered_v3")},
    }
    missing = [name for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing attack inputs: {missing}")
    out.mkdir(parents=True, exist_ok=False)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    domias = json.loads(inputs["domias_v3"].read_text(encoding="utf-8"))
    gda = json.loads(inputs["gda_v3"].read_text(encoding="utf-8"))
    population = json.loads(
        inputs["population_linkage_v3"].read_text(encoding="utf-8")
    )
    ordered = json.loads(inputs["ordered_v3"].read_text(encoding="utf-8"))

    result_rows = []
    for row in domias["rows"]:
        result_rows.append(
            {
                "attack": row["attack"],
                "goal": "single-trajectory membership inference",
                "candidate_unit": "one complete trajectory",
                "candidates_per_class": row["members"],
                "primary_metric": "AUC",
                "value": row["aucroc"],
                "ci95_low": row["auc_bootstrap95_low"],
                "ci95_high": row["auc_bootstrap95_high"],
                "low_fpr_result": f"TPR@1%FPR={row['tpr_at_1pct_fpr']:.3f}; TPR@5%FPR={row['tpr_at_5pct_fpr']:.3f}",
                "random_reference": 0.5,
            }
        )
    row = gda["rows"][0]
    result_rows.append(
        {
            "attack": "GDA-MIA",
            "goal": "single-trajectory membership inference",
            "candidate_unit": "one complete trajectory",
            "candidates_per_class": row["candidate_members_per_class"],
            "primary_metric": "AUC",
            "value": row["auc"],
            "ci95_low": row["auc_bootstrap95_low"],
            "ci95_high": row["auc_bootstrap95_high"],
            "low_fpr_result": f"TPR@1%FPR={row['tpr_at_1pct_fpr']:.3f}; TPR@5%FPR={row['tpr_at_5pct_fpr']:.3f}",
            "random_reference": 0.5,
        }
    )
    result_rows.append(
        {
            "attack": "Population nearest-landmark linkage",
            "goal": "member/nonmember separation by nearest released trajectory",
            "candidate_unit": "one complete trajectory",
            "candidates_per_class": population["members"],
            "primary_metric": "AUC",
            "value": population["auc"],
            "ci95_low": population["auc_bootstrap95"][0],
            "ci95_high": population["auc_bootstrap95"][1],
            "low_fpr_result": "not recorded",
            "random_reference": 0.5,
        }
    )
    order_row = ordered["rows"][0]
    result_rows.extend(
        [
            {
                "attack": "Ordered-interface origin linkage",
                "goal": "recover the claimed input trajectory from a 50-item candidate group",
                "candidate_unit": "one complete trajectory at the same slot index",
                "candidates_per_class": 50,
                "primary_metric": "Top-1",
                "value": order_row["origin_top1"],
                "ci95_low": "",
                "ci95_high": "",
                "low_fpr_result": f"Top-5={order_row['origin_top5']:.3f}; MRR={order_row['origin_mrr']:.3f}",
                "random_reference": order_row["origin_random_top1"],
            },
            {
                "attack": "Ordered-interface trajectory-fingerprint linkage",
                "goal": "recover the claimed input trajectory from a 50-item candidate group",
                "candidate_unit": "one complete trajectory at the same slot index",
                "candidates_per_class": 50,
                "primary_metric": "Top-1",
                "value": order_row["fingerprint_top1"],
                "ci95_low": "",
                "ci95_high": "",
                "low_fpr_result": f"Top-5={order_row['fingerprint_top5']:.3f}; MRR={order_row['fingerprint_mrr']:.3f}",
                "random_reference": order_row["fingerprint_random_top1"],
            },
        ]
    )
    result_path = out / "privacy_attack_results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    markdown = f"""# Threat model and empirical attacks

## Formal release threat model

| Item | Fixed scope |
|---|---|
| Protected unit | One preprocessed complete trajectory. The guarantee is trajectory-level, not user-level when one person contributes multiple trajectories. |
| Adjacency | Add/remove one complete trajectory under a fixed public capacity and fixed output-slot count. |
| Released objects | Synthetic coordinate trajectories and their directed-road witness sidecars. |
| Public information | Algorithm, source code, total and per-query budgets, public OSM graph, bbox, partition rules, bucket boundaries, decoder parameters, and output count. |
| Adversary | Computationally unbounded and allowed arbitrary auxiliary information; may choose any neighboring datasets covered by the adjacency definition. |
| Hidden state | Exact private query vectors and mechanism randomness. Reproducible fixed seeds in research artifacts are not part of a certifiable deployment and must be replaced by hidden cryptographic randomness. |
| Excluded claim | The mechanism does not claim user-level DP across all trajectories belonging to one person. |

## Empirical attack interfaces

| Attack family | Attacker observes | Auxiliary data | Goal | Access not granted | Random reference |
|---|---|---|---|---|---|
| DOMIAS Eq. 1/2 and GAN-Leaks | Complete synthetic trajectory release and public protocol | 1,000 labeled member candidates, 1,000 disjoint nonmember candidates, and {split['reference_count']} disjoint same-city reference trajectories | Infer whether one candidate trajectory belonged to the private training set | No exact measurements, gradients, decoder state, record/output pairing, or adaptive mechanism queries | AUC 0.5 |
| GDA-MIA | Complete synthetic trajectory release and public protocol | Released trajectories as positive attack-training examples; disjoint same-city reference trajectories as negatives; the same frozen member/nonmember candidates | Infer candidate membership using a fixed classifier | No model query, record/output pairing, or private conditions | AUC 0.5 |
| Population nearest-landmark linkage | Complete synthetic trajectory release | Member and disjoint nonmember candidate trajectories | Separate members from nonmembers by nearest released trajectory fingerprint | No slot correspondence or model access | AUC 0.5 |
| Ordered-interface linkage | Complete release with its public slot order | A 50-item candidate group containing the claimed private input for each slot | Recover the claimed input from origin or eight-landmark fingerprint | No exact measurements or decoder state; the attack is deliberately stronger because it assumes an input/output slot correspondence that MTR-GSRT does not publish | Top-1 0.02; Top-5 0.10 |

The frozen split contains {split['member_training_count']:,} training trajectories. Member candidates are sampled from that training partition; nonmembers and reference trajectories are disjoint subsets of the {split['attack_candidate_count_per_class'] + split['reference_count']:,}-trajectory test partition. Every attack therefore uses the same protected unit and does not reuse a trajectory across member, nonmember, and reference roles.

## Interpretation rule

The empirical attacks are finite diagnostics, not the privacy proof. AUC confidence intervals covering 0.5 or retrieval rates near the candidate-set random reference mean that these particular attacks did not distinguish the evaluated release. They do not strengthen the formal privacy parameter. The formal guarantee follows from the stated adjacency, bounded query sensitivity, sequential composition, and post-processing.
"""
    (out / "threat_model.md").write_text(markdown, encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "classification": "ASSEMBLED_EXISTING_EMPIRICAL_ATTACK_RESULTS_NO_ATTACK_RERUN",
        "input_hashes": {name: sha256_file(path) for name, path in inputs.items()},
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "output_hashes": {name: sha256_file(out / name) for name in
                          ("privacy_attack_results.csv", "threat_model.md")},
    }, indent=2) + "\n", encoding="utf-8")
    print(markdown)
    print(json.dumps(result_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
