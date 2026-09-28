"""Report frozen-release identities without guessing budgets from file names."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWN_PRIVTRACE = "76538f38dbd78420ed0594b391f0a26bc0958d905d1bf0d98144873b684d6c73"


def describe(entry: dict) -> dict:
    result = dict(entry)
    result["batch"] = "imported_frozen_release"
    result["generation_epsilon"] = None
    result["budget_evidence"] = "No hash-bound generation protocol included; directory names are not budget evidence."
    if entry["sha256"] == KNOWN_PRIVTRACE:
        result["generation_epsilon"] = "1"
        result["generation_variant"] = "official-stageA+p30"
        result["budget_evidence"] = {
            "report": "public_release/check/reproduction_audit_20260717/REPORT.md",
            "source_filename": "geolife-1_p30_eps1p0_l112500_l2120_part0.2-0.4-0.4_generated_tras.txt",
            "binding": "Parser output compared elementwise with the imported pickle in the local audit.",
        }
    if "/train_only_baselines/" in entry["path"]:
        result["evaluation_role"] = "train_only_release_requires_generator_input_binding"
    elif "/route_experiments/" in entry["path"]:
        result["evaluation_role"] = "public_routing_of_saved_release"
    else:
        result["evaluation_role"] = "frozen_full_corpus_or_generation_scope_not_bound"
    result["regeneration_contract"] = "New generator outputs are a separate batch; bind their own protocol and hashes."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="experiment_results/release_inventory",
                        help="Report directory, relative to this independent repository.")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out = (out if out.is_absolute() else ROOT / out).resolve()
    provenance = json.loads((ROOT / "manifests/synthetic_data_provenance.json").read_text(encoding="utf-8-sig"))
    rows = []
    for entry in provenance["datasets"]:
        path = (ROOT / entry["path"]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Invalid dataset path: {entry['path']}")
        payload = path.read_bytes()
        if path.suffix.lower() == ".json":
            payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry["sha256"] or len(payload) != entry["size"]:
            raise ValueError(f"Frozen release changed: {entry['path']}")
        rows.append(describe(entry))
    out.mkdir(parents=True, exist_ok=True)
    (out / "releases.json").write_text(json.dumps({
        "schema_version": 1,
        "packages_are_beijing_reproduction": True,
        "frozen_release_rows": rows,
        "privtrace_budget_replacement": "excluded_by_user",
        "scope": "Frozen file identities, not certification of unseen-test synthesis.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 保存的发布数据及批次", "",
             "|方法|文件|生成预算|批次|", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"|{row['method']}|{row['path']}|{row['generation_epsilon'] or '未绑定生成协议'}|{row['batch']}|")
    lines += ["", "新生成数据必须使用自己的 protocol、manifest 和指标重新作图；不能沿用保存批次的指标。",
              "PrivTrace 保存批次的生成预算为 1，原始生成配置是 official-stageA+p30；本轮不更换该数据。"]
    (out / "RELEASES_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "verified_frozen_files": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
