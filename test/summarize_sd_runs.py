"""Validate downloaded SD/Full-CoT artifacts and write a comparison summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.joinpath("evaluation_rows.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    ids = [row.get("id") for row in rows]
    if any(value is None for value in ids):
        raise ValueError(f"Missing id in {path}")
    files = {
        name: path.joinpath(name).is_file()
        for name in (
            "calibration_rows.jsonl",
            "evaluation_rows.jsonl",
            "full_results.json",
            "run_meta.json",
            "run.stdout.log",
        )
    }
    return rows, {
        "evaluation_rows": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "files_present": files,
        "all_required_files_present": all(files.values()),
    }


def read_results(path: Path) -> dict[str, Any]:
    return json.loads(path.joinpath("full_results.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd3b", type=Path, required=True)
    parser.add_argument("--full3b", type=Path, required=True)
    parser.add_argument("--full7b", type=Path, required=True)
    parser.add_argument("--old-gate3b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sd3b_rows, sd3b_check = load_rows(args.sd3b)
    full3b_rows, full3b_check = load_rows(args.full3b)
    full7b_rows, full7b_check = load_rows(args.full7b)
    old_gate3b_rows, old_gate3b_check = load_rows(args.old_gate3b)
    if any(
        check["duplicate_ids"] or not check["all_required_files_present"]
        for check in (sd3b_check, full3b_check, full7b_check, old_gate3b_check)
    ):
        raise ValueError("Artifact validation failed: duplicate IDs or missing files")

    sd3b = read_results(args.sd3b)
    full3b = read_results(args.full3b)
    full7b = read_results(args.full7b)
    old_gate3b = read_results(args.old_gate3b)
    baseline = sd3b["baseline_full"]
    method = sd3b["multilevel_method"]
    comparison = sd3b["comparison_vs_full"]
    target_token_reduction = 20.0
    target_accuracy_loss = 2.0
    full3b_summary = full3b.get("summary", full3b.get("baseline_full"))
    if full3b_summary is None:
        raise ValueError("3B Full-CoT reference has no summary")

    report = {
        "classification": (
            "Fresh inference comparison; not paper-reported and not claimed as "
            "official reproduction."
        ),
        "artifact_checks": {
            "3b_multilevel_sd": sd3b_check,
            "3b_full_cot_reference": full3b_check,
            "7b_full_cot": full7b_check,
            "3b_previous_gate": old_gate3b_check,
        },
        "runs": {
            "3b_multilevel_sd": {
                "path": str(args.sd3b),
                "result_classification": sd3b["result_classification"],
                "base_model": sd3b["base_model"],
                "precision": sd3b["precision"],
                "policy": sd3b["policy"],
                "selected_counts": sd3b["selected_counts"],
                "summary": method,
            },
            "3b_full_cot_reference": {
                "path": str(args.full3b),
                "result_classification": full3b.get(
                    "result_classification", "Fresh inference"
                ),
                "base_model": full3b["base_model"],
                "precision": full3b["precision"],
                "summary": full3b_summary,
            },
            "7b_full_cot": {
                "path": str(args.full7b),
                "result_classification": full7b["result_classification"],
                "base_model": full7b["base_model"],
                "precision": full7b["precision"],
                "summary": full7b["summary"],
            },
            "3b_previous_gate": {
                "path": str(args.old_gate3b),
                "result_classification": old_gate3b["result_classification"],
                "base_model": old_gate3b["base_model"],
                "precision": old_gate3b["precision"],
                "summary": old_gate3b["gated_method"],
                "comparison_vs_full": old_gate3b["comparison_vs_full"],
            },
        },
        "comparisons": {
            "3b_multilevel_vs_3b_full": comparison,
            "3b_previous_gate_vs_3b_full": old_gate3b["comparison_vs_full"],
            "7b_full_vs_3b_full_descriptive_only": {
                "accuracy_delta_pp": round(
                    (full7b["summary"]["accuracy"] - full3b_summary["accuracy"])
                    * 100,
                    4,
                ),
                "avg_reasoning_tokens_delta_pct": round(
                    (
                        full7b["summary"]["avg_reasoning_tokens"]
                        / full3b_summary["avg_reasoning_tokens"]
                        - 1
                    )
                    * 100,
                    6,
                ),
                "latency_delta_pct": round(
                    (
                        full7b["summary"]["total_latency_s"]
                        / full3b_summary["total_latency_s"]
                        - 1
                    )
                    * 100,
                    6,
                ),
                "note": (
                    "Different model sizes; descriptive only, not a compression "
                    "ablation."
                ),
            },
        },
        "illustrative_20pct_tokens_2pp_accuracy_check": {
            "target_token_reduction_pct": target_token_reduction,
            "maximum_accuracy_loss_pp": target_accuracy_loss,
            "3b_multilevel_token_reduction_pct": comparison[
                "reasoning_token_reduction_pct_vs_full"
            ],
            "3b_multilevel_accuracy_loss_pp": comparison["accuracy_loss_pp_vs_full"],
            "passes": (
                comparison["reasoning_token_reduction_pct_vs_full"]
                >= target_token_reduction
                and comparison["accuracy_loss_pp_vs_full"] <= target_accuracy_loss
            ),
            "note": (
                "This is a diagnostic against the teacher's earlier example, not "
                "a claim that the example is a mandatory threshold."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
