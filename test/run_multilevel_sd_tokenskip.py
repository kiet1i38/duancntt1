"""Calibration-only multi-level Selective Difficulty-Conditioned TokenSkip.

The policy keeps the TokenSkip backbone and official ratio candidates fixed.
Calibration rows choose the most compressed feasible ratio for the easy and
medium difficulty bins; hard rows are forced to Full-CoT.  Test answers are
never used to choose the policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_original_ratio_gate import generate_mode, load_calibration, summarize
from tokenskip_pilot_core import (
    complexity,
    load_model,
    load_official_evaluator,
    read_jsonl,
)


CANDIDATE_RATIOS = (0.5, 0.7, 0.8, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--cal-data", required=True, type=Path)
    parser.add_argument("--eval-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--full-reference",
        type=Path,
        help="Existing evaluation_rows.jsonl containing a full result for every test row.",
    )
    parser.add_argument("--n-cal", type=int, default=9)
    parser.add_argument("--n-eval", type=int, default=1319)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=288)
    parser.add_argument("--cal-accuracy-tolerance", type=float, default=0.02)
    parser.add_argument("--precision", choices=("4bit", "fp16"), default="fp16")
    return parser.parse_args()


def difficulty_bins(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = sorted(float(row["difficulty_score"]) for row in rows)
    if len(scores) < 3:
        raise ValueError("Need at least three calibration rows for three difficulty bins")
    return {
        "easy_max": scores[max(0, len(scores) // 3 - 1)],
        "medium_max": scores[max(0, 2 * len(scores) // 3 - 1)],
    }


def assign_bin(score: float, boundaries: dict[str, float]) -> str:
    if score <= boundaries["easy_max"]:
        return "easy"
    if score <= boundaries["medium_max"]:
        return "medium"
    return "hard"


def load_full_reference(
    path: Path,
    evaluation: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get("full")
        if not isinstance(value, dict):
            raise ValueError(f"Full reference row {row.get('id')} has no full result")
        by_id[row["id"]] = value
    expected = {row["id"] for row in evaluation}
    if set(by_id) != expected:
        raise ValueError("Full reference IDs do not exactly match evaluation IDs")
    return by_id


def choose_ratio(
    bin_name: str,
    rows: list[dict[str, Any]],
    candidate_results: dict[float, dict[str, dict[str, Any]]],
    tolerance: float,
) -> float:
    if bin_name == "hard":
        return 1.0
    full_summary = summarize([candidate_results[1.0][row["id"]] for row in rows])
    feasible: list[tuple[float, dict[str, Any]]] = []
    for ratio in CANDIDATE_RATIOS:
        summary = summarize([candidate_results[ratio][row["id"]] for row in rows])
        accuracy_loss = full_summary["accuracy"] - summary["accuracy"]
        if accuracy_loss <= tolerance:
            feasible.append((ratio, summary))
    if not feasible:
        return 1.0
    return min(feasible, key=lambda item: (item[1]["avg_reasoning_tokens"], item[0]))[0]


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.cal_data, args.n_cal)
    evaluation = read_jsonl(args.eval_data)[: args.n_eval]
    boundaries = difficulty_bins(calibration)
    for row in calibration:
        row["difficulty_score"] = complexity(row["question"])
        row["difficulty_bin"] = assign_bin(row["difficulty_score"], boundaries)
    for row in evaluation:
        row["difficulty_score"] = complexity(row["question"])
        row["difficulty_bin"] = assign_bin(row["difficulty_score"], boundaries)

    print(
        f"Loading Qwen2.5-3B + official TokenSkip adapter in {args.precision}...",
        flush=True,
    )
    model, tokenizer = load_model(
        args.base_model,
        args.adapter_path,
        use_4bit=args.precision == "4bit",
    )
    extract_answer, evaluate_answer = load_official_evaluator()

    calibration_results: dict[float, dict[str, dict[str, Any]]] = {}
    for ratio in CANDIDATE_RATIOS:
        calibration_results[ratio] = generate_mode(
            model,
            tokenizer,
            calibration,
            ratio,
            args.max_new_tokens,
            args.batch_size,
            extract_answer,
            evaluate_answer,
        )

    calibration_by_bin = {
        name: [row for row in calibration if row["difficulty_bin"] == name]
        for name in ("easy", "medium", "hard")
    }
    policy = {
        name: choose_ratio(
            name,
            rows,
            calibration_results,
            args.cal_accuracy_tolerance,
        )
        for name, rows in calibration_by_bin.items()
    }
    for row in calibration:
        row["selected_ratio"] = policy[row["difficulty_bin"]]
    for row in evaluation:
        row["selected_ratio"] = policy[row["difficulty_bin"]]
        row["selected_source"] = f"calibration_{row['difficulty_bin']}"

    full_reference: dict[str, dict[str, Any]] | None = None
    if args.full_reference:
        full_reference = load_full_reference(args.full_reference, evaluation)
    else:
        full_reference = generate_mode(
            model,
            tokenizer,
            evaluation,
            1.0,
            args.max_new_tokens,
            args.batch_size,
            extract_answer,
            evaluate_answer,
        )

    selected_results: dict[str, dict[str, Any]] = {}
    for ratio in sorted(set(float(row["selected_ratio"]) for row in evaluation)):
        rows = [row for row in evaluation if row["selected_ratio"] == ratio]
        if ratio == 1.0 and args.full_reference:
            selected_results.update({row["id"]: full_reference[row["id"]] for row in rows})
        else:
            selected_results.update(
                generate_mode(
                    model,
                    tokenizer,
                    rows,
                    ratio,
                    args.max_new_tokens,
                    args.batch_size,
                    extract_answer,
                    evaluate_answer,
                )
            )

    output_rows = []
    for row in evaluation:
        output_rows.append(
            {
                **row,
                "full": full_reference[row["id"]],
                "selected": selected_results[row["id"]],
            }
        )

    baseline = summarize([row["full"] for row in output_rows])
    selected_summary = summarize([row["selected"] for row in output_rows])
    candidate_summaries = {
        name: {
            str(ratio): summarize(
                [calibration_results[ratio][row["id"]] for row in rows]
            )
            for ratio in CANDIDATE_RATIOS
        }
        for name, rows in calibration_by_bin.items()
    }
    selected_counts = {
        str(ratio): sum(row["selected_ratio"] == ratio for row in evaluation)
        for ratio in sorted(set(row["selected_ratio"] for row in evaluation))
    }
    results = {
        "method": "Multi-Level Selective Difficulty-Conditioned TokenSkip",
        "result_classification": "Extension result; fresh inference for selected modes",
        "base_model": args.base_model,
        "adapter": args.adapter_path,
        "precision": args.precision,
        "calibration_n": len(calibration),
        "evaluation_n": len(evaluation),
        "candidate_ratios": list(CANDIDATE_RATIOS),
        "calibration_accuracy_tolerance": args.cal_accuracy_tolerance,
        "difficulty_boundaries": boundaries,
        "policy": policy,
        "selected_counts": selected_counts,
        "calibration_candidate_summaries": candidate_summaries,
        "baseline_source": str(args.full_reference) if args.full_reference else "fresh_full_generation",
        "baseline_full": baseline,
        "multilevel_method": selected_summary,
        "comparison_vs_full": {
            "accuracy_loss_pp_vs_full": round(
                (baseline["accuracy"] - selected_summary["accuracy"]) * 100, 6
            ),
            "reasoning_token_reduction_pct_vs_full": round(
                (1 - selected_summary["avg_reasoning_tokens"] / baseline["avg_reasoning_tokens"])
                * 100,
                6,
            ),
            "latency_delta_pct_vs_full": round(
                (selected_summary["total_latency_s"] / baseline["total_latency_s"] - 1) * 100,
                6,
            ),
        },
        "caveat": (
            "Policy is selected from calibration rows only; hard rows are forced to Full-CoT. "
            "This is an empirical extension and not the paper's reported result."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibration_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in calibration) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "evaluation_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "full_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "run_meta.json").write_text(
        json.dumps(
            {
                **results,
                "max_new_tokens": args.max_new_tokens,
                "batch_size_requested": args.batch_size,
                "eos_trim": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
