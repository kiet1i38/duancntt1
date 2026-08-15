"""Pilot for Selective Difficulty-Conditioned TokenSkip (SD-TokenSkip).

The policy is locked before evaluation:
    easy   -> TokenSkip ratio 0.5
    medium -> Full-CoT (ratio 1.0)
    hard   -> Full-CoT (ratio 1.0)

Difficulty is estimated only from the input.  Calibration is used to set the
bin thresholds, never to inspect evaluation answers.  The policy is a
conservative empirical extension/baseline, not an absolute novelty claim.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from tokenskip_pilot_core import (
    RATIOS,
    fit_bins,
    generate_one,
    load_model,
    load_official_evaluator,
    model_device,
    read_jsonl,
    select_calibration_rows,
    set_seed,
    summarize,
)


LOCKED_POLICY = {"easy": 0.5, "medium": 1.0, "hard": 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--cal-data", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cal-pool", type=int, default=0)
    parser.add_argument("--n-cal", type=int, default=9)
    parser.add_argument("--n-eval", type=int, default=27)
    parser.add_argument("--eval-ids", default="")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not run the 3B pilot on CPU.")
    extract_answer, evaluate_answer = load_official_evaluator()
    calibration_pool = read_jsonl(
        args.cal_data, None if args.cal_pool <= 0 else args.cal_pool
    )
    calibration = select_calibration_rows(calibration_pool, args.n_cal)
    if args.eval_ids.strip():
        wanted = {item.strip() for item in args.eval_ids.split(",") if item.strip()}
        evaluation = [row for row in read_jsonl(args.eval_data) if row.get("id") in wanted]
        if len(evaluation) != len(wanted):
            found = {row.get("id") for row in evaluation}
            raise ValueError(f"Missing evaluation ids: {sorted(wanted - found)}")
        evaluation = evaluation[: args.n_eval]
    else:
        evaluation = read_jsonl(args.eval_data, args.n_eval)
    if len(evaluation) < args.n_eval:
        raise ValueError(f"Need {args.n_eval} evaluation rows, got {len(evaluation)}")
    thresholds = fit_bins(calibration, evaluation)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen2.5-3B + official TokenSkip adapter in 4-bit...", flush=True)
    model, tokenizer = load_model(args.base_model, args.adapter_path)
    print(f"Loaded on {model_device(model)}", flush=True)

    for set_name, rows in (("cal", calibration), ("eval", evaluation)):
        for index, row in enumerate(rows, 1):
            print(
                f"[{set_name} {index}/{len(rows)}] {row['id']} "
                f"({row['difficulty_bin']})",
                flush=True,
            )
            row["full"] = generate_one(
                model,
                tokenizer,
                row["question"],
                1.0,
                args.max_new_tokens,
                extract_answer,
                evaluate_answer,
                row["answer"],
            )
            for ratio in RATIOS:
                row[f"ratio_{ratio:.1f}"] = generate_one(
                    model,
                    tokenizer,
                    row["question"],
                    ratio,
                    args.max_new_tokens,
                    extract_answer,
                    evaluate_answer,
                    row["answer"],
                )

    for row in evaluation:
        ratio = LOCKED_POLICY[row["difficulty_bin"]]
        source = "full" if ratio >= 1.0 else f"ratio_{ratio:.1f}"
        row["sd_tokenskip"] = copy.deepcopy(row[source])
        row["sd_tokenskip"]["selected_ratio"] = ratio
        row["sd_tokenskip"]["policy_bin"] = row["difficulty_bin"]

    summaries = {"full": summarize(evaluation, "full")}
    for ratio in RATIOS:
        summaries[f"ratio_{ratio:.1f}"] = summarize(evaluation, f"ratio_{ratio:.1f}")
    summaries["sd_tokenskip"] = summarize(evaluation, "sd_tokenskip")
    policy_report = {
        name: {
            "ratio": ratio,
            "rule": "compressed only for easy inputs; medium/hard fall back to Full-CoT",
            "locked_before_evaluation": True,
        }
        for name, ratio in LOCKED_POLICY.items()
    }
    result = {
        "pilot_only": True,
        "method": "Selective Difficulty-Conditioned TokenSkip (SD-TokenSkip)",
        "base_model": args.base_model,
        "adapter": args.adapter_path,
        "calibration_dataset": "official TokenSkip GSM8K train subset",
        "evaluation_dataset": "official TokenSkip GSM8K test subset",
        "official_prompt": "TokenSkip/evaluation.py Qwen prompt",
        "official_evaluator": "TokenSkip/data_processing + eval evaluator",
        "calibration_n": len(calibration),
        "evaluation_n": len(evaluation),
        "evaluation_selection": (
            "declared input-complexity-stratified official ids"
            if args.eval_ids.strip()
            else "first official test rows"
        ),
        "candidate_ratios": list(RATIOS),
        "max_new_tokens": args.max_new_tokens,
        "difficulty_proxy": "input-only word/number/operator/clause score",
        "thresholds": thresholds,
        "policy": policy_report,
        "summaries": summaries,
        "caveat": (
            "Not an official reproduction; 4-bit inference-only pilot, finite "
            "evaluation subset, heuristic difficulty proxy, and no confidence interval. "
            "The model, official adapter, prompt, evaluator, and GSM8K files are held "
            "constant. This conservative policy is an empirical baseline/extension and "
            "must not be described as a guaranteed novel method."
        ),
    }
    (output_dir / "pilot_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "pilot_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in calibration + evaluation:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
