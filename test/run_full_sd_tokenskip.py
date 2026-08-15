"""Full GSM8K evaluation for SD-TokenSkip with row-level resume support.

This runner keeps the TokenSkip base model, adapter, prompt and evaluator
fixed.  It evaluates Full-CoT plus fixed TokenSkip ratios on the complete
evaluation split and derives the locked SD policy without reading test labels.
The resulting file is a full-split proposed-method evaluation, not an
official/reproduced result until all paper revisions and protocol details have
been independently pinned and verified.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
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
    parser.add_argument("--n-eval", type=int, default=1319)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if row_id:
                rows[row_id] = row
    return rows


def append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def apply_locked_policy(row: dict) -> None:
    ratio = LOCKED_POLICY[row["difficulty_bin"]]
    source = "full" if ratio >= 1.0 else f"ratio_{ratio:.1f}"
    row["sd_tokenskip"] = copy.deepcopy(row[source])
    row["sd_tokenskip"]["selected_ratio"] = ratio
    row["sd_tokenskip"]["policy_bin"] = row["difficulty_bin"]


def generate_row(
    model,
    tokenizer,
    row: dict,
    max_new_tokens: int,
    extract_answer,
    evaluate_answer,
) -> dict:
    row["full"] = generate_one(
        model,
        tokenizer,
        row["question"],
        1.0,
        max_new_tokens,
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
            max_new_tokens,
            extract_answer,
            evaluate_answer,
            row["answer"],
        )
    return row


def per_bin_summary(rows: list[dict], key: str) -> dict:
    result: dict[str, dict] = {}
    for bin_name in ("easy", "medium", "hard"):
        subset = [row for row in rows if row["difficulty_bin"] == bin_name]
        result[bin_name] = summarize(subset, key)
    return result


def main() -> None:
    args = parse_args()
    set_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not run the 3B evaluation on CPU.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cal_path = output_dir / "calibration_rows.jsonl"
    eval_path = output_dir / "evaluation_rows.jsonl"
    meta_path = output_dir / "run_meta.json"

    extract_answer, evaluate_answer = load_official_evaluator()
    calibration_pool = read_jsonl(
        args.cal_data, None if args.cal_pool <= 0 else args.cal_pool
    )
    calibration = select_calibration_rows(calibration_pool, args.n_cal)
    evaluation = read_jsonl(args.eval_data, args.n_eval)
    if len(evaluation) != args.n_eval:
        raise ValueError(f"Expected {args.n_eval} evaluation rows, got {len(evaluation)}")
    if len({row["id"] for row in evaluation}) != len(evaluation):
        raise ValueError("Evaluation ids are not unique")

    thresholds = fit_bins(calibration, evaluation)
    meta = {
        "method": "Selective Difficulty-Conditioned TokenSkip (SD-TokenSkip)",
        "result_classification": "full-split proposed-method evaluation; not reproduced result",
        "base_model": args.base_model,
        "adapter": args.adapter_path,
        "calibration_dataset": "official TokenSkip GSM8K train subset",
        "evaluation_dataset": "official TokenSkip GSM8K complete test split",
        "calibration_n": len(calibration),
        "evaluation_n": len(evaluation),
        "candidate_ratios": list(RATIOS),
        "max_new_tokens": args.max_new_tokens,
        "seed": 42,
        "official_prompt": "TokenSkip/evaluation.py Qwen prompt",
        "official_evaluator": "TokenSkip/data_processing + eval evaluator",
        "difficulty_proxy": "input-only word/number/operator/clause score",
        "thresholds": thresholds,
        "locked_policy": LOCKED_POLICY,
        "resume_files": [str(cal_path), str(eval_path)],
        "caveat": (
            "Full GSM8K test split is evaluated, but this run uses 4-bit local "
            "inference and must still be distinguished from official repository "
            "and reproduced results until revisions, hashes and the complete paper "
            "protocol are independently pinned."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    cal_done = load_rows(cal_path)
    eval_done = load_rows(eval_path)
    calibration_by_id = {row["id"]: row for row in calibration}
    evaluation_by_id = {row["id"]: row for row in evaluation}
    calibration = [cal_done.get(row["id"], row) for row in calibration]
    evaluation = [eval_done.get(row["id"], row) for row in evaluation]
    calibration_by_id = {row["id"]: row for row in calibration}
    evaluation_by_id = {row["id"]: row for row in evaluation}

    pending_cal = [row for row in calibration if "full" not in row]
    pending_eval = [row for row in evaluation if "full" not in row]
    print(
        f"Resume state: calibration_done={len(calibration)-len(pending_cal)}/{len(calibration)} "
        f"evaluation_done={len(evaluation)-len(pending_eval)}/{len(evaluation)}",
        flush=True,
    )
    print("Loading Qwen2.5-3B + official TokenSkip adapter in 4-bit...", flush=True)
    model, tokenizer = load_model(args.base_model, args.adapter_path)
    print(f"Loaded on {model_device(model)}", flush=True)

    for index, row in enumerate(calibration, 1):
        if "full" in row:
            continue
        print(f"[cal {index}/{len(calibration)}] {row['id']} ({row['difficulty_bin']})", flush=True)
        row = generate_row(
            model, tokenizer, row, args.max_new_tokens, extract_answer, evaluate_answer
        )
        append_row(cal_path, row)
        calibration_by_id[row["id"]] = row

    for index, row in enumerate(evaluation, 1):
        if "full" in row:
            continue
        print(f"[eval {index}/{len(evaluation)}] {row['id']} ({row['difficulty_bin']})", flush=True)
        row = generate_row(
            model, tokenizer, row, args.max_new_tokens, extract_answer, evaluate_answer
        )
        apply_locked_policy(row)
        append_row(eval_path, row)
        evaluation_by_id[row["id"]] = row

    evaluation = [evaluation_by_id[row["id"]] for row in read_jsonl(args.eval_data, args.n_eval)]
    for row in evaluation:
        if "sd_tokenskip" not in row:
            apply_locked_policy(row)

    summaries = {"full": summarize(evaluation, "full")}
    for ratio in RATIOS:
        summaries[f"ratio_{ratio:.1f}"] = summarize(evaluation, f"ratio_{ratio:.1f}")
    summaries["sd_tokenskip"] = summarize(evaluation, "sd_tokenskip")
    result = {
        **meta,
        "summaries": summaries,
        "per_bin": {
            "full": per_bin_summary(evaluation, "full"),
            "ratio_0.5": per_bin_summary(evaluation, "ratio_0.5"),
            "ratio_0.7": per_bin_summary(evaluation, "ratio_0.7"),
            "ratio_0.9": per_bin_summary(evaluation, "ratio_0.9"),
            "sd_tokenskip": per_bin_summary(evaluation, "sd_tokenskip"),
        },
    }
    (output_dir / "full_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
