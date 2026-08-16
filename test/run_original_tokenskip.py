"""Fresh FP16 evaluation of the original TokenSkip ratio on the full test split.

This runner is intentionally separate from the adaptive extension runner.  It
uses the official TokenSkip ratio (0.5) for every evaluation row, while keeping
the same model, adapter, prompt construction, evaluator, and EOS trimming.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_original_ratio_gate import (
    ORIGINAL_RATIO,
    generate_mode,
    load_calibration,
    summarize,
)
from tokenskip_pilot_core import (
    complexity,
    load_model,
    load_official_evaluator,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--cal-data", required=True, type=Path)
    parser.add_argument("--eval-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-cal", type=int, default=9)
    parser.add_argument("--n-eval", type=int, default=1319)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--precision", choices=("4bit", "fp16"), default="fp16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.cal_data, args.n_cal)
    evaluation = read_jsonl(args.eval_data)[: args.n_eval]
    for row in evaluation:
        row["difficulty_score"] = complexity(row["question"])
        row["selected_ratio"] = ORIGINAL_RATIO
        row["selected_source"] = "official_ratio_0.5"

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
    original_results = generate_mode(
        model,
        tokenizer,
        evaluation,
        ORIGINAL_RATIO,
        args.max_new_tokens,
        args.batch_size,
        extract_answer,
        evaluate_answer,
    )

    output_rows = []
    for row in evaluation:
        output_rows.append(
            {
                **row,
                "selected": original_results[row["id"]],
            }
        )

    summary = summarize([row["selected"] for row in output_rows])
    results = {
        "method": "Original TokenSkip fixed-ratio baseline",
        "result_classification": (
            "Fresh inference of the original-ratio baseline; not the paper's reported number"
        ),
        "base_model": args.base_model,
        "adapter": args.adapter_path,
        "precision": args.precision,
        "calibration_n": len(calibration),
        "evaluation_n": len(evaluation),
        "original_tokenskip_ratio": ORIGINAL_RATIO,
        "policy": "ratio_0.5 for every evaluation row",
        "summary": summary,
        "caveat": (
            f"{args.precision} inference; this is a fresh local reproduction of the "
            "original-ratio protocol, not the paper's reported result."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "calibration_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in calibration:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "evaluation_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "full_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "run_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **results,
                "max_new_tokens": args.max_new_tokens,
                "batch_size_requested": args.batch_size,
                "eos_trim": True,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
