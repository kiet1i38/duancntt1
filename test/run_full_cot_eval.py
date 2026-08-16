"""Fresh Full-CoT evaluation for a TokenSkip-supported model scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_original_ratio_gate import generate_mode, load_calibration, summarize
from tokenskip_pilot_core import load_model, load_official_evaluator, read_jsonl


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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", choices=("4bit", "fp16"), default="fp16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.cal_data, args.n_cal)
    evaluation = read_jsonl(args.eval_data)[: args.n_eval]

    print(
        f"Loading {args.base_model} + TokenSkip adapter in {args.precision}; Full-CoT...",
        flush=True,
    )
    model, tokenizer = load_model(
        args.base_model,
        args.adapter_path,
        use_4bit=args.precision == "4bit",
    )
    extract_answer, evaluate_answer = load_official_evaluator()
    full_results = generate_mode(
        model,
        tokenizer,
        evaluation,
        1.0,
        args.max_new_tokens,
        args.batch_size,
        extract_answer,
        evaluate_answer,
    )
    output_rows = [{**row, "selected": full_results[row["id"]]} for row in evaluation]
    summary = summarize([row["selected"] for row in output_rows])
    results = {
        "method": "Full-CoT baseline",
        "result_classification": "Fresh inference; not the paper's reported number",
        "base_model": args.base_model,
        "adapter": args.adapter_path,
        "precision": args.precision,
        "compression_ratio": 1.0,
        "calibration_n": len(calibration),
        "evaluation_n": len(evaluation),
        "summary": summary,
        "caveat": (
            "The TokenSkip adapter is loaded but disabled for Full-CoT. "
            "This is a fresh same-dataset comparison baseline, not a paper-reported number."
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
