"""Fresh evaluation for a calibration-locked original-ratio gate.

The method keeps the official TokenSkip ratio at 0.5.  An input-only
difficulty gate applies that ratio to low-risk questions and Full-CoT to the
remaining questions.  Full-CoT is generated for every evaluation row so the
comparison uses the same test split and the same inference environment.

This runner is intentionally separate from the previous multi-candidate SD
runner.  It trims the generated sequence at the first Qwen EOS token so that
post-EOS padding/repetition cannot inflate reasoning-token counts.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from tokenskip_pilot_core import (
    complexity,
    load_model,
    load_official_evaluator,
    model_device,
    official_qwen_prompt,
    read_jsonl,
)


ORIGINAL_RATIO = 0.5


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
    parser.add_argument("--precision", choices=("4bit", "fp16"), default="4bit")
    return parser.parse_args()


def load_calibration(path: Path, n_cal: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    ranked = sorted(rows, key=lambda row: complexity(row["question"]))
    if len(ranked) < n_cal:
        raise ValueError(f"Need {n_cal} calibration rows, found {len(ranked)}")
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for index in range(n_cal):
        position = round(index * (len(ranked) - 1) / max(1, n_cal - 1))
        while position in used and position + 1 < len(ranked):
            position += 1
        used.add(position)
        row = dict(ranked[position])
        row["difficulty_score"] = complexity(row["question"])
        selected.append(row)
    return selected


def derive_threshold(calibration: list[dict[str, Any]]) -> dict[str, float]:
    scores = sorted(float(row["difficulty_score"]) for row in calibration)
    easy_index = max(0, math.ceil(len(scores) / 3) - 1)
    easy_max = scores[easy_index]
    higher = [score for score in scores if score > easy_max]
    if not higher:
        raise ValueError("Calibration does not contain a higher-difficulty boundary")
    first_non_easy = higher[0]
    return {
        "easy_max_from_calibration": easy_max,
        "first_non_easy_from_calibration": first_non_easy,
        "gate_threshold": (easy_max + first_non_easy) / 2.0,
    }


def eos_ids(tokenizer) -> set[int]:
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, (list, tuple)):
            ids.update(int(value) for value in tokenizer.eos_token_id)
        else:
            ids.add(int(tokenizer.eos_token_id))
    generation_config = getattr(tokenizer, "generation_config", None)
    configured = getattr(generation_config, "eos_token_id", None)
    if isinstance(configured, (list, tuple)):
        ids.update(int(value) for value in configured)
    elif configured is not None:
        ids.add(int(configured))
    return ids


def generate_mode(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    ratio: float,
    max_new_tokens: int,
    batch_size: int,
    extract_answer,
    evaluate_answer,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    stop_ids = eos_ids(tokenizer)
    device = model_device(model)

    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [official_qwen_prompt(row["question"], ratio) for row in chunk]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_width = encoded["input_ids"].shape[1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            if ratio >= 1.0:
                with model.disable_adapter():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=list(stop_ids) if len(stop_ids) > 1 else next(iter(stop_ids)),
                    )
            else:
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=list(stop_ids) if len(stop_ids) > 1 else next(iter(stop_ids)),
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        per_row_latency = elapsed / len(chunk)

        for index, row in enumerate(chunk):
            new_ids = generated[index, input_width:]
            if stop_ids:
                stop_positions = [
                    position
                    for position, token_id in enumerate(new_ids.tolist())
                    if int(token_id) in stop_ids
                ]
                if stop_positions:
                    new_ids = new_ids[: stop_positions[0]]
            output = tokenizer.decode(new_ids, skip_special_tokens=False)
            cot = output.split("\n\nThe final answer is:")[0]
            cot_length = int(tokenizer(cot, return_tensors="pt")["input_ids"].shape[1])
            prediction = extract_answer(row["question"], output, task="cot")
            prediction = prediction if isinstance(prediction, str) else ""
            correct = evaluate_answer(
                {"prediction": prediction, "answer": row["answer"].replace(",", "")}
            )
            results[row["id"]] = {
                "ratio": ratio,
                "output": output,
                "prediction": prediction,
                "correct": bool(correct),
                "generated_tokens": int(new_ids.shape[0]),
                "reasoning_tokens": cot_length,
                "latency_s": round(per_row_latency, 6),
                "batch_latency_s": round(elapsed, 6),
                "batch_size": len(chunk),
            }

    return results


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(item["correct"]) for item in items)
    tokens = sum(float(item["reasoning_tokens"]) for item in items)
    latency = sum(float(item["latency_s"]) for item in items)
    return {
        "n": len(items),
        "correct": correct,
        "accuracy": round(correct / len(items), 6),
        "avg_reasoning_tokens": round(tokens / len(items), 6),
        "total_latency_s": round(latency, 6),
        "tokens_per_correct": round(tokens / correct, 6) if correct else None,
    }


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.cal_data, args.n_cal)
    evaluation = read_jsonl(args.eval_data)[: args.n_eval]
    threshold_info = derive_threshold(calibration)
    threshold = threshold_info["gate_threshold"]
    for row in evaluation:
        row["difficulty_score"] = complexity(row["question"])
        row["selected_ratio"] = ORIGINAL_RATIO if row["difficulty_score"] <= threshold else 1.0
        row["selected_source"] = "ratio_0.5" if row["selected_ratio"] < 1.0 else "full"

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
    ratio_rows = [row for row in evaluation if row["selected_ratio"] == ORIGINAL_RATIO]
    ratio_results = generate_mode(
        model,
        tokenizer,
        ratio_rows,
        ORIGINAL_RATIO,
        args.max_new_tokens,
        args.batch_size,
        extract_answer,
        evaluate_answer,
    )

    output_rows: list[dict[str, Any]] = []
    for row in evaluation:
        selected = ratio_results[row["id"]] if row["selected_ratio"] < 1.0 else full_results[row["id"]]
        output_rows.append(
            {
                **row,
                "full": full_results[row["id"]],
                "ratio_0.5": ratio_results.get(row["id"]),
                "selected": selected,
            }
        )

    baseline = summarize([row["full"] for row in output_rows])
    selected_summary = summarize([row["selected"] for row in output_rows])
    ratio_count = sum(row["selected_ratio"] < 1.0 for row in evaluation)
    full_count = len(evaluation) - ratio_count
    results = {
        "method": "Calibration-Locked Original-Ratio Difficulty Gate (CL-OR-DG-TokenSkip)",
        "result_classification": "Extension result; fresh inference",
        "base_model": args.base_model,
        "adapter": args.adapter_path,
        "precision": args.precision,
        "calibration_n": len(calibration),
        "evaluation_n": len(evaluation),
        "original_tokenskip_ratio": ORIGINAL_RATIO,
        "policy": "difficulty_score <= gate_threshold -> ratio_0.5; otherwise -> Full-CoT",
        "threshold_source": "midpoint between calibration easy maximum and first higher calibration score",
        "threshold_info": threshold_info,
        "selection_counts": {"ratio_0.5": ratio_count, "full": full_count},
        "baseline_full": baseline,
        "gated_method": selected_summary,
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
            f"{args.precision} inference; calibration has 9 rows; the policy is an empirical "
            "extension and is not the original paper's reported result."
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
                "precision": args.precision,
                "eos_trim": True,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
