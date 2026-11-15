#!/usr/bin/env python3
"""Run SD-TokenSkip calibration and one-pass evaluation.

This script intentionally imports the official TokenSkip data-processing and
evaluation helpers from the clean-room source checkout, but does not modify
that checkout. It uses Transformers/PEFT directly so that the model is loaded
once per invocation and test rows can be grouped by the frozen policy gamma.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


GAMMAS = tuple(round(index / 10.0, 1) for index in range(1, 11))
BIN_LABELS = ("easy", "medium", "hard")
GAMMA_TO_BIN = {
    **{round(index / 10.0, 1): "easy" for index in range(1, 6)},
    **{round(index / 10.0, 1): "medium" for index in range(6, 8)},
    **{round(index / 10.0, 1): "hard" for index in range(8, 11)},
}
DIFFICULTY_QUANTILES = tuple(index / 10.0 for index in range(1, 10))
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
OPERATOR_RE = re.compile(r"[+\-*/%]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_gsm8k_rows(raw_rows: Sequence[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    from data_processing.process_utils import process_gsm8k_test

    prepared: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        item = dict(raw)
        item.setdefault("id", f"gsm8k-{split}-{index}")
        processed = list(process_gsm8k_test(item))
        if len(processed) != 1:
            raise RuntimeError(f"Expected one processed row for {item['id']}")
        example = processed[0]
        example["source_row_index"] = index
        example["source_id"] = item["id"]
        messages = copy.deepcopy(example["messages"])
        example["reference"] = example.get("reference", "") or [
            message["content"] for message in messages if message["role"] == "assistant"
        ]
        for message in messages:
            if message["role"] == "assistant":
                message["content"] = ""
        example["messages"] = messages
        prepared.append(example)
    return prepared


def build_prompt(example: Dict[str, Any], tokenizer: Any, gamma: float) -> str:
    prompt = ""
    for message in example["messages"]:
        if message["role"] == "user":
            content = message["content"]
            prompt += (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\nPlease reason step by step, and put your final "
                "answer within \\boxed{} .".replace("{} .", "{}.")
            )
            if gamma < 1.0:
                prompt += f"\n{content}<|eot_id|>{gamma}<|eot_id|><|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt += f"\n{content}<|im_end|>\n<|im_start|>assistant\n"
        elif message["role"] == "assistant":
            prompt += message["content"].rstrip()
        prompt = prompt.lstrip()
    return prompt


def question_text(example: Dict[str, Any]) -> str:
    for message in example["messages"]:
        if message["role"] == "user":
            return message["content"]
    raise ValueError("Prepared row has no user message")


def raw_features(example: Dict[str, Any], tokenizer: Any) -> Dict[str, float]:
    question = question_text(example)
    token_count = len(tokenizer(question, add_special_tokens=False)["input_ids"])
    return {
        "question_token_count": float(token_count),
        "numeric_span_count": float(len(NUMBER_RE.findall(question))),
        "operator_count": float(len(OPERATOR_RE.findall(question))),
    }


def fit_difficulty(calibration: Sequence[Dict[str, Any]], tokenizer: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    feature_names = ("question_token_count", "numeric_span_count", "operator_count")
    raw = [raw_features(row, tokenizer) for row in calibration]
    means = {name: float(np.mean([item[name] for item in raw])) for name in feature_names}
    stds = {name: float(np.std([item[name] for item in raw])) for name in feature_names}
    safe_stds = {name: (value if value > 0 else 1.0) for name, value in stds.items()}
    weights = {"question_token_count": 0.70, "numeric_span_count": 0.20, "operator_count": 0.10}

    scored: List[Dict[str, Any]] = []
    for row, feature_values in zip(calibration, raw):
        score = sum(
            weights[name] * (feature_values[name] - means[name]) / safe_stds[name]
            for name in feature_names
        )
        scored.append({"id": row["id"], "features": feature_values, "difficulty_score": float(score)})

    scores = np.asarray([item["difficulty_score"] for item in scored], dtype=np.float64)
    thresholds = [
        float(np.quantile(scores, probability, method="linear"))
        for probability in DIFFICULTY_QUANTILES
    ]
    spec = {
        "feature_names": list(feature_names),
        "weights": weights,
        "calibration_means": means,
        "calibration_stds": stds,
        "safe_stds": safe_stds,
        "difficulty_quantiles": list(DIFFICULTY_QUANTILES),
        "gamma_score_thresholds": thresholds,
        "quantile_method": "linear",
        "gamma_assignment_rule": (
            "Use the first calibration score threshold greater than or equal to the input score; "
            "map thresholds q10..q90 in order to gamma 0.1..0.9; scores above q90 map to gamma 1.0. "
            "Ties use the lowest matching gamma."
        ),
        "difficulty_group_rule": (
            "easy: gamma 0.1..0.5; medium: gamma 0.6..0.7; hard: gamma 0.8..1.0"
        ),
    }
    for item in scored:
        item["gamma"] = assign_gamma(item["difficulty_score"], spec)
        item["bin"] = assign_bin(item["gamma"])
    return spec, scored


def score_one(example: Dict[str, Any], tokenizer: Any, spec: Dict[str, Any]) -> Dict[str, Any]:
    features = raw_features(example, tokenizer)
    score = sum(
        spec["weights"][name]
        * (features[name] - spec["calibration_means"][name])
        / spec["safe_stds"][name]
        for name in spec["feature_names"]
    )
    gamma = assign_gamma(float(score), spec)
    return {
        "id": example["id"],
        "features": features,
        "difficulty_score": float(score),
        "gamma": gamma,
        "bin": assign_bin(gamma),
    }


def assign_gamma(score: float, spec: Dict[str, Any]) -> float:
    for index, threshold in enumerate(spec["gamma_score_thresholds"]):
        if score <= float(threshold):
            return GAMMAS[index]
    return GAMMAS[-1]


def assign_bin(gamma: float) -> str:
    return GAMMA_TO_BIN[round(float(gamma), 1)]


def generate_group(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Dict[str, Any]],
    gamma: float,
    batch_size: int,
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], float]:
    from data_processing.answer_extraction import extract_last_single_answer
    from eval.eval_script import eval_last_single_answer
    from eval.utils import generate_completions

    if not rows:
        return [], 0.0
    prompts = [build_prompt(row, tokenizer, gamma) for row in rows]
    start = time.time()
    outputs, _ = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        batch_size=batch_size,
        stop_id_sequences=[[tokenizer.eos_token_id]] if tokenizer.eos_token_id is not None else None,
        end_of_generation_id_sequence=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )
    elapsed = time.time() - start
    if len(outputs) != len(rows):
        raise RuntimeError(
            f"Generation returned {len(outputs)} outputs for {len(rows)} input rows at gamma={gamma:.1f}"
        )
    empty_ids = [row["id"] for row, output in zip(rows, outputs) if not isinstance(output, str) or not output.strip()]
    if empty_ids:
        preview = ", ".join(empty_ids[:10])
        raise RuntimeError(
            f"Generation returned empty output for {len(empty_ids)} rows at gamma={gamma:.1f}; IDs: {preview}"
        )
    results: List[Dict[str, Any]] = []
    for row, prompt, output in zip(rows, prompts, outputs):
        cot = output.split("\n\nThe final answer is:")[0]
        cot_length = int(tokenizer(cot, return_tensors="pt")["input_ids"].shape[1])
        prediction = extract_last_single_answer(question_text(row), output, task="cot")
        result = {
            "id": row["id"],
            "source_row_index": row["source_row_index"],
            "prompt": prompt,
            "gamma": float(gamma),
            "model_output": output,
            "prediction": prediction,
            "answer": row["answer"],
            "cot_length": cot_length,
        }
        result["accuracy"] = bool(prediction) and bool(eval_last_single_answer(result))
        results.append(result)
    return results, elapsed


def summarize(rows: Sequence[Dict[str, Any]], elapsed: float) -> Dict[str, Any]:
    count = len(rows)
    return {
        "n_samples": count,
        "correct": int(sum(bool(row["accuracy"]) for row in rows)),
        "accuracy": (sum(bool(row["accuracy"]) for row in rows) / count) if count else None,
        "avg_cot_length": (sum(row["cot_length"] for row in rows) / count) if count else None,
        "sample_latency": elapsed / count if count else None,
        "elapsed_seconds": elapsed,
        "invalid_outputs": int(sum(not row["prediction"] for row in rows)),
    }


def aggregate_by_bin(rows: Sequence[Dict[str, Any]], assignments: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_id = {item["id"]: item["bin"] for item in assignments}
    grouped: Dict[str, List[Dict[str, Any]]] = {label: [] for label in BIN_LABELS}
    for row in rows:
        grouped[by_id[row["id"]]].append(row)
    result: Dict[str, Any] = {}
    for label in BIN_LABELS:
        summary = summarize(grouped[label], 0.0)
        summary.pop("sample_latency", None)
        summary.pop("elapsed_seconds", None)
        result[label] = summary
    return result


def select_policy_rows(
    gamma_results: Dict[str, List[Dict[str, Any]]],
    assignments: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows_by_gamma = {
        gamma: {row["id"]: row for row in rows}
        for gamma, rows in gamma_results.items()
    }
    selected: List[Dict[str, Any]] = []
    for assignment in assignments:
        key = f"{float(assignment['gamma']):.1f}"
        row = copy.deepcopy(rows_by_gamma[key][assignment["id"]])
        row["difficulty_score"] = float(assignment["difficulty_score"])
        row["bin"] = assignment["bin"]
        selected.append(row)
    return selected


def metrics_without_latency(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    result = summarize(rows, 0.0)
    result.pop("sample_latency", None)
    result.pop("elapsed_seconds", None)
    return result


def summarize_calibration(
    policy_rows: Sequence[Dict[str, Any]],
    gamma_one_rows: Sequence[Dict[str, Any]],
    assignments: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    policy_by_id = {row["id"]: row for row in policy_rows}
    baseline_by_id = {row["id"]: row for row in gamma_one_rows}
    if policy_by_id.keys() != baseline_by_id.keys():
        raise ValueError("Calibration policy and gamma=1.0 rows do not have matching IDs")
    policy_ordered = [policy_by_id[key] for key in sorted(policy_by_id)]
    baseline_ordered = [baseline_by_id[key] for key in sorted(baseline_by_id)]
    policy_metrics = metrics_without_latency(policy_ordered)
    baseline_metrics = metrics_without_latency(baseline_ordered)
    accuracy_delta_pp = 100.0 * (policy_metrics["accuracy"] - baseline_metrics["accuracy"])
    token_reduction = (
        100.0 * (baseline_metrics["avg_cot_length"] - policy_metrics["avg_cot_length"])
        / baseline_metrics["avg_cot_length"]
        if baseline_metrics["avg_cot_length"]
        else None
    )
    assignment_by_id = {row["id"]: row for row in assignments}
    per_group: Dict[str, Any] = {}
    for label in BIN_LABELS:
        selected_group = [row for row in policy_ordered if assignment_by_id[row["id"]]["bin"] == label]
        baseline_group = [row for row in baseline_ordered if assignment_by_id[row["id"]]["bin"] == label]
        selected_metrics = metrics_without_latency(selected_group)
        baseline_group_metrics = metrics_without_latency(baseline_group)
        group_accuracy_delta_pp = (
            100.0 * (selected_metrics["accuracy"] - baseline_group_metrics["accuracy"])
            if selected_metrics["accuracy"] is not None and baseline_group_metrics["accuracy"] is not None
            else None
        )
        group_token_reduction = (
            100.0 * (baseline_group_metrics["avg_cot_length"] - selected_metrics["avg_cot_length"])
            / baseline_group_metrics["avg_cot_length"]
            if baseline_group_metrics["avg_cot_length"]
            else None
        )
        per_group[label] = {
            "policy_metrics": selected_metrics,
            "gamma_1_reference_metrics": baseline_group_metrics,
            "paired_accuracy_delta_percentage_points": group_accuracy_delta_pp,
            "cot_length_reduction_percent": group_token_reduction,
            "selected_gamma_counts": dict(sorted(Counter(
                f"{float(assignment_by_id[row['id']]['gamma']):.1f}" for row in selected_group
            ).items())),
        }
    return {
        "calibration_only": True,
        "policy_metrics": policy_metrics,
        "gamma_1_reference_metrics": baseline_metrics,
        "paired_accuracy_delta_percentage_points": accuracy_delta_pp,
        "cot_length_reduction_percent": token_reduction,
        "per_difficulty_group": per_group,
        "interpretation": "Descriptive calibration measurements only; they do not determine whether test evaluation runs.",
    }


def load_model(args: argparse.Namespace) -> Tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, device_map="auto")
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, choices=["qwen2.5-3b", "qwen2.5-7b"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--cleanroom-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    cleanroom = Path(args.cleanroom_root).resolve()
    config_path = cleanroom / "configs" / "sd_tokenskip.json"
    with config_path.open("r", encoding="utf-8") as handle:
        protocol_config = json.load(handle)
    protocol = protocol_config["protocol"]
    calibration_size = int(protocol_config["calibration"]["rows_per_model"])
    evaluation_size = int(protocol_config["evaluation"]["rows"])
    batch_size = int(protocol["evaluation_batch_size"])
    max_new_tokens = int(protocol["max_new_tokens"])
    seed = int(protocol["seed"])
    configured_gammas = tuple(round(float(value), 1) for value in protocol_config["candidate_gammas"])
    if args.model_name not in protocol_config["models"]:
        raise ValueError(f"Model {args.model_name!r} is not in the SD-TokenSkip protocol config")
    if configured_gammas != GAMMAS or calibration_size != 180 or evaluation_size != 1319:
        raise ValueError("SD-TokenSkip protocol config does not match the locked gamma or row contract")
    if protocol["dtype"] != "float16" or protocol["decoding"] != "greedy" or float(protocol["temperature"]) != 0.0:
        raise ValueError("SD-TokenSkip protocol config does not match FP16 greedy decoding")
    expected_groups = {
        "easy": [0.1, 0.2, 0.3, 0.4, 0.5],
        "medium": [0.6, 0.7],
        "hard": [0.8, 0.9, 1.0],
    }
    configured_groups = {
        label: [round(float(value), 1) for value in values]
        for label, values in protocol_config["difficulty"]["groups"].items()
    }
    if configured_groups != expected_groups:
        raise ValueError("SD-TokenSkip protocol config has unexpected gamma-to-group boundaries")
    expected_weights = {
        "question_token_count": 0.7,
        "numeric_span_count": 0.2,
        "operator_count": 0.1,
    }
    configured_weights = {
        name: float(value)
        for name, value in protocol_config["difficulty"]["features"].items()
    }
    if configured_weights != expected_weights or protocol_config["difficulty"]["quantile_method"] != "linear":
        raise ValueError("SD-TokenSkip protocol config does not match the locked difficulty feature contract")

    model_short = "3b" if args.model_name.endswith("3b") else "7b"
    expected_run_prefix = f"sd-{model_short}-main-"
    if Path(args.run_name).name != args.run_name or not args.run_name.startswith(expected_run_prefix):
        raise ValueError(f"--run-name must be a new main-run name beginning with {expected_run_prefix!r}")
    run_dir = (cleanroom / "runs" / args.run_name).resolve()
    if cleanroom not in run_dir.parents:
        raise ValueError("Run directory must remain inside the clean-room root")
    log_path = cleanroom / "logs" / f"{args.run_name}.log"
    if run_dir.exists() or log_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run or log: {args.run_name}")

    official = cleanroom / "source" / "TokenSkip-official"
    sys.path.insert(0, str(official))
    train_path = official / "datasets" / "gsm8k" / "train.jsonl"
    test_path = official / "datasets" / "gsm8k" / "test.jsonl"
    raw_train = read_jsonl(train_path)[: calibration_size]
    calibration = prepare_gsm8k_rows(raw_train, "train")
    calibration_ids = [row["id"] for row in calibration]
    if len(calibration) != 180 or len(set(calibration_ids)) != 180:
        raise ValueError("Calibration must contain exactly 180 unique GSM8K training rows")

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        def log_line(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        log_line(json.dumps({
            "event": "start",
            "method_id": "SD-TokenSkip",
            "model_name": args.model_name,
            "run_name": args.run_name,
            "calibration_size": calibration_size,
            "evaluate_test_requested": args.evaluate_test,
            "seed": seed,
        }))
        set_seed(seed)
        log_line("Loading model and tokenizer...")
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required by the locked FP16 protocol")
        model, tokenizer = load_model(args)
        log_line("Model loaded.")

        method_spec, calibration_assignments = fit_difficulty(calibration, tokenizer)
        policy = {
            "method_id": "SD-TokenSkip",
            "policy_type": "input_only_score_quantile_decile_to_gamma",
            "policy_locked_before_calibration_generations": True,
            "policy_locked_before_test": True,
            "calibration_size": calibration_size,
            "candidate_gammas": list(GAMMAS),
            "difficulty_quantiles": list(DIFFICULTY_QUANTILES),
            "gamma_score_thresholds": method_spec["gamma_score_thresholds"],
            "gamma_to_difficulty_group": {f"{gamma:.1f}": GAMMA_TO_BIN[gamma] for gamma in GAMMAS},
            "assignment_rule": method_spec["gamma_assignment_rule"],
            "group_rule": method_spec["difficulty_group_rule"],
            "calibration_outcomes_used_to_change_policy": False,
            "test_labels_used_to_fit_policy": False,
        }
        write_json(run_dir / "calibration_assignments.json", calibration_assignments)
        write_json(run_dir / "difficulty_spec.json", method_spec)
        write_json(run_dir / "frozen_policy.json", policy)
        write_json(run_dir / "calibration_rows.json", [
            {"id": row["id"], "source_row_index": row["source_row_index"], "answer": row["answer"]}
            for row in calibration
        ])

        calibration_results: Dict[str, List[Dict[str, Any]]] = {}
        generation_summaries: Dict[str, Any] = {}
        for gamma in GAMMAS:
            key = f"{gamma:.1f}"
            gamma_rows, elapsed = generate_group(model, tokenizer, calibration, gamma, batch_size, max_new_tokens)
            if len(gamma_rows) != len(calibration) or {row["id"] for row in gamma_rows} != set(calibration_ids):
                raise RuntimeError(f"Calibration gamma={key} did not produce exactly one row per calibration ID")
            calibration_results[key] = gamma_rows
            write_jsonl(run_dir / "calibration" / f"gamma-{gamma:.1f}" / "predictions.jsonl", gamma_rows)
            generation_summaries[f"calibration_gamma_{key}"] = summarize(gamma_rows, elapsed)
            log_line(json.dumps({"event": "calibration_complete", "gamma": gamma, **generation_summaries[f"calibration_gamma_{key}"]}))

        calibration_policy_rows = select_policy_rows(calibration_results, calibration_assignments)
        calibration_metrics = summarize_calibration(
            calibration_policy_rows,
            calibration_results["1.0"],
            calibration_assignments,
        )
        generation_summaries["calibration_adaptive_policy"] = calibration_metrics
        write_jsonl(run_dir / "calibration" / "adaptive-policy-predictions.jsonl", calibration_policy_rows)
        write_json(run_dir / "calibration" / "adaptive-policy-metrics.json", calibration_metrics)
        log_line(json.dumps({"event": "calibration_metrics_complete", **calibration_metrics}))

        test_rows: List[Dict[str, Any]] = []
        test_summary: Dict[str, Any] = {"evaluated": False, "status": "not_requested"}
        if args.evaluate_test:
            raw_test = read_jsonl(test_path)[: evaluation_size]
            test_rows = prepare_gsm8k_rows(raw_test, "test")
            test_ids = [row["id"] for row in test_rows]
            if len(test_rows) != 1319 or len(set(test_ids)) != 1319:
                raise ValueError("Evaluation must contain exactly 1,319 unique GSM8K test rows")
            test_assignments = [score_one(row, tokenizer, method_spec) for row in test_rows]
            write_json(run_dir / "test_assignments.json", test_assignments)
            by_gamma: Dict[str, List[Dict[str, Any]]] = {str(gamma): [] for gamma in GAMMAS}
            for row, assignment in zip(test_rows, test_assignments):
                selected_gamma = float(assignment["gamma"])
                by_gamma[f"{selected_gamma:.1f}"].append(row)
            test_results: List[Dict[str, Any]] = []
            for gamma in GAMMAS:
                key = f"{gamma:.1f}"
                group_results, elapsed = generate_group(model, tokenizer, by_gamma[key], gamma, batch_size, max_new_tokens)
                test_results.extend(group_results)
                generation_summaries[f"test_gamma_{key}"] = summarize(group_results, elapsed)
                log_line(json.dumps({"event": "test_group_complete", "gamma": gamma, "n_rows": len(group_results), **generation_summaries[f"test_gamma_{key}"]}))
            expected_ids = {row["id"] for row in test_rows}
            actual_ids = {row["id"] for row in test_results}
            if actual_ids != expected_ids or len(test_results) != len(expected_ids):
                raise RuntimeError("Test output IDs are not a one-to-one match")
            test_results.sort(key=lambda row: int(row["id"].rsplit("-", 1)[-1]))
            test_elapsed = sum(item["elapsed_seconds"] for key, item in generation_summaries.items() if key.startswith("test_gamma_"))
            test_summary = summarize(test_results, test_elapsed)
            test_summary["evaluated"] = True
            test_summary["status"] = "completed"
            test_summary["one_generation_per_test_row"] = True
            write_jsonl(run_dir / "test" / "predictions.jsonl", test_results)
            write_json(run_dir / "test" / "metrics.json", test_summary)
            write_json(run_dir / "test" / "metrics_by_bin.json", aggregate_by_bin(test_results, test_assignments))

        manifest = {
            "method_id": "SD-TokenSkip",
            "model_name": args.model_name,
            "run_name": args.run_name,
              "source_revision": args.source_revision,
              "runner": {
                  "path": str(Path(__file__).resolve()),
                  "sha256": sha256_file(Path(__file__).resolve()),
              },
              "runtime_environment": {
                  "python": sys.version,
                  "torch": torch.__version__,
                  "cuda": torch.version.cuda,
                  "gpus": [
                      {
                          "index": index,
                          "name": torch.cuda.get_device_name(index),
                          "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                      }
                      for index in range(torch.cuda.device_count())
                  ],
              },
            "model_path": args.model_path,
            "model_revision": args.model_revision,
            "tokenizer_path": args.tokenizer_path,
            "adapter_path": args.adapter_path,
            "adapter_revision": args.adapter_revision,
            "protocol_config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "protocol": {
                "dtype": "float16",
                "seed": seed,
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "candidate_gammas": list(GAMMAS),
                "calibration_size": calibration_size,
                "evaluation_size": len(test_rows),
                "policy_locked_before_test": True,
                "one_generation_per_test_row": True,
                "gamma_assignment": "calibration input-score deciles map monotonically to gamma 0.1..1.0",
                "difficulty_groups": policy["group_rule"],
                "calibration_metrics_reported": True,
                "evaluation_status": test_summary["status"],
            },
            "source_files": {
                "train_jsonl": {"path": str(train_path), "sha256": sha256_file(train_path)},
                "protocol_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            },
            "calibration_ids": [row["id"] for row in calibration],
            "test_ids": [row["id"] for row in test_rows],
            "generation_summaries": generation_summaries,
            "test_summary": test_summary,
            "artifacts": {},
        }
        if test_rows:
            manifest["source_files"]["test_jsonl"] = {
                "path": str(test_path),
                "sha256": sha256_file(test_path),
            }
        for path in run_dir.rglob("*"):
            if path.is_file():
                manifest["artifacts"][str(path.relative_to(run_dir))] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        write_json(run_dir / "run_manifest.json", manifest)
        log_line(json.dumps({"event": "complete", "run_dir": str(run_dir), "test_summary": test_summary}))


if __name__ == "__main__":
    main()
