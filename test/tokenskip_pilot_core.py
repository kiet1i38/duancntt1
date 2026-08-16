"""Shared inference utilities for the TokenSkip screening pilots.

This module contains only the fixed TokenSkip protocol helpers.  The active
policy is defined in a separate runner so that a failed policy can be archived
without changing the model, prompt, evaluator, or data-loading code.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


RATIOS = (0.5, 0.7, 0.9)


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str, count: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("id", f"{Path(path).stem}-{len(rows)}")
            rows.append(row)
            if count is not None and len(rows) >= count:
                break
    return rows


def complexity(question: str) -> float:
    """Input-only difficulty proxy; gold answers and gold CoT are not read."""
    words = re.findall(r"\w+", question)
    numbers = re.findall(r"\d+(?:\.\d+)?", question)
    operators = re.findall(
        r"[+*/%=-]|\b(?:half|twice|double|percent|per)\b", question.lower()
    )
    clauses = re.findall(
        r"\b(?:if|then|after|before|each|every|remaining|total)\b",
        question.lower(),
    )
    return float(len(words) + 2 * len(numbers) + 2 * len(operators) + 3 * len(clauses))


def select_calibration_rows(pool: list[dict], n_cal: int) -> list[dict]:
    if len(pool) < n_cal:
        raise ValueError(f"Need at least {n_cal} calibration rows, got {len(pool)}")
    ranked = sorted(pool, key=lambda row: complexity(row["question"]))
    selected: list[dict] = []
    used: set[int] = set()
    for index in range(n_cal):
        position = round(index * (len(ranked) - 1) / max(1, n_cal - 1))
        while position in used and position + 1 < len(ranked):
            position += 1
        used.add(position)
        selected.append(ranked[position])
    return selected


def fit_bins(calibration: list[dict], evaluation: list[dict]) -> dict:
    for row in calibration + evaluation:
        row["difficulty_score"] = complexity(row["question"])
    scores = sorted(row["difficulty_score"] for row in calibration)
    if not scores:
        raise ValueError("Calibration set is empty")
    first_index = max(0, math.ceil(len(scores) / 3) - 1)
    second_index = max(0, math.ceil(2 * len(scores) / 3) - 1)
    thresholds = {"easy_max": scores[first_index], "medium_max": scores[second_index]}

    def label(score: float) -> str:
        if score <= thresholds["easy_max"]:
            return "easy"
        if score <= thresholds["medium_max"]:
            return "medium"
        return "hard"

    for row in calibration + evaluation:
        row["difficulty_bin"] = label(row["difficulty_score"])
    return thresholds


def official_qwen_prompt(question: str, ratio: float) -> str:
    prefix = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nPlease reason step by step, and put your final answer "
        "within \\boxed{}.\n"
    )
    if ratio < 1.0:
        return (
            prefix
            + question
            + f"<|eot_id|>{ratio}<|eot_id|><|im_end|>\n<|im_start|>assistant\n"
        )
    return prefix + question + "<|im_end|>\n<|im_start|>assistant\n"


def load_official_evaluator() -> tuple:
    repo = Path(__file__).resolve().parent / "TokenSkip-official"
    sys.path.insert(0, str(repo))
    from data_processing.answer_extraction import extract_last_single_answer
    from eval.eval_script import eval_last_single_answer

    return extract_last_single_answer, eval_last_single_answer


def model_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(base_model: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    return model, tokenizer


def generate_one(
    model,
    tokenizer,
    question: str,
    ratio: float,
    max_new_tokens: int,
    extract_answer,
    evaluate_answer,
    gold_answer: str,
) -> dict:
    prompt = official_qwen_prompt(question, ratio)
    encoded = tokenizer(prompt, return_tensors="pt")
    device = model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
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
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        else:
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_ids = generated[0, encoded["input_ids"].shape[1] :]
    output = tokenizer.decode(new_ids, skip_special_tokens=False)
    cot = output.split("\n\nThe final answer is:")[0]
    cot_length = int(tokenizer(cot, return_tensors="pt")["input_ids"].shape[1])
    prediction = extract_answer(question, output, task="cot")
    prediction = prediction if isinstance(prediction, str) else ""
    correct = evaluate_answer(
        {"prediction": prediction, "answer": gold_answer.replace(",", "")}
    )
    return {
        "ratio": ratio,
        "output": output,
        "prediction": prediction,
        "correct": bool(correct),
        "generated_tokens": int(new_ids.shape[0]),
        "reasoning_tokens": cot_length,
        "latency_s": round(elapsed, 4),
    }


def generate_batch(
    model,
    tokenizer,
    questions: list[str],
    ratio: float,
    max_new_tokens: int,
    extract_answer,
    evaluate_answer,
    gold_answers: list[str],
) -> list[dict]:
    """Generate a fixed TokenSkip ratio for several rows at once.

    The prompt, ratio, evaluator, and decoding settings are unchanged from
    ``generate_one``.  Latency is recorded as amortized wall time per row and
    the full batch wall time is retained as ``batch_latency_s``.
    """
    if not questions:
        return []
    if len(questions) != len(gold_answers):
        raise ValueError("questions and gold_answers must have the same length")

    prompts = [official_qwen_prompt(question, ratio) for question in questions]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    device = model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
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
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        else:
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    input_width = encoded["input_ids"].shape[1]
    amortized_latency = elapsed / len(questions)
    results: list[dict] = []
    for index, (question, gold_answer) in enumerate(zip(questions, gold_answers)):
        new_ids = generated[index, input_width:]
        output = tokenizer.decode(new_ids, skip_special_tokens=False)
        cot = output.split("\n\nThe final answer is:")[0]
        cot_length = int(tokenizer(cot, return_tensors="pt")["input_ids"].shape[1])
        prediction = extract_answer(question, output, task="cot")
        prediction = prediction if isinstance(prediction, str) else ""
        correct = evaluate_answer(
            {"prediction": prediction, "answer": gold_answer.replace(",", "")}
        )
        results.append(
            {
                "ratio": ratio,
                "output": output,
                "prediction": prediction,
                "correct": bool(correct),
                "generated_tokens": int(new_ids.shape[0]),
                "reasoning_tokens": cot_length,
                "latency_s": round(amortized_latency, 4),
                "batch_latency_s": round(elapsed, 4),
                "batch_size": len(questions),
            }
        )
    return results


def summarize(rows: list[dict], key: str) -> dict:
    items = [row[key] for row in rows]
    correct = sum(item["correct"] for item in items)
    tokens = sum(item["reasoning_tokens"] for item in items)
    latency = sum(item["latency_s"] for item in items)
    return {
        "n": len(items),
        "correct": correct,
        "accuracy": round(correct / len(items), 4) if items else None,
        "avg_reasoning_tokens": round(tokens / len(items), 2) if items else None,
        "total_latency_s": round(latency, 3),
        "tokens_per_correct": round(tokens / correct, 2) if correct else None,
    }
