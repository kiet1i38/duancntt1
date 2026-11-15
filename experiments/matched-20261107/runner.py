"""Matched fixed-gamma and frozen SD evaluation; no fitting on test data.

Uses the archived prompt, answer extractor, correctness evaluator, and CoT
token definition. Generation uses Transformers' native EOS termination rather
than the archived redundant Python scan over every generated token prefix.
All fresh conditions share this engine and are reported separately from old runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "bundle"
sys.path.insert(0, str(BUNDLE))
N = 1319
STOP = False


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def emit(**value):
    print(json.dumps({"utc": now(), **value}, ensure_ascii=False), flush=True)


def request_stop(*_):
    global STOP
    STOP = True


def bundle_check():
    expected = read(BUNDLE / "bundle_manifest.json")
    for name, sha in expected.items():
        assert digest(BUNDLE / name) == sha, name
    assert digest(BUNDLE / "test.jsonl") == "fd32e8e549626b0dcce350eec7bbace9c463ddbcfa1c0303ba5abe9122450839"
    return expected


def load_batches(out, method):
    rows = []
    for path in sorted((out / "batches" / method).glob("*.json")):
        rows.extend(read(path)["rows"])
    return sorted(rows, key=lambda r: r["source_row_index"])


def export(out, names):
    metrics = {}
    for name in names:
        rows = load_batches(out, name)
        assert len({r["source_row_index"] for r in rows}) == len(rows)
        if rows:
            p = out / f"{name}.jsonl"
            tmp = p.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            tmp.replace(p)
        metrics[name] = {"n": len(rows), "complete": len(rows) == N,
                         "correct": sum(int(r["accuracy"]) for r in rows),
                         "accuracy_percent": 100 * sum(int(r["accuracy"]) for r in rows) / len(rows) if rows else None,
                         "mean_cot_tokens": sum(r["cot_length"] for r in rows) / len(rows) if rows else None,
                         "gamma_counts": dict(Counter(str(r["gamma"]) for r in rows))}
    atomic(out / "metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("qwen2.5-3b", "qwen2.5-7b"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--deadline-utc", type=float, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    expected = bundle_check()
    plan = read(ROOT / "experiment_plan.json")
    method_order = plan["method_order"][args.model]
    if args.preflight_only:
        emit(event="bundle_preflight_pass", inputs=len(expected), model=args.model)
        return
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.export_only:
        emit(event="export", metrics=export(out, method_order))
        return
    if time.time() >= args.deadline_utc or (ROOT / "STOP").exists():
        emit(event="stop_before_loading", model=args.model)
        return
    import numpy as np
    import torch
    import common
    import v3
    from data_processing.answer_extraction import extract_last_single_answer
    from eval.eval_script import eval_last_single_answer
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert args.batch_size == plan["batch_size"] == 16
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_properties(0).total_memory >= 23 * 1024 ** 3
    common.set_seed(42)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    config = read(BUNDLE / "source_protocol.json")["models"][args.model]
    policy = read(BUNDLE / f"policies/{args.model}.json")
    environment = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0),
                   "packages": {k: importlib.metadata.version(k) for k in
                                ("transformers", "peft", "accelerate", "numpy", "sympy", "regex", "tokenizers", "huggingface-hub")}}
    contract = {"model": config, "model_name": args.model, "dtype": "float16", "seed": 42,
                "batch_size": args.batch_size, "max_new_tokens": 512, "do_sample": False,
                "attn_implementation": "sdpa", "adapter": "merged once for every condition, including gamma=1.0",
                "generation_engine": "Transformers native EOS; full decoded prompt removed as in upstream helper",
                "cot_tokens": "tokenizer(output before final-answer separator), same archived definition",
                "runner_sha256": digest(Path(__file__)), "bundle_sha256": digest(BUNDLE / "bundle_manifest.json"),
                "policy_sha256": digest(BUNDLE / f"policies/{args.model}.json"),
                "experiment_plan_sha256": digest(ROOT / "experiment_plan.json"), "environment": environment}
    contract_hash = stable(contract)
    manifest_path = out / "manifest.json"
    old = read(manifest_path) if manifest_path.exists() else {}
    if old:
        assert old["contract_sha256"] == contract_hash, "Cannot mix changed source, environment, or settings"
    manifest = {"contract": contract, "contract_sha256": contract_hash,
                "started_utc": old.get("started_utc", now()), "last_resume_utc": now(), "status": "loading"}
    atomic(manifest_path, manifest)
    emit(event="loading_model", model=args.model, environment=environment)
    started_load = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["model_revision"],
                                                torch_dtype=torch.float16, device_map={"": "cuda:0"},
                                                attn_implementation="sdpa", low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, config["adapter_id"], revision=config["adapter_revision"])
    model = model.merge_and_unload()
    model.eval()
    assert all(p.device.type == "cuda" for p in model.parameters())
    assert all(p.dtype == torch.float16 for p in model.parameters() if p.is_floating_point())
    emit(event="model_loaded", model=args.model, seconds=time.monotonic() - started_load,
         allocated_gb=torch.cuda.memory_allocated() / 1e9)
    raw = common.read_jsonl(BUNDLE / "test.jsonl")
    rows = common.prepare_gsm8k_rows(raw, "test")
    assert len(rows) == N
    assignments = v3.build_v3_assignments(rows, tokenizer, common, policy["difficulty_spec"])
    gammas = {i: v3.v3_gamma_for_assignment(assignments[i], policy) for i in range(N)}
    archive = {r["source_row_index"]: r for r in common.read_jsonl(ROOT / "archived_routes" / f"{args.model}.jsonl")}
    assert set(archive) == set(assignments) == set(range(N))
    for i in range(N):
        assert assignments[i]["question_sha256"] == archive[i]["question_sha256"]
        assert gammas[i] == archive[i]["gamma"], (i, gammas[i], archive[i]["gamma"])
    atomic(out / "routing.json", [{**assignments[i], "gamma": gammas[i]} for i in range(N)])

    def generate(selected, gamma):
        prompts = [common.build_prompt(r, tokenizer, gamma) for r in selected]
        tokenized = tokenizer(prompts, padding="longest", return_tensors="pt", add_special_tokens=False)
        batch_input_ids = tokenized.input_ids.to("cuda")
        mask = tokenized.attention_mask.to("cuda")
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            outputs = model.generate(input_ids=batch_input_ids, attention_mask=mask,
                                     max_new_tokens=512, do_sample=False, temperature=0.0, top_p=1.0,
                                     use_cache=True, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
        # Preserve upstream decoded-string prompt removal and final EOS cleanup.
        decoded_full = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
        generated = [a[len(b):] for a, b in zip(decoded_full, decoded_prompts)]
        output_rows = []
        for j, (row, prompt, text) in enumerate(zip(selected, prompts, generated)):
            assert text.strip(), f"Empty generation: {row['source_row_index']}"
            cot = text.split("\n\nThe final answer is:")[0]
            prediction = extract_last_single_answer(common.question_text(row), text, task="cot")
            token_ids = outputs[j, batch_input_ids.shape[1]:].tolist()
            eos_ids = model.generation_config.eos_token_id
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
            first_eos = next((k for k, token_id in enumerate(token_ids) if token_id in eos_ids), None)
            result = {"id": row["id"], "source_row_index": row["source_row_index"],
                      "question_sha256": assignments[row["source_row_index"]]["question_sha256"],
                      "prompt": prompt, "gamma": gamma, "model_output": text, "prediction": prediction,
                      "answer": row["answer"], "cot_length": int(tokenizer(cot, return_tensors="pt")["input_ids"].shape[1]),
                      "generated_token_count_excluding_eos": first_eos if first_eos is not None else len(token_ids),
                      "finish_reason": "eos" if first_eos is not None else "length"}
            result["accuracy"] = bool(prediction) and bool(eval_last_single_answer(result))
            output_rows.append(result)
        return output_rows, seconds

    # Pilot validates worst-case batch and engine parity against the archived helper.
    pilot_path = out / "pilot.json"
    if not pilot_path.exists():
        if time.time() >= args.deadline_utc or STOP:
            manifest["status"] = "stopped_before_pilot"; atomic(manifest_path, manifest); return
        pilot, seconds = generate(rows[:16], 1.0)
        # Three same-batch examples check native EOS against the unchanged upstream engine.
        check_old, _ = common.generate_group(model, tokenizer, rows[:3], 0.8, 3, 512)
        check_new, _ = generate(rows[:3], 0.8)
        parity = all(a["model_output"] == b["model_output"] and a["cot_length"] == b["cot_length"]
                     and a["accuracy"] == b["accuracy"] for a, b in zip(check_old, check_new))
        atomic(pilot_path, {"rows": pilot, "batch_size": 16, "generation_seconds": seconds,
                           "upstream_same_batch_parity_n": 3, "upstream_same_batch_parity": parity,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        assert parity, "Native EOS pilot diverged from upstream; stop before full inference"
        emit(event="pilot_pass", model=args.model, seconds_per_16=seconds,
             rough_full_seven_conditions_hours=seconds / 16 * N * 7 / 3600,
             peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, upstream_parity=parity)

    manifest["status"] = "running"; atomic(manifest_path, manifest)
    progress = {m: len(load_batches(out, m)) for m in method_order}
    # Conditions ordered in the frozen plan; SD groups each gamma into batches.
    for name in method_order:
        if name == "sd_frozen":
            groups = {g: [r for r in rows if gammas[r["source_row_index"]] == g] for g in sorted(set(gammas.values()))}
        else:
            gamma = float(name.removeprefix("fixed_"))
            groups = {gamma: rows}
        for gamma, group in groups.items():
            for start in range(0, len(group), args.batch_size):
                chosen = group[start:start + args.batch_size]
                target = out / "batches" / name / f"gamma-{gamma:.1f}-{start:05d}.json"
                indices = [r["source_row_index"] for r in chosen]
                if target.exists():
                    saved = read(target)
                    assert saved["contract_sha256"] == contract_hash
                    assert [r["source_row_index"] for r in saved["rows"]] == indices
                    assert all(r["gamma"] == gamma and r["model_output"].strip()
                               and r["question_sha256"] == assignments[r["source_row_index"]]["question_sha256"]
                               for r in saved["rows"])
                    continue
                if STOP or time.time() >= args.deadline_utc or (ROOT / "STOP").exists():
                    manifest.update(status="stopped_at_checkpoint", updated_utc=now())
                    atomic(manifest_path, manifest); export(out, method_order)
                    emit(event="checkpoint_stop", model=args.model, reason="deadline_or_stop_request")
                    return
                torch.cuda.reset_peak_memory_stats()
                generated, seconds = generate(chosen, gamma)
                atomic(target, {"contract_sha256": contract_hash, "utc": now(), "rows": generated,
                                "generation_seconds": seconds, "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
                progress[name] += len(generated)
                atomic(out / "progress.json", {"utc": now(), "model": args.model, "counts": progress,
                                                "last_method": name, "last_generation_seconds": seconds})
                emit(event="batch_complete", model=args.model, method=name, gamma=gamma,
                     n=len(generated), seconds=round(seconds, 3), counts=progress)
        metrics = export(out, method_order)
        assert metrics[name]["complete"], name
        emit(event="condition_complete", model=args.model, method=name, metrics=metrics[name])
    metrics = export(out, method_order)
    assert all(m["complete"] for m in metrics.values())
    manifest.update(status="complete", completed_utc=now())
    atomic(manifest_path, manifest)
    emit(event="model_complete", model=args.model, metrics=metrics)


if __name__ == "__main__":
    main()
