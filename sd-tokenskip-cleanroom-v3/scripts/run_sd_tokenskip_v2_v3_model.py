#!/usr/bin/env python3
"""Run a V2 frozen-policy test continuation and the registered V3 pipeline for one model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


GAMMAS = tuple(round(index / 10.0, 1) for index in range(5, 11))
GAMMA_BANDS = (
    (0, 4, (0.5,), "easy"),
    (5, 6, (0.6, 0.7), "medium"),
    (7, 9, (0.8, 0.9, 1.0), "hard"),
)
BASE_INPUT_FEATURES = ("question_token_count", "numeric_span_count", "operator_count")
ADDITIONAL_INPUT_FEATURES = ("clause_connector_count", "numeric_unit_pair_count", "relation_marker_count")
ALL_INPUT_FEATURES = BASE_INPUT_FEATURES + ADDITIONAL_INPUT_FEATURES
POLICY_GROUPS = ("easy", "medium", "hard")
MIN_GROUP_POLICY_SUPPORT = 30
CLAUSE_CONNECTOR_RE = re.compile(r"[;:]|,(?=\s)|\b(?:and|then|but|while|before|after|if|when)\b", re.IGNORECASE)
NUMERIC_UNIT_PAIR_RE = re.compile(
    r"(?<![\w.])(?:[$€£]\s*)?\d[\d,]*(?:\.\d+)?\s*(?:\b(?:dollars?|cents?|miles?|kilometers?|km|meters?|feet|foot|inches?|pounds?|lbs?|kilograms?|kg|grams?|liters?|gallons?|hours?|minutes?|seconds?|days?|weeks?|months?|years?|percent)\b|%)",
    re.IGNORECASE,
)
RELATION_MARKER_RE = re.compile(
    r"\b(?:times\s+as\s+(?:many|much)|more\s+than|less\s+than|in\s+all|altogether|each|per|remaining|left|ratio|half|twice|double|total|every)\b",
    re.IGNORECASE,
)
TRAIN_SHA256 = "eb28ef6d91a70f791eec906f9a7af0bac57f6dedb94ad0c75a296d490c09ead5"
TEST_SHA256 = "fd32e8e549626b0dcce350eec7bbace9c463ddbcfa1c0303ba5abe9122450839"
SOURCE_REVISION = "7b958cdb98e0b2a84056e26e885b260987f2e64f"
EVALUATOR_SHA256 = "1c1f18c1ec86a8f6bd1e81e20c87104b7b7ed8f55e2221fc74b430423880b4b3"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def source_index(row: dict[str, Any]) -> int:
    if "source_row_index" in row:
        return int(row["source_row_index"])
    return int(str(row["id"]).rsplit("-", 1)[-1])


def question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def selected_gamma_values() -> list[float]:
    return [float(value) for value in GAMMAS]


def policy_hash(policy: dict[str, Any]) -> str:
    return stable_json_hash(policy)


def expected_model_entry(config: dict[str, Any], model_name: str) -> dict[str, str]:
    return config["models"][model_name]


def local_source_revision(official: Path) -> str | None:
    if not (official / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(official), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_preflight(repo_root: Path, config: dict[str, Any], split_manifest: dict[str, Any], model_name: str) -> dict[str, Any]:
    official = repo_root / "sd-tokenskip-cleanroom" / "source" / "TokenSkip-official"
    train_path = official / "datasets" / "gsm8k" / "train.jsonl"
    test_path = official / "datasets" / "gsm8k" / "test.jsonl"
    evaluator_path = official / "evaluation.py"
    if not train_path.is_file() or not test_path.is_file() or not evaluator_path.is_file():
        raise FileNotFoundError("Official TokenSkip source and GSM8K files must already exist in the clean-room")
    train_hash = sha256_file(train_path)
    test_hash = sha256_file(test_path)
    evaluator_hash = sha256_file(evaluator_path)
    if train_hash != TRAIN_SHA256 or test_hash != TEST_SHA256 or evaluator_hash != EVALUATOR_SHA256:
        raise ValueError("Official train/test/evaluator hashes differ from the audited V2 artifacts")
    revision = local_source_revision(official)
    if revision is not None and revision != SOURCE_REVISION:
        raise ValueError(f"Official source revision mismatch: expected {SOURCE_REVISION}, found {revision}")
    if split_manifest["dataset"]["sha256"] != train_hash:
        raise ValueError("V3 splits were not made from the current official GSM8K train file")
    source_inventory_path = repo_root / "sd-tokenskip-cleanroom-v3" / "manifests" / "source_inventory.json"
    preserved_path = repo_root / "sd-tokenskip-cleanroom-v3" / "manifests" / "preserved_v1_v2_hashes.json"
    source_inventory = read_json(source_inventory_path)
    preserved_manifest = read_json(preserved_path)
    for item in source_inventory.get("v3_code_files", []):
        code_path = repo_root / item["path"]
        if not code_path.is_file() or sha256_file(code_path) != item["sha256"]:
            raise ValueError(f"V3 runner/code differs from the frozen source inventory: {item['path']}")
    split_sizes = {name: int(split_manifest["splits"][name]["n_rows"]) for name in ("fit", "tuning", "validation")}
    if split_sizes != {"fit": 540, "tuning": 180, "validation": 180}:
        raise ValueError(f"V3 split sizes do not match the registered contract: {split_sizes}")
    split_sets = {
        name: set(int(value) for value in split_manifest["splits"][name]["source_row_indices"])
        for name in ("fit", "tuning", "validation")
    }
    if split_sets["fit"] & split_sets["tuning"] or split_sets["fit"] & split_sets["validation"] or split_sets["tuning"] & split_sets["validation"]:
        raise ValueError("V3 fit/tuning/validation source IDs overlap")
    if not set(range(180)).issubset(split_sets["fit"]):
        raise ValueError("V2 180-row calibration is not fully contained in V3 fit")

    entry = expected_model_entry(config, model_name)
    v2_short = "3b" if model_name.endswith("3b") else "7b"
    v2_run = repo_root / "sd-tokenskip-cleanroom-v2" / "runs" / f"sd-{v2_short}-v2-main"
    manifest_path = v2_run / "run_manifest.json"
    frozen_path = v2_run / "frozen_policy.json"
    spec_path = v2_run / "difficulty_spec.json"
    v2_manifest = read_json(manifest_path)
    frozen_policy = read_json(frozen_path)
    difficulty_spec = read_json(spec_path)
    if v2_manifest.get("model_revision") != entry["model_revision"] or v2_manifest.get("adapter_revision") != entry["adapter_revision"]:
        raise ValueError("V2 model/adapter revision does not match the approved model-specific revision")
    if v2_manifest.get("source_revision") != SOURCE_REVISION:
        raise ValueError("V2 manifest source revision does not match the audited source")
    if frozen_policy.get("method_id") != "SD-TokenSkip-v2" or not frozen_policy.get("policy_locked_before_test"):
        raise ValueError("Existing V2 frozen policy is missing or was not locked before test")
    original_v2_gammas = tuple(round(float(value), 1) for value in frozen_policy.get("candidate_gammas", []))
    if not set(GAMMAS).issubset(set(original_v2_gammas)):
        raise ValueError("V2 frozen policy does not contain every gamma in the restricted 0.5-1.0 range")
    expected_hashes = v2_manifest.get("artifacts", {})
    for relative, path in (("frozen_policy.json", frozen_path), ("difficulty_spec.json", spec_path)):
        if relative in expected_hashes and expected_hashes[relative].get("sha256") != sha256_file(path):
            raise ValueError(f"V2 {relative} does not match its original manifest hash")
    calibration_dir = v2_run / "calibration"
    calibration_hashes: dict[str, str] = {}
    for gamma in GAMMAS:
        key = f"{gamma:.1f}"
        predictions_path = calibration_dir / f"gamma-{key}" / "predictions.jsonl"
        if not predictions_path.is_file():
            raise FileNotFoundError(f"Missing V2 calibration gamma artifact: {predictions_path}")
        rows = read_jsonl(predictions_path)
        ids = [str(row["id"]) for row in rows]
        if len(rows) != 180 or len(set(ids)) != 180 or set(ids) != {f"gsm8k-train-{index}" for index in range(180)}:
            raise ValueError(f"V2 calibration gamma={key} does not contain the expected 180 unique fit rows")
        if any(round(float(row["gamma"]), 1) != gamma for row in rows):
            raise ValueError(f"V2 calibration file gamma label mismatch for {key}")
        calibration_hashes[key] = sha256_file(predictions_path)
        relative = f"calibration/gamma-{key}/predictions.jsonl"
        if relative in expected_hashes and expected_hashes[relative].get("sha256") != calibration_hashes[key]:
            raise ValueError(f"V2 calibration output changed from the original manifest for gamma={key}")

    if len(frozen_policy.get("gamma_score_thresholds", [])) != 9:
        raise ValueError("V2 frozen policy score thresholds are incomplete")
    if frozen_policy["gamma_score_thresholds"] != difficulty_spec.get("gamma_score_thresholds"):
        raise ValueError("V2 frozen policy and difficulty specification have different thresholds")
    config_model = entry["model_id"]
    if not config_model.endswith("3B-Instruct") and not config_model.endswith("7B-Instruct"):
        raise ValueError(f"Unexpected model id in V3 configuration: {config_model}")
    return {
        "official_source_revision": revision or SOURCE_REVISION,
        "train_sha256": train_hash,
        "test_sha256": test_hash,
        "evaluator_sha256": evaluator_hash,
        "v2_original_runner_source_available_locally": (repo_root / "sd-tokenskip-cleanroom-v2" / "scripts" / "run_sd_tokenskip.py").is_file(),
        "v2_original_protocol_config_available_locally": (repo_root / "sd-tokenskip-cleanroom-v2" / "configs" / "sd_tokenskip_v2.json").is_file(),
        "v2_frozen_policy_sha256": sha256_file(frozen_path),
        "v2_difficulty_spec_sha256": sha256_file(spec_path),
        "v2_calibration_prediction_sha256_by_gamma": calibration_hashes,
        "v2_original_candidate_gammas": list(original_v2_gammas),
        "v2_effective_candidate_gammas": list(GAMMAS),
        "v2_gamma_floor": min(GAMMAS),
        "v2_calibration_reuse_count": 180 * len(GAMMAS),
        "v3_split_sizes": split_sizes,
        "v3_config_sha256": sha256_file(repo_root / "sd-tokenskip-cleanroom-v3" / "configs" / "protocol_v3.json"),
        "v3_split_manifest_sha256": sha256_file(repo_root / "sd-tokenskip-cleanroom-v3" / "data" / "splits_v3.json"),
        "v3_source_inventory_sha256": sha256_file(source_inventory_path),
        "preserved_v1_v2_manifest_sha256": sha256_file(preserved_path),
        "preserved_v1_v2_manifest_file_count": int(preserved_manifest["file_count"]),
    }


def load_common_helpers(repo_root: Path) -> Any:
    official = repo_root / "sd-tokenskip-cleanroom" / "source" / "TokenSkip-official"
    helper_dir = repo_root / "sd-tokenskip-cleanroom" / "scripts"
    sys.path.insert(0, str(official))
    sys.path.insert(0, str(helper_dir))
    import run_sd_tokenskip as common

    return common


def additional_question_features(text: str) -> dict[str, int]:
    """Cheap deterministic features from the question text only."""
    return {
        "clause_connector_count": 1 + len(CLAUSE_CONNECTOR_RE.findall(text)),
        "numeric_unit_pair_count": len(NUMERIC_UNIT_PAIR_RE.findall(text)),
        "relation_marker_count": len(RELATION_MARKER_RE.findall(text)),
    }


def out_of_fit_feature_envelope(features: dict[str, Any], spec: dict[str, Any]) -> bool:
    envelope = spec["out_of_fit_feature_envelope"]["bounds_by_feature"]
    return any(
        float(features[name]) < float(envelope[name]["lower"])
        or float(features[name]) > float(envelope[name]["upper"])
        for name in ALL_INPUT_FEATURES
    )


def fit_v3_difficulty(rows: list[dict[str, Any]], tokenizer: Any, common: Any) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    import numpy as np

    feature_names = BASE_INPUT_FEATURES
    weights = {"question_token_count": 0.7, "numeric_span_count": 0.2, "operator_count": 0.1}
    router_rows = [
        {
            "id": row["id"],
            "source_row_index": source_index(row),
            "messages": [message for message in row["messages"] if message["role"] == "user"],
        }
        for row in rows
    ]
    raw_features = []
    for router_row in router_rows:
        text = common.question_text(router_row)
        raw_features.append({**common.raw_features(router_row, tokenizer), **additional_question_features(text)})
    means = {name: float(np.mean([values[name] for values in raw_features])) for name in ALL_INPUT_FEATURES}
    stds = {name: float(np.std([values[name] for values in raw_features])) for name in ALL_INPUT_FEATURES}
    safe_stds = {name: value if value > 0.0 else 1.0 for name, value in stds.items()}

    assignments: dict[int, dict[str, Any]] = {}
    scores: list[float] = []
    feature_risk_scores: list[float] = []
    feature_risk_weights = {name: 1.0 / len(ALL_INPUT_FEATURES) for name in ALL_INPUT_FEATURES}
    for row, router_row, features in zip(rows, router_rows, raw_features):
        score = sum(weights[name] * (features[name] - means[name]) / safe_stds[name] for name in feature_names)
        feature_risk_score = sum(
            feature_risk_weights[name] * (features[name] - means[name]) / safe_stds[name]
            for name in ALL_INPUT_FEATURES
        )
        index = source_index(row)
        scores.append(float(score))
        feature_risk_scores.append(float(feature_risk_score))
        assignments[index] = {"id": row["id"], "source_row_index": index, "features": features, "difficulty_score": float(score)}
        assignments[index]["question_sha256"] = question_hash(common.question_text(router_row))
    quantiles = [index / 10.0 for index in range(1, 10)]
    thresholds = [float(np.quantile(np.asarray(scores, dtype=np.float64), q, method="linear")) for q in quantiles]
    feature_risk_thresholds = [
        float(np.quantile(np.asarray(feature_risk_scores, dtype=np.float64), q, method="linear")) for q in quantiles
    ]
    feature_envelope = {
        name: {
            "lower": float(np.quantile(np.asarray([row[name] for row in raw_features], dtype=np.float64), 0.01, method="linear")),
            "upper": float(np.quantile(np.asarray([row[name] for row in raw_features], dtype=np.float64), 0.99, method="linear")),
        }
        for name in ALL_INPUT_FEATURES
    }
    spec = {
        "feature_names": list(feature_names),
        "weights": weights,
        "normalization_fit_split": "fit inputs only",
        "fit_means": means,
        "fit_stds": stds,
        "safe_stds": safe_stds,
        "score_decile_quantiles": quantiles,
        "score_decile_thresholds": thresholds,
        "additional_feature_names": list(ADDITIONAL_INPUT_FEATURES),
        "additional_feature_definitions": {
            "clause_connector_count": "1 plus count of commas/semicolons/colons and selected clause connectors in the user question",
            "numeric_unit_pair_count": "count of numeric amounts explicitly paired with a fixed currency, distance, mass, volume, or time unit, percent word, or percent sign",
            "relation_marker_count": "count of a fixed list of ratio, comparison, remainder, multiplicative, and total markers in the user question",
        },
        "feature_risk_score": {
            "feature_names": list(ALL_INPUT_FEATURES),
            "weights": feature_risk_weights,
            "normalization_fit_split": "fit inputs only",
            "score_decile_thresholds": feature_risk_thresholds,
            "interpretation": "A lightweight structural feature ablation; not a learned correctness predictor.",
        },
        "out_of_fit_feature_envelope": {
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "bounds_fit_split": "fit inputs only",
            "bounds_by_feature": feature_envelope,
            "fallback_gamma": 1.0,
        },
        "tie_rule": "Assign the lowest decile whose upper score threshold is greater than or equal to the score; scores above q90 use decile 10.",
        "question_only_features": True,
    }
    for assignment, feature_risk_score in zip(assignments.values(), feature_risk_scores):
        assignment["score_decile"] = decile_for_score(float(assignment["difficulty_score"]), spec)
        assignment["difficulty_band"] = band_for_decile(assignment["score_decile"])
        assignment["feature_risk_score"] = feature_risk_score
        assignment["feature_risk_decile"] = decile_for_score(feature_risk_score, {"score_decile_thresholds": feature_risk_thresholds})
        assignment["feature_risk_group"] = band_for_decile(assignment["feature_risk_decile"])
        assignment["feature_out_of_range"] = out_of_fit_feature_envelope(assignment["features"], spec)
    return spec, assignments


def decile_for_score(score: float, spec: dict[str, Any]) -> int:
    for index, threshold in enumerate(spec["score_decile_thresholds"]):
        if score <= float(threshold):
            return index
    return 9


def band_for_decile(decile: int) -> str:
    for low, high, _gammas, label in GAMMA_BANDS:
        if low <= decile <= high:
            return label
    raise ValueError(f"Invalid score decile: {decile}")


def allowed_gammas(decile: int) -> tuple[float, ...]:
    for low, high, values, _label in GAMMA_BANDS:
        if low <= decile <= high:
            return values
    raise ValueError(f"Invalid score decile: {decile}")


def v3_gamma_for_assignment(assignment: dict[str, Any], policy: dict[str, Any]) -> float:
    if "gamma_by_group" in policy:
        group_field = policy["group_assignment_field"]
        group = str(assignment[group_field])
        fallback = policy.get("fallback_policy")
        if fallback:
            if bool(assignment.get("feature_out_of_range")):
                return round(float(fallback["gamma"]), 1)
            if group in fallback.get("fallback_groups", []):
                return round(float(fallback["gamma"]), 1)
        return round(float(policy["gamma_by_group"][group]), 1)

    decile = int(assignment["score_decile"])
    gamma = round(float(policy["gamma_by_score_decile"][str(decile + 1)]), 1)
    if gamma not in allowed_gammas(decile):
        raise ValueError(f"Frozen V3 policy selects gamma {gamma} outside the registered band for decile {decile + 1}")
    return gamma


def records_by_index(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = source_index(row)
        if index in result:
            raise ValueError(f"Duplicate prediction row index {index}")
        result[index] = row
    return result


def validate_records(rows: list[dict[str, Any]], expected_indices: set[int], expected_gamma: float) -> None:
    indexed = records_by_index(rows)
    if set(indexed) != expected_indices or len(rows) != len(expected_indices):
        raise ValueError(f"Prediction rows do not match expected indices for gamma={expected_gamma:.1f}")
    if any(round(float(row["gamma"]), 1) != expected_gamma for row in rows):
        raise ValueError(f"Prediction gamma labels do not match gamma={expected_gamma:.1f}")
    if any(not isinstance(row.get("model_output"), str) or not row["model_output"].strip() for row in rows):
        raise ValueError(f"Empty model output found at gamma={expected_gamma:.1f}")


def add_generation_metadata(rows: list[dict[str, Any]], assignments: dict[int, dict[str, Any]], method_id: str, run_id: str) -> list[dict[str, Any]]:
    decorated: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        index = source_index(row)
        assignment = assignments.get(index, {})
        row["source_row_index"] = index
        if "question_sha256" in assignment:
            row["question_sha256"] = assignment["question_sha256"]
        if "difficulty_score" in assignment:
            row["difficulty_score"] = float(assignment["difficulty_score"])
        if "score_decile" in assignment:
            row["score_decile"] = int(assignment["score_decile"]) + 1
            row["difficulty_band"] = assignment["difficulty_band"]
        row["method_id"] = method_id
        row["run_id"] = run_id
        decorated.append(row)
    return decorated


def write_gamma_files(
    *,
    stage_dir: Path,
    rows_for_gamma: dict[float, list[dict[str, Any]]],
    assignments: dict[int, dict[str, Any]],
    common: Any,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    max_new_tokens: int,
    method_id: str,
    run_id: str,
    reuse_for_gamma: dict[float, dict[int, dict[str, Any]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    reuse_for_gamma = reuse_for_gamma or {}
    outputs_by_gamma: dict[str, list[dict[str, Any]]] = {}
    for gamma in GAMMAS:
        gamma_key = f"{gamma:.1f}"
        gamma_started = time.monotonic()
        expected_rows = sorted(rows_for_gamma.get(gamma, []), key=source_index)
        expected_indices = {source_index(row) for row in expected_rows}
        gamma_dir = stage_dir / f"gamma-{gamma_key}"
        final_path = gamma_dir / "predictions.jsonl"
        if final_path.is_file():
            saved = read_jsonl(final_path)
            validate_records(saved, expected_indices, gamma)
            outputs_by_gamma[gamma_key] = sorted(saved, key=source_index)
            continue

        print(json.dumps({"event": "gamma_start", "stage": stage_dir.as_posix(), "gamma": gamma, "n_rows": len(expected_rows)}), flush=True)

        reused = reuse_for_gamma.get(gamma, {})
        reused_indices = set(reused) & expected_indices
        reused_path = gamma_dir / "reused-v2-calibration.jsonl"
        if reused_indices:
            if reused_path.is_file():
                reused_rows = read_jsonl(reused_path)
                validate_records(reused_rows, reused_indices, gamma)
            else:
                reused_rows = add_generation_metadata(
                    [reused[index] for index in sorted(reused_indices)], assignments, method_id, run_id
                )
                atomic_jsonl(reused_path, reused_rows)
        else:
            reused_rows = []

        rows_to_generate = [row for row in expected_rows if source_index(row) not in reused_indices]
        batch_dir = gamma_dir / "generated-batches"
        generated_rows: list[dict[str, Any]] = []
        for batch_number, offset in enumerate(range(0, len(rows_to_generate), batch_size)):
            batch = rows_to_generate[offset : offset + batch_size]
            expected_batch = {source_index(row) for row in batch}
            batch_path = batch_dir / f"batch-{batch_number:05d}.jsonl"
            if batch_path.is_file():
                batch_rows = read_jsonl(batch_path)
                validate_records(batch_rows, expected_batch, gamma)
            else:
                batch_rows, _elapsed = common.generate_group(model, tokenizer, batch, gamma, batch_size, max_new_tokens)
                batch_rows = add_generation_metadata(batch_rows, assignments, method_id, run_id)
                validate_records(batch_rows, expected_batch, gamma)
                atomic_jsonl(batch_path, batch_rows)
            generated_rows.extend(batch_rows)

        combined = sorted(reused_rows + generated_rows, key=source_index)
        if expected_indices:
            validate_records(combined, expected_indices, gamma)
        elif combined:
            raise ValueError(f"Unexpected generation output for empty gamma group {gamma_key}")
        atomic_jsonl(final_path, combined)
        outputs_by_gamma[gamma_key] = combined
        print(json.dumps({"event": "gamma_complete", "stage": stage_dir.as_posix(), "gamma": gamma, "n_rows": len(combined), "elapsed_seconds": round(time.monotonic() - gamma_started, 2)}), flush=True)
    return outputs_by_gamma


def score_utility(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], lam: float) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Utility comparison requires nonempty paired rows")
    base = sorted(baseline, key=source_index)
    cand = sorted(candidate, key=source_index)
    if [source_index(row) for row in base] != [source_index(row) for row in cand]:
        raise ValueError("Utility rows are not aligned by source row index")
    base_cot = sum(int(row["cot_length"]) for row in base) / len(base)
    cand_cot = sum(int(row["cot_length"]) for row in cand) / len(cand)
    base_correct = sum(bool(row.get("accuracy")) for row in base)
    cand_correct = sum(bool(row.get("accuracy")) for row in cand)
    base_accuracy = 100.0 * base_correct / len(base)
    cand_accuracy = 100.0 * cand_correct / len(cand)
    reduction = 100.0 * (1.0 - cand_cot / base_cot) if base_cot else 0.0
    return {
        "n": len(base),
        "baseline_correct": base_correct,
        "candidate_correct": cand_correct,
        "baseline_accuracy_percent": base_accuracy,
        "candidate_accuracy_percent": cand_accuracy,
        "accuracy_delta_percentage_points": cand_accuracy - base_accuracy,
        "baseline_mean_cot_tokens": base_cot,
        "candidate_mean_cot_tokens": cand_cot,
        "cot_token_reduction_percent": reduction,
        "utility": cand_accuracy + lam * reduction,
    }


def selected_records(grid: dict[str, list[dict[str, Any]]], assignments: dict[int, dict[str, Any]], gamma_for_index: dict[int, float]) -> list[dict[str, Any]]:
    maps = {key: records_by_index(rows) for key, rows in grid.items()}
    selected: list[dict[str, Any]] = []
    for index in sorted(gamma_for_index):
        gamma = round(float(gamma_for_index[index]), 1)
        selected.append(maps[f"{gamma:.1f}"][index])
    return selected


def policy_gamma_indices(assignments: dict[int, dict[str, Any]], spec: dict[str, Any], policy: dict[str, Any]) -> dict[int, float]:
    if policy.get("policy_family") == "SD-TokenSkip-v2-frozen-reference-with-gamma-floor":
        thresholds = policy["gamma_score_thresholds"]
        mapping = policy["gamma_by_decile"]
        result: dict[int, float] = {}
        for index, assignment in assignments.items():
            score = float(assignment["difficulty_score"])
            decile = 10
            for candidate_decile, threshold in enumerate(thresholds, 1):
                if score <= float(threshold):
                    decile = candidate_decile
                    break
            result[index] = round(float(mapping[str(decile)]), 1)
        return result
    return {index: v3_gamma_for_assignment(assignment, policy) for index, assignment in assignments.items()}


def fit_group_policy(
    *,
    family_key: str,
    family_name: str,
    group_field: str,
    fit_grid: dict[str, list[dict[str, Any]]],
    fit_assignments: dict[int, dict[str, Any]],
    lam: float,
    fallback_out_of_range: bool,
) -> dict[str, Any]:
    fit_indices = sorted(fit_assignments)
    gamma_maps = {key: records_by_index(rows) for key, rows in fit_grid.items()}
    baseline = sorted((gamma_maps["1.0"][index] for index in fit_indices), key=source_index)
    baseline_total_tokens = sum(int(row["cot_length"]) for row in baseline)
    if baseline_total_tokens <= 0:
        raise ValueError("Gamma 1.0 fit outputs have invalid CoT token totals")

    gamma_by_group: dict[str, float] = {}
    support_by_group: dict[str, int] = {}
    fallback_groups: list[str] = []
    for group in POLICY_GROUPS:
        group_indices = [
            index for index in fit_indices
            if str(fit_assignments[index][group_field]) == group
        ]
        eligible_indices = [
            index for index in group_indices
            if not fallback_out_of_range or not bool(fit_assignments[index].get("feature_out_of_range"))
        ]
        support_by_group[group] = len(eligible_indices)
        if not eligible_indices or (
            fallback_out_of_range and len(eligible_indices) < MIN_GROUP_POLICY_SUPPORT
        ):
            gamma_by_group[group] = 1.0
            fallback_groups.append(group)
            continue

        choices: list[tuple[tuple[float, int, float], float]] = []
        for gamma in GAMMAS:
            rows_by_index = gamma_maps[f"{gamma:.1f}"]
            correct = sum(bool(rows_by_index[index].get("accuracy")) for index in eligible_indices)
            cot_sum = sum(int(rows_by_index[index]["cot_length"]) for index in eligible_indices)
            utility_component = (
                (100.0 / len(fit_indices)) * correct
                - (100.0 * lam / baseline_total_tokens) * cot_sum
            )
            choices.append(((round(utility_component, 12), correct, gamma), gamma))
        gamma_by_group[group] = max(choices, key=lambda item: item[0])[1]

    selected: list[dict[str, Any]] = []
    for index in fit_indices:
        assignment = fit_assignments[index]
        group = str(assignment[group_field])
        use_fallback = fallback_out_of_range and (
            bool(assignment.get("feature_out_of_range")) or group in fallback_groups
        )
        gamma = 1.0 if use_fallback else gamma_by_group[group]
        selected.append(gamma_maps[f"{gamma:.1f}"][index])
    fit_metrics = score_utility(baseline, selected, lam)

    policy: dict[str, Any] = {
        "policy_family": family_name,
        "family_key": family_key,
        "group_assignment_field": group_field,
        "gamma_by_group": gamma_by_group,
        "fit_group_support": support_by_group,
        "fit_utility": fit_metrics["utility"],
        "fit_metrics": fit_metrics,
        "uses_all_registered_candidate_gamma_values": True,
        "monotonicity_constraint": False,
    }
    if fallback_out_of_range:
        policy["fallback_policy"] = {
            "gamma": 1.0,
            "fallback_groups": fallback_groups,
            "minimum_nonfallback_fit_rows_per_group": MIN_GROUP_POLICY_SUPPORT,
            "trigger": "input feature outside fit q1-q99 envelope or fewer than 30 in-envelope fit rows in the assigned group",
            "fit_envelope_quantiles": [0.01, 0.99],
        }
    return policy


def policy_family_tie_key(name: str, policy: dict[str, Any]) -> tuple[int, tuple[float, ...]]:
    priority = {
        "fit_optimized_three_group_all_gammas": 0,
        "fit_optimized_nondecreasing": 1,
        "fit_optimized_independent": 2,
        "fit_optimized_feature_risk_group_with_fallback": 3,
    }
    if "gamma_by_group" in policy:
        gamma_order = tuple(float(policy["gamma_by_group"][group]) for group in POLICY_GROUPS)
    else:
        gamma_order = tuple(
            float(policy["gamma_by_score_decile"][str(decile)])
            for decile in range(1, 11)
        )
    return -priority.get(name, 99), gamma_order


def fit_policy_families(
    fit_grid: dict[str, list[dict[str, Any]]],
    fit_assignments: dict[int, dict[str, Any]],
    lam: float,
) -> dict[str, dict[str, Any]]:
    fit_indices = sorted(fit_assignments)
    n = len(fit_indices)
    gamma_maps = {key: records_by_index(rows) for key, rows in fit_grid.items()}
    baseline_total_tokens = sum(int(gamma_maps["1.0"][index]["cot_length"]) for index in fit_indices)
    if baseline_total_tokens <= 0:
        raise ValueError("Gamma 1.0 fit outputs have invalid CoT token totals")

    contribution: dict[int, dict[float, tuple[float, int]]] = {}
    for decile in range(10):
        row_indices = [index for index in fit_indices if int(fit_assignments[index]["score_decile"]) == decile]
        contribution[decile] = {}
        for gamma in allowed_gammas(decile):
            output_rows = gamma_maps[f"{gamma:.1f}"]
            correct = sum(bool(output_rows[index].get("accuracy")) for index in row_indices)
            cot_sum = sum(int(output_rows[index]["cot_length"]) for index in row_indices)
            # Maximizing this sum is equivalent to maximizing accuracy% + lambda * token reduction%.
            utility_without_constant = (100.0 / n) * correct - (100.0 * lam / baseline_total_tokens) * cot_sum
            contribution[decile][gamma] = (utility_without_constant, correct)

    independent: list[float] = []
    for decile in range(10):
        choice = max(
            allowed_gammas(decile),
            key=lambda gamma: (round(contribution[decile][gamma][0], 12), contribution[decile][gamma][1], gamma),
        )
        independent.append(choice)

    # Dynamic program over score deciles; harder deciles cannot receive a lower gamma.
    states: dict[float, tuple[float, int, tuple[float, ...]]] = {}
    for gamma in allowed_gammas(0):
        score, correct = contribution[0][gamma]
        states[gamma] = (score, correct, (gamma,))
    for decile in range(1, 10):
        next_states: dict[float, tuple[float, int, tuple[float, ...]]] = {}
        for gamma in allowed_gammas(decile):
            feasible = [value for prior_gamma, value in states.items() if prior_gamma <= gamma]
            if not feasible:
                continue
            prior = max(feasible, key=lambda value: (round(value[0], 12), value[1], value[2]))
            add_score, add_correct = contribution[decile][gamma]
            next_states[gamma] = (prior[0] + add_score, prior[1] + add_correct, prior[2] + (gamma,))
        if not next_states:
            raise RuntimeError(f"No nondecreasing policy path reaches score decile {decile + 1}")
        states = next_states
    monotonic_state = max(states.values(), key=lambda value: (round(value[0], 12), value[1], value[2]))

    families = {
        "fit_optimized_nondecreasing": {
            "policy_family": "fit-optimized nondecreasing gamma by score decile",
            "gamma_by_score_decile": {str(index + 1): gamma for index, gamma in enumerate(monotonic_state[2])},
            "fit_objective_component_sum": monotonic_state[0],
        },
        "fit_optimized_independent": {
            "policy_family": "fit-optimized independent gamma by score decile within registered bands",
            "gamma_by_score_decile": {str(index + 1): gamma for index, gamma in enumerate(independent)},
            "fit_objective_component_sum": sum(contribution[index][gamma][0] for index, gamma in enumerate(independent)),
        },
    }
    families["fit_optimized_three_group_all_gammas"] = fit_group_policy(
        family_key="fit_optimized_three_group_all_gammas",
        family_name="fit-optimized independent gamma by easy/medium/hard difficulty group over the registered gamma range",
        group_field="difficulty_band",
        fit_grid=fit_grid,
        fit_assignments=fit_assignments,
        lam=lam,
        fallback_out_of_range=False,
    )
    families["fit_optimized_feature_risk_group_with_fallback"] = fit_group_policy(
        family_key="fit_optimized_feature_risk_group_with_fallback",
        family_name="fit-optimized independent gamma by six-feature risk group with input-only fallback",
        group_field="feature_risk_group",
        fit_grid=fit_grid,
        fit_assignments=fit_assignments,
        lam=lam,
        fallback_out_of_range=True,
    )
    return families


def v2_policy_reference(frozen_policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_family": "SD-TokenSkip-v2-frozen-reference-with-gamma-floor",
        "gamma_score_thresholds": list(frozen_policy["gamma_score_thresholds"]),
        "gamma_by_decile": {
            str(index + 1): max(min(GAMMAS), round((index + 1) / 10.0, 1))
            for index in range(10)
        },
        "candidate_gammas": list(GAMMAS),
        "minimum_gamma": min(GAMMAS),
        "gamma_floor_rule": "Map any gamma below the registered 0.5 minimum to 0.5 before inference.",
        "gamma_to_difficulty_group": {
            key: value
            for key, value in frozen_policy["gamma_to_difficulty_group"].items()
            if round(float(key), 1) in GAMMAS
        },
        "locked_policy_sha256": stable_json_hash(frozen_policy),
    }


def evaluate_grid_policy(
    grid: dict[str, list[dict[str, Any]]],
    assignments: dict[int, dict[str, Any]],
    policy: dict[str, Any],
    difficulty_spec: dict[str, Any],
    lam: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chosen_gammas = policy_gamma_indices(assignments, difficulty_spec, policy)
    selected = selected_records(grid, assignments, chosen_gammas)
    baseline = sorted(records_by_index(grid["1.0"]).values(), key=source_index)
    return score_utility(baseline, selected, lam), selected


def build_v3_assignments(rows: list[dict[str, Any]], tokenizer: Any, common: Any, difficulty_spec: dict[str, Any]) -> dict[int, dict[str, Any]]:
    assignments: dict[int, dict[str, Any]] = {}
    for row in rows:
        # The router receives only identity and the user-question message, never answer/reference/output fields.
        router_row = {"id": row["id"], "source_row_index": source_index(row), "messages": copy.deepcopy(row["messages"])}
        router_row["messages"] = [message for message in router_row["messages"] if message["role"] == "user"]
        question = common.question_text(router_row)
        features = {**common.raw_features(router_row, tokenizer), **additional_question_features(question)}
        score = sum(
            float(difficulty_spec["weights"][name])
            * (features[name] - float(difficulty_spec["fit_means"][name]))
            / float(difficulty_spec["safe_stds"][name])
            for name in difficulty_spec["feature_names"]
        )
        decile = decile_for_score(float(score), difficulty_spec)
        risk_spec = difficulty_spec["feature_risk_score"]
        feature_risk_score = sum(
            float(risk_spec["weights"][name])
            * (features[name] - float(difficulty_spec["fit_means"][name]))
            / float(difficulty_spec["safe_stds"][name])
            for name in risk_spec["feature_names"]
        )
        feature_risk_decile = decile_for_score(
            float(feature_risk_score),
            {"score_decile_thresholds": risk_spec["score_decile_thresholds"]},
        )
        index = source_index(row)
        assignments[index] = {
            "id": row["id"],
            "source_row_index": index,
            "question_sha256": question_hash(question),
            "features": features,
            "difficulty_score": float(score),
            "score_decile": decile,
            "difficulty_band": band_for_decile(decile),
            "feature_risk_score": float(feature_risk_score),
            "feature_risk_decile": feature_risk_decile,
            "feature_risk_group": band_for_decile(feature_risk_decile),
            "feature_out_of_range": out_of_fit_feature_envelope(features, difficulty_spec),
        }
    return assignments


def build_v2_assignments(rows: list[dict[str, Any]], tokenizer: Any, common: Any, difficulty_spec: dict[str, Any]) -> dict[int, dict[str, Any]]:
    assignments: dict[int, dict[str, Any]] = {}
    for row in rows:
        router_row = {"id": row["id"], "source_row_index": source_index(row), "messages": [message for message in row["messages"] if message["role"] == "user"]}
        item = common.score_one(router_row, tokenizer, difficulty_spec)
        index = source_index(row)
        gamma_before_floor = round(float(item["gamma"]), 1)
        gamma = max(min(GAMMAS), gamma_before_floor)
        if gamma not in GAMMAS:
            raise ValueError(f"V2 frozen policy produced unsupported gamma={gamma_before_floor:.1f}")
        assignments[index] = {
            **item,
            "gamma": gamma,
            "gamma_before_candidate_floor": gamma_before_floor,
            "candidate_gamma_floor_applied": gamma != gamma_before_floor,
            "source_row_index": index,
            "question_sha256": question_hash(common.question_text(router_row)),
            "score_decile": next((d for d, threshold in enumerate(difficulty_spec["gamma_score_thresholds"]) if float(item["difficulty_score"]) <= float(threshold)), 9),
            "difficulty_band": item["bin"],
        }
    return assignments


def load_reused_v2_fit_grid(repo_root: Path, v2_run: Path, fit_rows: list[dict[str, Any]]) -> dict[float, dict[int, dict[str, Any]]]:
    fit_indices = {source_index(row) for row in fit_rows}
    reusable_indices = fit_indices & set(range(180))
    if reusable_indices != set(range(180)):
        raise ValueError("V3 fit no longer contains all 180 V2 calibration rows")
    results: dict[float, dict[int, dict[str, Any]]] = {}
    for gamma in GAMMAS:
        file_path = v2_run / "calibration" / f"gamma-{gamma:.1f}" / "predictions.jsonl"
        rows = read_jsonl(file_path)
        indexed = records_by_index(rows)
        if set(indexed) != set(range(180)):
            raise ValueError(f"V2 fit reuse grid is incomplete for gamma={gamma:.1f}")
        results[gamma] = indexed
    return results


def stage_grid(
    *,
    run_dir: Path,
    split_name: str,
    rows: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
    common: Any,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    max_new_tokens: int,
    run_id: str,
    reuse_grid: dict[float, dict[int, dict[str, Any]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_gamma = {gamma: sorted(rows, key=source_index) for gamma in GAMMAS}
    decorated_assignment = assignments
    return write_gamma_files(
        stage_dir=run_dir / "development" / split_name,
        rows_for_gamma=rows_by_gamma,
        assignments=decorated_assignment,
        common=common,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        method_id="SD-TokenSkip-v3-development-grid",
        run_id=run_id,
        reuse_for_gamma=reuse_grid,
    )


def save_manifest(manifest_path: Path, manifest: dict[str, Any], completed_stage: str | None = None) -> None:
    if completed_stage:
        completed = manifest.setdefault("completed_stages", [])
        if completed_stage not in completed:
            completed.append(completed_stage)
    manifest["last_updated_utc"] = now_utc()
    atomic_json(manifest_path, manifest)


def load_run_dir(path: Path, resume: bool, manifest_identity: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    manifest_path = path / "run_manifest.json"
    if path.exists() and not resume:
        raise FileExistsError(f"Run directory already exists; pass --resume only for the same run: {path}")
    if path.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"Existing run has no manifest; refusing to reuse: {path}")
        manifest = read_json(manifest_path)
        for key, value in manifest_identity.items():
            if manifest.get(key) != value:
                raise ValueError(f"Resume identity mismatch for {key}")
        if manifest.get("status") == "complete":
            raise FileExistsError(f"Run is already complete: {path}")
        return manifest, manifest_path
    path.mkdir(parents=True, exist_ok=False)
    manifest = {
        **manifest_identity,
        "status": "running",
        "started_utc": now_utc(),
        "completed_stages": [],
        "test_answers_or_rationales_used_for_routing": False,
        "one_generation_per_test_row_and_condition": True,
        "test_benchmark_status": "reused GSM8K benchmark",
    }
    save_manifest(manifest_path, manifest)
    return manifest, manifest_path


def run(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    v3_root = repo_root / "sd-tokenskip-cleanroom-v3"
    config_path = v3_root / "configs" / "protocol_v3.json"
    split_path = v3_root / "data" / "splits_v3.json"
    config = read_json(config_path)
    split_manifest = read_json(split_path)
    entry = expected_model_entry(config, args.model)
    preflight = verify_preflight(repo_root, config, split_manifest, args.model)
    if args.model_revision != entry["model_revision"] or args.adapter_revision != entry["adapter_revision"]:
        raise ValueError("Command model/adapter revisions do not match the locked V3 config")
    if args.run_id and not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError("Run ID must contain only letters, digits, dot, underscore, or hyphen")
    run_id = args.run_id
    model_label = "3b" if args.model.endswith("3b") else "7b"
    run_dir = v3_root / "runs" / run_id / args.model
    v2_run = repo_root / "sd-tokenskip-cleanroom-v2" / "runs" / f"sd-{model_label}-v2-main"
    protocol = config["protocol"]
    identity = {
        "schema_version": "1.0",
        "run_id": run_id,
        "model_name": args.model,
        "model_id": entry["model_id"],
        "model_revision": entry["model_revision"],
        "adapter_id": entry["adapter_id"],
        "adapter_revision": entry["adapter_revision"],
        "source_revision": SOURCE_REVISION,
        "test_dataset_sha256": TEST_SHA256,
        "train_dataset_sha256": TRAIN_SHA256,
        "v3_config_sha256": sha256_file(config_path),
        "split_manifest_sha256": sha256_file(split_path),
        "v2_frozen_policy_sha256": preflight["v2_frozen_policy_sha256"],
        "v3_runner_sha256": sha256_file(Path(__file__).resolve()),
        "shared_generation_helper_sha256": sha256_file(repo_root / "sd-tokenskip-cleanroom" / "scripts" / "run_sd_tokenskip.py"),
        "v3_source_inventory_sha256": preflight["v3_source_inventory_sha256"],
        "preserved_v1_v2_manifest_sha256": preflight["preserved_v1_v2_manifest_sha256"],
    }
    manifest, manifest_path = load_run_dir(run_dir, args.resume, identity)
    manifest["preflight"] = preflight
    manifest["protocol"] = protocol
    manifest["utility"] = config["utility"]
    manifest["runtime_paths"] = {
        "model_path": str(Path(args.model_path).resolve()),
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
        "adapter_path": str(Path(args.adapter_path).resolve()),
    }
    save_manifest(manifest_path, manifest)

    common = load_common_helpers(repo_root)
    import torch

    common.set_seed(int(protocol["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no GPU inference was started")
    model_args = argparse.Namespace(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        adapter_path=args.adapter_path,
    )
    model, tokenizer = common.load_model(model_args)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token; cannot use the locked generation stopping rule")
    manifest["runtime_environment"] = {
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
    }
    save_manifest(manifest_path, manifest)

    official = repo_root / "sd-tokenskip-cleanroom" / "source" / "TokenSkip-official"
    raw_test = common.read_jsonl(official / "datasets" / "gsm8k" / "test.jsonl")
    test_rows = common.prepare_gsm8k_rows(raw_test, "test")
    if len(test_rows) != 1319 or len({row["id"] for row in test_rows}) != 1319:
        raise ValueError("GSM8K test file does not have exactly 1,319 unique rows")
    test_questions = {source_index(row): common.question_text(row) for row in test_rows}
    if set(test_questions) != set(range(1319)):
        raise ValueError("Prepared GSM8K test source indices do not cover 0 through 1,318")

    v2_policy = read_json(v2_run / "frozen_policy.json")
    v2_spec = read_json(v2_run / "difficulty_spec.json")
    v2_reference = v2_policy_reference(v2_policy)
    atomic_json(run_dir / "v2-effective-policy.json", v2_reference)
    effective_v2_policy_sha256 = sha256_file(run_dir / "v2-effective-policy.json")
    v2_assignments = build_v2_assignments(test_rows, tokenizer, common, v2_spec)
    v2_route_by_gamma: dict[float, list[dict[str, Any]]] = {gamma: [] for gamma in GAMMAS}
    v2_routing_records: list[dict[str, Any]] = []
    for row in test_rows:
        assignment = v2_assignments[source_index(row)]
        gamma = round(float(assignment["gamma"]), 1)
        v2_route_by_gamma[gamma].append(row)
        v2_routing_records.append({
            **assignment,
            "gamma": gamma,
            "policy_sha256": effective_v2_policy_sha256,
            "base_policy_sha256": preflight["v2_frozen_policy_sha256"],
        })
    atomic_jsonl(run_dir / "v2-test" / "routing_assignments.jsonl", v2_routing_records)
    v2_grid = write_gamma_files(
        stage_dir=run_dir / "v2-test",
        rows_for_gamma=v2_route_by_gamma,
        assignments=v2_assignments,
        common=common,
        model=model,
        tokenizer=tokenizer,
        batch_size=int(protocol["batch_size"]),
        max_new_tokens=int(protocol["max_new_tokens"]),
        method_id="SD-TokenSkip-v2-test-continuation",
        run_id=run_id,
    )
    v2_test_predictions = sorted(
        [row for gamma_rows in v2_grid.values() for row in gamma_rows], key=source_index
    )
    if len(v2_test_predictions) != 1319 or {source_index(row) for row in v2_test_predictions} != set(range(1319)):
        raise ValueError("V2 continuation did not produce exactly one test prediction for every row")
    atomic_jsonl(run_dir / "v2-test" / "predictions.jsonl", v2_test_predictions)
    save_manifest(manifest_path, manifest, "v2-frozen-policy-test")

    raw_train = common.read_jsonl(official / "datasets" / "gsm8k" / "train.jsonl")
    train_rows = common.prepare_gsm8k_rows(raw_train, "train")
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("fit", "tuning", "validation"):
        indices = [int(value) for value in split_manifest["splits"][split_name]["source_row_indices"]]
        split_rows[split_name] = [train_rows[index] for index in indices]
        if len(split_rows[split_name]) != int(split_manifest["splits"][split_name]["n_rows"]):
            raise ValueError(f"Prepared rows do not match V3 {split_name} split size")

    difficulty_spec, fit_assignments = fit_v3_difficulty(split_rows["fit"], tokenizer, common)
    stage_assignments: dict[str, dict[int, dict[str, Any]]] = {"fit": fit_assignments}
    for split_name in ("tuning", "validation"):
        stage_assignments[split_name] = build_v3_assignments(split_rows[split_name], tokenizer, common, difficulty_spec)
    v2_stage_assignments = {
        split_name: build_v2_assignments(split_rows[split_name], tokenizer, common, v2_spec)
        for split_name in ("tuning", "validation")
    }
    atomic_json(run_dir / "v3-difficulty-spec.json", difficulty_spec)
    for split_name, assignments in stage_assignments.items():
        atomic_jsonl(
            run_dir / "development" / split_name / "routing_assignments.jsonl",
            [assignments[index] for index in sorted(assignments)],
        )

    reused_grid = load_reused_v2_fit_grid(repo_root, v2_run, split_rows["fit"])
    fit_grid = stage_grid(
        run_dir=run_dir,
        split_name="fit",
        rows=split_rows["fit"],
        assignments=fit_assignments,
        common=common,
        model=model,
        tokenizer=tokenizer,
        batch_size=int(protocol["batch_size"]),
        max_new_tokens=int(protocol["max_new_tokens"]),
        run_id=run_id,
        reuse_grid=reused_grid,
    )
    save_manifest(manifest_path, manifest, "v3-fit-grid")
    lambda_value = float(config["utility"]["lambda"])
    families = fit_policy_families(fit_grid, fit_assignments, lambda_value)
    atomic_json(run_dir / "fit-policy-candidates.json", families)

    tuning_grid = stage_grid(
        run_dir=run_dir,
        split_name="tuning",
        rows=split_rows["tuning"],
        assignments=stage_assignments["tuning"],
        common=common,
        model=model,
        tokenizer=tokenizer,
        batch_size=int(protocol["batch_size"]),
        max_new_tokens=int(protocol["max_new_tokens"]),
        run_id=run_id,
    )
    tuning_results: dict[str, Any] = {}
    for name, policy in families.items():
        tuning_results[name], _ = evaluate_grid_policy(
            tuning_grid, stage_assignments["tuning"], policy, difficulty_spec, lambda_value
        )
    tuning_results["v2_frozen_reference"], _ = evaluate_grid_policy(
        tuning_grid, v2_stage_assignments["tuning"], v2_reference, v2_spec, lambda_value
    )
    selected_family = max(
        families,
        key=lambda name: (
            round(float(tuning_results[name]["utility"]), 12),
            float(tuning_results[name]["candidate_accuracy_percent"]),
            *policy_family_tie_key(name, families[name]),
        ),
    )
    frozen_policy = {
        "method_id": "SD-TokenSkip-v3",
        **families[selected_family],
        "selected_family_key": selected_family,
        "difficulty_spec": difficulty_spec,
        "difficulty_spec_sha256": sha256_file(run_dir / "v3-difficulty-spec.json"),
        "fit_rows": 540,
        "tuning_rows": 180,
        "selected_using": "tuning utility only",
        "selection_tie_break": "higher tuning utility; higher tuning accuracy; prefer three-group policy, then nondecreasing decile policy, then independent decile policy, then feature-risk fallback; finally higher gamma map",
        "tuning_metrics": tuning_results,
        "utility": config["utility"],
        "locked_before_validation": True,
        "locked_before_test": True,
        "test_answers_or_rationales_used_for_policy": False,
    }
    frozen_path = run_dir / "frozen-v3-policy.json"
    if frozen_path.exists():
        existing = read_json(frozen_path)
        if policy_hash(existing) != policy_hash(frozen_policy):
            raise ValueError("Recomputed V3 policy differs from the previously frozen policy; refusing to continue")
    else:
        atomic_json(frozen_path, frozen_policy)
    manifest["v3_policy_sha256"] = sha256_file(frozen_path)
    manifest["v3_selected_policy_family"] = selected_family
    save_manifest(manifest_path, manifest, "v3-policy-frozen-before-validation-and-test")

    validation_grid = stage_grid(
        run_dir=run_dir,
        split_name="validation",
        rows=split_rows["validation"],
        assignments=stage_assignments["validation"],
        common=common,
        model=model,
        tokenizer=tokenizer,
        batch_size=int(protocol["batch_size"]),
        max_new_tokens=int(protocol["max_new_tokens"]),
        run_id=run_id,
    )
    validation_metrics: dict[str, Any] = {}
    validation_metrics["selected_v3"], _ = evaluate_grid_policy(
        validation_grid, stage_assignments["validation"], frozen_policy, difficulty_spec, lambda_value
    )
    validation_metrics["v2_frozen_reference"], _ = evaluate_grid_policy(
        validation_grid, v2_stage_assignments["validation"], v2_reference, v2_spec, lambda_value
    )
    atomic_json(run_dir / "validation" / "metrics.json", validation_metrics)
    save_manifest(manifest_path, manifest, "v3-validation-reported-no-gate")

    v3_test_assignments = build_v3_assignments(test_rows, tokenizer, common, difficulty_spec)
    v3_gamma_by_index = policy_gamma_indices(v3_test_assignments, difficulty_spec, frozen_policy)
    v3_route_by_gamma: dict[float, list[dict[str, Any]]] = {gamma: [] for gamma in GAMMAS}
    v3_routing_records: list[dict[str, Any]] = []
    for row in test_rows:
        index = source_index(row)
        assignment = v3_test_assignments[index]
        gamma = v3_gamma_by_index[index]
        v3_route_by_gamma[gamma].append(row)
        v3_routing_records.append({**assignment, "gamma": gamma, "policy_sha256": sha256_file(frozen_path)})
    atomic_jsonl(run_dir / "v3-test" / "routing_assignments.jsonl", v3_routing_records)
    v3_test_grid = write_gamma_files(
        stage_dir=run_dir / "v3-test",
        rows_for_gamma=v3_route_by_gamma,
        assignments=v3_test_assignments,
        common=common,
        model=model,
        tokenizer=tokenizer,
        batch_size=int(protocol["batch_size"]),
        max_new_tokens=int(protocol["max_new_tokens"]),
        method_id="SD-TokenSkip-v3-test",
        run_id=run_id,
    )
    v3_test_predictions = sorted([row for rows in v3_test_grid.values() for row in rows], key=source_index)
    if len(v3_test_predictions) != 1319 or {source_index(row) for row in v3_test_predictions} != set(range(1319)):
        raise ValueError("V3 did not produce exactly one test prediction for every GSM8K row")
    atomic_jsonl(run_dir / "v3-test" / "predictions.jsonl", v3_test_predictions)
    test_comparison = score_utility(v2_test_predictions, v3_test_predictions, lambda_value)
    test_comparison["comparison"] = "V3 minus V2 on paired GSM8K test rows"
    test_comparison["interpretation"] = "Descriptive point estimate; paired 95% intervals are created by analyze_v2_v3.py."
    atomic_json(run_dir / "v3-test" / "v2_vs_v3_point_estimate.json", test_comparison)
    save_manifest(manifest_path, manifest, "v2-v3-test-inference")

    manifest["status"] = "complete"
    manifest["completed_utc"] = now_utc()
    manifest["artifacts"] = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json" and not path.name.endswith(".tmp"):
            manifest["artifacts"][path.relative_to(run_dir).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    save_manifest(manifest_path, manifest)
    print(json.dumps({
        "run_dir": str(run_dir),
        "model": args.model,
        "status": manifest["status"],
        "selected_v3_family": selected_family,
        "tuning_utility": tuning_results[selected_family]["utility"],
        "validation": validation_metrics,
        "test_v2_vs_v3": test_comparison,
        "evidence_label": "Fresh inference; not Reproduced result",
        "accuracy_gate": "none",
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen2.5-3b", "qwen2.5-7b"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not RUN_ID_RE.fullmatch(args.run_id):
        parser.error("--run-id contains unsupported characters")
    if args.dry_run:
        root = args.repo_root.resolve()
        config = read_json(root / "sd-tokenskip-cleanroom-v3" / "configs" / "protocol_v3.json")
        splits = read_json(root / "sd-tokenskip-cleanroom-v3" / "data" / "splits_v3.json")
        entry = expected_model_entry(config, args.model)
        if args.model_revision != entry["model_revision"] or args.adapter_revision != entry["adapter_revision"]:
            raise SystemExit("Dry-run failed: model or adapter revision does not match protocol_v3.json")
        summary = verify_preflight(root, config, splits, args.model)
        print(json.dumps({"dry_run": True, "model": args.model, "preflight": summary, "gpu_loaded": False, "generation_started": False}, indent=2))
        return
    try:
        run(args)
    except BaseException as error:
        run_dir = args.repo_root.resolve() / "sd-tokenskip-cleanroom-v3" / "runs" / args.run_id / args.model
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = read_json(manifest_path)
                manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                manifest["failure"] = {"type": type(error).__name__, "message": str(error), "recorded_utc": now_utc()}
                save_manifest(manifest_path, manifest)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
