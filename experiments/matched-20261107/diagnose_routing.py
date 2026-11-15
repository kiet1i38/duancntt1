"""Separate allocation value, compression sensitivity, and oracle headroom."""
from pathlib import Path
import json

import numpy as np

from analyze_results import read_rows
from reproduce import verify

ROOT = Path(__file__).resolve().parent
GAMMAS = (.5, .6, .7, .8, .9, 1.)


def auc(scores, positive):
    differences = scores[positive, None] - scores[None, ~positive]
    return float(((differences > 0) + .5 * (differences == 0)).mean())


def permutation_test(utility, chosen, permutable, seed=20261107):
    """Permute only the given positions, holding the ratio histogram fixed."""
    observed = float(utility[np.arange(len(chosen)), chosen].mean())
    rng = np.random.default_rng(seed)
    permuted = chosen.copy()
    scores = np.empty(10_000)
    for i in range(len(scores)):
        permuted[permutable] = rng.permutation(chosen[permutable])
        scores[i] = utility[np.arange(len(chosen)), permuted].mean()
    return {
        "observed_utility": observed,
        "permutation_mean": float(scores.mean()),
        "observed_minus_random_mean": float(observed - scores.mean()),
        "central_95_percent_of_random_utilities": np.quantile(scores, [.025, .975]).tolist(),
        "upper_tail_mc_p": float((1 + (scores >= observed).sum()) / (1 + len(scores))),
        "permutations": len(scores), "seed": seed,
    }


def diagnose(model):
    path = ROOT / "results" / model
    grid = [read_rows(path / f"fixed_{gamma:.1f}.jsonl") for gamma in GAMMAS]
    routes = sorted(json.loads((path / "routing.json").read_text(encoding="utf-8")),
                    key=lambda row: row["source_row_index"])
    y = np.array([[row["accuracy"] for row in condition] for condition in grid], dtype=bool).T
    tokens = np.array([[row["cot_length"] for row in condition] for condition in grid]).T
    utility = 100 * y + 10 * (1 - tokens / tokens[:, -1].mean())
    chosen = np.array([GAMMAS.index(row["gamma"]) for row in routes])
    fallback = np.array([row["feature_out_of_range"] for row in routes])
    scores = np.array([row["feature_risk_score"] for row in routes])
    baseline_correct = y[:, -1]
    oracle = utility.argmax(axis=1)
    return {
        "risk_auc_for_error_at_1.0": auc(scores, ~baseline_correct),
        "risk_auc_for_compression_loss_among_gamma1_correct": {
            f"{gamma:.1f}": auc(scores[baseline_correct], ~y[baseline_correct, j])
            for j, gamma in enumerate(GAMMAS[:-1])
        },
        "all_ratio_assignments_shuffled": permutation_test(utility, chosen, np.ones(len(y), bool)),
        "fallback_positions_fixed": permutation_test(utility, chosen, ~fallback),
        "fallback_count": int(fallback.sum()),
        "oracle": {"accuracy_percent": 100 * float(y[np.arange(len(y)), oracle].mean()),
                   "mean_cot_tokens": float(tokens[np.arange(len(y)), oracle].mean()),
                   "utility": float(utility.max(axis=1).mean()),
                   "utility_minus_best_fixed_on_this_grid": float(utility.max(axis=1).mean() - utility.mean(axis=0).max())},
    }


if __name__ == "__main__":
    verify()
    result = {"scope": "Exploratory on reused test grid. Oracle uses labels. Permutation p values are unadjusted and conditional on recorded outputs.",
              "models": {model: diagnose(model) for model in ("qwen2.5-3b", "qwen2.5-7b")}}
    output = ROOT / "analysis/routing_diagnostics.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))
