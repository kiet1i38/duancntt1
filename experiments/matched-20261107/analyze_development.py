"""Read-only analysis of saved SD-TokenSkip development outputs.

No inference, network calls, policy publication, or manuscript edits.
All comparisons and intervals are exploratory; fixed choices examined after
validation exposure are diagnostics, not newly registered confirmatory tests.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "analysis"
OUT.mkdir(exist_ok=True)
PROJECT = ROOT / "sd-tokenskip-cleanroom-v3"
RUN = PROJECT / "runs" / "sd-v2-v3-gamma05-10-20261025"
GAMMAS = tuple(round(g / 10, 1) for g in range(5, 11))
MODELS = ("qwen2.5-3b", "qwen2.5-7b")
RESAMPLES = 20_000
SEED = 20261107
INPUTS: dict[str, dict] = {}


def remember(path):
    blob = path.read_bytes()
    INPUTS[path.relative_to(ROOT).as_posix()] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
    return blob


def js(path):
    return json.loads(remember(path).decode("utf-8-sig"))


def jl(path):
    rows = [json.loads(s) for s in remember(path).decode("utf-8-sig").splitlines() if s]
    assert len({r["source_row_index"] for r in rows}) == len(rows), path
    return sorted(rows, key=lambda r: r["source_row_index"])


script = PROJECT / "scripts" / "run_sd_tokenskip_v2_v3_model.py"
remember(script)
spec = importlib.util.spec_from_file_location("saved_runner", script)
runner = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(runner)


def arrays(rows):
    return (np.asarray([int(r["accuracy"]) for r in rows]),
            np.asarray([int(r["cot_length"]) for r in rows]))


def measure(rows, baseline):
    y, t = arrays(rows)
    _, bt = arrays(baseline)
    return {"n": len(rows), "correct": int(y.sum()), "accuracy_percent": float(y.mean() * 100),
            "mean_cot_tokens": float(t.mean()),
            "reduction_vs_gamma1_percent": float(100 * (1 - t.mean() / bt.mean())),
            "utility_lambda_0.1": float(100 * y.mean() + 10 * (1 - t.mean() / bt.mean()))}


def paired(candidate, reference, baseline, label):
    assert [r["source_row_index"] for r in candidate] == [r["source_row_index"] for r in reference]
    assert [r["question_sha256"] for r in candidate] == [r["question_sha256"] for r in reference]
    cy, ct = arrays(candidate)
    ry, rt = arrays(reference)
    _, bt = arrays(baseline)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(candidate), size=(RESAMPLES, len(candidate)))
    da = (cy[idx].mean(axis=1) - ry[idx].mean(axis=1)) * 100
    dt = ct[idx].mean(axis=1) - rt[idx].mean(axis=1)
    reduction = (1 - ct[idx].mean(axis=1) / rt[idx].mean(axis=1)) * 100
    du = da - 10 * dt / bt[idx].mean(axis=1)
    gained = int(((cy == 1) & (ry == 0)).sum())
    lost = int(((cy == 0) & (ry == 1)).sum())
    n_discord = gained + lost
    p = min(1., 2 * sum(math.comb(n_discord, k) for k in range(min(gained, lost) + 1)) / 2 ** n_discord)
    ci = lambda a: [float(v) for v in np.quantile(a, [.025, .975])]
    return {"label": label, "n": len(candidate), "candidate_correct_reference_wrong": gained,
            "candidate_wrong_reference_correct": lost,
            "delta_accuracy_pp": float(100 * (cy.mean() - ry.mean())), "delta_accuracy_pp_ci95": ci(da),
            "delta_mean_cot_tokens": float(ct.mean() - rt.mean()), "delta_mean_cot_tokens_ci95": ci(dt),
            "token_reduction_vs_reference_percent": float(100 * (1 - ct.mean() / rt.mean())),
            "token_reduction_vs_reference_percent_ci95": ci(reduction),
            "delta_utility_lambda_0.1": float(100 * (cy.mean() - ry.mean()) - 10 * (ct.mean() - rt.mean()) / bt.mean()),
            "delta_utility_ci95": ci(du), "mcnemar_exact_two_sided_p_unadjusted": p,
            "scope": "Exploratory pointwise intervals; not adjusted for selection or multiple comparisons."}


def load_stage(rd, stage, n):
    rows_by_gamma = {f"{g:.1f}": jl(rd / "development" / stage / f"gamma-{g:.1f}" / "predictions.jsonl") for g in GAMMAS}
    routes = jl(rd / "development" / stage / "routing_assignments.jsonl")
    assert len(routes) == n
    ids = [r["source_row_index"] for r in routes]
    hashes = [r["question_sha256"] for r in routes]
    for g, rows in rows_by_gamma.items():
        assert len(rows) == n
        assert [r["source_row_index"] for r in rows] == ids
        assert [r["question_sha256"] for r in rows] == hashes
        assert all(float(r["gamma"]) == float(g) for r in rows)
    return rows_by_gamma, {r["source_row_index"]: r for r in routes}


summary = {"date": "2026-11-07", "scope": "Exploratory CPU recomputation of archived outputs; no fresh inference.",
           "resamples": RESAMPLES, "seed": SEED, "models": {}}
all_metrics = []
all_sensitivity = []

for model in MODELS:
    rd = RUN / model
    frozen = js(rd / "frozen-v3-policy.json")
    fit_candidates = js(rd / "fit-policy-candidates.json")
    difficulty = js(rd / "v3-difficulty-spec.json")
    stages = {stage: load_stage(rd, stage, n) for stage, n in (("fit", 540), ("tuning", 180), ("validation", 180))}
    result = {"stages": {}, "comparisons": [], "family_selection": {}, "medium_sensitivity": {}}
    stage_selected = {}
    for stage, (grid, routes) in stages.items():
        _, chosen = runner.evaluate_grid_policy(grid, routes, frozen, difficulty, .1)
        stage_selected[stage] = chosen
        measured = {f"fixed_{g}": measure(rows, grid["1.0"]) for g, rows in grid.items()}
        measured["SD_frozen"] = measure(chosen, grid["1.0"])
        result["stages"][stage] = measured
        for method, m in measured.items():
            all_metrics.append({"model": model, "split": stage, "method": method, **m})
        if stage in ("fit", "tuning"):
            ids = {r["source_row_index"] for r in grid["1.0"]}
            assert not ids & {r["source_row_index"] for r in stages["validation"][0]["1.0"]}

    tuning_grid, tuning_routes = stages["tuning"]
    val_grid, val_routes = stages["validation"]
    best_u = max(GAMMAS, key=lambda g: (result["stages"]["tuning"][f"fixed_{g:.1f}"]["utility_lambda_0.1"],
                                       result["stages"]["tuning"][f"fixed_{g:.1f}"]["correct"], g))
    best_acc = max(GAMMAS, key=lambda g: (result["stages"]["tuning"][f"fixed_{g:.1f}"]["correct"],
                                         -result["stages"]["tuning"][f"fixed_{g:.1f}"]["mean_cot_tokens"], g))
    result["tuning_selected_fixed"] = {"utility": best_u, "accuracy_then_fewer_tokens": best_acc}
    diagnostic_gamma = .8 if model.endswith("3b") else .6
    result["comparisons"].append(paired(val_grid[f"{diagnostic_gamma:.1f}"], stage_selected["validation"], val_grid["1.0"],
                                        f"validation fixed {diagnostic_gamma} minus frozen SD; teacher diagnostic"))
    result["comparisons"].append(paired(val_grid[f"{best_u:.1f}"], stage_selected["validation"], val_grid["1.0"],
                                        f"validation tuning-selected fixed {best_u} minus frozen SD"))
    result["comparisons"].append(paired(tuning_grid[f"{best_u:.1f}"], stage_selected["tuning"], tuning_grid["1.0"],
                                        f"tuning best-fixed {best_u} minus frozen SD; in-sample selection"))

    family_rows = {k: runner.evaluate_grid_policy(tuning_grid, tuning_routes, p, difficulty, .1)[1]
                   for k, p in fit_candidates.items()}
    ranked = sorted(family_rows, key=lambda k: measure(family_rows[k], tuning_grid["1.0"])["utility_lambda_0.1"], reverse=True)
    assert ranked[0] == frozen["selected_family_key"]
    for key, rows in family_rows.items():
        saved = frozen["tuning_metrics"][key]
        actual = measure(rows, tuning_grid["1.0"])
        assert abs(actual["utility_lambda_0.1"] - saved["utility"]) < 1e-9
    family_compare = paired(family_rows[ranked[0]], family_rows[ranked[1]], tuning_grid["1.0"],
                            "original winning router family minus original runner-up; conditional on fitted maps")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 180, size=(RESAMPLES, 180))
    _, bt = arrays(tuning_grid["1.0"])
    scores = []
    for k in ranked:
        y, t = arrays(family_rows[k])
        scores.append(100 * y[idx].mean(axis=1) + 10 * (1 - t[idx].mean(axis=1) / bt[idx].mean(axis=1)))
    wins = np.argmax(np.stack(scores), axis=0)
    result["family_selection"] = {"original_ranked_keys": ranked, "original_gap_comparison": family_compare,
                                  "bootstrap_winner_fractions_conditional_on_original_fit_maps":
                                  {k: float((wins == i).mean()) for i, k in enumerate(ranked)},
                                  "note": "Diagnostic only. Does not re-fit maps or account for post-selection uncertainty; exact ties prefer original rank."}

    medium = [i for i, route in val_routes.items() if route["feature_risk_group"] == "medium"]
    for gamma in ("0.5", "1.0"):
        subset = [r for r in val_grid[gamma] if r["source_row_index"] in medium]
        result["medium_sensitivity"][gamma] = {"n": len(subset), "correct": sum(r["accuracy"] for r in subset),
                                               "mean_cot_tokens": float(np.mean([r["cot_length"] for r in subset]))}
    base_y = {r["source_row_index"]: r["accuracy"] for r in val_grid["1.0"]}
    low_y = {r["source_row_index"]: r["accuracy"] for r in val_grid["0.5"]}
    result["medium_sensitivity"]["lost_base_correct"] = sum(base_y[i] and not low_y[i] for i in medium)
    result["medium_sensitivity"]["recovered_base_wrong"] = sum(not base_y[i] and low_y[i] for i in medium)

    fit_grid, fit_routes = stages["fit"]
    for lam in (0., .025, .05, .1, .15, .2, .3, .5):
        candidates = runner.fit_policy_families(fit_grid, fit_routes, lam)
        scored = {k: runner.evaluate_grid_policy(tuning_grid, tuning_routes, p, difficulty, lam)[0]
                  for k, p in candidates.items()}
        winner = max(candidates, key=lambda k: (round(scored[k]["utility"], 12),
                                                scored[k]["candidate_accuracy_percent"],
                                                runner.policy_family_tie_key(k, candidates[k])))
        fixed_scores = {g: runner.score_utility(tuning_grid["1.0"], tuning_grid[f"{g:.1f}"], lam) for g in GAMMAS}
        fg = max(fixed_scores, key=lambda g: (round(fixed_scores[g]["utility"], 12), fixed_scores[g]["candidate_accuracy_percent"], g))
        frozen_family = runner.fit_group_policy(family_key=frozen["family_key"], family_name=frozen["policy_family"],
                                                group_field="feature_risk_group", fit_grid=fit_grid,
                                                fit_assignments=fit_routes, lam=lam, fallback_out_of_range=True)
        if lam == .1:
            assert frozen_family["gamma_by_group"] == frozen["gamma_by_group"]
        all_sensitivity.append({"model": model, "lambda": lam, "best_router_family": winner,
                                "best_router_tuning_utility": scored[winner]["utility"], "best_fixed_gamma": fg,
                                "best_fixed_tuning_utility": fixed_scores[fg]["utility"],
                                "fixed_minus_router_tuning_utility": fixed_scores[fg]["utility"] - scored[winner]["utility"],
                                "six_feature_refitted_map": json.dumps(frozen_family["gamma_by_group"], sort_keys=True),
                                "scope": "Retrospective sensitivity, fit refit and tuning rerank only; no validation/test selection."})

    summary["models"][model] = result

summary["inputs"] = INPUTS
(OUT / "saved_development_analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
for name, rows in (("development_comparisons.csv", all_metrics), ("lambda_sensitivity.csv", all_sensitivity)):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

print(json.dumps({"output": str(OUT), "input_file_count": len(INPUTS), "models": {
    m: {"validation_teacher_diagnostic": r["comparisons"][0], "tuning_fixed_choices": r["tuning_selected_fixed"],
        "family_gap": r["family_selection"]["original_gap_comparison"],
        "family_bootstrap_winners": r["family_selection"]["bootstrap_winner_fractions_conditional_on_original_fit_maps"]}
    for m, r in summary["models"].items()}}, ensure_ascii=False, indent=2))
