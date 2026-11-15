"""CPU analysis of complete fresh matched outputs; JSON/CSV only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    rows = sorted(rows, key=lambda r: r["source_row_index"])
    assert [r["source_row_index"] for r in rows] == list(range(1319)), str(path)
    return rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def arrays(rows):
    return np.array([bool(r["accuracy"]) for r in rows], dtype=float), np.array([r["cot_length"] for r in rows], dtype=float)


def wilson(k, n):
    z = 1.959963984540054
    p = k / n; den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [100*(center-half), 100*(center+half)]


def summary(rows, base):
    y, t = arrays(rows); _, bt = arrays(base)
    k = int(y.sum()); n = len(y)
    return {"n": n, "correct": k, "accuracy_percent": float(y.mean()*100),
            "accuracy_wilson_ci95": wilson(k, n), "mean_cot_tokens": float(t.mean()),
            "reduction_vs_gamma1_percent": float(100*(1-t.mean()/bt.mean())),
            "utility_lambda_0.1": float(y.mean()*100 + 10*(1-t.mean()/bt.mean())),
            "length_limited_outputs": sum(r.get("finish_reason") == "length" for r in rows),
            "invalid_predictions": sum(not r.get("prediction") for r in rows)}


def paired(a, b, baseline=None):
    assert [r["question_sha256"] for r in a] == [r["question_sha256"] for r in b]
    ay, at = arrays(a); by, bt = arrays(b)
    _, baseline_t = arrays(b if baseline is None else baseline)
    rng = np.random.default_rng(4242)
    da, red, du = [], [], []
    for _ in range(100):
        ix = rng.integers(0, len(a), size=(200, len(a)))
        delta_a = (ay[ix]-by[ix]).mean(axis=1)*100
        da.extend(delta_a)
        red.extend((1-at[ix].mean(axis=1)/bt[ix].mean(axis=1))*100)
        du.extend(delta_a + 10*(bt[ix]-at[ix]).mean(axis=1)/baseline_t[ix].mean(axis=1))
    lost = int(((ay == 0) & (by == 1)).sum())
    gained = int(((ay == 1) & (by == 0)).sum())
    n = lost + gained
    p = min(1., 2*sum(math.comb(n, k) for k in range(min(lost,gained)+1))/2**n)
    return {"n": len(a), "delta_accuracy_pp": float((ay-by).mean()*100),
            "delta_accuracy_ci95": [float(x) for x in np.quantile(da,[.025,.975])],
            "cot_reduction_percent": float(100*(1-at.mean()/bt.mean())),
            "cot_reduction_ci95": [float(x) for x in np.quantile(red,[.025,.975])],
            "delta_utility_lambda_0.1":float((ay-by).mean()*100 + 10*(bt-at).mean()/baseline_t.mean()),
            "delta_utility_ci95":[float(x) for x in np.quantile(du,[.025,.975])],
            "a_wrong_b_correct": lost, "a_correct_b_wrong": gained,
            "mcnemar_exact_two_sided_p": p,
            "ci_scope": "pointwise question bootstrap; conditional on frozen policies and assignments"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = read(Path(__file__).parent / "experiment_plan.json")
    result = {"scope": "Fresh matched inference on reused GSM8K; no policy fitting or test-based selection",
              "plan_sha256": sha(Path(__file__).parent / "experiment_plan.json"), "models": {}}
    table = []; pairs_table = []; group_table = []; provenance = {}
    primary = []
    for model in plan["models"]:
        folder = args.results / model
        if not (folder / "manifest.json").exists():
            result["models"][model] = {"status": "NO_LOCAL_MANIFEST"}; continue
        manifest = read(folder / "manifest.json")
        if manifest["status"] != "complete":
            result["models"][model] = {"status": manifest["status"], "complete": False}; continue
        names = plan["method_order"][model]
        methods = {name: read_rows(folder / f"{name}.jsonl") for name in names}
        for name in names:
            provenance[f"{model}/{name}.jsonl"] = sha(folder / f"{name}.jsonl")
        base = methods["fixed_1.0"]
        direct = methods["sd_frozen"]
        assert read(folder / "pilot.json")["upstream_same_batch_parity"]
        routes = sorted(read(folder / "routing.json"), key=lambda r:r["source_row_index"])
        assert [r["question_sha256"] for r in routes] == [r["question_sha256"] for r in direct]
        assert all(r["gamma"] == d["gamma"] for r,d in zip(routes,direct))
        def replay(gammas):
            return [methods[f"fixed_{g:.1f}"][i] for i,g in enumerate(gammas)]
        g = [r["gamma"] for r in routes]
        grid_sd = replay(g)
        methods["sd_grid_replay"] = grid_sd
        different = [i for i,(a,b) in enumerate(zip(direct,grid_sd)) if a["model_output"] != b["model_output"]]
        correct_different = [i for i,(a,b) in enumerate(zip(direct,grid_sd)) if a["accuracy"] != b["accuracy"]]
        for seed in plan["offline_controls"]["random_histogram_seeds"]:
            shuffled = list(g); random.Random(seed).shuffle(shuffled)
            methods[f"random_histogram_{seed}"] = replay(shuffled)
            nonfallback = [i for i,r in enumerate(routes) if not r["feature_out_of_range"]]
            sub = [g[i] for i in nonfallback]; random.Random(seed).shuffle(sub)
            pinned = list(g)
            for i,value in zip(nonfallback,sub): pinned[i] = value
            methods[f"random_fallback_pinned_{seed}"] = replay(pinned)
        metrics = {name:summary(rows,base) for name,rows in methods.items()}
        comparisons = {}
        for name,rows in methods.items():
            if name == "sd_frozen": continue
            comparisons[f"sd_frozen_vs_{name}"] = paired(direct,rows,base)
            pairs_table.append({"model":model,"a":"sd_frozen","b":name,
                                **{k:json.dumps(v) if isinstance(v,list) else v for k,v in comparisons[f"sd_frozen_vs_{name}"].items()}})
        # The routing-value comparison uses grid-replay SD for matching engine/batch origin.
        for name,rows in methods.items():
            if name.startswith("random_"):
                comparisons[f"sd_grid_replay_vs_{name}"] = paired(grid_sd,rows,base)
        primary.append((model, comparisons["sd_frozen_vs_fixed_0.5"]["mcnemar_exact_two_sided_p"]))
        for name,m in metrics.items():
            table.append({"model":model,"method":name,"origin":"direct inference" if name in names else "offline grid replay",
                          **{k:json.dumps(v) if isinstance(v,list) else v for k,v in m.items()}})
        for group in ("easy","medium","hard"):
            indices = [i for i,r in enumerate(routes) if r["feature_risk_group"] == group]
            for name in names:
                subset = [methods[name][i] for i in indices]
                group_base = [base[i] for i in indices]
                m = summary(subset, group_base)
                losses = sum(base[i]["accuracy"] and not methods[name][i]["accuracy"] for i in indices)
                recoveries = sum(not base[i]["accuracy"] and methods[name][i]["accuracy"] for i in indices)
                baseline_correct = sum(bool(base[i]["accuracy"]) for i in indices)
                loss_ci = wilson(losses, baseline_correct) if baseline_correct else [None, None]
                group_table.append({"model":model,"group":group,"method":name,"n":m["n"],"correct":m["correct"],
                                    "accuracy_percent":m["accuracy_percent"],"mean_cot_tokens":m["mean_cot_tokens"],
                                    "lost_vs_gamma1":losses,"recovered_vs_gamma1":recoveries,
                                    "baseline_correct":baseline_correct,
                                    "loss_rate_among_baseline_correct_percent":100*losses/baseline_correct if baseline_correct else None,
                                    "loss_rate_wilson_ci95":json.dumps(loss_ci),
                                    "delta_accuracy_vs_gamma1_pp":100*(recoveries-losses)/len(indices),
                                    "feature_out_of_range_n":sum(bool(routes[i]["feature_out_of_range"]) for i in indices),
                                    "scope":"Exploratory group description; frozen test assignments"})
        result["models"][model] = {"status":"complete","metrics":metrics,"comparisons":comparisons,
                                   "direct_sd_vs_grid_replay": {"output_differences":len(different),
                                      "correctness_differences":len(correct_different),"different_ids":different},
                                   "manifest_sha256":sha(folder / "manifest.json")}
    sorted_primary = sorted(primary, key=lambda item:item[1])
    adjusted = {}; previous = 0.
    for rank,(model,pvalue) in enumerate(sorted_primary):
        previous = max(previous, min(1., (len(sorted_primary)-rank)*pvalue))
        adjusted[model] = previous
    result["primary_accuracy_tests"] = {"comparison":"direct SD vs fixed 0.5 selected on archived tuning",
                                       "holm_adjusted_pvalues":adjusted,
                                       "note":"Only complete downloaded models are included; family must contain both models for final report."}
    result["source_hashes"] = provenance
    atomic(args.output / "analysis.json",result)
    csv_write(args.output / "metrics.csv",table)
    csv_write(args.output / "paired_comparisons.csv",pairs_table)
    csv_write(args.output / "group_comparisons.csv",group_table)
    print(json.dumps({"models":{k:v["status"] for k,v in result["models"].items()},"rows":len(table),"output":str(args.output)},indent=2))


if __name__ == "__main__":
    main()
