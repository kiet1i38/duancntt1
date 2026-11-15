"""Plot only complete matched results; no inferred or partial test points."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "matched-20261107"
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--analysis", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
data = json.loads(args.analysis.read_text(encoding="utf-8"))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
for ax, model in zip(axes, ("qwen2.5-3b", "qwen2.5-7b")):
    result = data["models"][model]
    ax.set_title(model.replace("qwen2.5-", "Qwen2.5 ").upper())
    ax.set_xlabel("Mean CoT tokens")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=.2)
    if result["status"] != "complete":
        ax.text(.5, .5, "Complete matched results unavailable", transform=ax.transAxes,
                ha="center", va="center")
        continue
    metrics = result["metrics"]
    fixed = [(float(k.removeprefix("fixed_")), v) for k, v in metrics.items() if k.startswith("fixed_")]
    fixed.sort(key=lambda x: x[1]["mean_cot_tokens"])
    ax.plot([v["mean_cot_tokens"] for _, v in fixed],
            [v["accuracy_percent"] for _, v in fixed], "o-", color="#0072B2", label="Fixed gamma")
    for g, value in fixed:
        ax.annotate(f"{g:.1f}", (value["mean_cot_tokens"], value["accuracy_percent"]),
                    xytext=(4, 6), textcoords="offset points", fontsize=9)
    sd = metrics["sd_frozen"]
    ax.scatter(sd["mean_cot_tokens"], sd["accuracy_percent"], marker="*", s=190,
               color="#D55E00", edgecolor="black", linewidth=.5, label="Frozen SD", zorder=5)
    ax.legend(loc="best", frameon=False)
fig.suptitle("Fresh matched GSM8K evaluation · 1,319 questions per condition", fontsize=12)
args.output.mkdir(parents=True, exist_ok=True)
for ext in ("png", "svg"):
    target = args.output / f"matched_tradeoff.{ext}"
    fig.savefig(target, dpi=180, metadata={"Date":None} if ext=="svg" else None)
    if ext == "svg":
        target.write_text("\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8",newline="\n")
plt.close(fig)
print(args.output / "matched_tradeoff.png")
