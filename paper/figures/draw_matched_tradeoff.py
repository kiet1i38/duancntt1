"""Draw a vector figure at the manuscript's column width."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FIGURES = Path(__file__).resolve().parent
RESULTS = FIGURES.parents[1] / "experiments/matched-20261107/analysis/analysis.json"


def main():
    models = json.loads(RESULTS.read_text(encoding="utf-8"))["models"]
    with plt.rc_context({"font.size": 8, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.15), layout="constrained")
        for ax, model, title in zip(axes, models, ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct")):
            assert models[model]["status"] == "complete"
            metrics = models[model]["metrics"]
            ratios = [f"{g / 10:.1f}" for g in range(5, 11)]
            fixed = [metrics[f"fixed_{g}"] for g in ratios]
            x = [m["mean_cot_tokens"] for m in fixed]
            y = [m["accuracy_percent"] for m in fixed]
            ax.plot(x, y, "o-", color="#0072b2", linewidth=1.1, markersize=3.5, label="Fixed ratio")
            for gamma, xt, yt in zip(ratios, x, y):
                offset = (5, -13) if model == 'qwen2.5-3b' and gamma == '0.8' else (-5, 5)
                ax.annotate(gamma, (xt, yt), xytext=offset, textcoords="offset points", fontsize=7)
            sd = metrics["sd_frozen"]
            ax.plot(sd["mean_cot_tokens"], sd["accuracy_percent"], "*", color="#bf4200",
                    markersize=8, label="Frozen SD", zorder=4)
            ax.set_title(title, fontsize=8.5)
            ax.set_xlabel("Mean CoT tokens")
            ax.set_ylabel("Accuracy (%)")
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
            ax.margins(x=.08, y=.15)
            ax.grid(alpha=.22)
            ax.legend(loc="upper left", fontsize=7, frameon=False, borderaxespad=.3)
        fig.savefig(FIGURES / "matched_tradeoff.pdf", metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)


if __name__ == "__main__":
    main()
