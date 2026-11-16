"""Render the development ablation from the saved exploratory analysis."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES = Path(__file__).resolve().parent
METHODS = [('length_only','Length only'),('three_features','Three features'),
           ('six_features','Six features'),('six_no_fallback','Six, no envelope'),
           ('fold_selected_fixed','Fold-selected fixed')]


def main():
    data=json.loads((ROOT/'experiments/review-20261113/analysis.json').read_text(encoding='utf8'))
    for size in ('3b','7b'):
        metrics=data['development_ablation']['qwen2.5-'+size]['methods']
        lines=['% Generated from the review analysis.']
        for key,label in METHODS:
            m=metrics[key];lo,hi=m['delta_utility_ci95']
            lines.append(f"{label} & {m['correct']}/{m['n']} & {m['accuracy_percent']:.2f} & {m['mean_cot_tokens']:.2f} & {m['delta_utility_vs_fixed']:+.2f} [{lo:+.2f}, {hi:+.2f}]"+r' \\')
        lines.append(r'\bottomrule')
        (FIGURES/f'review_{size}_rows.tex').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')


if __name__ == '__main__': main()
