"""Bootstrap the 540-question fit split and refit the six-feature gamma map."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'sd-tokenskip-cleanroom-v3/runs/sd-v2-v3-gamma05-10-20261025'
FEATURES = ('question_token_count', 'numeric_span_count', 'operator_count',
            'clause_connector_count', 'numeric_unit_pair_count', 'relation_marker_count')
GAMMAS = (.5, .6, .7, .8, .9, 1.)
SEED, DRAWS = 20261115, 5000


def read(path):
    return sorted((json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s),
                  key=lambda r: r['source_row_index'])


def mapping(x, y, t, indices):
    x, y, t = x[indices], y[indices], t[indices]
    sd = x.std(axis=0)
    sd[sd == 0] = 1
    score = ((x - x.mean(axis=0)) / sd).mean(axis=1)
    bins = np.searchsorted(np.quantile(score, [.5, .7]), score, side='left')
    lower, upper = np.quantile(x, [.01, .99], axis=0)
    eligible = np.all((x >= lower) & (x <= upper), axis=1)
    baseline = t[:, -1].sum()
    chosen = []
    for group in range(3):
        mask = (bins == group) & eligible
        if mask.sum() < 30:
            chosen.append(1.0)
            continue
        correct = y[mask].sum(axis=0)
        score = 100 * correct / len(indices) - 10 * t[mask].sum(axis=0) / baseline
        j = max(range(6), key=lambda j: (round(float(score[j]), 12), int(correct[j]), GAMMAS[j]))
        chosen.append(GAMMAS[j])
    return tuple(chosen)


def main():
    results = {}
    for model in ('qwen2.5-3b', 'qwen2.5-7b'):
        fit = RUN / model / 'development/fit'
        route = read(fit / 'routing_assignments.jsonl')
        grid = [read(fit / f'gamma-{g:.1f}/predictions.jsonl') for g in GAMMAS]
        assert len(route) == 540 and all(len(g) == 540 for g in grid)
        assert all([r['question_sha256'] for r in g] == [r['question_sha256'] for r in route] for g in grid)
        x = np.array([[r['features'][f] for f in FEATURES] for r in route], dtype=float)
        y = np.array([[int(g[i]['accuracy']) for g in grid] for i in range(540)])
        t = np.array([[g[i]['cot_length'] for g in grid] for i in range(540)])
        policy = json.loads((RUN / model / 'frozen-v3-policy.json').read_text(encoding='utf8'))
        original = mapping(x, y, t, np.arange(540))
        assert original == tuple(policy['gamma_by_group'][g] for g in ('easy', 'medium', 'hard')), (model, original)
        rng = np.random.default_rng(SEED)
        counts = Counter(mapping(x, y, t, rng.integers(0, 540, 540)) for _ in range(DRAWS))
        results[model] = {'original_map': original, 'bootstrap_draws': DRAWS,
                          'map_counts': {','.join(map(str, k)): v for k, v in counts.most_common()},
                          'original_map_frequency': counts[original] / DRAWS}
    out = Path(__file__).with_name('fit_stability.json')
    out.write_text(json.dumps({'seed': SEED, 'scope': 'Exploratory fit resampling; refits normalization, quantiles, envelope and gamma map within the fixed six-feature family; does not reselect policy family or evaluate test.', 'models': results}, indent=2) + '\n', encoding='utf8')
    print(out)


if __name__ == '__main__':
    main()
