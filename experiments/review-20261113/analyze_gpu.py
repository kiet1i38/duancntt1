"""Validate completed paired GPU batches and compute exploratory intervals."""
from pathlib import Path
import argparse
import json
import numpy as np

CONDITIONS = ('fixed_1.0', 'fixed_0.5', 'sd_frozen')


def load_condition(root, dataset, cap, condition, expected_n, contract):
    rows = []
    for path in sorted((root / dataset / str(cap) / condition).glob('*.json')):
        batch = json.loads(path.read_text(encoding='utf8'))
        assert batch['contract_sha256'] == contract, path
        rows.extend(batch['rows'])
    rows.sort(key=lambda r: r['source_row_index'])
    assert len(rows) == expected_n and [r['source_row_index'] for r in rows] == list(range(expected_n)), (root, dataset, cap, condition, len(rows))
    assert all(r['model_output'].strip() and r['finish_reason'] in ('eos', 'length') for r in rows)
    return rows


def compare(a, b, seed):
    assert [r['question_sha256'] for r in a] == [r['question_sha256'] for r in b]
    rng = np.random.default_rng(seed)
    n = len(a)
    da = np.array([int(x['accuracy']) - int(y['accuracy']) for x, y in zip(a, b)], dtype=float) * 100
    dt = np.array([x['cot_length'] - y['cot_length'] for x, y in zip(a, b)], dtype=float)
    indices = rng.integers(0, n, size=(10000, n))
    return {'delta_accuracy_pp': float(da.mean()), 'accuracy_ci95': np.quantile(da[indices].mean(axis=1), [.025, .975]).tolist(),
            'delta_cot_tokens': float(dt.mean()), 'cot_ci95': np.quantile(dt[indices].mean(axis=1), [.025, .975]).tolist(),
            'n': n}


def analyze(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
    assert manifest['status'] == 'complete', (root, manifest['status'])
    contract = manifest['contract_sha256']
    result, data = {}, {}
    for dataset, n, caps in (('gsm8k', 1319, (512, 1024)), ('svamp', 1000, (512,))):
        result[dataset] = {}
        for cap in caps:
            rows = {c: load_condition(root, dataset, cap, c, n, contract) for c in CONDITIONS}
            assert all([r['question_sha256'] for r in rows[c]] == [r['question_sha256'] for r in rows['fixed_1.0']] for c in CONDITIONS)
            data[(dataset, cap)] = rows
            metrics = {c: {'correct': sum(bool(x['accuracy']) for x in r), 'n': n,
                           'accuracy_percent': 100 * sum(bool(x['accuracy']) for x in r) / n,
                           'mean_cot_tokens': sum(x['cot_length'] for x in r) / n,
                           'length_stops': sum(x['finish_reason'] == 'length' for x in r)} for c, r in rows.items()}
            result[dataset][str(cap)] = {'metrics': metrics,
                'sd_minus_fixed_05': compare(rows['sd_frozen'], rows['fixed_0.5'], 42),
                'sd_minus_fixed_10': compare(rows['sd_frozen'], rows['fixed_1.0'], 42)}
        if dataset == 'gsm8k':
            result[dataset]['1024_minus_512'] = {c: compare(data[(dataset, 1024)][c], data[(dataset, 512)][c], 42) for c in CONDITIONS}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='one model run directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analysis = analyze(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + '\n', encoding='utf8')
