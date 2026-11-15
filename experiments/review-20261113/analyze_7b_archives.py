"""Analyze the completed 7B follow-up from the three recovered archives."""
import json
import tarfile
from pathlib import Path
from analyze_gpu import compare

ROOT = Path(__file__).resolve().parent / 'recovered-20261115' / 'checkpoints'
ARCHIVES = ROOT / 'source-archives'
CONTRACT = '09076b113302d54c8e90c078e7eb6860a1da418a504e52b196f7983defd4971c'
SOURCES = {
    ('gsm8k', 512, c): ('60768567-checkpoint.tar.gz', 'qwen2.5-7b')
    for c in ('fixed_1.0', 'fixed_0.5', 'sd_frozen')
}
SOURCES.update({
    ('gsm8k', 1024, 'fixed_1.0'): ('60768567-checkpoint.tar.gz', 'qwen2.5-7b'),
    ('gsm8k', 1024, 'fixed_0.5'): ('60778087-gsm8k-fixed05-svamp.tar.gz', 'gsm8k-cap1024-fixed05'),
    ('gsm8k', 1024, 'sd_frozen'): ('60778013-gsm8k-cap1024-sd.tar.gz', 'gsm8k-cap1024'),
})
SOURCES.update({('svamp', 512, c): ('60778087-gsm8k-fixed05-svamp.tar.gz', 'svamp-cap512')
                for c in ('fixed_1.0', 'fixed_0.5', 'sd_frozen')})


def load(key):
    dataset, cap, condition = key
    archive, prefix = SOURCES[key]
    directory = f'{prefix}/{dataset}/{cap}/{condition}/'
    rows = []
    with tarfile.open(ARCHIVES / archive, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.startswith(directory) and member.name.endswith('.json'):
                batch = json.load(tar.extractfile(member))
                assert batch['contract_sha256'] == CONTRACT
                rows.extend(batch['rows'])
    rows.sort(key=lambda row: row['source_row_index'])
    n = 1319 if dataset == 'gsm8k' else 1000
    assert [r['source_row_index'] for r in rows] == list(range(n)), key
    assert len({r['id'] for r in rows}) == n
    assert all(r['model_output'].strip() and r['finish_reason'] in ('eos', 'length') for r in rows)
    return rows


def main():
    data = {key: load(key) for key in SOURCES}
    results = {}
    for dataset, caps in [('gsm8k', (512, 1024)), ('svamp', (512,))]:
        results[dataset] = {}
        for cap in caps:
            rows = {c: data[(dataset, cap, c)] for c in ('fixed_1.0', 'fixed_0.5', 'sd_frozen')}
            assert all([r['question_sha256'] for r in v] == [r['question_sha256'] for r in rows['fixed_1.0']] for v in rows.values())
            metrics = {}
            for condition, items in rows.items():
                n = len(items)
                correct = sum(bool(r['accuracy']) for r in items)
                metrics[condition] = {'correct': correct, 'n': n, 'accuracy_percent': 100 * correct / n,
                                      'mean_cot_tokens': sum(r['cot_length'] for r in items) / n,
                                      'length_stops': sum(r['finish_reason'] == 'length' for r in items)}
            results[dataset][str(cap)] = {'metrics': metrics,
                'sd_minus_fixed_05': compare(rows['sd_frozen'], rows['fixed_0.5'], 42),
                'sd_minus_fixed_10': compare(rows['sd_frozen'], rows['fixed_1.0'], 42)}
        if dataset == 'gsm8k':
            results[dataset]['1024_minus_512'] = {c: compare(data[(dataset, 1024, c)], data[(dataset, 512, c)], 42)
                                                  for c in ('fixed_1.0', 'fixed_0.5', 'sd_frozen')}
    output = ROOT.parent / 'full-7b-cap-svamp-paired-analysis.json'
    output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf8')
    print(output)


if __name__ == '__main__':
    main()
