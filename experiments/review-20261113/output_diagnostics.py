"""Summarize termination and missing-answer-separator diagnostics by run."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'experiments' / 'matched-20261107' / 'results'
OUTPUT = Path(__file__).resolve().parent / 'output_diagnostics.csv'


def main():
    records = []
    for model in ('qwen2.5-3b', 'qwen2.5-7b'):
        for condition in [*(f'fixed_{g / 10:.1f}' for g in range(5, 11)), 'sd_frozen']:
            path = RUNS / model / f'{condition}.jsonl'
            rows = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line]
            assert len(rows) == 1319 and len({r['source_row_index'] for r in rows}) == 1319
            missing = [r for r in rows if '\n\nThe final answer is:' not in r['model_output']]
            records.append({'model': model, 'condition': condition, 'n': len(rows),
                            'length_stops': sum(r.get('finish_reason') == 'length' for r in rows),
                            'missing_separator': len(missing),
                            'missing_separator_correct': sum(bool(r['accuracy']) for r in missing)})
    with OUTPUT.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(OUTPUT)


if __name__ == '__main__':
    main()
