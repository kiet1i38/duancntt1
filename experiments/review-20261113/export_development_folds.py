"""Export the question-level folds used by the controlled ablation."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT = ROOT / 'sd-tokenskip-cleanroom-v3/runs/sd-v2-v3-gamma05-10-20261025'
OUTPUT = Path(__file__).resolve().parent / 'development-folds.csv'
SEED = 20261113


def read_assignments(model):
    result = []
    for split in ('fit', 'tuning', 'validation'):
        path = DEVELOPMENT / model / 'development' / split / 'routing_assignments.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
        rows.sort(key=lambda row: row['source_row_index'])
        result.extend((split, row) for row in rows)
    assert len(result) == 900
    assert len({row['question_sha256'] for _, row in result}) == 900
    return result


def main():
    reference = None
    with OUTPUT.open('w', encoding='utf8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model', 'original_split', 'source_row_index', 'question_sha256', 'ablation_fold'])
        for model in ('qwen2.5-3b', 'qwen2.5-7b'):
            rows = read_assignments(model)
            identities = [(split, row['source_row_index'], row['question_sha256']) for split, row in rows]
            if reference is None:
                reference = identities
            else:
                assert identities == reference
            for fold, indices in enumerate(np.array_split(np.random.default_rng(SEED).permutation(len(rows)), 5), 1):
                assert len(indices) == 180
                for i in indices:
                    split, row = rows[int(i)]
                    writer.writerow([model, split, row['source_row_index'], row['question_sha256'], fold])
    print(OUTPUT)


if __name__ == '__main__':
    main()
