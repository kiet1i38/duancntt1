# Follow-up audit: training inputs and validation groups

## Official TokenSkip data preparation

On 16 November 2026, we inspected TokenSkip repository commit
`7b958cdb98e0b2a84056e26e885b260987f2e64f`. Its
[`get_llamafactory_input.py`](https://github.com/hemingkx/TokenSkip/blob/7b958cdb98e0b2a84056e26e885b260987f2e64f/get_llamafactory_input.py)
selects the original input for the 1.0 branch and appends the ratio marker for
the compressed branches. The original input is copied from a prior prediction;
the script alone does not prove what that copied text contains.

The repository provides reference processed training files for 3B and 7B in
[`datasets/gsm8k/llamafactory_inputs`](https://github.com/hemingkx/TokenSkip/tree/7b958cdb98e0b2a84056e26e885b260987f2e64f/datasets/gsm8k/llamafactory_inputs).
Inspection of the JSON `input` fields found:

| Reference file | Total rows | No `<\|eot_id\|>` marker | SHA-256 of file bytes |
|---|---:|---:|---|
| Qwen 3B | 6,638 | 1,061 | `df73ea50eeaa8ad6bd38cb2b101cc46193d5134e0c5783f1977cf0c8c4200ff3` |
| Qwen 7B | 6,002 | 971 | `f4b1d626151a1e5ddc92cc184d09e6081fdecc4ff4dfcbb29972b5f815cdfad1` |

All other rows have a trailing marker with one of 0.5, 0.6, 0.7, 0.8 or
0.9; no other marker format was found. These are evidence about the published
reference files, not proof that these exact file bytes trained the released
adapters. The repository describes them as reference processed data; no
verified training dataset hash linking them to either adapter was located.

## Validation-group reconstruction

For each model, using its frozen six-feature score thresholds, recomputed
deciles and groups match every saved development routing assignment. The stored
decile index is zero-based. Source indices are unique within each split and
the three split ID sets do not intersect. Counts, ordered easy / medium / hard:

| Split | Count | Group counts |
|---|---:|---|
| Fit | 540 | 270 / 108 / 162 |
| Tuning | 180 | 95 / 37 / 48 |
| Validation | 180 | 102 / 21 / 57 |

The split manifest states that prior fit rows and duplicate groups were
preserved, then tuning and validation were filled in source-index order; the
procedure did not stratify by the score group. Fit quantile boundaries applied
to a different set therefore do not enforce 50/20/30 proportions on it. The
audit finds no discrepancy between saved labels and frozen thresholds; it does
not establish the statistical cause of the small medium group.
