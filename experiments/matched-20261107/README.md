# Matched TokenSkip controls, 7 November 2026

This package contains 18,466 recorded generations: two Qwen2.5 adapters, seven
conditions per model, and all 1,319 GSM8K test questions per condition. The six
fixed ratios are 0.5, 0.6, 0.7, 0.8, 0.9, and 1.0. `sd_frozen` is the unchanged
six-feature policy selected before these runs. Previous results are retained
elsewhere in the repository; these outputs form a separate matched evaluation.

## Verify and regenerate

From the repository root, with Python 3.11 or newer for the pinned analysis
environment (verification alone also supports Python 3.10):

```bash
python experiments/matched-20261107/reproduce.py --verify-only
python -m pip install -r experiments/matched-20261107/requirements-analysis.txt
python experiments/matched-20261107/reproduce.py
python experiments/matched-20261107/analyze_development.py
python experiments/matched-20261107/diagnose_routing.py
```

Verification uses only the standard library. Analysis uses NumPy and Matplotlib.
Raw-output hashes, row sets, question hashes, gamma assignments, counts, and
token means are checked before any table is regenerated. `sha256.json` protects
the inference inputs and results, and `.gitattributes` preserves their bytes.

The two diagnostic commands refit the original candidate families on saved
development outcomes and compute fixed-grid AUROCs, 10,000 ratio shuffles,
and label-informed oracle headroom. They use the archived development runner
and the recorded fixed grid, respectively. Both analyses run on CPU.

`analysis/metrics.csv` distinguishes direct inference from offline grid replay.
`paired_comparisons.csv` contains 20,000-resample question bootstrap intervals
and exact McNemar tests. The two primary accuracy comparisons use fixed 0.5,
chosen by the archived tuning utility at lambda 0.1, with Holm correction across
models. Report all six ratios; do not select a new deployment policy on test.

## Protocol and interpretation

- FP16, PyTorch 2.3.0+cu121, Transformers 4.47.0, greedy, seed 42, batch 16,
  SDPA, maximum 512 new tokens; one RTX 3090 for all final conditions.
- Model/adapter revisions and the inference environment are in each manifest.
  Exact inference packages are in `requirements-inference.lock.txt`.
- Every ratio, including 1.0, uses the merged TokenSkip adapter. The ratio-1.0
  condition is not an adapter-disabled Full-CoT baseline.
- Native Transformers EOS stopping replaces the redundant upstream Python
  prefix scan. The same-batch pilot matched the upstream helper on three
  questions per model before full inference.
- Same-ratio batching means batch composition differs between fixed and routed
  conditions. Direct SD and grid-replayed SD have equal aggregate accuracy but
  differ in correctness on 10/2 individual questions for 3B/7B. Random-control
  comparisons use grid-replayed SD to match their source batches.
- The 512-token ceiling censors some outputs. Token reduction alone does not
  establish latency, energy, or end-to-end speedup.
- GSM8K test was previously inspected. These matched reruns do not create a new
  untouched benchmark for a redesigned router.

Original raw outputs are unchanged. Runtime rental credentials, SSH keys,
account balances, and browser screenshots are excluded. The budget in the
frozen execution plan records the initial operational limit, not a scientific
hyperparameter. It was subsequently increased without changing this policy.

## Re-running inference

Use the pinned revisions in `bundle/source_protocol.json` and install the
recorded inference environment. `runner.py` exposes `--model`, `--output`,
`--batch-size 16`, and `--deadline-utc UNIX_EPOCH`. Choose a **new** output folder,
such as `fresh-runs/qwen2.5-3b`; never overwrite `results/`. The published package
omits duplicate batch bodies; all individual predictions are in the JSONL files.
`batch_timings.jsonl` preserves batch timing metadata and is not a row-latency
measurement. Model weights are obtained from their public Hugging Face repos.

## Third-party material

TokenSkip-derived helper code is distributed under its Apache-2.0 license
(`LICENSE-TokenSkip.txt`). GSM8K question/answer data is from OpenAI's
grade-school-math repository under MIT (`LICENSE-GSM8K.txt`). The adapter/model
licenses remain those of the pinned upstream repositories. The `bundle/` helpers
are historical execution snapshots, not newly authored libraries.
