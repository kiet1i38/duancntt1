# SD-TokenSkip: Evaluation of Adaptive Reasoning Compression

This repository contains the manuscript, code, frozen policies and recorded
predictions for an evaluation of question-conditioned TokenSkip against fixed
compression ratios. The study uses Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct
with their TokenSkip GSM8K adapters. Download model weights and adapters from
their upstream repositories at the revisions in the manifests.

## Contents

- `experiments/matched-20261107/`: inference code, environment, policies,
  manifests, question-level GSM8K outputs and matched-study analysis.
- `experiments/review-20261113/`: additional evaluation scripts and protocol;
  `output_diagnostics.csv` gives termination and missing-separator counts.
- `results/additional-3b/`: completed 3B follow-up and paired analysis.
- `experiments/review-20261113/recovered-20261115/`: 7B follow-up row-level
  outputs and paired analysis, including GSM8K cap sensitivity and SVAMP.
  Run `python -B experiments/review-20261113/analyze_7b_archives.py` to
  recompute the 7B results; install the analysis requirements first.
  GSM8K conditions used multiple GPU hosts, so host and cap effects may be
  confounded. SVAMP conditions were run on one host.
- `sd-tokenskip-cleanroom-v3/` and `sd-tokenskip-utility-router/`: archived
  development inputs and analysis code retained at their original paths.
- `paper/`: manuscript and figures.

To compile the manuscript, obtain `llncs.cls` from the Springer Nature LNCS
author instructions and place it in `paper/` before running your LaTeX toolchain.
The class file is not included in this release.

Archived experiment directories keep their original names because their hashes
and source paths are part of the research record.

## Verify and analyze the recorded GSM8K results

Python 3.10 or later suffices for the integrity check:

```sh
python experiments/matched-20261107/reproduce.py --verify-only
```

The verifier checks 46 file hashes and 18,466 question-aligned generations.
To recompute the analyses and figures:

```sh
python -m pip install -r experiments/matched-20261107/requirements-analysis.txt
python experiments/matched-20261107/reproduce.py
python experiments/review-20261113/output_diagnostics.py
```

For development sensitivity and controlled ablations, install the additional
packages from `experiments/review-20261113/requirements.txt`, then run
`python experiments/matched-20261107/analyze_development.py` and
`python experiments/review-20261113/analyze.py`. These analyses are exploratory.
`python experiments/review-20261113/fit_stability.py` resamples fit questions
and refits the six-feature map without reselecting the policy family.

See `experiments/matched-20261107/README.md` for the inference environment and
fresh inference instructions. GPU inference requires the model and adapter
weights, sufficient GPU memory and compatible software. Write new inference to
a separate folder so the recorded results remain unchanged.

## Additional evaluation

`results/additional-3b/followup-3b-final.tgz` includes 3B question-level outputs
for GSM8K at caps 512 and 1,024 and SVAMP at cap 512. The paired summary is in
`full-3b-cap-svamp-paired-analysis.json`. Some GSM8K batches were resumed on
different RTX 3090 hosts; cap and host effects cannot be fully separated.
The three SVAMP conditions ran on the same instance. Token counts do not measure
end-to-end latency.

## Data and third-party code

GSM8K and TokenSkip-derived helpers retain their upstream notices in
`experiments/matched-20261107/LICENSE-GSM8K.txt` and `LICENSE-TokenSkip.txt`.
The SVAMP upstream MIT license is in `results/additional-3b/LICENSE-SVAMP.txt`.
Model and adapter licenses are governed by their upstream repositories. No
project-wide license is asserted for contributions whose licensing is unsettled.
