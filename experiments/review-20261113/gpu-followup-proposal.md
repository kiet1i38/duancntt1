# GPU follow-up — protocol frozen, inference not yet executed

The stored 512-token results cannot answer whether a longer generation limit
changes the conclusions. Nor does an analysis of GSM8K establish transfer to
another benchmark. The configuration and source hashes are recorded in
`gpu-protocol-lock.json` before GPU inference. No new result exists yet.

## Paired generation-limit sensitivity

For each model, regenerate the complete 1,319-question GSM8K test set at caps
512 and 1,024 using the same frozen SD policy and the fixed ratios 0.5 and 1.0.
These are 7,914 generations per model, or 15,828 total. Use the same model,
adapter revisions and prompt as the matched run, FP16, greedy decoding and seed
42. Keep the adapter on at all ratios, and use the same engine, ordering and
batching strategy at both caps. Record each output, question ID and hash,
prompt, gamma, token counts and termination reason. Compare caps within the
same method and model on the same questions with paired intervals; then compare
SD with fixed ratios at each cap. If 1,024 causes an out-of-memory error, do
not quietly change one condition: revise and document the shared batch design
before collecting the comparisons.

This is sensitivity on an already examined test set, not a fresh holdout.
Longer caps can change runtime and batch behavior; they do not guarantee a
correct final answer. Preserve the original 512-token runs and do not treat
changes between separate runs as pure effects of the cap without controlling
the execution setup.

## External evaluation candidate

Use the official [SVAMP challenge set](https://github.com/arkilpatel/SVAMP)
only after pinning the source revision and data hash, verifying all 1,000 IDs,
and testing the question conversion and numeric answer evaluator. Compare the
unchanged SD policy with tuning-selected fixed gamma 0.5 and adapter-enabled
gamma 1.0, separately for each model, at the previously used 512 cap: 6,000
generations total. Keep the mapping and evaluation rules frozen before reading
SVAMP outcome metrics. Record paired accuracy and token differences and their
intervals. Public availability and possible pretraining overlap mean this is an
external benchmark, not proof that the model has never seen its examples.
The source is commit `78e727689e1c1bebfc4be39c446898e8e10b0518`;
the input is the upstream `SVAMP.json` (MIT-licensed upstream repository).
Each prompt concatenates `Body`, one space, then `Question`. Numeric answers
are converted to strings for the archived TokenSkip extraction and evaluator;
answers are excluded from the routing input. The recorded SHA-256 and code
hashes are checked by the runner before loading models.

## Readiness and spending

The archived runner hard-codes GSM8K and 512 tokens. A separate resumable
runner checks input hashes and writes cap-specific batches and manifests. CPU
preflight, question-only routing smoke checks and answer-scoring examples pass;
the model-loading and engine-parity pilot still require a GPU. The runner stops
at the specified deadline between batches, but this does not itself stop cloud
billing: the instance must also be terminated after preserving results. Pilot
runtime and transfer charges are uncertain;
before any paid rental, present the actual offer, GPU/VRAM, RAM, disk, hourly
and transfer costs, expected duration and total budget for user confirmation.
If the budget cannot cover all conditions, preserve completed outputs and
report the missing portions rather than presenting partial runs as complete.

After environment setup, the commands from the extracted package root are:

```sh
bash experiments/review-20261113/setup_gpu.sh
python -B experiments/review-20261113/gpu_runner.py --model qwen2.5-7b --output /workspace/followup/qwen2.5-7b --deadline-utc 9999999999 --preflight-only
bash experiments/review-20261113/run_gpu.sh /workspace/followup DEADLINE_UTC_UNIX
```

The deadline is a placeholder to replace with the actual Unix timestamp.
`analyze_gpu.py` refuses to analyze a model until all planned batches and row
IDs are present, then calculates matched differences and paired bootstrap
intervals. Both timing and cost must be monitored outside the instance; a
checkpoint alone leaves the instance billable.
