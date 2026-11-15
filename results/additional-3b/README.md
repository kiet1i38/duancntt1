# GPU follow-up results and checkpoint history

Recovered from the last locally saved archive before instance 60523197 was
deleted. Archive SHA-256:
`4e72fdb08000de9b5ecb36bdaf5e01c25dddd3c8d978a5f9daaf5fee3751644e`.

For Qwen2.5-3B, GSM8K, cap 512, the archive contains 1,319 unique rows for
fixed gamma 1.0 and 72 unique rows for fixed gamma 0.5. The model's engine
parity pilot passed, and every recovered batch matches the saved contract hash.
The second condition and the remaining planned conditions are incomplete; do
not use this checkpoint as a completed paired comparison. More rows may have
been generated on the deleted instance after this archive was saved, but they
were not recovered. No 7B follow-up output was recovered here.

`checkpoint-08.tgz` is a later backup after resuming on a second host. It
contains all 1,319 unique GSM8K rows for each fixed gamma at cap 512 and 40
unique SD rows. The 3B gamma 0.5 run was resumed across two GPU hosts; this
fact must be disclosed when interpreting or reporting these outputs. The
runner's three-question parity pilot passed separately on both hosts, but this
does not establish bitwise equivalence for all examples. The SD condition and
all cap-1024 and SVAMP conditions remain incomplete in this backup.

`checkpoint-15.tgz` contains completed 3B GSM8K cap-512 outputs for all three
conditions. Each has 1,319 distinct ordered row indices, matching question
hashes and contract hashes. Counts and mean CoT tokens are: fixed 1.0,
1,096/1,319 and 318.892; fixed 0.5, 989/1,319 and 169.009; frozen SD,
1,087/1,319 and 264.739. Paired bootstrap intervals (10,000 resamples, seed 42)
for SD minus fixed 1.0 are -0.682 percentage points [-2.199, 0.834] and
-54.153 tokens [-57.539, -50.849]. These data alone do not establish
equivalence. The fixed 1.0 output is from the first host, fixed 0.5 was resumed
across hosts, and SD ran on the second host; consequently the comparisons are
paired by question but not strictly same-host. This snapshot does not complete
the cap-1024 or SVAMP study, and
does not contain a completed 7B follow-up.

`checkpoint-22.tgz` additionally contains all 1,319 unique 3B GSM8K fixed
gamma 1.0 predictions at cap 1024. Their question hashes match the cap-512
predictions. The 1024 run has 1,112 correct, 323.177 mean CoT tokens, and four
length stops; its paired accuracy change relative to 512 is +1.213 percentage
points (bootstrap 95% CI [0.682, 1.820], 10,000 resamples, seed 42). The
1024 and 512 runs used different hosts, so this difference cannot be attributed
solely to the cap. The remaining 1024 conditions and SVAMP were not complete
when this snapshot was taken.

`checkpoint-38.tgz` completes all six 3B GSM8K conditions (three policies at
each cap), with 1,319 unique aligned question IDs per condition. The full
recomputed metrics and paired intervals are in `gsm8k-3b-512-1024-analysis.json`.
At cap 1024, SD is correct on 1,095/1,319 questions (83.02%) with mean CoT
length 266.94 tokens. Its accuracy difference from cap 512 is +0.607 percentage
points (paired bootstrap 95% CI [0.227, 1.061]); its mean CoT length rises by
2.20 tokens (95% CI [1.08, 3.55]). SD cap 1024 was resumed on a third host
after 861 rows; therefore cap and host effects cannot be separated cleanly.
Matching runtime contracts and a three-question parity pilot do not establish
bitwise equivalence across hosts. The SVAMP and 7B follow-up remain separate.

`checkpoint-29.tgz` additionally contains 1,319 distinct 3B GSM8K fixed
gamma 0.5 predictions at cap 1024. All question and contract hashes match
the cap-512 condition. The 1024 run has 989 correct, 173.736 mean CoT tokens,
and 12 length stops. The paired accuracy difference from 512 is 0.000
percentage points (bootstrap 95% CI [0.000, 0.000]); the mean CoT difference
is +4.727 tokens (95% CI [2.346, 7.444], 10,000 resamples, seed 42). This
interval does not establish general equivalence, and the earlier cap-512
condition was resumed across hosts. SD at cap 1024 and SVAMP were incomplete
when this snapshot was taken.

## Final 3B follow-up — completed

The final bundle is `followup-3b-final.tgz` (SHA-256
`9855da02c009031961e0bf333895fe305366d71e7e8c822b25fb771b6a4050ae`). Its
manifest is `complete`, with contract hash
`9d4e7515eb9b7ff76fe9567fa6a24db2df07935349366d777c87883006e0fe8b`. The
archive contains the row-level batches and runner log. `full-3b-cap-svamp-paired-analysis.json`
contains the recomputed metrics and paired bootstrap intervals.

All six GSM8K conditions are present with 1,319 unique aligned rows each:
fixed gamma 1.0, fixed gamma 0.5, and frozen SD at generation caps 512 and
1024. The three SVAMP conditions are present with 1,000 unique aligned rows
each at cap 512. The SVAMP results are 898/1,000 correct and 223.098 mean CoT
tokens for fixed gamma 1.0; 830/1,000 and 107.760 tokens for fixed gamma 0.5;
and 886/1,000 and 168.995 tokens for SD.

On SVAMP, SD minus fixed gamma 1.0 is -1.20 accuracy points (paired bootstrap
95% CI [-3.0, 0.6]) and -54.103 mean CoT tokens (95% CI [-58.185, -50.078]).
SD minus fixed gamma 0.5 is +5.60 accuracy points (95% CI [3.3, 7.8]) and
+61.235 CoT tokens (95% CI [57.027, 65.539]). These comparisons show a
trade-off; they do not establish that SD is better than fixed gamma 1.0 or
that the two are equivalent. The intervals are paired percentile-bootstrap
intervals with 10,000 resamples and seed 42, not a McNemar test.

The SVAMP conditions were generated on the same RTX 3090 instance. GSM8K
outputs include batches resumed from earlier RTX 3090 instances; those paired
comparisons are not strict same-host controls, and the three-question parity
pilots do not prove bitwise equivalence across hosts. The 7B follow-up was not
run. The final instance, 60739489, was stopped after the archive was downloaded
and verified.
