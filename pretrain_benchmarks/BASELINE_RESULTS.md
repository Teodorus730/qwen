# Versioned benchmark baseline results

This index records completed baseline evaluations for the untouched
`Qwen/Qwen3.5-0.8B-Base` model at revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.

Core lm-eval hardware and protocol: Intel Arc A770 16 GB, XPU (`xpu:0`),
BF16, seed 42, and `max_length=2048`. The lm-eval tasks are zero-shot; its
automatic batch size resolved to 32. The C-Eval and MMMLU sections below record
their own fixed batch and length settings.

## lm-eval Core run — 20260807T061546Z

| Task | Metrics |
|---|---|
| WikiText | word PPL 19.2195992561004; byte PPL 1.7380673704772327; bits/byte 0.7974840045744408 |
| LAMBADA OpenAI | acc 0.5076654376091597; perplexity 11.66613371683548 |
| HellaSwag | acc 0.42013543118900615; acc_norm 0.5483967337183828 |
| PIQA | acc 0.7018498367791077; acc_norm 0.7154515778019587 |
| ARC-Easy | acc 0.70496632996633; acc_norm 0.6746632996632996 |

Artifacts: [aggregate/config JSON](baseline_results/Qwen__Qwen3.5-0.8B-Base/20260807T061546Z.json)
and [environment snapshot](baseline_results/Qwen__Qwen3.5-0.8B-Base/20260807T061546Z.environment.txt).

## Multilingual Wikipedia BPB/PPL — 20260808T121142Z

The evaluation reuses committed, tokenizer-independent corpus manifests from
the pinned `wikimedia/wikipedia` snapshot. About 1 MiB raw UTF-8 text is
scored for each language.

| Language | BPB | PPL | Scored tokens |
|---|---:|---:|---:|
| English (EN) | 0.8721543396364448 | 11.248555473500607 | 261891 |
| Chinese (ZH) | 1.171477019179626 | 20.104978941889723 | 283723 |
| Russian (RU) | 0.5583387021488371 | 8.870595759848115 | 185909 |

Artifacts: [canonical manifests](baseline_results/multilingual_wikipedia/manifests/),
[result JSON](baseline_results/multilingual_wikipedia/20260808T121142Z/result.json),
and [environment snapshot](baseline_results/multilingual_wikipedia/20260808T121142Z/environment.txt).

BPB is the primary normalized comparison metric for model changes within one
language on this fixed corpus. Token PPL is tokenizer-dependent, and absolute
BPB values across EN/ZH/RU are not a language-knowledge ranking.

## Multilingual / knowledge baselines

### C-Eval validation — 20260809T112434Z

The completed `ceval-valid` run evaluates all 1,346 validation examples from
the pinned `ceval/ceval-exam` revision
`617524a00b307ff6f9933702f724131fe12ca7ce`. It uses native zero-shot task
prompts, lm-eval 0.4.12, XPU/BF16, and fixed batch size 8.

| Benchmark | Samples | acc | acc_norm |
|---|---:|---:|---:|
| C-Eval validation | 1,346 | 0.549777117384844 | 0.549777117384844 |

Artifacts: [aggregate/config JSON](baseline_results/Qwen__Qwen3.5-0.8B-Base/20260809T112434Z.json)
and [environment snapshot](baseline_results/Qwen__Qwen3.5-0.8B-Base/20260809T112434Z.environment.txt).

### MMMLU full — exact merge of complementary selections

This is a completed full `openai/MMMLU` test-set baseline, not a 5% subset or
a third full-dataset model run. The pinned dataset revision is
`325a01dc3e173cac1578df94120499aaca2e2504`; it covers 14 locales, 57 subjects
per locale, and 798 locale-by-subject strata. Both stages use native zero-shot
prompts and the reference lm-eval 0.4.12 path on XPU/BF16.

| Benchmark | Samples | Correct (`acc` / `acc_norm`) | acc | acc_norm |
|---|---:|---:|---:|---:|
| MMMLU full | 196,588 | 78,977 / 78,977 | 0.40173866156632143 | 0.40173866156632143 |

The full result is the exact integer-counter merge of deterministic,
non-overlapping selections: Stage 1 has 9,828 examples (3,989 correct) and
Stage 2 is its exact complement with 186,760 examples (74,988 correct).
Their union is the complete pinned test set; metrics are computed from summed
integer counters, not by averaging stage accuracies.

Reproducibility identity: model SHA
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`; Stage-1 selection SHA
`d5d1fc9851094bc498df5eaaa2f4293bf849e64f58e0fe4d2b094ef90dc15aaf`;
Stage-2 selection SHA
`b69444c25dde246edda2edbce6b189e85127f562a1823cae5a12f8fb38e14b80`;
and progressive manifest SHA
`001b5824114013b92629413abc142a5a201dfa89be0ac9a3df4f9c374d5cc306`.
Stage 2 used fixed batch size 2, `max_length=4096`, and no limit.

Artifacts: [Stage-1 summary](baseline_results/mmmlu/20260809T113000Z/stage_summary.json),
[Stage-2 summary](baseline_results/mmmlu/20260810T134134Z/stage_summary.json),
[progressive manifest](baseline_results/mmmlu/manifests/mmmlu_progressive_manifest.json),
and [full merged result](baseline_results/mmmlu/full_result.json).

Stage-2 model evaluation completed successfully. Its 798 raw JSONL files
(186,760 records) were valid, but compact-summary post-processing initially
used `str.splitlines()`, which treats Unicode U+2028 in a JSON string as a
line boundary. The parser now splits JSONL only on physical LF; the recovered
summary was independently checked against the retained raw artifacts, so no
model rerun was required.

An experimental memory-efficient MMMLU path was investigated but failed a
strict reference-equivalence regression and was reverted. The canonical result
above uses the reference lm-eval 0.4.12 execution path.

## Older lm-eval Core-run caveat

The lm-eval aggregate preserves task versions, task configuration, environment,
and model SHA, but its `dataset_revision` fields are `null` and
`task_hashes` is empty. This does not invalidate the recorded scores or require
a rerun; it limits exact source pinning for a future recreation of that older
lm-eval run. Raw harness/sample outputs remain ignored under `results/`.
