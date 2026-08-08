# Versioned Core baseline results

This index records the completed **Core: Base/pretraining evaluation** for the
untouched `Qwen/Qwen3.5-0.8B-Base` model at revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.

Hardware and protocol: Intel Arc A770 16 GB, XPU (`xpu:0`), BF16, seed 42,
and `max_length=2048`. The lm-eval tasks are zero-shot; its automatic batch
size resolved to 32.

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

## Caveat

The lm-eval aggregate preserves task versions, task configuration, environment,
and model SHA, but its `dataset_revision` fields are `null` and
`task_hashes` is empty. This does not invalidate the recorded scores or require
a rerun; it limits exact source pinning for a future recreation of that older
lm-eval run. Raw harness/sample outputs remain ignored under `results/`.
