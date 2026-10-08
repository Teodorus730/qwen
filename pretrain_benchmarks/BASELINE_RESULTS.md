# Versioned benchmark baseline results

Technical source of truth for the completed baseline of
`Qwen/Qwen3.5-0.8B-Base` at exact model revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.

## Completed baseline at a glance

| Direction | Benchmark | Coverage | Primary results |
|---|---|---:|---|
| Core / intrinsic LM | WikiText | Full test split | bits/byte 0.7974840045744408; byte PPL 1.7380673704772327; word PPL 19.2195992561004 |
| Core / intrinsic LM | Wikipedia EN/ZH/RU | Fixed ~1 MiB UTF-8 corpus per language | BPB: EN 0.8721543396364448; ZH 1.171477019179626; RU 0.5583387021488371 |
| Core / base-model diagnostic | LAMBADA OpenAI | 5,153 | acc 0.5076654376091597; PPL 11.66613371683548 |
| Core / base-model diagnostic | HellaSwag | 10,042 | acc 0.42013543118900615; acc_norm 0.5483967337183828 |
| Core / base-model diagnostic | PIQA | 1,838 | acc 0.7018498367791077; acc_norm 0.7154515778019587 |
| Core / base-model diagnostic | ARC-Easy | 2,376 | acc 0.70496632996633; acc_norm 0.6746632996632996 |
| Multilingual / knowledge | C-Eval validation | 1,346 | acc = acc_norm = 0.549777117384844 |
| Multilingual / knowledge | MMMLU full | 196,588 | acc = acc_norm = 0.40173866156632143 |

All reported runs use the untouched Qwen Base checkpoint, XPU on an Intel Arc
A770 16 GB, and BF16. Unless stated otherwise, lm-eval tasks are zero-shot.
Smoke/probe runs and limited scores are not part of this registry.

## Core: Base/pretraining evaluation

The Core suite combines intrinsic language-model measurements with downstream
diagnostics that can be applied directly to a base model. It does not claim
that every task is a pure pretraining metric.

### lm-eval Core run — 20260807T061546Z

Protocol: lm-eval 0.4.12, zero-shot, seed 42, `max_length=2048`, XPU/BF16;
automatic batch size resolved to 32.

| Task | Samples | Metrics |
|---|---:|---|
| WikiText | 62 documents | word PPL 19.2195992561004; byte PPL 1.7380673704772327; bits/byte 0.7974840045744408 |
| LAMBADA OpenAI | 5,153 | acc 0.5076654376091597; perplexity 11.66613371683548 |
| HellaSwag | 10,042 | acc 0.42013543118900615; acc_norm 0.5483967337183828 |
| PIQA | 1,838 | acc 0.7018498367791077; acc_norm 0.7154515778019587 |
| ARC-Easy | 2,376 | acc 0.70496632996633; acc_norm 0.6746632996632996 |

Artifacts: [aggregate/config JSON](baseline_results/Qwen__Qwen3.5-0.8B-Base/20260807T061546Z.json)
and [environment snapshot](baseline_results/Qwen__Qwen3.5-0.8B-Base/20260807T061546Z.environment.txt).

### Multilingual Wikipedia BPB/PPL — 20260808T121142Z

Source: `wikimedia/wikipedia`, revision
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, configurations `20231101.en`,
`20231101.zh`, and `20231101.ru`. The evaluator reuses committed,
tokenizer/model-independent corpus manifests with about 1 MiB raw UTF-8 text
per language. Tokenization and chunk metadata are recorded per run.

| Language | BPB | Token PPL | Scored tokens |
|---|---:|---:|---:|
| English (EN) | 0.8721543396364448 | 11.248555473500607 | 261,891 |
| Chinese (ZH) | 1.171477019179626 | 20.104978941889723 | 283,723 |
| Russian (RU) | 0.5583387021488371 | 8.870595759848115 | 185,909 |

Artifacts: [canonical corpus manifests](baseline_results/multilingual_wikipedia/manifests/),
[result JSON](baseline_results/multilingual_wikipedia/20260808T121142Z/result.json),
and [environment snapshot](baseline_results/multilingual_wikipedia/20260808T121142Z/environment.txt).

BPB is the primary normalized metric for comparing model changes within one
language on the same corpus. Token PPL is tokenizer-dependent. Absolute BPB
differences across EN/ZH/RU are not a ranking of language knowledge because
the corpora and language encoding properties differ.

## Multilingual / knowledge extension

### C-Eval validation — 20260809T112434Z

Protocol: public validation split, native zero-shot prompts, all 52 subjects,
lm-eval task version 2.0, fixed batch size 8, XPU/BF16. This is C-Eval
validation, not the closed official test set.

Dataset: `ceval/ceval-exam`, exact revision
`617524a00b307ff6f9933702f724131fe12ca7ce`.

| Samples | acc | acc_norm |
|---:|---:|---:|
| 1,346 | 0.549777117384844 | 0.549777117384844 |

Artifacts: [aggregate/config JSON](baseline_results/Qwen__Qwen3.5-0.8B-Base/20260809T112434Z.json)
and [environment snapshot](baseline_results/Qwen__Qwen3.5-0.8B-Base/20260809T112434Z.environment.txt).

### MMMLU full

Protocol: full `openai/MMMLU` test set, native zero-shot prompts, 14 locales,
57 subjects per locale, 798 locale-by-subject strata, reference lm-eval 0.4.12
path, XPU/BF16, canonical `max_length=4096`, and no limit. The two recorded
stages used fixed batches 4 and 2 respectively. The retained Stage-1 command
used `max_length=2048`; its audited maximum prompt length was 1,840 tokens, so
no Stage-1 input was truncated. Stage 2 used the adopted 4,096-token bound.

Dataset exact revision: `325a01dc3e173cac1578df94120499aaca2e2504`.

| Samples | Correct acc | Correct acc_norm | acc | acc_norm |
|---:|---:|---:|---:|---:|
| 196,588 | 78,977 | 78,977 | 0.40173866156632143 | 0.40173866156632143 |

The full result is an exact merge of two deterministic, non-overlapping
selections, not a third full-dataset model run:

| Selection | Samples | Correct acc | Correct acc_norm | Selection SHA-256 |
|---|---:|---:|---:|---|
| Stage 1 | 9,828 | 3,989 | 3,989 | `d5d1fc9851094bc498df5eaaa2f4293bf849e64f58e0fe4d2b094ef90dc15aaf` |
| Stage 2, exact complement | 186,760 | 74,988 | 74,988 | `b69444c25dde246edda2edbce6b189e85127f562a1823cae5a12f8fb38e14b80` |

Stage 1 ∩ Stage 2 is empty and their union is the complete pinned dataset.
The full metrics come from summed integer counters (`78,977 / 196,588`), not
from averaging stage-level floating-point accuracies. The progressive manifest
SHA-256 is
`001b5824114013b92629413abc142a5a201dfa89be0ac9a3df4f9c374d5cc306`.

Artifacts: [progressive manifest and selections](baseline_results/mmmlu/manifests/),
[Stage-1 summary](baseline_results/mmmlu/20260809T113000Z/stage_summary.json),
[Stage-2 summary](baseline_results/mmmlu/20260810T134134Z/stage_summary.json),
and [full merged result](baseline_results/mmmlu/full_result.json).

## Reproducibility record

The compact artifacts preserve the exact model SHA, task configuration and
versions, dataset revisions when available, seed, few-shot setting, backend,
device, dtype, batch/context policy, commands, metrics, and package versions.
Canonical Wikipedia and MMMLU selections are additionally protected by
manifest and selection hashes. Heavy per-sample/raw harness outputs remain
ignored under `results/`.

### Known caveats and audit notes

- The older Core lm-eval artifact preserves task versions/configuration,
  environment, model SHA, and all aggregate metrics, but its
  `dataset_revision` fields are `null` and `task_hashes` is empty. This limits
  exact future source pinning for those datasets; it does not invalidate the
  recorded baseline or require a rerun.
- Stage-2 MMMLU model evaluation completed successfully and produced 798 valid
  JSONL files with 186,760 records. Compact-summary parsing initially failed
  because Python `str.splitlines()` treats Unicode U+2028 inside a valid JSON
  string as a line boundary. The parser was corrected to split only on the
  physical LF delimiter, and the summary was independently reconstructed and
  verified from saved raw outputs; no model rerun was required.
- An experimental memory-efficient MMMLU evaluator failed strict
  reference-equivalence testing and was reverted. The reported canonical
  MMMLU result uses the reference lm-eval 0.4.12 path.
