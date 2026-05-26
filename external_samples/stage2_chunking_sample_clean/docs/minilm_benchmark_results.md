# MiniLM benchmark results

Date: 2026-05-27

Scope: first real local MiniLM/Sentence-Transformers nearest-label run on the synthetic classifier benchmark.

No HF streaming, NLL/logprob scoring, model training, push, or commit was performed in this analysis step.

## Environment

Embedding Python:

```text
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe
```

Packages:

| Package | Version |
| --- | --- |
| `torch` | `2.12.0+cpu` |
| `sentence-transformers` | `5.5.1` |
| `transformers` | `5.9.0` |
| `huggingface_hub` | `1.16.4` |
| `numpy` | `2.4.6` |

Local model path:

```text
C:\models\sentence-transformers\all-MiniLM-L6-v2
```

## MiniLM run command

```powershell
cd C:\Users\pervo\PycharmProjects\qwen\external_samples\stage2_chunking_sample_clean
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe scripts\classify_chunks_embedding_baseline.py `
  --input data_samples\classifier_benchmark_chunks.jsonl `
  --labels taxonomy\simple_domain_labels.json `
  --output data_samples\classifier_benchmark_minilm_labeled.jsonl `
  --model C:\models\sentence-transformers\all-MiniLM-L6-v2 `
  --batch-size 32 `
  --top-k 3
```

Run summary:

```text
records_read: 42
records_written: 42
records_classified_by_embedding: 42
low_confidence_count: 6
```

## Raw MiniLM output counts

`source_type`:

```text
unknown: 42
```

`domain`:

```text
science: 9
null: 6
stem: 6
infrastructure: 5
web: 5
multilingual: 3
government: 2
software: 2
education: 1
media: 1
reference: 1
unknown: 1
```

`field`:

```text
null: 6
mathematics: 6
environmental_science: 5
urban_systems: 5
boilerplate_or_navigation: 4
mixed_language: 3
biology: 2
legal_notice: 2
physics: 2
programming: 2
encyclopedic_article: 1
forum_qa: 1
general_education: 1
news: 1
unknown: 1
```

`label_method`:

```text
embedding_nearest_label_minilm: 42
```

## Raw MiniLM metrics

Against expected benchmark labels:

```text
source_type_accuracy: 0.0714
domain_accuracy: 0.7619
field_accuracy: 0.7381
subfield_accuracy: 0.7857
full_label_accuracy: 0.0714
```

Validation:

- without `--require-labels`: passed;
- with `--require-labels`: failed with 12 errors;
- failure reason: 6 low-confidence records intentionally have `domain = null` and `field = null`, producing 2 errors each.

## Why full_label_accuracy is misleading

`full_label_accuracy` includes `source_type`.

The MiniLM classifier currently preserves input `source_type`. The raw benchmark chunks have:

```text
source_type: unknown
```

for all 42 records before classification.

`taxonomy/simple_domain_labels.json` contains `domain`, `field`, `subfield`, descriptions, and keywords. It does not contain source-type labels or source-type descriptions.

Therefore MiniLM is currently solving domain/field/subfield nearest-label classification, not source-type classification. The low `full_label_accuracy` mostly measures that `source_type` was not predicted.

This is consistent with the classifier contract: source metadata should not be silently rewritten by the embedding classifier.

## Low-confidence records

Low-confidence abstentions at `min_confidence = 0.35`:

| chunk_id | confidence | expected label | top-1 before abstain |
| --- | ---: | --- | --- |
| `local_classifier_benchmark_000003_000` | 0.3213 | software / programming / documentation | unknown / unknown / null |
| `local_classifier_benchmark_000004_000` | 0.3022 | commercial / product_page / retail | unknown / unknown / null |
| `local_classifier_benchmark_000005_000` | 0.1743 | web / forum_qa / discussion | unknown / unknown / null |
| `local_classifier_benchmark_000019_000` | 0.3404 | software / programming / documentation | software / programming / documentation |
| `local_classifier_benchmark_000028_000` | 0.2925 | software / programming / documentation | unknown / unknown / null |
| `local_classifier_benchmark_000031_000` | 0.2157 | commercial / product_page / retail | science / environmental_science / article |

## Domain/field/subfield performance

MiniLM exactly matched `domain/field/subfield` for 28 of 42 records while leaving `source_type = unknown`.

True topic-label misses: 11 of 42.

Breakdown:

- 6 low-confidence abstentions;
- 5 confident-but-wrong topic predictions:
  - reference reservoir article -> infrastructure;
  - transit news report -> infrastructure;
  - forum Q&A about cookie banners -> boilerplate/navigation;
  - commercial product page with educational wording -> unknown;
  - climate/water-quality news article -> environmental science.

This suggests MiniLM is useful as a semantic topic classifier, but it can over-focus on topical vocabulary and miss document format/frame.

## Validation interpretation

Question: should low-confidence embedding output pass `--require-labels`?

Answer: not in the current strict mode.

`--require-labels` means every record must have non-empty `domain`, `field`, `confidence`, and `label_method`. Low-confidence embedding output deliberately abstains with `domain = null` and `field = null`, so strict validation should fail.

Recommended usage:

- raw MiniLM output: validate without `--require-labels`;
- hybrid or fallback-filled output: validate with `--require-labels`;
- future improvement: add a validator mode for embedding outputs that permits low-confidence null labels while still checking embedding metadata.

## Source type diagnosis

Why `source_type` is almost always `unknown`:

- raw benchmark chunks have `source_type = unknown`;
- embedding classifier preserves input `source_type`;
- taxonomy has no source-type classes;
- nearest-label embedding only chooses `domain/field/subfield`;
- this behavior avoids hidden source metadata rewrites.

Should MiniLM predict `source_type`?

Not in the current MVP. `source_type` means format/type/origin, while `domain/field/subfield` means knowledge/topic label. The current taxonomy is designed for the latter.

## Recommended architecture

Recommended MVP strategy: Option C, hybrid.

Use:

- `source_type` from rule-based classifier;
- `domain/field/subfield` from MiniLM when confidence is high enough and labels are non-null;
- fallback to rule-based `domain/field/subfield` when MiniLM abstains or confidence is low.

Why not Option B immediately:

- adding source-type descriptions to taxonomy blurs source metadata and topic labels;
- it would require taxonomy redesign and new evaluation semantics.

Why not Option D immediately:

- two-stage source-type then topic classifier is cleaner long-term, but it is more code and test surface than the current MVP needs.

## Hybrid run

Command:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe scripts\build_hybrid_labels.py `
  --rule-based data_samples\classifier_benchmark_labeled.jsonl `
  --embedding data_samples\classifier_benchmark_minilm_labeled.jsonl `
  --output data_samples\classifier_benchmark_hybrid_labeled.jsonl `
  --min-embedding-confidence 0.35
```

Summary:

```text
records_written: 42
hybrid_rule_source_type_minilm_domain: 32
hybrid_rule_fallback_low_confidence: 10
missing_embedding_records: 0
extra_embedding_records: 0
```

Hybrid validation:

```text
VALIDATION PASSED with --require-labels
records checked: 42
errors count: 0
```

Hybrid metrics:

```text
source_type_accuracy: 1.0000
domain_accuracy: 0.9286
field_accuracy: 0.9048
subfield_accuracy: 0.9524
full_label_accuracy: 0.9048
```

Hybrid mismatches:

- `local_classifier_benchmark_000012_000`: reference reservoir article predicted as infrastructure.
- `local_classifier_benchmark_000014_000`: transit news report predicted as infrastructure.
- `local_classifier_benchmark_000015_000`: forum Q&A about cookie banners predicted as boilerplate/navigation.
- `local_classifier_benchmark_000035_000`: environmental-science news article predicted as environmental science instead of news.

## Interpretation

On the synthetic benchmark, rule-based remains strongest because the benchmark was built around transparent rules and expected labels.

MiniLM is not yet a drop-in replacement for rule-based labels. It is a useful semantic comparison signal and a candidate for hybrid topic labels, especially on real samples where rules may be brittle.

Hybrid output is the safest MVP comparison artifact because:

- it keeps source-type behavior stable;
- it avoids null labels in strict benchmark mode;
- it preserves MiniLM topic predictions when confidence is high;
- it creates a comparable JSONL file for validators/evaluators.

## Next steps

1. Commit MiniLM raw benchmark output separately from code/docs if the team wants to track it.
2. Commit hybrid builder and hybrid output separately.
3. Compare hybrid behavior on FineWeb-Edu and FineMath tiny samples only after reviewing benchmark misses.
4. Consider an embedding-output validator mode that allows low-confidence null labels.
5. Consider improving taxonomy descriptions for source/frame-sensitive labels such as news, forum Q&A, product pages, and reference articles.
6. Keep OpenWebMath optional_later.
7. Keep NLL/logprob/effective context outside this MiniLM comparison step.
