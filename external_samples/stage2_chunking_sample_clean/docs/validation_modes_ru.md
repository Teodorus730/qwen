# Validation modes

Дата: 2026-05-26

Stage2 has different validation/evaluation rules for synthetic benchmark data and real samples.

## Synthetic benchmark

Synthetic benchmark files have expected labels.

Use for:

- strict schema checks;
- classifier regression checks;
- accuracy/mismatch reports;
- comparing rule-based, lexical, and future embedding classifiers under controlled conditions.

Appropriate:

```bash
python scripts\validate_chunks.py --input data_samples\classifier_benchmark_labeled.jsonl --require-labels
```

Accuracy is meaningful here because expected labels exist.

## Edge-case local chunks

Edge-case local chunks are small smoke-test artifacts.

Raw chunks:

```bash
python scripts\validate_chunks.py --input data_samples\edge_case_chunks_sample.jsonl
```

Labeled chunks:

```bash
python scripts\validate_chunks.py --input data_samples\edge_case_chunks_labeled.jsonl --require-labels
```

## Real samples

Real samples do not have gold labels.

Use for:

- chunk-quality review;
- classifier sanity checks;
- rule-based vs lexical disagreement review;
- later rule-based vs lexical vs embedding comparison.

Do not treat agreement with a classifier as accuracy.

Usually validate without `--require-labels`:

```bash
python scripts\validate_chunks.py --input data_samples\real_samples\fineweb_edu_sample10bt_chunks.jsonl
python scripts\validate_chunks.py --input data_samples\real_samples\finemath_chunks.jsonl
```

For already labeled real outputs, schema validation is useful, but labels are still predictions, not gold truth.

## MiniLM stage

MiniLM should use:

- the same chunks;
- the same taxonomy;
- the same output contract;
- separate output files;
- comparison against rule-based and lexical outputs.

Review metrics:

- agreement rate;
- disagreement examples;
- low-confidence count;
- top-k ambiguity if available;
- manual review notes.

Do not report real-sample classifier agreement as model accuracy.
