# FineMath tiny sample report

## Purpose

Second real HF streaming sample and first math-domain sample for the current MVP.

## Source

- dataset id: `HuggingFaceTB/finemath`
- config: `finemath-4plus`
- split: `train`
- text_field: `text`
- max_docs: 20
- command used:

```bash
python scripts\sample_fineweb_chunks.py --use-hf-streaming --dataset HuggingFaceTB/finemath --config finemath-4plus --split train --dataset-label FineMath --source-type math --text-field text --max-docs 20 --out data_samples\real_samples\finemath_chunks.jsonl --stats-out data_samples\real_samples\finemath_run_stats.json
```

## Chunking results

- docs_seen: 20
- docs_with_text: 20
- docs_missing_text_field: 0
- chunks_written: 101
- token_count min/mean/max: 81 / 169.99 / 465
- stats path: `data_samples\real_samples\finemath_run_stats.json`

## Labeling results

Rule-based output:

- records: 101
- source_type distribution: `math=68`, `educational=31`, `commercial_product=1`, `forum_qa=1`
- domain distribution: `null=44`, `education=30`, `stem=24`, `commercial=1`, `science=1`, `web=1`
- label_method distribution: `rule_based_keyword_v1=57`, `rule_based_unknown=44`

Lexical output:

- records: 101
- source_type distribution: `math=101`
- domain distribution: `null=56`, `stem=34`, `education=7`, `science=2`, `commercial=1`, `unknown=1`
- label_method distribution: `lexical_nearest_label_low_confidence=56`, `lexical_nearest_label_v1=45`

Rule-based vs lexical comparison:

- common chunk_ids: 101
- matching full labels: 49
- differing full labels: 52
- agreement rate: 0.4851
- source_type disagreements: 33
- domain disagreements: 46
- field disagreements: 47
- subfield disagreements: 46

## Manual observations

- Texts look math-oriented overall: worksheets, curriculum descriptions, word problems, algebra questions, substitution problems, fractions, geometry, and applied lessons.
- LaTeX/formula-like material appears, including escaped dollar signs and compact equations.
- Some rows contain noisy web artifacts, product/resource listings, cookie/privacy text, repeated links, and teaching-resource sales language.
- Chunking produced more chunks than FineWeb-Edu because FineMath rows are often structured as many short exercises or sections.
- The `text` field worked for all 20 streamed rows; no missing text field issue appeared.
- Some chunks are very long after fallback tokenization, with a maximum of 465 tokens.

## Comparison with FineWeb-Edu

- FineWeb-Edu: 20 docs -> 44 chunks; FineMath: 20 docs -> 101 chunks.
- FineWeb-Edu token_count min/mean/max: 83 / 321.07 / 416.
- FineMath token_count min/mean/max: 81 / 169.99 / 465.
- FineMath has more chunks per document and shorter average chunks, but a slightly longer maximum chunk.
- FineWeb-Edu rule-vs-lexical agreement: 0.5455.
- FineMath rule-vs-lexical agreement: 0.4851.
- FineMath lexical labeling finds more `stem/mathematics` labels than rule-based, while rule-based often maps math lessons to broad `education/general_education`.

## Limitations

- Only 20 documents.
- First math-domain sanity check, not representative.
- No human gold labels.
- No embedding classifier.
- No NLL/logprob scoring.
- Null domain/field/subfield values are expected for low-confidence real sample labels.

## Next recommendation

Manually review the FineMath chunks and disagreements before changing rules. The most useful next classifier improvement is likely a better distinction between math source type, math domain labels, and general educational lesson text. After review, either improve math taxonomy/rules lightly or proceed to a very small observed-token NLL pilot on selected chunks.
