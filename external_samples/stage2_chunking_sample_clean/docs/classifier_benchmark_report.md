# Rule-based classifier benchmark report

## Purpose

Synthetic local benchmark for validating chunk label plumbing and the simple rule-based classifier. It is meant to catch obvious regressions before real HF samples or probability profiling.

## Data

- Synthetic documents: 30.
- Generated chunks: 30 with current default chunker settings.
- Categories covered: math calculus, math algebra, code docs, commercial product pages, forum Q&A, biology, physics, environmental science, urban infrastructure, multilingual English/Russian, boilerplate/navigation, wiki/reference, legal/government notices, and news-like articles.
- Easy cases: direct single-topic examples with obvious keywords.
- Hard cases: Q&A mentioning cookies, useful articles with footer noise, product pages with educational explanation, math with code, code with formulas, heading-heavy news, table-like content, markdown-heavy docs, and LaTeX-heavy math.

## Pipeline

Chunker command:

```bash
python scripts\sample_fineweb_chunks.py --local-input examples\local_docs_classifier_benchmark.jsonl --max-docs 100 --out data_samples\classifier_benchmark_chunks.jsonl --stats-out data_samples\classifier_benchmark_run_stats.json
```

Classifier command:

```bash
python scripts\classify_chunks_rule_based.py --input data_samples\classifier_benchmark_chunks.jsonl --output data_samples\classifier_benchmark_labeled.jsonl
```

Evaluator command:

```bash
python scripts\evaluate_chunk_labels.py --input data_samples\classifier_benchmark_labeled.jsonl
```

## Results

Final local metrics after rule updates:

- source_type_accuracy: 1.0000
- domain_accuracy: 1.0000
- field_accuracy: 1.0000
- subfield_accuracy: 1.0000
- full_label_accuracy: 1.0000

Predicted source_type counts:

- `educational`: 9
- `math`: 5
- `code`: 4
- `boilerplate_or_noise`: 2
- `commercial_product`: 2
- `forum_qa`: 2
- `news`: 2
- `unknown`: 2
- `legal_government`: 1
- `wiki_reference`: 1

Main confusions:

- Initial baseline confused algebra with boilerplate because the word `terms` appeared in math text.
- Initial baseline confused product pages with code because `returns` matched the loose `return` signal.
- Initial baseline over-labeled useful articles with footer/privacy text as boilerplate.
- Initial baseline had no dedicated wiki/reference, legal/government, news, science-subfield, or infrastructure rules.
- Final run had no mismatches on this synthetic benchmark.

## Manual observations

The updated rules work well for obvious local smoke cases. Forum Q&A is checked before boilerplate, so cookie-banner discussions remain forum content. Product pages are detected before environmental/educational content, which helps mixed product explainers. Math rules now distinguish calculus and algebra with small keyword clusters.

Fragile areas remain: useful text plus boilerplate is still classified as a whole chunk, not as spans; source_type and domain can disagree in real pages; and synthetic keyword coverage is easier than real web text.

## Limitations

- Synthetic benchmark is not real FineWeb.
- Rules are heuristic.
- Source_type and domain labels can be mixed in real documents.
- Boilerplate detection currently labels whole chunks, not spans.
- Expected labels are approximate human-authored benchmark labels.

## Next improvements

- Add `contains_boilerplate` flag.
- Add stronger cleaning/boilerplate removal.
- Run the optional embedding nearest-label baseline when dependencies and local model files exist.
- Replace/extend the small local taxonomy with an OpenAlex-like taxonomy.
- Add real streaming sample later, outside this local smoke test.
