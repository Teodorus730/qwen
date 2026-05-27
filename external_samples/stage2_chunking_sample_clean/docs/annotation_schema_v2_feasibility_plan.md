# Annotation schema v2 feasibility plan

## Purpose

This document is a design feasibility plan for the next annotation schema after `real_hf_benchmark_v1`.

It does not implement a new pipeline and does not tune the current classifiers. The goal is to decide which annotation fields are practical for the corpus preparation layer, how each field can be filled, and which fields should be included in the next MVP before downstream NLL/logprob/perplexity profiling.

## 1. Why `source_type` single-label failed

The old single-label `source_type` tried to answer too many different questions at once:

- provenance: where the text came from, for example FineWeb, FineWeb-Edu, or FineMath;
- genre/function: article, reference page, tutorial, forum answer, patent-style description, product documentation;
- format: code, math notation, list-heavy text, boilerplate/navigation;
- quality/noise: useful content vs generated/SEO-like/noisy chunks;
- topic: mathematics, patents, chemistry, art history, social issues, product pages.

The real benchmark made this visible. A chunk could be from FineMath but actually look like Maple documentation, a low-quality SEO page, a statistics lesson, or a math Q&A. A FineWeb-Edu chunk could be an educational explanation, a news-like article, an encyclopedia-style reference, or a personal essay. A FineWeb chunk could contain patent claims, product documentation, commercial text, or general web prose.

As a result, `source_type` became an overloaded full-label component. It was strict enough to break `full_label_accuracy`, but not well-defined enough for lexical, MiniLM, or hybrid classifiers to predict reliably. The source type experiment confirmed this: a separate source type MiniLM/lexical/hybrid setup did not beat the original rule-based source type overall, and many errors came from ambiguous boundaries rather than one obvious classifier bug.

The main lesson is:

```text
dataset, format, quality, topic, and genre/function should be separate axes.
```

## 2. Proposed axes

Annotation schema v2 should split the current overloaded label space into independent fields.

### Provenance

Provenance describes where the record came from and how it entered the local corpus. It should not be inferred from text.

Examples:

- dataset;
- source dataset/config/split;
- local sample id;
- sampling method;
- chunking method/version.

### Text Stats

Text stats are deterministic measurements of the chunk.

Examples:

- character count;
- byte count;
- line count;
- token count once a tokenizer is fixed;
- token-per-byte ratio once tokenizer output is available.

### Surface Features

Surface features are observable text signals that do not require semantic classification.

Examples:

- math notation markers;
- code/API/markup markers;
- numeric density;
- symbol density;
- boilerplate markers;
- URL/link/menu-like patterns.

### Quality / Noise

Quality/noise describes whether a chunk looks clean enough for downstream profiling or likely contains boilerplate, navigation, spam, generated text, duplicated fragments, or low-value content.

This should be a heuristic score or coarse level, not a deep semantic label.

### Topic

Topic describes the semantic subject matter:

- domain;
- field;
- subfield;
- confidence;
- method;
- abstention / unknown.

Taxonomy v2 showed that topic classification improves when the taxonomy actually covers the observed data. Topic should remain, but it should be treated as weak labeling unless reviewed.

### Optional Genre / Function

Genre/function describes what the text is doing:

- tutorial;
- encyclopedic reference;
- news article;
- product documentation;
- patent claim/description;
- forum Q&A;
- personal essay;
- legal notice.

This is useful, but hard. It should not be required in the first schema v2 MVP.

## 3. Feasibility by field

| Field | Method | Difficulty | Expected reliability | Needs model? | MVP/Later |
| --- | --- | --- | --- | --- | --- |
| `provenance.dataset` | Loader metadata / local sample metadata | Easy | High | No | MVP |
| `provenance.source_dataset` | Original HF dataset id or local source name | Easy | High | No | MVP |
| `provenance.source_config` | Dataset config/name when available | Easy | High | No | MVP |
| `provenance.source_split` | Loader split metadata | Easy | High | No | MVP |
| `provenance.sample_id` | Deterministic id from sampler | Easy | High | No | MVP |
| `provenance.chunk_id` | Existing chunk id | Easy | High if stable | No | MVP |
| `provenance.chunking_method` | Pipeline metadata / fixed string | Easy | High | No | MVP |
| `text_stats.char_count` | `len(text)` | Easy | High | No | MVP |
| `text_stats.byte_count` | UTF-8 encoded byte length | Easy | High | No | MVP |
| `text_stats.line_count` | Count line breaks / split lines | Easy | High | No | MVP |
| `text_stats.token_count` | Fixed tokenizer later | Medium | High if tokenizer/version fixed | Yes, tokenizer only | MVP/later |
| `text_stats.token_per_byte` | `token_count / byte_count` after tokenizer | Medium | High if tokenizer/version fixed | Yes, tokenizer only | Later if tokenizer not ready |
| `text_stats.avg_line_length` | Deterministic line stats | Easy | High | No | MVP |
| `surface.has_math_notation` | Regex for formulas, operators, TeX-ish markers, equation density | Easy/medium | Medium/high | No | MVP |
| `surface.has_code` | Regex for code/API/markup patterns, braces, imports, stack traces, function signatures | Medium | Medium | No | MVP |
| `surface.has_numbers` | Digit presence/density | Easy | High for presence, medium for meaning | No | MVP |
| `surface.has_boilerplate_markers` | Keyword/regex for cookie, privacy, menu, subscribe, nav, copyright | Easy | Medium | No | MVP |
| `surface.has_urls_or_links` | Regex for URLs/domains/link-like fragments | Easy | High for explicit URLs, medium for link fragments | No | MVP |
| `surface.symbol_density` | Count non-alphanumeric/non-whitespace symbols divided by text length | Easy | High as a statistic, medium as quality signal | No | MVP |
| `surface.digit_density` | Digit count divided by text length | Easy | High as a statistic, medium as topic signal | No | MVP |
| `surface.repetition_score` | Repeated lines/tokens/simple n-gram heuristics | Medium | Medium | No | Later/MVP if simple |
| `quality.noise_level` | Heuristic from boilerplate markers, URL/menu density, repetitive short lines, very low text quality signals | Medium | Medium | No | MVP |
| `quality.noise_reasons` | List of heuristic triggers | Medium | Medium/high as explanation | No | MVP |
| `topic.domain` | Weak labeling: lexical + MiniLM top-k + keywords + provenance priors + abstain | Hard | Medium only on confident cases | Optional MiniLM | MVP with abstention |
| `topic.domain_confidence` | Classifier score calibrated only as similarity/heuristic confidence | Hard | Medium/low calibration | Optional MiniLM | MVP |
| `topic.field` | Weak labeling only when confident | Hard | Lower than domain | Optional MiniLM | Later / optional MVP |
| `topic.subfield` | Weak labeling only when confident and taxonomy covers it | Hard | Lower | Optional MiniLM | Later |
| `topic.method` | Pipeline metadata | Easy | High | No | MVP |
| `topic.top_k` | Lexical/MiniLM diagnostic candidates | Medium | Useful for review, not truth | Optional MiniLM | MVP for diagnostics |
| `genre_or_function` | Optional classifier or LLM/pseudo-gold review | Hard | Low/medium | Usually yes | Later/exploratory |
| `educational_value` | Requires rubric or LLM/human review | Hard | Low/medium | Usually yes | Later |

## 4. Topic strategy

### Should we still use an OpenAlex-like taxonomy?

Yes, but as vocabulary inspiration rather than a direct target.

OpenAlex-like labels are useful because they provide a familiar hierarchy and make downstream grouping easier. But the real benchmark showed that a small fixed taxonomy can become a bottleneck: if a real chunk is about patents, statistics, chemistry, measurement, or topology and the taxonomy does not contain those labels, both lexical and embedding classifiers are forced into bad nearest matches or abstentions.

For the MVP, the taxonomy should be pragmatic:

- coarse enough to avoid fragile overfitting;
- broad enough to cover expected real chunks;
- explicit about `unknown` / abstention;
- versioned when changed.

### Should MVP use coarse domains first?

Yes. `topic.domain` should be the primary semantic label in the next MVP.

`topic.field` and `topic.subfield` should be optional or confident-only because they are much more sensitive to taxonomy coverage. The taxonomy v2 experiment improved semantic labels, but it also used benchmark v1 as a development set. More fine-grained labels should not become the main KPI until a held-out benchmark exists.

### How should field/subfield work?

Field/subfield can be emitted when the classifier is confident and the taxonomy label is specific enough. Otherwise they should be nullable or `unknown`, with an explicit reason:

```text
topic.domain = stem
topic.field = mathematics
topic.subfield = null
topic.abstain_reason = low_subfield_confidence
```

This is better than forcing a nearest subfield that downstream users may treat as ground truth.

### How do we avoid expensive ChatGPT labeling?

LLM labeling should not be used for the full corpus.

The scalable path is:

1. deterministic features for text stats, surface signals, and noise;
2. weak topic labels from lexical/MiniLM/keywords;
3. abstention for low-confidence cases;
4. small pseudo-gold benchmarks for evaluation and calibration;
5. optional LLM/human review only for benchmark records, disagreement samples, taxonomy audits, and ambiguous label policy.

This keeps full-corpus annotation cheap and reproducible.

### How should LLM review be used?

LLM review is useful for:

- creating pseudo-gold labels on small samples;
- explaining disagreement groups;
- identifying missing taxonomy labels;
- writing review notes for ambiguous records;
- auditing whether current labels are useful for downstream NLL/profiling.

LLM review should not be treated as an automatic full-corpus labeler in the current stage.

## 5. MVP proposal

The next implementation step should not replace the whole pipeline. It should add a small schema v2 feature layer next to existing chunk records.

### Required MVP fields

```text
provenance.dataset
text_stats.char_count
text_stats.byte_count
text_stats.line_count
surface.has_math_notation
surface.has_code
surface.has_numbers
surface.has_boilerplate_markers
surface.symbol_density
surface.digit_density
quality.noise_level
topic.domain
topic.domain_confidence
topic.domain_method
topic.domain_abstained
```

Recommended shape:

```json
{
  "provenance": {
    "dataset": "FineMath",
    "source_dataset": "HuggingFaceTB/finemath",
    "source_config": "finemath-4plus",
    "source_split": "train",
    "chunk_id": "FineMath_000002_003"
  },
  "text_stats": {
    "char_count": 1234,
    "byte_count": 1260,
    "line_count": 8
  },
  "surface": {
    "has_math_notation": true,
    "has_code": false,
    "has_numbers": true,
    "has_boilerplate_markers": false,
    "symbol_density": 0.042,
    "digit_density": 0.031
  },
  "quality": {
    "noise_level": "low",
    "noise_reasons": []
  },
  "topic": {
    "domain": "stem",
    "confidence": 0.62,
    "method": "weak_topic_v1",
    "abstained": false
  }
}
```

### Optional / later fields

```text
topic.field
topic.subfield
genre_or_function
educational_value
full OpenAlex-like labels
token_count
token_per_byte
```

These are useful, but they should not block the MVP. They need stronger evaluation and clearer label policy.

## 6. Evaluation policy

Schema v2 should not use old `full_label_accuracy` as the main KPI. Full-label accuracy is still useful as a diagnostic, but it punishes independent axes as if they were one monolithic label.

Recommended metrics:

| Area | Metrics |
| --- | --- |
| Surface flags | Precision/recall on a small pseudo-gold set; examples for false positives and false negatives |
| Noise level | Agreement with pseudo-gold coarse labels; confusion among `low`, `medium`, `high` |
| Topic domain | Accuracy only on confident predictions; coverage/abstention rate; per-domain confusion |
| Topic field/subfield | Optional diagnostic accuracy on records where the classifier emits non-abstained labels |
| Provenance/text stats | Schema completeness and deterministic consistency checks |
| Downstream support | Does the field create useful groups for NLL/logprob/perplexity profiling? Are group sizes large enough? |

For topic labels, report both:

```text
accuracy_on_answered
coverage_rate
```

A classifier that answers 20% of records with high precision may be more useful for corpus profiling than one that forces labels on every chunk with low precision.

## 7. Next implementation step

Do one small coding step only:

```text
implement deterministic feature extractor for:
- text_stats
- surface_features
- simple noise_level
```

Then run it on the existing `real_hf_benchmark_v1` chunks and inspect distributions.

This step should not tune MiniLM, should not change semantic taxonomy, and should not replace the existing classifiers. It should answer:

- Are the deterministic features stable?
- Do math/code/boilerplate flags match human intuition on the benchmark?
- Does `noise_level` separate useful text from obvious low-quality chunks?
- Are the fields useful for grouping before NLL/probability profiling?

Only after this feature layer is inspected should the project decide whether to add a new weak topic labeling layer on top.

## 8. Risks

### Heuristic false positives

Regexes can mark normal prose as code or math if they are too broad. Patent text, product documentation, and formulas embedded in prose are especially risky.

### Regex language/domain bias

English web patterns, programming syntax, and math notation are easier to detect than multilingual text, informal explanations, or OCR-like fragments. Rules may underperform on non-English or non-standard content.

### Topic remains hard

Taxonomy v2 improved coverage, but topic labeling is still not solved. Topic classification depends on label descriptions, benchmark coverage, threshold policy, and chunk quality.

### Chunk size affects labels

Short chunks may lack enough context. Long chunks may mix multiple functions and topics. Schema v2 should keep chunking metadata so label quality can be analyzed by chunk size.

### Overfitting to benchmark v1

`real_hf_benchmark_v1` is now a development benchmark. It should not be used indefinitely to add labels, tune thresholds, and claim final quality. A held-out v2-test sample is needed before presenting model-quality claims.

## Recommendation

The next MVP should pivot from one overloaded `source_type` to a multi-axis annotation schema:

- deterministic provenance and text stats;
- deterministic or heuristic surface features;
- simple explainable noise level;
- weak `topic.domain` with confidence and abstention;
- optional field/subfield and genre/function later.

For downstream NLL/logprob/perplexity profiling, this is safer than forcing every chunk into a single combined label. It gives the project usable grouping metadata while keeping uncertainty explicit.
