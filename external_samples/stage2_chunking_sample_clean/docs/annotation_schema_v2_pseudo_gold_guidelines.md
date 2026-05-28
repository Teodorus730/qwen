# Annotation schema v2 pseudo-gold guidelines

## Purpose

`real_hf_benchmark_v1_annotation_v2_pseudo_gold.jsonl` is a small development reference set for the new multi-axis annotation schema v2. It is intended for evaluating deterministic surface features, quality/noise labels, and the future weak `topic.domain` classifier with confidence and abstention.

This is not a replacement for held-out human gold. It is a reviewed pseudo-gold layer for the same 120 real HF benchmark chunks.

## Difference from old pseudo-gold

The old pseudo-gold used a strict full-label tuple:

```text
source_type + domain + field + subfield
```

That tuple was useful for discovering failure modes, but it overloaded one label space with provenance, topic, genre/function, surface format, and quality. `source_type` in particular became too ambiguous to be the central metric.

Annotation schema v2 separates labels into independent axes:

- provenance is metadata;
- text stats are deterministic measurements;
- surface features describe visible format signals;
- quality/noise describes residue and usefulness;
- topic is a coarse semantic grouping with confidence/abstain;
- field/subfield are optional diagnostics, not the primary KPI.

## Required review fields

Every pseudo-gold record should include:

```text
review_topic_domain
review_topic_confidence
review_topic_abstained
review_topic_note
review_has_math_notation
review_has_code
review_is_symbol_heavy
review_has_scientific_formula
review_has_api_or_command_syntax
review_noise_level
review_has_ui_residue
review_has_forum_residue
review_confidence
review_note
```

The record should also preserve:

- `chunk_id`;
- `dataset`;
- text or text preview;
- old `expected_source_type/domain/field/subfield` for traceability;
- deterministic/tokenized `annotation_v2` fields when available.

## Topic axis

Primary fields:

- `review_topic_domain`
- `review_topic_confidence`
- `review_topic_abstained`
- `review_topic_note`

Allowed coarse domains:

- `stem`
- `science`
- `technology`
- `software`
- `humanities`
- `social_sciences`
- `commercial`
- `government`
- `media`
- `reference`
- `education`
- `unknown`

Rules:

- Use coarse topic labels for future NLL grouping.
- Do not force field/subfield everywhere.
- If the topic is unclear, mixed, or the chunk is mostly boilerplate/noise, use `unknown` and set `review_topic_abstained=true`.
- If the chunk is clearly about a broad topic but field/subfield is uncertain, keep a coarse domain with lower confidence.
- Old `expected_domain/field/subfield` can inform the decision but should not be treated as automatically correct.
- Old field/subfield may be kept as optional traceability/diagnostic fields, not as primary evaluation targets.

## Surface review axis

Fields:

- `review_has_math_notation`
- `review_has_code`
- `review_is_symbol_heavy`
- `review_has_scientific_formula`
- `review_has_api_or_command_syntax`

Rules:

- `review_has_math_notation=true` only when the text contains visible formulas, symbolic expressions, TeX-like notation, equations, derivations, probability notation, or similar math surface.
- Plain numbers alone do not imply math notation.
- `review_has_code=true` only when there is visible code, markup, SQL, command output, API/help syntax, function signatures, stack traces, or code-like snippets.
- Technical patent prose is not code unless it contains actual code/API/command syntax.
- `review_is_symbol_heavy=true` when symbols, formulas, equations, tables, or compact notation materially affect tokenization/format.
- `review_has_scientific_formula=true` for chemical formulas, units-heavy scientific formulas, TeX scientific notation, or compact scientific expressions.
- `review_has_api_or_command_syntax=true` for function signatures, command flags, library help examples, CLI/API docs, or programming-oriented call syntax.

## Quality review axis

Fields:

- `review_noise_level`
- `review_has_ui_residue`
- `review_has_forum_residue`

Allowed `review_noise_level` values:

- `clean`
- `partial_noise`
- `mostly_noise`
- `unknown`

Rules:

- `clean`: primary text is useful and has no meaningful boilerplate/residue.
- `partial_noise`: useful primary content exists, but UI, forum, navigation, footer, link list, attribution, menu, or repeated boilerplate is mixed in.
- `mostly_noise`: most of the chunk is navigation, cookie/footer, SEO junk, repeated menu/list residue, or unusable boilerplate.
- `unknown`: text is too unclear or too short to judge.
- Symbol-heavy or formula-heavy useful text should remain `clean` unless actual residue is present.
- `review_has_ui_residue=true` for visible navigation/menu/search/login/share/report/breadcrumb/footer-like text.
- `review_has_forum_residue=true` for posted-by, reply, quote, edited, permalink, comments, thread, upvote/downvote, or Q&A forum metadata.

## Confidence scale

Use:

- `1.0`: obvious;
- `0.8`: likely;
- `0.6`: ambiguous but usable;
- `0.4`: weak;
- `0.2`: mostly unknown.

`review_topic_confidence` describes only the topic decision. `review_confidence` describes the overall review confidence for the record.

## Abstain and unknown policy

Abstain when:

- the topic is truly mixed;
- the chunk is mostly noise;
- the useful text is too short;
- the topic is not inferable from the available text;
- old labels only forced a nearest taxonomy match.

Do not use abstention for clean, understandable chunks merely because field/subfield is uncertain. In that case, use a coarse domain with lower confidence.

## Evaluation policy

Future weak topic evaluation should report:

- accuracy on answered records;
- coverage / abstention rate;
- confusion by coarse topic domain;
- calibration bins if confidence is available.

Do not use old full-label accuracy as the primary KPI for annotation schema v2.
