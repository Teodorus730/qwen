# Semantic topic and genre/function pseudo-gold report

## Purpose

This report summarizes the cleaned dev pseudo-gold created for `real_hf_benchmark_v1`.

The goal was to create a better development target for future schema cleanup and embedding experiments by separating:

- what the text is about: `semantic_topic_domain`;
- what kind/function of text it is: `genre_function`.

This was needed because the old `annotation_v2.topic.domain` mixed semantic topics with function-like labels such as `reference`, `education`, `media`, `commercial`, and `unknown`.

## Inputs

Primary input:

- `data_samples/real_hf_benchmark_v1_annotation_v2_pseudo_gold_patched.jsonl`

Taxonomy inputs:

- `taxonomy/semantic_topic_domain_v1_descriptions.json`
- `taxonomy/genre_function_v1_descriptions.json`

No v2-test data was used. No classifier, feature extractor, tokenizer stats, embedding model, MiniLM, AutoModel, HF streaming, or NLL/logits step was run.

## Output

Created:

- `data_samples/real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl`
- `data_samples/real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold_summary.json`

Records:

- total: 120;
- FineWeb: 40;
- FineWeb-Edu: 40;
- FineMath: 40.

Integrity checks:

- 120 records: pass;
- no duplicate `chunk_id`: pass;
- required fields present: pass;
- allowed labels only: pass;
- confidence values in `[0.0, 1.0]`: pass.

## Axis definitions

`semantic_topic_domain` answers:

> What is the main subject matter of the text?

`genre_function` answers:

> What kind of text is this, or what function does it serve?

Examples:

- clothing storefront: `semantic_topic_domain=lifestyle_everyday`, `genre_function=product_commercial`;
- OED/dictionary entry: `semantic_topic_domain=humanities_arts`, `genre_function=reference`;
- science classroom activity: `semantic_topic_domain=natural_science`, `genre_function=education_tutorial`;
- automotive review: `semantic_topic_domain=engineering_technology`, `genre_function=blog_opinion` or `news_media`.

## Semantic topic distribution

| semantic_topic_domain | Count |
| --- | ---: |
| `engineering_technology` | 26 |
| `math` | 23 |
| `natural_science` | 23 |
| `humanities_arts` | 13 |
| `lifestyle_everyday` | 10 |
| `law_government_policy` | 7 |
| `computer_science_software` | 5 |
| `medicine_health` | 5 |
| `social_sciences` | 5 |
| `business_economics` | 1 |
| `other_unclear` | 2 |

Semantic abstention:

- count: 2;
- share: 0.016667.

## Genre/function distribution

| genre_function | Count |
| --- | ---: |
| `reference` | 52 |
| `education_tutorial` | 29 |
| `blog_opinion` | 18 |
| `news_media` | 8 |
| `documentation_api` | 4 |
| `forum_qa` | 3 |
| `product_commercial` | 3 |
| `academic_paper_like` | 1 |
| `other_function` | 2 |

Genre/function abstention:

- count: 2;
- share: 0.016667.

## Interesting cross-tabs

High-signal cross-axis combinations:

| semantic_topic_domain | genre_function | Count | Interpretation |
| --- | --- | ---: | --- |
| `engineering_technology` | `reference` | 22 | Mostly patent-like POS/video rental system technical descriptions. |
| `natural_science` | `education_tutorial` | 15 | Science lessons and classroom activities. |
| `math` | `reference` | 15 | Math/statistics explanatory/reference fragments. |
| `math` | `education_tutorial` | 6 | Worked math/statistics teaching material. |
| `humanities_arts` | `reference` | 7 | Dictionary, art/history, religion, or biography reference-like content. |
| `lifestyle_everyday` | `blog_opinion` | 6 | Sports, food, training, and personal commentary. |
| `law_government_policy` | `education_tutorial` | 2 | Civics/election educational app material. |
| `computer_science_software` | `documentation_api` | 3 | Web server and Maple/API-style documentation. |

These combinations show why the split matters: the same semantic topic can appear in multiple functions, and the same function can carry multiple semantic topics.

## Old mixed-label splits

### Old `commercial`

Examples:

- `FineWeb_000008_000`: clothing product page -> `semantic_topic_domain=lifestyle_everyday`, `genre_function=product_commercial`;
- `FineWeb_000005_000`: boots product/category page -> `semantic_topic_domain=lifestyle_everyday`, `genre_function=product_commercial`;
- `FineWeb_000000_001`: services marketing explanation -> `semantic_topic_domain=business_economics`, `genre_function=education_tutorial`.

Takeaway: commercial intent is not a semantic topic.

### Old `reference`

Examples:

- `FineWeb-Edu_000018_000`: OED / English-language dictionary -> `semantic_topic_domain=humanities_arts`, `genre_function=reference`;
- `FineWeb-Edu_000024_000`: Buddhist sutta entries -> `semantic_topic_domain=humanities_arts`, `genre_function=reference`.

Takeaway: reference is a document function; subject comes from the referenced content.

### Old `education`

Examples:

- `FineWeb-Edu_000013_000`: literacy/letter-sound learning -> `semantic_topic_domain=humanities_arts`, `genre_function=education_tutorial`;
- `FineWeb-Edu_000020_000`: presidential election app for students -> `semantic_topic_domain=law_government_policy`, `genre_function=education_tutorial`;
- `FineWeb-Edu_000010_013`: athlete/teamwork reflection -> `semantic_topic_domain=lifestyle_everyday`, `genre_function=blog_opinion`.

Takeaway: education often marks pedagogical intent, but not always the final genre after reading the text.

### Old `media`

Examples:

- `FineWeb_000026_000`: Camaro article -> `semantic_topic_domain=engineering_technology`, `genre_function=news_media`;
- `FineWeb_000026_001`: Camaro review continuation -> `semantic_topic_domain=engineering_technology`, `genre_function=blog_opinion`;
- `FineWeb_000025_000`: chef/restaurant article -> `semantic_topic_domain=lifestyle_everyday`, `genre_function=blog_opinion`;
- `FineWeb_000007_000`: local crash report -> `semantic_topic_domain=lifestyle_everyday`, `genre_function=news_media`.

Takeaway: media is a publishing mode, not the subject.

## Implications for embedding bake-off

The next embedding bake-off should target `semantic_topic_domain_v1`, not the old mixed `topic.domain`.

Reasons:

- the semantic labels now describe one conceptual axis;
- label descriptions can be used cleanly for embedding similarity;
- genre/function detection can be evaluated separately;
- current `weak_topic_domain_v2.1` can remain as a transparent legacy baseline rather than the target schema.

Recommended next experiment:

1. Use this v1-dev pseudo-gold as the development target for `semantic_topic_domain_v1`.
2. Compare embedding label-description strategies against semantic topic labels only.
3. Keep `genre_function_v1` available for separate rules/LLM/manual experiments.
4. Do not reuse v2-test directly until deciding whether to relabel it under cleaned axes or create a fresh v3 holdout.

## Limitations

- This is v1-dev only.
- This is pseudo-gold, not human gold.
- This is not held out.
- It should not be reported as final classifier quality.
- Several labels remain coarse because the source chunks are partial web extracts.
- Genre/function is single-label in this MVP, even though some chunks naturally have multiple functions.

## Next step

Proceed with **embedding bake-off on `semantic_topic_domain_v1`** as the dev target.

Handle `genre_function_v1` separately by rules, LLM-assisted review, or manual labeling later.

Keep v2-test untouched until the project decides whether to relabel it under the cleaned axes or create a new v3 held-out split.
