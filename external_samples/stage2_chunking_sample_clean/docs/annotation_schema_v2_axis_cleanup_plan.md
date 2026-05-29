# Annotation schema v2 axis cleanup plan

## Purpose

This document defines a cleanup direction for `annotation_v2` before using topic labels for an embedding bake-off or stronger NLL/logprob/perplexity profiling claims.

The current metadata layer is useful for pilot profiling, but the `topic.domain` field still combines several different concepts:

- semantic subject matter;
- genre or document function;
- quality/noise state;
- surface cues;
- source/provenance priors.

The cleanup goal is not to replace the existing weak baseline immediately. The goal is to separate axes so future experiments measure the intended property.

## Why cleanup is needed

`weak_topic_domain_v2.1` improved the dev benchmark but degraded substantially on held-out v2-test:

| Split | Coverage | Accuracy on answered | Strict accuracy |
| --- | ---: | ---: | ---: |
| v1 patched dev | 0.9083 | 0.8165 | 0.7417 |
| v2-test held-out | 0.7889 | 0.6197 | 0.4889 |

The held-out drop indicates that the keyword/surface/prior baseline is brittle outside the dev set. This is especially visible on FineWeb, where v2-test reached:

- coverage: 0.7333;
- accuracy on answered: 0.3182;
- strict accuracy: 0.2333.

FineWeb failures are a warning against continuing to tune keyword rules on the same target. They also expose a schema problem: several labels currently used as `topic.domain` are often not semantic topics at all.

The current `topic.domain` field mixes semantic topics with genre/function labels. For example, an educational science explainer is not semantically "education"; its semantic topic may be `natural_science`, while its function is `education_tutorial`. A commercial product page is not semantically "commercial"; it may be about industrial equipment, medicine, finance, software, or lifestyle goods.

## Current problematic labels

### `reference`

`reference` describes how content is organized or used. Encyclopedic entries, glossaries, API references, directories, and tables can all be reference-like, but their semantic topic depends on the content.

Examples:

- a botanical encyclopedia entry is `semantic_topic_domain=natural_science` and `genre_function=reference`;
- an API class reference is `semantic_topic_domain=computer_science_software` and may be `genre_function=documentation_api`;
- a city directory is likely `semantic_topic_domain=lifestyle_everyday` or `law_government_policy`, depending on content, with `genre_function=list_table_directory`.

### `education`

`education` usually describes pedagogical intent, not the subject. A tutorial, lesson, worksheet, course page, or study guide can cover math, science, humanities, law, health, or software.

Examples:

- a calculus lesson is `semantic_topic_domain=math` and `genre_function=education_tutorial`;
- a biology worksheet is `semantic_topic_domain=natural_science` and `genre_function=education_tutorial`;
- a civics classroom resource is `semantic_topic_domain=law_government_policy` or `social_sciences`, with `genre_function=education_tutorial`.

### `media`

`media` is usually a publishing format or distribution context, not the semantic subject. News articles, interviews, reviews, transcripts, and entertainment snippets can cover any domain.

Examples:

- a news article about elections is `semantic_topic_domain=law_government_policy` and `genre_function=news_media`;
- a review of a phone is `semantic_topic_domain=engineering_technology` or `lifestyle_everyday`, with `genre_function=blog_opinion` or `news_media`;
- a science magazine excerpt is `semantic_topic_domain=natural_science` and `genre_function=news_media`.

### `commercial`

`commercial` describes intent or page function: selling, advertising, pricing, service promotion, product listing, lead generation, or affiliate content. It is not a semantic subject.

Examples:

- an industrial sensor product page is `semantic_topic_domain=engineering_technology` and `genre_function=product_commercial`;
- a SaaS pricing page is `semantic_topic_domain=computer_science_software` or `business_economics`, with `genre_function=product_commercial`;
- a supplement storefront may be `semantic_topic_domain=medicine_health` or `lifestyle_everyday`, with `genre_function=product_commercial`.

### `unknown`

`unknown` is an abstention or uncertainty state. It should not be treated as a positive semantic topic. It can mean the content is too short, too noisy, too mixed, boilerplate-heavy, or outside the current taxonomy.

In a cleaned schema, unresolved content should use `semantic_topic_domain=other_unclear` only when a label is required, or abstain when the labeling workflow supports abstention.

## Proposed cleaned axes

### A. `semantic_topic_domain`

This axis should answer: "What is the main subject matter of the text?"

Recommended initial labels:

- `math`
- `natural_science`
- `computer_science_software`
- `engineering_technology`
- `medicine_health`
- `business_economics`
- `law_government_policy`
- `social_sciences`
- `humanities_arts`
- `lifestyle_everyday`
- `other_unclear`

Labeling policy:

- Prefer the concrete subject over document function.
- Use `other_unclear` when the text is too mixed, too short, too noisy, or not covered by the taxonomy.
- Do not use commercial, educational, media, or reference status as the semantic label.

### B. `genre_function`

This axis should answer: "What kind of document or communicative function is this?"

Recommended labels:

- `reference`
- `education_tutorial`
- `news_media`
- `forum_qa`
- `blog_opinion`
- `product_commercial`
- `documentation_api`
- `academic_paper_like`
- `list_table_directory`
- `boilerplate_navigation`
- `other_function`

Labeling policy:

- Assign based on structure, purpose, and discourse mode.
- Do not infer semantic topic from genre alone.
- Allow future workflows to use multi-label genre/function if needed, but keep v1 single-label for MVP simplicity.

### C. `quality_noise`

This axis should continue to represent cleanliness and extraction quality.

Keep or extend labels:

- `clean`
- `partial_noise`
- `mostly_noise`
- `spam_adult_seo`
- `ui_residue`
- `duplicated_template`

Labeling policy:

- Treat quality/noise as orthogonal to topic and genre.
- A product page can be clean or spammy.
- A scientific text can contain boilerplate residue.
- A noisy chunk should not automatically become `semantic_topic_domain=other_unclear` if the subject is still clear, but uncertainty should be recorded.

### D. `surface_features`

Keep existing deterministic flags as non-semantic evidence:

- `has_math_notation`
- `has_code`
- `is_symbol_heavy`
- `has_scientific_formula`
- `has_api_or_command_syntax`
- `has_table_or_list`
- `has_urls_or_links`
- `has_boilerplate_markers`

Labeling policy:

- Surface flags should support profiling and diagnostics.
- They should not be treated as final topic labels.
- For example, `has_math_notation=true` may support `math` or `natural_science`, but it can also appear in software documentation, finance, or noisy extraction.

## Compatibility with current `annotation_v2`

This cleanup should be additive.

Do not delete the current `annotation_v2.topic.domain` field. It remains useful for backward compatibility, historical comparison, and reproducing the current weak baseline.

Recommended compatibility naming:

- treat current `annotation_v2.topic.domain` as `legacy weak_topic_domain`;
- add future cleaned semantic labels under `annotation_v2.semantic_topic`;
- add future genre/function labels under `annotation_v2.genre_function`.

Possible future shape:

```json
{
  "annotation_v2": {
    "topic": {
      "domain": "legacy_value"
    },
    "semantic_topic": {
      "domain": "natural_science",
      "confidence": 0.74,
      "method": "future_method_name"
    },
    "genre_function": {
      "label": "education_tutorial",
      "confidence": 0.82,
      "method": "future_method_name"
    }
  }
}
```

The exact schema can be finalized later. The important decision is axis separation.

## Impact on NLL/profiling

Cleaned axes reduce false claims when interpreting model behavior.

If a profiling report says a model has high NLL on "commercial" text, that could mean many different things: product pages, noisy SEO pages, e-commerce templates, medicine ads, industrial equipment listings, or software pricing pages. That mixes topic, genre, and quality.

With separated axes:

- a commercial product page can have `semantic_topic_domain=engineering_technology` and `genre_function=product_commercial`;
- an educational science text can have `semantic_topic_domain=natural_science` and `genre_function=education_tutorial`;
- an API reference can have `semantic_topic_domain=computer_science_software` and `genre_function=documentation_api`;
- a noisy legal terms page can have `semantic_topic_domain=law_government_policy`, `genre_function=boilerplate_navigation`, and `quality_noise=partial_noise` or `mostly_noise`.

This allows NLL comparisons such as:

- topic-controlled genre comparisons;
- genre-controlled topic comparisons;
- quality/noise stratification;
- deterministic surface-feature slices independent of weak topic labels.

For near-term NLL pilots, prefer robust axes first:

- provenance/dataset;
- deterministic text stats;
- tokenizer stats;
- surface flags;
- quality/noise fields with caveats.

Use legacy `topic.domain` only as weak exploratory metadata with confidence, abstention, and split-specific caveats.

## Impact on embedding bake-off

Future embedding bake-off should target `semantic_topic_domain`, not the current mixed `topic.domain`.

Reasons:

- embedding label descriptions work best when labels describe one conceptual axis;
- current labels like `commercial`, `education`, `media`, and `reference` would reward function detection rather than subject detection;
- a mixed target makes it hard to interpret whether embeddings are good at semantic similarity, genre recognition, or noise detection.

Recommended policy:

- build or remap pseudo-gold for `semantic_topic_domain` first;
- evaluate genre/function separately, using rules, LLM/manual labeling, or a later classifier;
- keep v2-test held out and do not tune on it;
- if the schema changes substantially, create a new dev/test split or map v2-test carefully with explicit audit notes.

## Recommended next step

Create a small v1-dev pseudo-gold pass for:

- `annotation_v2.semantic_topic.domain`;
- `annotation_v2.genre_function.label`.

Then run an embedding bake-off against `semantic_topic_domain` only. In parallel, NLL pilot plumbing can start with deterministic/provenance/tokenization/quality axes that are already robust enough for exploratory grouping.
