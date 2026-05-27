# Real HF benchmark v1 pseudo-gold summary

## Scope

This document summarizes pseudo-gold labels for `data_samples/real_hf_benchmark_v1_review_candidates.jsonl`.

The labeled output is:

- `data_samples/real_hf_benchmark_v1_pseudo_gold.jsonl`

The labels were assigned from the chunk text. Existing rule-based, lexical, MiniLM, and hybrid predictions were treated only as review hints.

## Counts

Total records labeled: 120

| Dataset | Records |
| --- | ---: |
| FineWeb | 40 |
| FineWeb-Edu | 40 |
| FineMath | 40 |

## Expected source type distribution

| expected_source_type | Count |
| --- | ---: |
| educational | 39 |
| reference | 35 |
| math | 15 |
| web_general | 9 |
| news | 7 |
| forum_qa | 5 |
| commercial_product | 4 |
| boilerplate_or_noise | 2 |
| code | 2 |
| legal_government | 1 |
| wiki_reference | 1 |

## Expected domain distribution

| expected_domain | Count |
| --- | ---: |
| science | 31 |
| software | 27 |
| stem | 23 |
| education | 13 |
| reference | 12 |
| media | 7 |
| commercial | 2 |
| infrastructure | 2 |
| unknown | 2 |
| government | 1 |

## Expected field distribution

| expected_field | Count |
| --- | ---: |
| programming | 27 |
| mathematics | 23 |
| general_education | 13 |
| physics | 13 |
| encyclopedic_article | 12 |
| environmental_science | 12 |
| news | 7 |
| biology | 6 |
| product_page | 2 |
| unknown | 2 |
| urban_systems | 2 |
| legal_notice | 1 |

## Common taxonomy gaps

The current taxonomy is intentionally tiny, so many real chunks had to use the closest available label with a `missing_taxonomy:` note.

Most frequent gaps:

- `patents/POS_systems`: 17
- `statistics`: 5
- `patents/hardware_networking`: 4
- `arithmetic/measurement`: 4
- `chemistry`: 3
- `art_history`: 2
- `automotive`: 2
- `food`: 2
- `geometry`: 2
- `probability/statistics`: 2
- `spam/low_quality_math`: 2
- `sports/coaching`: 2
- `topology`: 2

Other one-off gaps include medicine, psychology, international relations, religion, clinical trials, gender studies, weather, crafts, computer science algorithms, biography, public health, finance, and business/marketing.

## Dataset notes

FineWeb is a broad web mix. The sampled candidates include product pages, local news, blog-like articles, environmental advice, and many patent-style technical chunks. Several patent chunks are semantically about software/POS systems, but the current taxonomy has no patent or technical-legal category.

FineWeb-Edu is not automatically educational. The sample contains educational activities and environmental articles, but also book reviews, art history, political history, biography/reference material, and advocacy/social-issue articles.

FineMath is not automatically pure math. Many chunks are mathematical, but the sample also contains physics lessons, chemistry tutorials, forum-style probability questions, and low-quality SEO-like math text. FineMath still remains the only current MVP math dataset.

## Ambiguous cases

Ambiguity was common in these cases:

- patent-style technical text: `source_type=reference`, domain closest to `software`;
- health and psychology articles: closest labels often use `science/biology/article`;
- chemistry tutorials: closest label uses `science/physics/article` because chemistry is absent;
- statistics/probability: closest label uses `stem/mathematics/algebra`;
- sports, food, automotive, and personal essays: usually mapped to `media/news`, `education/general_education`, or `unknown` depending on the chunk;
- low-quality generated/SEO math pages: labeled as `boilerplate_or_noise` with `unknown` topic.

## Limitations

These are pseudo-gold labels produced by a coding agent, not adjudicated human labels. They are useful for pipeline debugging, relative comparison, and taxonomy gap discovery. They should not be treated as final benchmark ground truth without human review.

The taxonomy is small and forces some imperfect nearest-label choices. Metrics against this pseudo-gold should be interpreted as a readiness signal, not as a final quality claim.
