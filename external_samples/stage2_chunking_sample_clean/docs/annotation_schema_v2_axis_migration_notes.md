# Annotation schema v2 axis migration notes

## Purpose

These notes describe how the current mixed `annotation_v2.topic.domain` values should be interpreted during a future migration to separated axes.

This is a planning document only. Do not auto-patch old pseudo-gold yet.

## Compatibility rule

Keep current `annotation_v2.topic.domain` intact as legacy weak metadata.

Recommended future additions:

- `annotation_v2.semantic_topic.domain`
- `annotation_v2.genre_function.label`

The current `topic.domain` should be treated as `legacy weak_topic_domain`, not as the final target for embedding bake-off or final NLL claims.

## Approximate mapping from old values

| Old `topic.domain` value | Approximate `semantic_topic_domain` | Approximate `genre_function` | Migration note |
| --- | --- | --- | --- |
| `stem` | split into `math`, `natural_science`, `engineering_technology`, or sometimes `computer_science_software` | depends on document function | Too broad for the cleaned schema. Requires content review or model-assisted remap. |
| `math` | `math` | depends on document function | Usually a direct semantic topic if math is the main subject. |
| `science` | `natural_science` or `medicine_health` | depends on document function | Direct only when non-clinical natural science is central. |
| `software` | `computer_science_software` | often `documentation_api`, `education_tutorial`, `forum_qa`, or `reference` | Usually semantic, but API/docs/tutorial function should move to genre. |
| `technology` | `engineering_technology` or `computer_science_software` | often `product_commercial`, `reference`, or `news_media` | Split hardware/systems from software. Product intent belongs to genre. |
| `business` | `business_economics` | depends on document function | Usually semantic when business/economics content is central. |
| `law` / `government` / policy-like labels | `law_government_policy` | often `reference`, `news_media`, or `boilerplate_navigation` | Legal terms pages may be substantive or boilerplate; review function and quality. |
| `social` / social-science-like labels | `social_sciences` | depends on document function | Separate empirical/social analysis from humanities or policy. |
| `humanities` / `arts` / history-like labels | `humanities_arts` | depends on document function | Usually semantic when cultural, historical, philosophical, literary, or arts content is central. |
| `lifestyle` | `lifestyle_everyday` | often `blog_opinion`, `product_commercial`, `news_media`, or `reference` | Usually semantic for everyday consumer/practical topics. |
| `commercial` | depends on content | `product_commercial` | Not a direct semantic topic. |
| `education` | depends on subject | `education_tutorial` | Not a direct semantic topic. |
| `reference` | depends on subject | `reference` | Not a direct semantic topic. |
| `media` | depends on subject | `news_media` or sometimes `blog_opinion` | Not a direct semantic topic. |
| `unknown` | `other_unclear` or abstain | `other_function` or abstain | Treat as uncertainty, not a positive topic. |

Exact old label names should be verified against the current taxonomy and pseudo-gold before any scripted migration. This table is a conceptual map, not a patch specification.

## Labels that are not direct semantic topics

The following old labels should not be carried forward as `semantic_topic_domain` values:

- `commercial`
- `education`
- `reference`
- `media`
- `unknown`

They describe function, source style, or uncertainty more often than subject matter.

## Required examples

### Old `commercial`

Old:

- `topic.domain=commercial`

Future:

- `semantic_topic_domain` depends on content;
- `genre_function=product_commercial`.

Examples:

- industrial pump page: `semantic_topic_domain=engineering_technology`, `genre_function=product_commercial`;
- SaaS pricing page: `semantic_topic_domain=computer_science_software` or `business_economics`, `genre_function=product_commercial`;
- skincare storefront: `semantic_topic_domain=lifestyle_everyday` or `medicine_health`, `genre_function=product_commercial`.

### Old `education`

Old:

- `topic.domain=education`

Future:

- `semantic_topic_domain` depends on subject;
- `genre_function=education_tutorial`.

Examples:

- algebra lesson: `semantic_topic_domain=math`, `genre_function=education_tutorial`;
- biology worksheet: `semantic_topic_domain=natural_science`, `genre_function=education_tutorial`;
- civics lesson: `semantic_topic_domain=law_government_policy` or `social_sciences`, `genre_function=education_tutorial`.

### Old `reference`

Old:

- `topic.domain=reference`

Future:

- `semantic_topic_domain` depends on subject;
- `genre_function=reference`, or `documentation_api` for API/software docs.

Examples:

- encyclopedia entry about a mineral: `semantic_topic_domain=natural_science`, `genre_function=reference`;
- Python API reference: `semantic_topic_domain=computer_science_software`, `genre_function=documentation_api`;
- legal glossary: `semantic_topic_domain=law_government_policy`, `genre_function=reference`.

### Old `unknown`

Old:

- `topic.domain=unknown`

Future:

- `semantic_topic_domain=other_unclear` only when a required label is needed;
- otherwise abstain;
- use `quality_noise` and `genre_function` separately when the chunk is noisy or boilerplate-heavy.

Examples:

- unreadable SEO residue: abstain or `semantic_topic_domain=other_unclear`, likely `quality_noise=spam_adult_seo` or `mostly_noise`;
- navigation-only chunk: `genre_function=boilerplate_navigation`, `semantic_topic_domain=other_unclear` or abstain;
- short but clear product snippet: assign concrete semantic topic if recoverable, plus `genre_function=product_commercial`.

## What not to do now

Do not:

- auto-patch old pseudo-gold;
- overwrite current `topic.domain`;
- retune `weak_topic_domain_v2.1` on v2-test;
- treat this mapping table as an evaluation target;
- remove old sweep outputs or early smoke artifacts.

## Recommended future task

Create new pseudo-gold labels on v1-dev for:

- `semantic_topic.domain`;
- `genre_function.label`.

Then decide how to evaluate:

1. Use v1-dev for schema and classifier development.
2. Keep v2-test held out unless doing a carefully documented remap.
3. If the schema changes substantially, create a fresh v3 held-out split for independent claims.

The safest next path is to build cleaned pseudo-gold on v1-dev, run embedding experiments against `semantic_topic_domain`, and reserve held-out evaluation for a split that matches the cleaned schema.
