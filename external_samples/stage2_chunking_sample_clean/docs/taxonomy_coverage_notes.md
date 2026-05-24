# Taxonomy coverage notes

## Purpose

This note compares rule-based classifier outputs with `taxonomy/simple_domain_labels.json`.

## Current result

Checked `data_samples/classifier_benchmark_labeled.jsonl` after the hardened benchmark run.

- records checked: 42
- unique predicted label tuples: 15
- labels covered by taxonomy: 15
- labels missing from taxonomy: 0
- unique source_type values: 10

## Covered predicted domain/field/subfield tuples

- `commercial/product_page/retail`
- `education/general_education/article`
- `government/legal_notice/public_information`
- `infrastructure/urban_systems/article`
- `media/news/article`
- `multilingual/mixed_language/null`
- `reference/encyclopedic_article/general`
- `science/biology/article`
- `science/environmental_science/article`
- `science/physics/article`
- `software/programming/documentation`
- `stem/mathematics/algebra`
- `stem/mathematics/calculus`
- `web/boilerplate_or_navigation/page_noise`
- `web/forum_qa/discussion`

## Source type values

- `boilerplate_or_noise`
- `code`
- `commercial_product`
- `educational`
- `forum_qa`
- `legal_government`
- `math`
- `news`
- `unknown`
- `wiki_reference`

## Naming notes

`source_type` is intentionally separate from taxonomy labels. For example, `math` maps to `stem/mathematics/*`, and `code` maps to `software/programming/documentation`.

No taxonomy changes were needed in this pass. The current MVP taxonomy covers all rule-based outputs on the local hardened benchmark.
