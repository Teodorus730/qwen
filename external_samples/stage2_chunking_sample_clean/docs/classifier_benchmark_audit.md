# Classifier benchmark audit

## Purpose

The synthetic classifier benchmark exists to verify local stage2 plumbing: chunk schema, expected-label propagation, rule-based classification, validation, and evaluation. It is a fast regression check before any real HF samples, embeddings, or probability profiling.

## Current benchmark coverage

Current benchmark after hardening: 42 documents and 42 default chunks.

By expected source_type:

- `educational`: 12
- `math`: 6
- `code`: 5
- `boilerplate_or_noise`: 3
- `commercial_product`: 3
- `forum_qa`: 3
- `news`: 3
- `unknown`: 3
- `legal_government`: 2
- `wiki_reference`: 2

By expected domain:

- `science`: 8
- `stem`: 6
- `web`: 6
- `software`: 5
- `commercial`: 3
- `infrastructure`: 3
- `media`: 3
- `multilingual`: 3
- `government`: 2
- `reference`: 2

## Easy cases

Several examples are intentionally obvious smoke-test cases:

- `bench_001_math_calculus`: direct derivative/limit/slope wording.
- `bench_002_math_algebra`: direct algebra, variable, polynomial, equation wording.
- `bench_003_code_python_docs`: fenced Python function and documentation phrases.
- `bench_005_product_page`: direct retail product, shipping, buy, warranty wording.
- `bench_012_pure_boilerplate`: pure navigation/cookie/footer text.
- `bench_030_latex_heavy_math`: dense LaTeX calculus notation.

These are useful regressions, but they make perfect accuracy easier than real web data.

## Tricky cases

The benchmark already includes useful mixed cases:

- `bench_016_forum_qa_with_cookies`: Q&A with cookie/privacy words.
- `bench_017_education_with_footer`: useful biology content plus footer noise.
- `bench_018_product_with_education`: product page with educational explanation.
- `bench_019_math_with_code`: math explanation plus small code block.
- `bench_020_code_with_formula`: code docs plus a formula.
- `bench_021_many_headings_not_educational`: heading-heavy news, not a lesson.
- `bench_025_long_noisy_article`: infrastructure article plus navigation/footer.
- `bench_028_table_like_content`: table-like environmental measurements.
- `bench_029_markdown_heavy`: markdown-heavy developer docs.
- `bench_031_education_mentions_product_words`: retail words used as classroom vocabulary examples.
- `bench_034_legal_with_navigation_words`: legal notice with navigation/account/service terms.
- `bench_036_news_with_science_words`: science vocabulary inside a news report.
- `bench_039_multilingual_technical_docs`: technical-doc signals inside mixed English/Russian text.
- `bench_040_science_article_with_footer`: useful environmental article plus footer/cookie language.

## Main risk

`1.0000` accuracy is a smoke-test success, not real classifier quality. The benchmark is synthetic, compact, and uses clear wording. It proves that obvious local examples and known edge cases are handled, but it does not prove robust classification on FineWeb-like noise.

## Document-level expected label limitation

Expected labels are stored at document level and copied to chunks. This is usually fine when one document becomes one chunk. It becomes unfair when aggressive chunking splits a document into topic-bearing chunks and generic context chunks. In the target-120 sweep, generic benchmark-context chunks inherited the original document label and became artificial mismatches.

## What this benchmark can prove

- schema works;
- chunker preserves expected labels;
- classifier runs;
- obvious cases are covered;
- regression checks catch broken rules;
- evaluation scripts report mismatches.

## What this benchmark cannot prove

- real FineWeb quality;
- robust domain classification;
- OpenAlex-level taxonomy quality;
- model-based classification quality;
- span-level boilerplate detection;
- calibrated confidence.

## Recommended next benchmark improvements

- Add adversarial cases where keywords appear in the wrong context.
- Add more article-vs-commercial and article-vs-boilerplate boundary cases.
- Add news with science words and legal notices with navigation words.
- Add multilingual technical notes that look like docs but are primarily mixed-language.
- Keep some imperfect outcomes documented instead of tuning rules to fake perfect accuracy.
- Consider chunk-level expected labels for aggressive chunking experiments.

## Final hardening observations

After adding 12 adversarial cases, the initial hardened run produced one mismatch: an educational media-literacy article was labeled as commercial because it mentioned `buy`, `features`, `product`, `discount`, and `warranty` as examples. A small negative-context guard fixed that clear rule issue.

Final hardened benchmark metrics are 1.0000 across source_type/domain/field/subfield/full label. This should still be interpreted as local smoke-test coverage, not evidence of real-world robustness.
