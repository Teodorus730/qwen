# Real HF review labeling guide

Date: 2026-05-27

Purpose: guide manual pseudo-gold labeling for real HF benchmark chunks.

Do not blindly trust predictions. Rule-based, lexical, MiniLM, and hybrid labels are hints, not truth.

## Fields to fill

For each review candidate, fill:

- `expected_source_type`;
- `expected_domain`;
- `expected_field`;
- `expected_subfield`;
- `review_note`;
- `review_confidence`.

Use `null` when a field cannot be decided.

## source_type

`source_type` describes format/content type, not the academic topic.

Examples:

- `web_general`: general web article or page;
- `educational`: lesson, explainer, classroom/student-facing material;
- `math`: math-focused text or problem/explanation;
- `code`: programming documentation or code-heavy text;
- `forum_qa`: question/answer discussion;
- `commercial_product`: product page or sales page;
- `boilerplate_or_noise`: navigation, cookie banners, footer, repeated low-content page chrome;
- `legal_government`: public notice, regulation, government announcement;
- `news`: recent-event reporting;
- `wiki_reference`: neutral reference/encyclopedic style;
- `unknown`: unclear or mixed beyond confident assignment.

If a FineMath chunk is clearly a math lesson, `source_type` can still be `math`. If it is mostly site navigation around a math page, use `boilerplate_or_noise`.

## domain / field / subfield

These describe semantic topic.

Use the current taxonomy from `taxonomy/simple_domain_labels.json`.

Examples:

- `stem / mathematics / calculus`;
- `stem / mathematics / algebra`;
- `science / biology / article`;
- `science / environmental_science / article`;
- `education / general_education / article`;
- `web / boilerplate_or_navigation / page_noise`;
- `web / forum_qa / discussion`;
- `software / programming / documentation`;
- `unknown / unknown / null`.

## Ambiguous cases

Prefer the dominant purpose of the chunk.

Examples:

- A news article about climate measurements: `source_type = news`, but topic may be `media/news/article` if reporting frame dominates.
- A classroom article about advertisements: `source_type = educational`, `domain = education`, not commercial product.
- A product page explaining water quality: `source_type = commercial_product`, even if it contains science words.
- A forum answer mentioning cookie banners: `source_type = forum_qa`, unless the chunk is mostly repeated banner/footer text.

## Boilerplate

Use boilerplate/noise labels when most of the chunk is:

- cookie banner;
- footer;
- privacy/terms links;
- navigation menus;
- repeated subscribe/search/account UI;
- low-content page chrome.

If a useful article has a small footer or cookie fragment, label the useful article topic and mention boilerplate in `review_note`.

## Mixed content

If the chunk contains multiple meaningful topics:

1. Pick the topic that dominates by purpose and text length.
2. If two topics are equally important, choose the one most relevant to corpus selection.
3. Use `review_note` to explain the ambiguity.
4. Use lower `review_confidence`.

## When to use unknown

Use `unknown` when:

- text is too short after cleaning;
- text is mostly corrupted;
- no taxonomy label fits;
- mixed content is impossible to resolve;
- language/script mixture makes the label unreliable.

Do not use `unknown` just because the model was low-confidence. Read the text first.

## review_confidence scale

Suggested scale:

- `1.0`: obvious label;
- `0.8`: likely label with minor noise;
- `0.6`: plausible but ambiguous;
- `0.4`: weak guess;
- `0.2`: mostly unknown/corrupted.

## Examples

Educational math:

```json
{
  "expected_source_type": "math",
  "expected_domain": "stem",
  "expected_field": "mathematics",
  "expected_subfield": "algebra",
  "review_note": "Math lesson about equations; small page boilerplate ignored.",
  "review_confidence": 0.9
}
```

News about climate:

```json
{
  "expected_source_type": "news",
  "expected_domain": "media",
  "expected_field": "news",
  "expected_subfield": "article",
  "review_note": "Reporting frame dominates despite environmental vocabulary.",
  "review_confidence": 0.8
}
```

Pure navigation:

```json
{
  "expected_source_type": "boilerplate_or_noise",
  "expected_domain": "web",
  "expected_field": "boilerplate_or_navigation",
  "expected_subfield": "page_noise",
  "review_note": "Repeated menu/privacy/cookie text; no substantive content.",
  "review_confidence": 1.0
}
```

Unclear:

```json
{
  "expected_source_type": "unknown",
  "expected_domain": "unknown",
  "expected_field": "unknown",
  "expected_subfield": null,
  "review_note": "Fragmented text; no reliable dominant topic.",
  "review_confidence": 0.3
}
```
