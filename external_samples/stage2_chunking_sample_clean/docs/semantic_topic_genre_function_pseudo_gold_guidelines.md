# Semantic topic and genre/function pseudo-gold guidelines

## Purpose

This guide defines dev pseudo-gold labeling rules for the cleaned annotation axes:

- `semantic_topic_domain_v1`
- `genre_function_v1`

The target split is `real_hf_benchmark_v1` only. These labels are development pseudo-gold for schema cleanup and future embedding experiments. They are not human gold and are not held-out evaluation truth.

## How this differs from old `topic.domain`

The old `annotation_v2.topic.domain` field mixed several concepts:

- semantic topic, such as math or software;
- genre/function, such as reference or education;
- page intent, such as commercial;
- uncertainty, such as unknown;
- surface/provenance cues.

The cleaned pseudo-gold separates two axes:

- `semantic_topic_domain` describes what the text is about;
- `genre_function` describes what kind of text it is or what function it serves.

Examples:

- a product page for industrial sensors can be `semantic_topic_domain=engineering_technology` and `genre_function=product_commercial`;
- a biology worksheet can be `semantic_topic_domain=natural_science` and `genre_function=education_tutorial`;
- an API reference can be `semantic_topic_domain=computer_science_software` and `genre_function=documentation_api`;
- a dictionary entry about a poet can be `semantic_topic_domain=humanities_arts` and `genre_function=reference`.

## Allowed `semantic_topic_domain` values

Use exactly the labels from `taxonomy/semantic_topic_domain_v1_descriptions.json`:

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

Labeling rule:

- choose the main subject matter;
- ignore whether the text is a tutorial, reference, news article, product page, or noisy page when choosing the semantic topic;
- use `other_unclear` when the subject is too short, mixed, noisy, or unrecoverable.

## Allowed `genre_function` values

Use exactly the labels from `taxonomy/genre_function_v1_descriptions.json`:

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

Labeling rule:

- choose the dominant document type, discourse mode, or page function;
- do not use genre/function as a proxy for semantic topic;
- use `other_function` when the function is unclear, mixed, or outside the current taxonomy.

## Confidence scale

Use a numeric confidence in `[0.0, 1.0]`.

Recommended interpretation:

- `1.0`: explicit, unambiguous label from visible text;
- `0.8`: strong label with minor ambiguity;
- `0.6`: plausible label, but the chunk is partial, mixed, or genre/topic boundary is uncertain;
- `0.4`: weak guess retained for traceability;
- `0.0`: abstained or no recoverable label.

Confidence is per axis. A record can have high semantic confidence and low genre confidence, or the reverse.

## Abstention and unclear policy

For semantic topic:

- set `semantic_topic_abstained=true` when the subject is not recoverable;
- set `semantic_topic_domain=other_unclear` for schema validity;
- set `semantic_topic_confidence=0.0` unless there is a reason to keep a weak non-abstained `other_unclear` label.

For genre/function:

- set `genre_function_abstained=true` when the document function is not recoverable;
- set `genre_function=other_function` for schema validity;
- set `genre_function_confidence=0.0` for true abstention.

Use `other_unclear` and `other_function` as explicit uncertainty buckets, not as hidden labels for difficult but recoverable content.

## Common old-label splits

### Old `commercial`

Usually:

- `genre_function=product_commercial`;
- semantic topic depends on product/content.

Examples:

- clothing storefront: `semantic_topic_domain=lifestyle_everyday`, `genre_function=product_commercial`;
- SaaS marketing page: `semantic_topic_domain=computer_science_software`, `genre_function=product_commercial`;
- industrial device page: `semantic_topic_domain=engineering_technology`, `genre_function=product_commercial`.

### Old `education`

Usually:

- `genre_function=education_tutorial`;
- semantic topic depends on subject.

Examples:

- statistics lesson: `semantic_topic_domain=math`, `genre_function=education_tutorial`;
- science classroom activity: `semantic_topic_domain=natural_science`, `genre_function=education_tutorial`;
- literacy activity: `semantic_topic_domain=humanities_arts`, `genre_function=education_tutorial`.

### Old `reference`

Usually:

- `genre_function=reference`;
- semantic topic depends on subject.

Examples:

- OED description: `semantic_topic_domain=humanities_arts`, `genre_function=reference`;
- Buddhist text index: `semantic_topic_domain=humanities_arts`, `genre_function=reference`;
- clinical registry table: `semantic_topic_domain=medicine_health`, `genre_function=list_table_directory`.

### Old `media`

Usually:

- `genre_function=news_media` for journalistic/reporting content;
- `genre_function=blog_opinion` for personal review/commentary;
- semantic topic depends on content.

Examples:

- car review article: `semantic_topic_domain=engineering_technology`, `genre_function=news_media` or `blog_opinion`;
- local crash report: `semantic_topic_domain=lifestyle_everyday`, `genre_function=news_media`;
- restaurant/chef article: `semantic_topic_domain=lifestyle_everyday`, `genre_function=blog_opinion`.

### Old `unknown`

Usually:

- `semantic_topic_abstained=true` with `semantic_topic_domain=other_unclear`, or a concrete topic if the text is recoverable after reading;
- `genre_function=other_function` or a concrete function such as `boilerplate_navigation`.

Do not preserve `unknown` as a semantic topic.

## Review note policy

Each pseudo-gold record should include short notes:

- `semantic_topic_note`: why the semantic topic was assigned or abstained;
- `genre_function_note`: why the genre/function was assigned or abstained;
- `review_note`: compact combined rationale, including old-label split where useful.

Notes should be brief but interpretable enough for later audit.
