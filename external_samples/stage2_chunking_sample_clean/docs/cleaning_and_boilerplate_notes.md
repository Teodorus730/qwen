# Cleaning and boilerplate notes

The current classifier can label a whole chunk as `boilerplate_or_noise`, but that is not the same thing as cleaning the text.

Many real web chunks will contain both useful content and boilerplate: an article paragraph plus cookie notice, a forum answer plus footer links, or a product explanation plus subscription blocks. Treating the whole chunk as noise loses useful text, while ignoring the boilerplate can distort later probability profiling.

Cleaning should be a separate stage before classification or a neighboring stage near chunking. The raw text should stay available, and any cleaned text should be stored separately.

## Future schema fields

- `contains_boilerplate`: bool
- `boilerplate_score`: float
- `quality_flags`: list[str]
- `cleaning_method`: string
- `text_cleaned`: optional string

## Possible future heuristics

- repeated navigation phrases;
- cookie/privacy/footer density;
- link/menu separator density;
- short repeated fragments;
- text-to-boilerplate ratio;
- preserve raw text and optional cleaned text separately.

For MVP probability profiling, very low NLL/perplexity may indicate repeated boilerplate rather than high-quality data, so preserving these flags will matter.
