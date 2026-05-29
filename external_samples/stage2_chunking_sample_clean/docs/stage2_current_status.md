# Stage2 current status

Stage2 is an isolated corpus preparation and annotation subproject under
`external_samples/stage2_chunking_sample_clean/`.

Current direction:

- prepare controlled corpus metadata for a from-scratch 0.8B Qwen-like model;
- use `Qwen/Qwen3.5-0.8B-Base` tokenizer as frozen infrastructure;
- support future NLL/logprob/perplexity profiling;
- use annotation schema v2 instead of the old single-label `source_type` framing.

Implemented layers:

- real HF dev/test samples for FineWeb, FineWeb-Edu, and FineMath;
- deterministic annotation_v2 features;
- refined surface and quality/noise features;
- Qwen3.5 tokenization-aware stats;
- annotation_v2 pseudo-gold for v1-dev and v2-test;
- `weak_topic_domain_v2.1` baseline with confidence and abstention.

Current decision:

- v1 is dev and should not be used for independent quality claims;
- v2-test is held-out and should not be used for tuning;
- `weak_topic_domain_v2.1` is an MVP baseline, not a final topic labeler;
- deterministic features and token stats are ready for pilot NLL grouping;
- topic labels must be used with confidence/coverage/abstention metadata.

Recommended next step:

- embedding-model bake-off for coarse `topic.domain`, or a small NLL grouping interface if the team wants to start profiling plumbing first.
