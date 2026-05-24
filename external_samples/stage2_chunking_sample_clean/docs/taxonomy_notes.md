# Taxonomy notes

`taxonomy/simple_domain_labels.json` is a small local MVP taxonomy for stage2 experiments. It is not a scientific taxonomy and should not be treated as final.

The file exists so later experiments can test an embedding/nearest-label baseline without inventing labels inside the classifier code. An OpenAlex-like taxonomy can replace or extend it later when the project moves beyond local smoke tests.

Important distinctions:

- `source_type` remains separate from the domain taxonomy. It describes the format or source type of the text, such as `forum_qa`, `code`, or `boilerplate_or_noise`.
- `domain`, `field`, and `subfield` describe the topic area.
- `subfield` may be null for broad or unknown labels.
- The current labels are deliberately small and readable for local regression tests.
