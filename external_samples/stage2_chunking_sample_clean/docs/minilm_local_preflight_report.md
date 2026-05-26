# MiniLM local preflight report

Date: 2026-05-26

Scope: local-only readiness check for the MiniLM/Sentence-Transformers embedding nearest-label classifier.

No HF streaming, dataset download, model download, dependency installation, NLL/logprob scoring, commit, or push was performed.

## Verdict

MiniLM is not ready for a real local run in the current environment.

The stage2 code and comparison pipeline are ready enough for a future run, but the runtime environment is missing the required Python packages and the local model cache.

## Environment checked

Python executable:

```text
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe
```

Python version:

```text
Python 3.14.1
```

Package availability in this environment:

| Package | Status |
| --- | --- |
| `sentence_transformers` | missing |
| `torch` | missing |
| `transformers` | missing |
| `huggingface_hub` | missing |
| `numpy` | missing |

Checked local cache paths:

| Path | Status |
| --- | --- |
| `C:\Users\pervo\.cache\huggingface\hub` | exists |
| `C:\Users\pervo\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2` | missing |
| `C:\Users\pervo\.cache\torch\sentence_transformers` | missing |
| `C:\Users\pervo\.cache\sentence_transformers` | missing |
| `data_samples\hf_cache_test` | exists, but this is not a MiniLM model cache |

## Safety assessment

Current `scripts/classify_chunks_embedding_baseline.py` is safe for dry-run:

- `--dry-run` does not import or load `sentence-transformers`;
- `--dry-run` does not write output;
- real runs request `local_files_only=True`;
- if `sentence-transformers` is missing, the script exits with a clear blocker;
- if the local model is missing, the script should fail closed instead of downloading.

Do not run real MiniLM inference until dependencies and a local model path/cache are approved and verified.

## Blockers

1. `sentence-transformers` is not installed in `.venv`.
2. `torch` is not installed in `.venv`.
3. `numpy` is not installed in `.venv`.
4. Local cache for `sentence-transformers/all-MiniLM-L6-v2` was not found.
5. No approved local model path is recorded yet.

## Commands that are safe now

Dry-run:

```bash
python scripts\classify_chunks_embedding_baseline.py --input data_samples\classifier_benchmark_chunks.jsonl --output data_samples\classifier_benchmark_labeled_embedding_dryrun.jsonl --labels taxonomy\simple_domain_labels.json --model sentence-transformers/all-MiniLM-L6-v2 --dry-run
```

Smoke test without a real model:

```bash
python scripts\smoke_test_embedding_baseline.py
```

Existing comparison readiness checks:

```bash
python scripts\validate_chunks.py --input data_samples\classifier_benchmark_labeled.jsonl --require-labels
python scripts\compare_label_runs.py --left data_samples\classifier_benchmark_labeled.jsonl --right data_samples\classifier_benchmark_lexical_labeled.jsonl --left-name rule_based --right-name lexical --limit 5
```

## Future commands after explicit approval

Install dependencies only after explicit approval. Exact install command should be chosen by the team/environment owner. Do not run it as part of routine preflight.

After dependencies and a local model are available, first verify imports:

```bash
python -c "import sentence_transformers, torch, numpy; print(sentence_transformers.__version__); print(torch.__version__)"
```

Then verify the model is local by running the real classifier on the synthetic benchmark only:

```bash
python scripts\classify_chunks_embedding_baseline.py --input data_samples\classifier_benchmark_chunks.jsonl --labels taxonomy\simple_domain_labels.json --output data_samples\classifier_benchmark_minilm_labeled.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --batch-size 32 --top-k 3
```

Then validate and compare:

```bash
python scripts\validate_chunks.py --input data_samples\classifier_benchmark_minilm_labeled.jsonl --require-labels
python scripts\evaluate_chunk_labels.py --input data_samples\classifier_benchmark_minilm_labeled.jsonl
python scripts\compare_label_runs.py --left data_samples\classifier_benchmark_labeled.jsonl --right data_samples\classifier_benchmark_minilm_labeled.jsonl --left-name rule_based --right-name embedding --limit 10
python scripts\compare_label_runs.py --left data_samples\classifier_benchmark_lexical_labeled.jsonl --right data_samples\classifier_benchmark_minilm_labeled.jsonl --left-name lexical --right-name embedding --limit 10
```

Only after the benchmark output is reviewed should FineWeb-Edu and FineMath tiny samples be labeled with MiniLM.

## Readiness for comparison

Available now:

- rule-based benchmark output;
- lexical benchmark output;
- compare script;
- validators;
- disagreement review pattern;
- dry-run embedding command;
- smoke test for embedding contract behavior.

Missing:

- real MiniLM output;
- local model/dependency availability;
- benchmark comparison involving embedding output.
