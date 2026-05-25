# Текущий статус stage2

Дата: 2026-05-26

Stage2 - это изолированный подпроект для подготовки корпуса и разметки chunks по доменам. Он живет в:

```text
external_samples/stage2_chunking_sample_clean/
```

Это сознательная safety boundary. Сейчас не переносим stage2 наверх и не меняем root repo без отдельного approval.

## Что входит в текущую задачу

- нарезка документов на логические chunks;
- сохранение `dataset` и `source_type`;
- заполнение `domain`, `field`, `subfield`;
- rule-based baseline;
- lexical nearest-label baseline;
- synthetic benchmark;
- tiny real samples FineWeb-Edu и FineMath;
- сравнение rule-based vs lexical;
- подготовка к MiniLM/Sentence-Transformers embedding classifier.

## Что не входит

- обучение Qwen-like модели;
- NLL/logprob scoring;
- effective context window;
- OpenWebMath как current MVP source;
- перенос stage2 в root repo;
- large-scale dataset processing.

## Что уже готово

| Area | Status |
| --- | --- |
| Local chunking | Работает на local JSONL fixtures |
| Metadata | `dataset` и `source_type` сохраняются |
| Rule-based classifier | Работает как прозрачный baseline |
| Lexical classifier | Работает как cheap nearest-label baseline |
| Synthetic benchmark | Есть expected labels, можно считать accuracy |
| FineWeb-Edu tiny sample | Completed, review mode |
| FineMath tiny sample | Completed, current MVP math source |
| Disagreement review | Есть первые review artifacts |
| MiniLM | Next milestone, не запускался в текущем cleanup |

## Где что лежит

| Path | Meaning |
| --- | --- |
| `scripts/` | chunking, validators, classifiers, planners, comparison tools |
| `examples/` | маленькие synthetic/local входные fixtures |
| `data_samples/` | маленькие outputs и tracked tiny samples |
| `data_samples/real_samples/` | FineWeb-Edu, FineMath, local real-like outputs |
| `taxonomy/simple_domain_labels.json` | текущая taxonomy для lexical/embedding labels |
| `config/dataset_sources.json` | registry/planning metadata for sources |
| `docs/` | stage2 docs and reports |
| `docs/repo_audit/` | audit reports and cleanup log |

## Текущие source decisions

- FineWeb-Edu: tiny sample completed.
- FineMath: tiny sample completed; единственный current MVP math source.
- OpenWebMath: optional_later only.
- FineWeb general: not current.
- Cosmopedia/SmolLM: optional/later.
- NLL/effective context: downstream.

## Safe local commands

Run from `external_samples/stage2_chunking_sample_clean/`:

```bash
python scripts\validate_chunks.py --input data_samples\edge_case_chunks_sample.jsonl
python scripts\validate_chunks.py --input data_samples\edge_case_chunks_labeled.jsonl --require-labels
python scripts\validate_chunks.py --input data_samples\classifier_benchmark_labeled.jsonl --require-labels
python scripts\smoke_test_rule_based_classifier.py
python scripts\run_local_benchmark_pipeline.py
python scripts\inspect_dataset_sources.py --registry config\dataset_sources.json
```

Real tiny samples are not gold-labeled benchmark data. Validate them without `--require-labels`.

## Next milestone

MiniLM/Sentence-Transformers embedding nearest-label classifier:

- same chunks;
- same taxonomy;
- comparable output schema;
- no implicit model download;
- compare rule-based vs lexical vs embedding;
- use disagreement as review signal on real samples.
