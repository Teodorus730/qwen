# FineMath tiny sample plan

## Purpose

Second real HF tiny sample and first math-domain sample for the MVP.

## Source

- dataset: `HuggingFaceTB/finemath`
- config: `finemath-4plus`
- split: `train`
- text_field: `text`
- source_type: `math`
- max_docs: 20

## Why FineMath

Current MVP uses FineMath as the only active math source. OpenWebMath is optional later only, as a backup or comparison source after the FineMath path is inspected.

## Planned command

Not run yet:

```bash
python scripts\sample_fineweb_chunks.py --use-hf-streaming --dataset HuggingFaceTB/finemath --config finemath-4plus --split train --dataset-label FineMath --source-type math --text-field text --max-docs 20 --out data_samples\real_samples\finemath_chunks.jsonl --stats-out data_samples\real_samples\finemath_run_stats.json
```

## Planned local processing

- validate chunks
- inspect chunks
- run rule-based labels
- run lexical labels
- validate labeled outputs without `--require-labels`
- compare rule-based vs lexical labels
- write a short sample report

## Risks

- LaTeX and formulas may interact poorly with fallback token counting.
- Math text may split oddly if formulas or derivations are long.
- Current taxonomy is small and may not represent all math topics cleanly.
- Lexical baseline may overweight direct formula or keyword overlap.
- Rule-based math labels are a transparent baseline, not final quality.
- NLL/logprob scoring is still not part of this step.
