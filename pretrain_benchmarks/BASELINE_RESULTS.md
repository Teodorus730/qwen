# Versioned full-baseline results

## Qwen3.5-0.8B-Base — 20260807T061546Z

- Model: `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- Hardware/backend: Intel Arc A770 16 GB, XPU (`xpu:0`), BF16
- Protocol: zero-shot, seed 42, batch size `auto` (effective 32), `max_length=2048`
- Environment: PyTorch `2.13.0+xpu`, Transformers `5.10.2`, lm-eval `0.4.12`

| Task | Metrics |
|---|---|
| WikiText | word PPL 19.2196; byte PPL 1.7381; bits/byte 0.7975 |
| LAMBADA OpenAI | acc 0.5077; perplexity 11.6661 |
| HellaSwag | acc 0.4201; acc_norm 0.5484 |
| ARC-Easy | acc 0.7050; acc_norm 0.6747 |
| PIQA | acc 0.7018; acc_norm 0.7155 |

The compact aggregate/config and environment snapshot are
`baseline_results/Qwen__Qwen3.5-0.8B-Base/20260807T061546Z.json` and its
adjacent `.environment.txt`. lm-eval reported task versions for all tasks;
its `task_hashes` field was empty for this run. Raw harness JSON and sample
logs remain in ignored `results/`.
