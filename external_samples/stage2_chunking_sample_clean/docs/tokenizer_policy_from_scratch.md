# Tokenizer policy for from-scratch Qwen-like MVP

## Project assumption

This project trains model weights from scratch for a small Qwen-like / Qwen-architecture model.

The tokenizer is a separate design choice. Using a Qwen-family tokenizer does **not** mean using Qwen pretrained weights, continued pretraining from Qwen weights, or fine-tuning an existing Qwen checkpoint.

For the current MVP, tokenizer training is out of scope unless explicitly revisited. Reusing a stable Qwen-family tokenizer is the practical path because it avoids turning tokenizer design, multilingual coverage, special token policy, and tokenizer evaluation into a separate large project.

## Inspection method

Checked with:

- `AutoTokenizer.from_pretrained(...)`
- no `AutoModel`
- no model weight loading
- tokenizer/cache metadata only

The default Hugging Face cache under `C:\Users\pervo\.cache\huggingface` raised a permissions error, so the inspection used a local stage2 cache:

- `.hf_tokenizer_cache/`

This cache is an inspection artifact and should not be committed.

## Candidate tokenizer metadata

| Candidate | Repo available | Revision inspected | Tokenizer class | Vocab size | `len(tokenizer)` | EOS | PAD | Chat template | `model_max_length` |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: |
| `Qwen/Qwen3.5-0.8B` | yes | `2fc06364715b967f1860aea9cf38778875588b17` | `Qwen2Tokenizer` | 248044 | 248077 | `<\|im_end\|>` / `248046` | `<\|endoftext\|>` / `248044` | yes | 262144 |
| `Qwen/Qwen3.5-0.8B-Base` | yes | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | `Qwen2Tokenizer` | 248044 | 248077 | `<\|endoftext\|>` / `248044` | `<\|endoftext\|>` / `248044` | yes | 262144 |
| `Qwen/Qwen3-0.6B` | yes | `c1899de289a04d12100db370d81485cdf75e47ca` | `Qwen2Tokenizer` | 151643 | 151669 | `<\|im_end\|>` / `151645` | `<\|endoftext\|>` / `151643` | yes | 131072 |
| `Qwen/Qwen3-0.6B-Base` | yes | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | `Qwen2Tokenizer` | 151643 | 151669 | `<\|endoftext\|>` / `151643` | `<\|endoftext\|>` / `151643` | yes | 131072 |

Tokenizer-related files observed:

- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `config.json`

The Qwen3.5 0.8B pair produced identical token counts and token ids on the representative strings, but special token policy differs between Base and non-Base. The Qwen3 0.6B pair behaved similarly within its pair, with a smaller vocabulary and different token ids.

## Representative string comparison

Token counts for the inspected examples:

| Example | Bytes | Qwen3.5 0.8B | Qwen3.5 0.8B Base | Qwen3 0.6B | Qwen3 0.6B Base |
| --- | ---: | ---: | ---: | ---: | ---: |
| English prose | 90 | 17 | 17 | 17 | 17 |
| Russian prose | 79 | 20 | 20 | 20 | 20 |
| Math formula | 36 | 20 | 20 | 20 | 20 |
| Chemical formula | 28 | 23 | 23 | 23 | 23 |
| Code | 23 | 8 | 8 | 8 | 8 |
| URL/boilerplate | 95 | 19 | 19 | 19 | 19 |
| Patent-like sentence | 98 | 20 | 20 | 20 | 20 |

Example tokenization behavior was broadly similar across candidates for this small probe. The main practical differences are vocabulary size, special token policy, maximum length metadata, and compatibility target.

Notable observations:

- English and patent-like prose tokenize compactly.
- Russian token count is reasonable in this small probe, but decoded token strings in console output are not reliable because byte-level tokens render poorly in PowerShell; ids/counts are still usable.
- Math and chemistry are token-heavy, especially formulas with many single-character symbols and digits.
- Code is compact for simple Python-like syntax.
- URL/boilerplate splits into expected URL components.

## Recommendation

Use this tokenizer as the canonical MVP tokenizer:

```text
Qwen/Qwen3.5-0.8B-Base
revision: dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
tokenizer_class: Qwen2Tokenizer
vocab_size: 248044
len_tokenizer: 248077
eos_token_id: 248044
pad_token_id: 248044
```

Reasoning:

- It is a Qwen-family tokenizer aligned with the intended Qwen-like architecture direction.
- The `Base` variant is cleaner for from-scratch base LM training than the non-Base/chat-oriented special-token policy.
- It is available through Transformers and has explicit tokenizer files.
- It handles English, Russian, math, chemistry, code, URLs, and patent-like text reasonably in the small probe.
- The exact revision can be pinned for reproducibility.

Important caveat:

- The 248k vocabulary is large. Before final training config is frozen, the model budget should explicitly account for token embedding and output head parameters. If the 0.8B parameter target becomes too tight, the fallback tokenizer should be `Qwen/Qwen3-0.6B-Base` with revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` and `len(tokenizer)=151669`.

Do not automatically switch to a newer Qwen tokenizer merely because it exists. Tokenizer changes alter token counts, sequence lengths, embedding sizes, and reproducibility.

## Integration plan for annotation_v2

Do not modify the deterministic feature extractor yet. The next tokenizer-aware implementation should add a separate token stats layer or extend `annotation_v2.text_stats` only after the tokenizer policy is accepted.

Future fields:

- `token_count`
- `token_per_byte`
- `tokens_per_char`
- `special_token_count` if special tokens are included in a later mode

Recommended output metadata:

```json
{
  "tokenizer": {
    "tokenizer_name": "Qwen/Qwen3.5-0.8B-Base",
    "revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    "tokenizer_class": "Qwen2Tokenizer",
    "vocab_size": 248044,
    "len_tokenizer": 248077,
    "bos_token_id": null,
    "eos_token_id": 248044,
    "pad_token_id": 248044,
    "unk_token_id": null,
    "add_special_tokens": false
  }
}
```

For corpus profiling and NLL/probability work, `add_special_tokens=false` should be the default for raw chunk token statistics. Any training-sample packing or chat/template formatting should be a separate downstream transformation with its own metadata.

## Safety policy

- Pin tokenizer repo and revision.
- Never infer tokenizer choice from "latest" model naming.
- Do not mix tokenizer stats from different tokenizer revisions in the same benchmark without explicit metadata.
- Do not commit tokenizer cache directories.
- Do not load model weights for tokenizer policy checks.
- Keep tokenizer training out of MVP unless the team explicitly reopens that scope.

## Next implementation step

After this policy is accepted, implement a small tokenizer-stats script that:

1. loads the pinned tokenizer;
2. reads existing `*_features_v2_refined.jsonl` or raw chunk files;
3. writes token stats and tokenizer metadata without changing old classifier outputs;
4. validates that token stats are reproducible from the pinned tokenizer.
