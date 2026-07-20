from __future__ import annotations

import math
from typing import Any

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from qwen_continuation.model import TeacherBundle


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, k=k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, -math.inf)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -math.inf)
        logits = torch.full_like(logits, -math.inf).scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_logits,
        )

    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def _detect_ngram_cycle(text: str, window_chars: int, ngram_chars: int) -> bool:
    if ngram_chars <= 0:
        return False
    if len(text) < ngram_chars * 2:
        return False
    window = text[-window_chars:]
    if len(window) <= ngram_chars:
        return False
    tail = window[-ngram_chars:]
    return tail in window[:-ngram_chars]


class _NgramCycleStopping(StoppingCriteria):
    def __init__(
        self,
        tokenizer: Any,
        prefix_len: int,
        window_chars: int,
        ngram_chars: int,
        min_chars: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.window_chars = window_chars
        self.ngram_chars = ngram_chars
        self.min_chars = min_chars

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        is_done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        for row in range(batch_size):
            generated_ids = input_ids[row, self.prefix_len:].tolist()
            if not generated_ids:
                continue
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if len(text) < self.min_chars:
                continue
            if _detect_ngram_cycle(text, self.window_chars, self.ngram_chars):
                is_done[row] = True
        return is_done


def _get_cycle_cfg(config: dict[str, Any]) -> dict[str, Any] | None:
    cfg = config["generation"].get("cycle_detection", {})
    if not cfg.get("enabled", False):
        return None
    return {
        "window_chars": int(cfg.get("window_chars", 100)),
        "ngram_chars": int(cfg.get("ngram_chars", 20)),
        "min_chars": int(cfg.get("min_chars", 50)),
    }


def generate_fixed(
    teacher: TeacherBundle,
    prefix_ids: list[int],
    config: dict[str, Any],
) -> tuple[list[int], list[float]]:
    return generate_fixed_batch(teacher, [prefix_ids], config)[0]


def generate_fixed_batch(
    teacher: TeacherBundle,
    prefix_ids_batch: list[list[int]],
    config: dict[str, Any],
) -> list[tuple[list[int], list[float]]]:
    generation = config["generation"]
    tokenizer = teacher.tokenizer
    model = teacher.model
    prefix_len = len(prefix_ids_batch[0])

    input_ids = torch.tensor(
        prefix_ids_batch, dtype=torch.long, device=teacher.input_device
    )
    attention_mask = torch.ones_like(input_ids)

    temperature = float(generation.get("temperature", 0.0))
    do_sample = temperature > 0

    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation["max_new_tokens"]),
        "do_sample": do_sample,
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        kwargs.update(
            {
                "temperature": temperature,
                "top_p": float(generation.get("top_p", 1.0)),
            }
        )
        top_k = int(generation.get("top_k", 0))
        if top_k > 0:
            kwargs["top_k"] = top_k

    cycle_cfg = _get_cycle_cfg(config)
    if cycle_cfg:
        kwargs["stopping_criteria"] = StoppingCriteriaList(
            [
                _NgramCycleStopping(
                    tokenizer=tokenizer,
                    prefix_len=prefix_len,
                    window_chars=cycle_cfg["window_chars"],
                    ngram_chars=cycle_cfg["ngram_chars"],
                    min_chars=cycle_cfg["min_chars"],
                )
            ]
        )

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    step_entropies = _entropy_from_logits(torch.stack(output.scores)).tolist()
    results = []
    for row in range(len(prefix_ids_batch)):
        generated = output.sequences[row, prefix_len:].tolist()
        if eos_id is not None and eos_id in generated:
            generated = generated[: generated.index(eos_id) + 1]
        elif pad_id is not None:
            while generated and generated[-1] == pad_id:
                generated.pop()
        entropies = [step[row] for step in step_entropies][: len(generated)]
        results.append((generated, entropies))
    return results


def generate_until_entropy(
    teacher: TeacherBundle,
    prefix_ids: list[int],
    config: dict[str, Any],
) -> tuple[list[int], list[float]]:
    return generate_until_entropy_batch(teacher, [prefix_ids], config)[0]


def generate_until_entropy_batch(
    teacher: TeacherBundle,
    prefix_ids_batch: list[list[int]],
    config: dict[str, Any],
) -> list[tuple[list[int], list[float]]]:
    generation = config["generation"]
    model = teacher.model
    tokenizer = teacher.tokenizer

    batch_size = len(prefix_ids_batch)
    max_new_tokens = int(generation["max_new_tokens"])
    threshold = float(generation["entropy_threshold"])
    min_before_stop = int(
        generation.get("min_generated_tokens_before_entropy_stop", 1)
    )
    temperature = float(generation.get("temperature", 0.0))
    top_p = float(generation.get("top_p", 1.0))
    top_k = int(generation.get("top_k", 0))
    cycle_cfg = _get_cycle_cfg(config)

    input_ids = torch.tensor(
        prefix_ids_batch, dtype=torch.long, device=teacher.input_device
    )
    attention_mask = torch.ones_like(input_ids)

    generated: list[list[int]] = [[] for _ in range(batch_size)]
    entropies: list[list[float]] = [[] for _ in range(batch_size)]
    finished = [False] * batch_size

    past_key_values = None
    current_input = input_ids

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            output = model(
                input_ids=current_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]

        next_tokens = _sample_next_token(
            logits, temperature=temperature, top_p=top_p, top_k=top_k
        )
        step_entropies = _entropy_from_logits(logits).tolist()
        step_tokens = next_tokens.squeeze(-1).tolist()

        for row in range(batch_size):
            if finished[row]:
                continue

            entropy = step_entropies[row]
            entropies[row].append(entropy)

            if len(generated[row]) >= min_before_stop and entropy > threshold:
                finished[row] = True
                continue

            token_id = step_tokens[row]
            generated[row].append(token_id)

            if not cycle_cfg and (
                tokenizer.eos_token_id is not None
                and token_id == tokenizer.eos_token_id
            ):
                finished[row] = True

        if cycle_cfg:
            active_rows = [
                row for row in range(batch_size) if not finished[row] and generated[row]
            ]
            if active_rows:
                texts = tokenizer.batch_decode(
                    [generated[row] for row in active_rows],
                    skip_special_tokens=True,
                )
                for row, text in zip(active_rows, texts):
                    if len(text) >= cycle_cfg["min_chars"] and _detect_ngram_cycle(
                        text, cycle_cfg["window_chars"], cycle_cfg["ngram_chars"]
                    ):
                        finished[row] = True

        if all(finished):
            break

        current_input = next_tokens
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ),
            dim=-1,
        )

    return list(zip(generated, entropies))


def generate_continuation(
    teacher: TeacherBundle,
    prefix_ids: list[int],
    config: dict[str, Any],
) -> tuple[list[int], list[float]]:
    mode = config["generation"]["mode"]
    if mode == "fixed":
        return generate_fixed(teacher, prefix_ids, config)
    if mode == "entropy":
        return generate_until_entropy(teacher, prefix_ids, config)
    raise ValueError(f"Unsupported generation mode: {mode}")


def generate_continuation_batch(
    teacher: TeacherBundle,
    prefix_ids_batch: list[list[int]],
    config: dict[str, Any],
) -> list[tuple[list[int], list[float]]]:
    mode = config["generation"]["mode"]
    if mode == "fixed":
        return generate_fixed_batch(teacher, prefix_ids_batch, config)
    if mode == "entropy":
        return generate_until_entropy_batch(teacher, prefix_ids_batch, config)
    raise ValueError(f"Unsupported generation mode: {mode}")
