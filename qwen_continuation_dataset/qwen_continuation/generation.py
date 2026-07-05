from __future__ import annotations

import math
from typing import Any

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from qwen_continuation.model import TeacherBundle


def _entropy_from_logits(logits: torch.Tensor) -> float:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return float(entropy.item())


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
    window_chars = int(cfg.get("window_chars", 100))
    ngram_chars = int(cfg.get("ngram_chars", 20))
    min_chars = int(cfg.get("min_chars", 50))
    if ngram_chars <= 0:
        raise ValueError("cycle_detection.ngram_chars must be positive")
    if window_chars <= ngram_chars:
        raise ValueError(
            "cycle_detection.window_chars must be greater than ngram_chars"
        )
    return {
        "window_chars": window_chars,
        "ngram_chars": ngram_chars,
        "min_chars": min_chars,
    }


def generate_fixed(
    teacher: TeacherBundle,
    prefix_ids: list[int],
    config: dict[str, Any],
) -> tuple[list[int], list[float]]:
    generation = config["generation"]
    tokenizer = teacher.tokenizer
    model = teacher.model

    input_ids = torch.tensor(
        [prefix_ids], dtype=torch.long, device=teacher.input_device
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
                    prefix_len=len(prefix_ids),
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

    sequence = output.sequences[0]
    generated = sequence[len(prefix_ids):].tolist()
    entropies = [
        _entropy_from_logits(step_logits[0])
        for step_logits in output.scores
    ]
    return generated, entropies


def generate_until_entropy(
    teacher: TeacherBundle,
    prefix_ids: list[int],
    config: dict[str, Any],
) -> tuple[list[int], list[float]]:
    generation = config["generation"]
    model = teacher.model
    tokenizer = teacher.tokenizer

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
        [prefix_ids], dtype=torch.long, device=teacher.input_device
    )
    attention_mask = torch.ones_like(input_ids)

    generated: list[int] = []
    entropies: list[float] = []

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        logits = output.logits[:, -1, :]
        entropy = _entropy_from_logits(logits[0])
        entropies.append(entropy)

        if len(generated) >= min_before_stop and entropy > threshold:
            break

        next_token = _sample_next_token(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        token_id = int(next_token.item())
        generated.append(token_id)

        if cycle_cfg:
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            if (
                len(decoded) >= cycle_cfg["min_chars"]
                and _detect_ngram_cycle(
                    decoded,
                    cycle_cfg["window_chars"],
                    cycle_cfg["ngram_chars"],
                )
            ):
                break

        input_ids = torch.cat((input_ids, next_token), dim=-1)
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ),
            dim=-1,
        )

        if (
            tokenizer.eos_token_id is not None
            and token_id == tokenizer.eos_token_id
        ):
            break

    return generated, entropies


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
