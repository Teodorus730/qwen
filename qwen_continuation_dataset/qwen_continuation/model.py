from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer, set_seed


@dataclass
class TeacherBundle:
    model: Any
    tokenizer: Any
    input_device: torch.device
    dtype_name: str


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def choose_dtype(device: torch.device | None = None) -> torch.dtype:
    device = device or choose_device()
    if device.type == "cpu":
        return torch.float32
    if device.type == "xpu" or torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _load_model_class():
    errors: list[str] = []

    try:
        from transformers import AutoModelForMultimodalLM
        return AutoModelForMultimodalLM
    except ImportError as error:
        errors.append(f"AutoModelForMultimodalLM: {error}")

    try:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM
    except ImportError as error:
        errors.append(f"AutoModelForCausalLM: {error}")

    raise RuntimeError(
        "Could not import a supported Transformers auto model class. "
        "Install dependencies from requirements.txt.\n" + "\n".join(errors)
    )


def load_teacher(config: dict[str, Any]) -> TeacherBundle:
    model_config = config["model"]
    generation_config = config["generation"]
    model_id = model_config["id"]
    trust_remote_code = bool(model_config.get("trust_remote_code", True))

    set_seed(int(generation_config.get("seed", 42)))

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_class = _load_model_class()
    device = choose_device()
    dtype = choose_dtype(device)

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "dtype": dtype,
    }

    if device.type != "cpu":
        device_map = model_config.get("device_map", "auto")
        load_kwargs["device_map"] = (
            {"": "xpu"} if device.type == "xpu" and device_map == "auto"
            else device_map
        )

    try:
        model = model_class.from_pretrained(model_id, **load_kwargs)
    except TypeError:
        # Compatibility fallback for releases that still use torch_dtype.
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = model_class.from_pretrained(model_id, **load_kwargs)

    if device.type == "cpu":
        model = model.to("cpu")

    model.eval()
    input_device = next(model.parameters()).device

    return TeacherBundle(
        model=model,
        tokenizer=tokenizer,
        input_device=input_device,
        dtype_name=str(dtype).replace("torch.", ""),
    )
