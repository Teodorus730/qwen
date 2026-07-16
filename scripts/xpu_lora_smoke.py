"""Small Intel XPU/Qwen/LoRA smoke test with one optimizer step."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Use the existing local model cache only.")
    args = parser.parse_args()

    results: dict[str, object] = {
        "torch": torch.__version__,
        "xpu_available": torch.xpu.is_available(),
        "xpu_count": torch.xpu.device_count(),
    }
    if not torch.xpu.is_available():
        raise RuntimeError("torch.xpu is not available")

    results["device"] = torch.xpu.get_device_name(0)

    left = torch.randn(256, 256, device="xpu", dtype=torch.float16, requires_grad=True)
    right = torch.randn(256, 256, device="xpu", dtype=torch.float16)
    fp16_loss = (left @ right).float().square().mean()
    fp16_loss.backward()
    torch.xpu.synchronize()
    results["fp16"] = {
        "loss": fp16_loss.detach().item(),
        "finite_gradient": torch.isfinite(left.grad).all().item(),
    }
    del left, right, fp16_loss

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=args.offline)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": "xpu"},
        low_cpu_mem_usage=True,
        local_files_only=args.offline,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
        weight_decay=0.0,
    )

    input_ids = tokenizer("Intel Arc LoRA backward smoke test.", return_tensors="pt")["input_ids"].to("xpu")
    trainable_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer.zero_grad(set_to_none=True)
    started = perf_counter()
    output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    forward_seconds = perf_counter() - started
    started = perf_counter()
    output.loss.backward()
    torch.xpu.synchronize()
    backward_seconds = perf_counter() - started

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    optimizer.step()
    torch.xpu.synchronize()
    changed = {
        name: (parameter.detach().cpu() != trainable_before[name]).any().item()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise RuntimeError("LoRA gradients are missing or non-finite")
    if not any(changed.values()):
        raise RuntimeError("The optimizer step did not change any LoRA weights")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    results["qwen_lora"] = {
        "model": MODEL_ID,
        "dtype": str(next(model.parameters()).dtype),
        "loss": output.loss.detach().item(),
        "forward_seconds": round(forward_seconds, 3),
        "backward_seconds": round(backward_seconds, 3),
        "trainable_parameters": trainable,
        "gradient_tensors": len(gradients),
        "finite_gradients": bool(gradients) and all(torch.isfinite(gradient).all().item() for gradient in gradients),
        "optimizer_step": True,
        "changed_parameter_tensors": sum(changed.values()),
        "any_trainable_tensor_changed": any(changed.values()),
        "all_trainable_tensors_changed": all(changed.values()),
        "peak_xpu_memory_mb": round(torch.xpu.max_memory_allocated() / 2**20, 1),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
