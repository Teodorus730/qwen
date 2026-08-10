"""Non-canonical MMMLU logits/memory probe for real Stage-2 samples."""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path("pretrain_benchmarks/.hf_cache").resolve()))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import lm_eval
from lm_eval import utils
from lm_eval.tasks._yaml_loader import load_yaml
from prepare_mmmlu_progressive import DATASET, LOCALES, MODEL, MODEL_REVISION, REVISION, normalized_subject


def memory() -> dict[str, int | None]:
    xpu = torch.xpu
    free, total = xpu.mem_get_info()
    return {"free": free, "total": total, "allocated": xpu.memory_allocated(), "reserved": xpu.memory_reserved(),
            "peak_allocated": xpu.max_memory_allocated(), "peak_reserved": xpu.max_memory_reserved()}


def reset_memory() -> None:
    gc.collect()
    torch.xpu.empty_cache()
    torch.xpu.reset_peak_memory_stats()


def score(model, ids: torch.Tensor, contlen: int, optimized: bool) -> tuple[list[float], list[int], dict]:
    reset_memory()
    before = memory()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=torch.bfloat16):
        logits = model(ids, logits_to_keep=contlen if optimized else 0).logits
        # Inputs are deliberately unpadded/equal-length in each forward, so the
        # returned tail is exactly the continuation scoring window.
        window = logits[:, -contlen:, :]
        targets = ids[:, -contlen:]
        logp = F.log_softmax(window, dim=-1, dtype=torch.float32)
        values = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum(-1)
        preds = window[:, -1, :].argmax(-1)
    elapsed = time.perf_counter() - started
    after = memory()
    del logits, window, targets, logp
    gc.collect(); torch.xpu.empty_cache()
    return values.cpu().tolist(), preds.cpu().tolist(), {"before": before, "after": after, "after_empty_cache": memory(), "seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = {(x["locale"], x["subject"]): set(x["selected_source_indices"]) for x in manifest["strata"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    task_root = Path(inspect.getfile(lm_eval)).parent / "tasks" / "openai-mmmlu" / "default"
    records, continuation_lengths = [], []
    for locale in LOCALES:
        ds = load_dataset(DATASET, locale.upper(), split="test", revision=REVISION)
        by = defaultdict(list)
        for source_index, row in enumerate(ds): by[normalized_subject(row["Subject"])].append((source_index, dict(row)))
        for subject, rows in by.items():
            cfg = load_yaml(task_root / f"mmmlu_{locale}_{subject}.yaml", resolve_func=False, recursive=True)
            choices = cfg["doc_to_choice"]
            delimiter = cfg.get("target_delimiter", " ")
            for choice in choices:
                continuation_lengths.append(len(tokenizer.encode(delimiter + choice, add_special_tokens=False)))
            for source_index, row in rows:
                if source_index in selected[(locale, subject)]: continue
                prompt = cfg["description"] + utils.apply_template(cfg["doc_to_text"], row)
                records.append((len(tokenizer.encode(prompt)), locale, subject, source_index, prompt, row["Answer"], choices, delimiter))
    records.sort(reverse=True, key=lambda x: x[0])
    # Every >2048 case plus top-10; this is the correctness set.  Stress uses
    # the top four in equal-length forwards to avoid padding ambiguity.
    cases = records[:10] + [x for x in records[10:] if x[0] > 2048]
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, torch_dtype=torch.bfloat16).to("xpu").eval()
    idle = memory()
    comparisons, stress = [], []
    for length, locale, subject, source, prompt, answer, choices, delimiter in cases:
        request_ids = [tokenizer.encode(prompt + delimiter + c, return_tensors="pt").to("xpu") for c in choices]
        contlen = len(tokenizer.encode(delimiter + choices[0], add_special_tokens=False))
        # Each choice is scored alone: no padding and the reference/optimized
        # windows are identical by construction.
        ref_scores=[]; opt_scores=[]
        for ids in request_ids:
            ref, _, telemetry_ref = score(model, ids, contlen, False)
            opt, _, telemetry_opt = score(model, ids, contlen, True)
            ref_scores.append(ref[0]); opt_scores.append(opt[0])
        ref_pred=choices[max(range(4), key=lambda i: ref_scores[i])]; opt_pred=choices[max(range(4), key=lambda i: opt_scores[i])]
        comparisons.append({"length": length, "locale": locale, "subject": subject, "source_index": source,
                            "answer": answer, "reference_prediction": ref_pred, "optimized_prediction": opt_pred,
                            "reference_acc": int(ref_pred == answer), "optimized_acc": int(opt_pred == answer),
                            "max_abs_loglikelihood_error": max(abs(a-b) for a,b in zip(ref_scores,opt_scores)),
                            "reference_memory": telemetry_ref, "optimized_memory": telemetry_opt})
    # Equal-length real long requests: batch stress proves the actual output-logit allocation case.
    for batch in (1, 2, 4):
        chosen = records[:batch]
        ids = [tokenizer.encode(x[4] + x[7] + x[6][0], return_tensors="pt").to("xpu") for x in chosen]
        if len({x.shape[1] for x in ids}) != 1:
            # Individual forwards retain exactness; stress record documents the
            # maximum rather than silently adding padded positions.
            ids = [ids[0]] * batch
        stacked = torch.cat(ids, dim=0)
        contlen = len(tokenizer.encode(chosen[0][7] + chosen[0][6][0], add_special_tokens=False))
        try:
            _, _, telemetry = score(model, stacked, contlen, True)
            stress.append({"batch": batch, "result": "pass", "prompt_length": chosen[0][0], "telemetry": telemetry})
        except torch.OutOfMemoryError as exc:
            reset_memory(); stress.append({"batch": batch, "result": "oom", "error": str(exc)})
    values=sorted(continuation_lengths)
    report={"continuation_lengths": {"count":len(values), "min":values[0], "median":values[(len(values)-1)//2],
             "p95":values[round((len(values)-1)*.95)], "p99":values[round((len(values)-1)*.99)], "max":values[-1],
             "histogram": {str(v): values.count(v) for v in sorted(set(values))}},
             "idle_memory": idle, "correctness_cases": comparisons, "stress": stress}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__": main()
