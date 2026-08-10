"""Runtime-only exact MMMLU single-token loglikelihood adapter.

It is deliberately opt-in and rejects any non-single-token continuation.
"""
from __future__ import annotations

from types import MethodType


def enable() -> None:
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from lm_eval.models.huggingface import HFLM
    from lm_eval.models.utils import Collator

    boundaries = ((0, 256, 64), (257, 512, 32), (513, 1024, 16), (1025, 2048, 8), (2049, 4096, 8))
    def optimized(self, requests, disable_tqdm=False, override_bs=None):
        if self.backend != "causal":
            raise RuntimeError("Optimized single-token path supports causal models only.")
        if any(len(continuation) != 1 for _, _, continuation in requests):
            raise RuntimeError("Optimized path requires verified one-token continuations.")
        def collate(req): return (-len(req[1] + req[2]), tuple(req[1] + req[2]))
        # Do not use lm-eval's logits cache here: it assumes a full sequence
        # logits tensor.  The one-token optimization is still exact without it.
        re_ord = Collator(requests, sort_fn=collate, group_by=None)
        batch_size = self.batch_size if self.batch_size != "auto" else override_bs
        if not batch_size: raise RuntimeError("Auto batching is forbidden for optimized MMMLU.")
        # Preserve lm-eval request identity/order; bucket membership is based
        # solely on the exact causal input after the standard truncation rule.
        buckets = {hi: [] for _, hi, _ in boundaries}
        for position, request in enumerate(requests):
            _, context, continuation = request
            length = len((context + continuation)[-(self.max_length + 1):]) - 1
            if len(context) + len(continuation) > self.max_length + 1 or length > 4096:
                raise RuntimeError("MMMLU optimized input exceeds canonical max_length=4096.")
            for lo, hi, fixed_batch in boundaries:
                if lo <= length <= hi:
                    buckets[hi].append((position, request)); break
            else: raise RuntimeError("MMMLU input did not match canonical bucket policy.")
        results = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(disable_tqdm or self.rank != 0), desc="Running optimized loglikelihood requests")
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eot_token_id
        for lo, hi, fixed_batch in boundaries:
          bucket = buckets[hi]
          for offset in range(0, len(bucket), fixed_batch):
            indexed = bucket[offset:offset + fixed_batch]
            chunk = [request for _, request in indexed]
            inputs=[]; targets=[]
            for _, context, continuation in chunk:
                combined=(context + continuation)[-(self.max_length + 1):]
                if len(combined) > self.max_length + 1:
                    raise RuntimeError("Unexpected context truncation")
                inputs.append(combined[:-1]); targets.append(combined[-1])
            width=max(map(len, inputs))
            ids=torch.full((len(inputs), width), pad_id, dtype=torch.long, device=self.device)
            mask=torch.zeros_like(ids)
            for row, item in enumerate(inputs):
                ids[row, width-len(item):]=torch.tensor(item, device=self.device)
                mask[row, width-len(item):]=1
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.mixed_precision_dtype, enabled=self.mixed_precision_dtype is not None):
                logits=self.model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1, :]
            logp=F.log_softmax(logits, dim=-1, dtype=self.softmax_dtype)
            target=torch.tensor(targets, device=self.device)
            scores=logp.gather(1,target[:,None]).squeeze(1); greedy=logits.argmax(-1)
            for (position, (request_str, _, _)), score, good in zip(indexed, scores, greedy == target, strict=True):
                answer=(float(score), bool(good))
                results[position] = answer
                if request_str is not None: self.cache_hook.add_partial("loglikelihood", request_str, answer)
                pbar.update(1)
        pbar.close()
        return results

    HFLM._loglikelihood_tokens = optimized
