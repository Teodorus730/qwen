"""Rerank saved semantic-topic embedding top-k with deterministic dev guards."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ENGINEERING_TERMS = re.compile(
    r"\b(patent|apparatus|embodiment|device|terminal|sensor|adapter|circuit|printer|keyboard|display|"
    r"invoice|transaction|pos|point of sale|physical system|infrastructure|appliance|industrial|"
    r"vehicle|engine|hardware|board|port|cable|data tap|lan adapter)\b",
    re.IGNORECASE,
)
SOFTWARE_TERMS = re.compile(
    r"\b(api|function|class|method|package|library|cli|command|parameter|python|javascript|sql|"
    r"code|programming|algorithm|complexity|database|server|configuration)\b",
    re.IGNORECASE,
)
SCIENCE_TERMS = re.compile(
    r"\b(biology|chemical|chemistry|physics|force|newton|molecule|atom|carbon|hydrogen|oxygen|"
    r"species|habitat|rainforest|climate|co2|cell|gene|autism|schizophrenia|bromine|iodine|"
    r"sucrose|mirror|gecko|ant|ecosystem|botany|tree|planet|astronomy)\b",
    re.IGNORECASE,
)
MATH_TERMS = re.compile(
    r"\b(theorem|proof|prove|equation|integral|derivative|topology|nowhere dense|matrix|vector|"
    r"statistics|median|mean|distribution|sigma|function|sin|cos|tan|probability|formula|"
    r"complexity|cholesky|decomposition)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def topk_domains(top_k: list[dict[str, Any]]) -> list[str]:
    return [item.get("domain") for item in top_k if isinstance(item, dict)]


def choose_with_guards(record: dict[str, Any], margin_threshold: float) -> tuple[str | None, str | None]:
    payload = record["semantic_topic_embedding"]
    top_k = payload.get("top_k") or []
    domains = topk_domains(top_k)
    if not top_k:
        return None, "no_top_k"

    current = top_k[0].get("domain")
    text = str(record.get("text") or record.get("text_preview") or "")
    has_engineering = bool(ENGINEERING_TERMS.search(text))
    has_software = bool(SOFTWARE_TERMS.search(text))
    has_science = bool(SCIENCE_TERMS.search(text))
    has_math = bool(MATH_TERMS.search(text))

    if "engineering_technology" in domains and "computer_science_software" in domains:
        if has_engineering and not (has_software and not has_engineering):
            return "engineering_technology", "engineering_vs_software_guard"
        if has_software and not has_engineering:
            return "computer_science_software", "software_vs_engineering_guard"

    if "natural_science" in domains and "math" in domains:
        if has_science and not (has_math and not has_science):
            return "natural_science", "science_vs_math_guard"
        if has_math and not has_science:
            return "math", "math_vs_science_guard"

    margin = float(payload.get("margin") or 0.0)
    if margin < margin_threshold:
        return None, f"low_margin:{margin:.6f}<{margin_threshold:.6f}"
    return current, "embedding_top1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    args = parser.parse_args()

    output = []
    for record in read_jsonl(args.input):
        enriched = dict(record)
        payload = dict(enriched["semantic_topic_embedding"])
        domain, reason = choose_with_guards(enriched, args.margin_threshold)
        payload["original_domain"] = payload.get("domain")
        payload["domain"] = domain
        payload["abstained"] = domain is None
        payload["abstain_reason"] = reason if domain is None else None
        payload["rerank_reason"] = reason
        payload["method"] = payload.get("method", "") + "+deterministic_topk_rerank_v1"
        enriched["semantic_topic_embedding"] = payload
        output.append(enriched)
    write_jsonl(args.output, output)
    print(f"Wrote {len(output)} reranked records to {args.output}")


if __name__ == "__main__":
    main()
