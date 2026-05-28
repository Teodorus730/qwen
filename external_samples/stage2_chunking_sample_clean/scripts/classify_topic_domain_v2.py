#!/usr/bin/env python
"""Weak coarse topic.domain v2 classifier for annotation_v2 records."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METHOD = "weak_topic_domain_v2_keyword_surface_prior"
ALLOWED_DOMAINS = [
    "stem",
    "science",
    "technology",
    "software",
    "humanities",
    "social_sciences",
    "commercial",
    "government",
    "media",
    "reference",
    "education",
    "unknown",
]


KEYWORD_PROFILES: dict[str, list[tuple[str, float]]] = {
    "stem": [
        ("solve", 1.2),
        ("equation", 1.5),
        ("matrix", 2.0),
        ("algebra", 1.6),
        ("calculus", 1.6),
        ("derivative", 1.8),
        ("trigonometry", 1.8),
        ("sin", 0.8),
        ("cos", 0.8),
        ("tan", 0.8),
        ("median", 1.4),
        ("mean", 1.1),
        ("variance", 1.5),
        ("probability", 1.4),
        ("statistics", 1.8),
        ("geometry", 1.5),
        ("topology", 1.8),
    ],
    "science": [
        ("biology", 1.5),
        ("species", 1.3),
        ("animal", 1.0),
        ("ape", 1.4),
        ("tamarin", 1.8),
        ("carbon", 1.3),
        ("oxygen", 1.1),
        ("co2", 1.4),
        ("chemical", 1.5),
        ("atom", 1.4),
        ("bromine", 2.0),
        ("iodine", 2.0),
        ("force", 1.2),
        ("volume", 1.0),
        ("newton", 1.3),
        ("climate", 1.4),
        ("flood", 1.2),
        ("ecosystem", 1.5),
        ("depression", 1.4),
        ("therapy", 1.2),
        ("medical", 1.2),
        ("health", 1.2),
    ],
    "technology": [
        ("patent", 2.0),
        ("terminal", 1.4),
        ("point-of-sale", 2.2),
        ("pos", 1.7),
        ("transaction", 1.5),
        ("data tap", 2.0),
        ("lan adapter", 2.2),
        ("interface", 1.2),
        ("circuit", 1.3),
        ("server", 0.9),
        ("computer", 0.9),
        ("display", 0.8),
        ("apparatus", 1.5),
        ("embodiment", 1.8),
        ("remote", 0.6),
        ("input", 0.5),
        ("output", 0.5),
        ("roundabout", 1.5),
        ("gyratory", 1.8),
    ],
    "software": [
        ("software", 1.7),
        ("program", 1.0),
        ("programming", 2.0),
        ("web server", 2.0),
        ("virtual hosting", 2.0),
        ("database", 1.0),
        ("api", 1.8),
        ("function", 1.2),
        ("parameter", 1.2),
        ("module", 1.4),
        ("html", 1.5),
        ("css", 1.5),
        ("http", 1.4),
        ("json", 1.5),
        ("command", 1.1),
        ("maple", 1.2),
    ],
    "humanities": [
        ("history", 1.2),
        ("political", 1.0),
        ("spain", 1.2),
        ("falange", 2.0),
        ("museum", 1.8),
        ("art", 1.6),
        ("artist", 1.4),
        ("religious", 1.2),
        ("faith", 1.0),
        ("language", 0.9),
        ("literature", 1.2),
    ],
    "social_sciences": [
        ("women", 1.7),
        ("gender", 1.8),
        ("inequality", 1.6),
        ("society", 1.0),
        ("social", 1.2),
        ("disaster trap", 2.2),
        ("policy", 1.0),
        ("community", 0.8),
    ],
    "commercial": [
        ("marketing", 2.0),
        ("service", 0.8),
        ("product", 1.0),
        ("customer", 1.2),
        ("sale", 1.0),
        ("rental", 1.3),
        ("invoice", 1.5),
        ("price", 1.0),
        ("tax", 0.8),
        ("store", 0.8),
    ],
    "government": [
        ("government", 1.7),
        ("legal", 1.6),
        ("law", 1.2),
        ("assessed", 1.4),
        ("council", 1.2),
        ("public information", 1.6),
        ("permit", 1.2),
        ("regulation", 1.2),
    ],
    "media": [
        ("news", 1.7),
        ("journalist", 1.4),
        ("television", 1.2),
        ("draft", 1.0),
        ("chevrolet", 1.8),
        ("camaro", 1.8),
        ("article", 0.6),
        ("published", 0.8),
        ("review", 0.8),
    ],
    "reference": [
        ("encyclopedia", 2.0),
        ("dictionary", 2.0),
        ("oed", 2.0),
        ("database", 0.9),
        ("definition", 1.5),
        ("bibliography", 1.5),
        ("catalog", 1.2),
        ("resource", 1.0),
        ("widely regarded", 1.5),
    ],
    "education": [
        ("lesson", 1.4),
        ("teacher", 1.4),
        ("student", 1.3),
        ("classroom", 1.5),
        ("preschool", 2.0),
        ("learning", 1.2),
        ("activity", 1.0),
        ("worksheet", 1.5),
        ("training", 0.9),
        ("school", 1.0),
    ],
}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def annotation(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("annotation_v2")
    return value if isinstance(value, dict) else {}


def section(record: dict[str, Any], name: str) -> dict[str, Any]:
    value = annotation(record).get(name)
    return value if isinstance(value, dict) else {}


def add_score(scores: dict[str, float], evidence: dict[str, list[str]], domain: str, amount: float, reason: str) -> None:
    scores[domain] += amount
    evidence[domain].append(f"{reason} (+{amount:g})")


def keyword_scores(text: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    normalized = text.lower()
    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[str]] = defaultdict(list)
    for domain, keywords in KEYWORD_PROFILES.items():
        for keyword, weight in keywords:
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b" if keyword.replace("-", "").replace(" ", "").isalnum() else re.escape(keyword.lower())
            hits = len(re.findall(pattern, normalized))
            if hits:
                amount = min(weight * hits, weight * 3)
                add_score(scores, evidence, domain, amount, f"keyword:{keyword} x{hits}")
    return scores, evidence


def apply_surface_scores(record: dict[str, Any], scores: dict[str, float], evidence: dict[str, list[str]]) -> None:
    surface = section(record, "surface")
    stats = section(record, "text_stats")
    if surface.get("has_math_notation"):
        add_score(scores, evidence, "stem", 1.2, "surface.has_math_notation")
    if surface.get("is_symbol_heavy"):
        add_score(scores, evidence, "stem", 0.8, "surface.is_symbol_heavy")
    if surface.get("has_code"):
        add_score(scores, evidence, "software", 1.4, "surface.has_code")
    if surface.get("has_api_or_command_syntax"):
        add_score(scores, evidence, "software", 0.8, "surface.has_api_or_command_syntax")
    if surface.get("has_scientific_formula"):
        add_score(scores, evidence, "science", 0.7, "surface.has_scientific_formula")
        add_score(scores, evidence, "stem", 0.4, "surface.has_scientific_formula")
    token_per_byte = stats.get("token_per_byte")
    if isinstance(token_per_byte, (int, float)) and token_per_byte >= 0.38:
        add_score(scores, evidence, "stem", 0.5, "high token_per_byte")


def apply_provenance_prior(record: dict[str, Any], scores: dict[str, float], evidence: dict[str, list[str]]) -> None:
    dataset = section(record, "provenance").get("dataset") or record.get("dataset")
    text = str(record.get("text", "")).lower()
    if dataset == "FineMath":
        if any(word in text for word in ["maple", "calling sequence", "parameters"]):
            add_score(scores, evidence, "software", 0.7, "FineMath prior overridden by tool/docs text")
        else:
            add_score(scores, evidence, "stem", 0.9, "FineMath weak provenance prior")
    elif dataset == "FineWeb-Edu":
        add_score(scores, evidence, "education", 0.25, "FineWeb-Edu weak provenance prior")


def classify_record(record: dict[str, Any], min_score: float, margin: float) -> dict[str, Any]:
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        return {
            "domain": "unknown",
            "confidence": 0.0,
            "method": METHOD,
            "abstained": True,
            "abstain_reason": "missing_text",
            "top_k": [],
            "evidence": {},
        }

    quality = section(record, "quality")
    if quality.get("noise_level") == "mostly_noise":
        return {
            "domain": "unknown",
            "confidence": 0.2,
            "method": METHOD,
            "abstained": True,
            "abstain_reason": "mostly_noise",
            "top_k": [],
            "evidence": {"quality": quality.get("noise_reasons", [])},
        }

    scores, evidence = keyword_scores(text)
    apply_surface_scores(record, scores, evidence)
    apply_provenance_prior(record, scores, evidence)

    ranked = sorted(
        ((domain, score) for domain, score in scores.items() if domain in ALLOWED_DOMAINS and domain != "unknown"),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return {
            "domain": "unknown",
            "confidence": 0.2,
            "method": METHOD,
            "abstained": True,
            "abstain_reason": "no_domain_evidence",
            "top_k": [],
            "evidence": {},
        }
    top_domain, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(max(score, 0.0) for _, score in ranked)
    confidence = round(top_score / total, 6) if total else 0.0
    top_k = [{"domain": domain, "score": round(score, 6)} for domain, score in ranked[:5]]

    if top_score < min_score:
        return {
            "domain": "unknown",
            "confidence": confidence,
            "method": METHOD,
            "abstained": True,
            "abstain_reason": "top_score_below_threshold",
            "top_k": top_k,
            "evidence": {domain: evidence.get(domain, [])[:6] for domain, _ in ranked[:5]},
        }
    if second_score and (top_score - second_score) < margin:
        return {
            "domain": "unknown",
            "confidence": confidence,
            "method": METHOD,
            "abstained": True,
            "abstain_reason": "top_two_domains_too_close",
            "top_k": top_k,
            "evidence": {domain: evidence.get(domain, [])[:6] for domain, _ in ranked[:5]},
        }

    return {
        "domain": top_domain,
        "confidence": confidence,
        "method": METHOD,
        "abstained": False,
        "abstain_reason": None,
        "top_k": top_k,
        "evidence": {domain: evidence.get(domain, [])[:6] for domain, _ in ranked[:5]},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify coarse annotation_v2 topic.domain.")
    parser.add_argument("--input", required=True, help="Input tokenized annotation_v2 JSONL.")
    parser.add_argument("--output", required=True, help="Output JSONL with annotation_v2.topic.")
    parser.add_argument("--min-score", type=float, default=1.4)
    parser.add_argument("--margin", type=float, default=0.3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(Path(args.input))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    abstained = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for record in rows:
            topic = classify_record(record, min_score=args.min_score, margin=args.margin)
            annotation(record)["topic"] = topic
            counts[topic["domain"]] += 1
            if topic["abstained"]:
                abstained += 1
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "input": args.input,
        "output": args.output,
        "records": len(rows),
        "domain_counts": dict(counts),
        "abstained": abstained,
        "method": METHOD,
        "min_score": args.min_score,
        "margin": args.margin,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
