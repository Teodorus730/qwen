#!/usr/bin/env python
"""Extract deterministic annotation schema v2 features for chunk JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


SCHEMA_VERSION = "annotation_v2_deterministic_features_v1"

TEX_MARKERS = re.compile(
    r"(\\frac|\\sum|\\int|\\lim|\\sqrt|\\alpha|\\beta|\\gamma|\\theta|\\pi|\\sigma|\\mu|\\Delta|\\partial|"
    r"\\mathrm|\\begin|\\end|\\left|\\right|\$\$|\\\(|\\\)|\\\[|\\\])"
)
MATH_SYMBOLS = re.compile(r"[∑∫√π∂∞≈≠≤≥±×÷]")
EQUATION_LIKE = re.compile(
    r"(?i)(?:\b[a-z]\s*(?:\^|_)\s*[\w{}]+|\b[a-z]\([a-z0-9_]+\)|\b[a-z]\s*[=<>]=?\s*[-+]?\d|\d+\s*[=<>]=?\s*[a-z])"
)
MATH_OPERATOR_CONTEXT = re.compile(r"\b(?:sin|cos|tan|log|ln|sqrt|median|mean|variance|matrix|vector)\b", re.I)

CODE_KEYWORDS = re.compile(
    r"(?m)(^\s*(?:def|class|import|from|return)\b|\b(?:function|const|let|var|status_code|endpoint|JSON)\b)"
)
API_HTTP = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+/", re.I)
MARKUP = re.compile(r"(```|</?[a-z][^>]{0,80}>|<script\b|<div\b|<code\b)", re.I)
SQL = re.compile(r"\bSELECT\b.+\bFROM\b", re.I | re.S)
API_DOC = re.compile(
    r"\b(?:Calling Sequence|Parameters|Programming Help|Online Help|function reference|API reference)\b",
    re.I,
)
MAPLE_OR_CAS = re.compile(r"(?:^\s*>\s*\$|\\mathrm\{(?:Matrix|Vector|RootOf)\}|:=|≔)", re.M)

URLS = re.compile(
    r"(https?://|www\.|[a-zA-Z0-9.-]+\.(?:com|org|net|edu|gov|io|ru|de|uk|fr|it|nl|info|biz)\b)"
)

BOILERPLATE_PATTERNS = [
    "cookie",
    "privacy policy",
    "terms of service",
    "terms and conditions",
    "subscribe",
    "login",
    "log in",
    "sign up",
    "skip to content",
    "all rights reserved",
    "copyright",
    "footer",
    "navigation",
    "main menu",
    "accept cookies",
    "reject all",
    "newsletter",
]
BOILERPLATE = re.compile("|".join(re.escape(term) for term in BOILERPLATE_PATTERNS), re.I)

BULLET_LINE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+\S", re.M)
TABLE_LINE = re.compile(r"^\s*\|.+\|\s*$", re.M)
WORD = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def get_text(record: dict[str, Any]) -> str:
    text = record.get("text")
    if text is None:
        return ""
    return str(text)


def build_provenance(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": record.get("dataset"),
        "source_dataset": record.get("source_dataset"),
        "source_config": record.get("source_config"),
        "source_split": record.get("source_split"),
        "chunk_id": record.get("chunk_id"),
        "document_id": (
            record.get("document_id")
            or record.get("doc_id")
            or record.get("source_doc_id")
            or record.get("id")
        ),
    }


def text_stats(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    nonempty_lines = [line for line in lines if line.strip()]
    char_count = len(text)
    byte_count = len(text.encode("utf-8"))
    line_count = len(lines) if text else 0
    return {
        "char_count": char_count,
        "byte_count": byte_count,
        "line_count": line_count,
        "avg_line_length": round(char_count / line_count, 2) if line_count else 0.0,
        "nonempty_line_count": len(nonempty_lines),
        "word_count_rough": len(WORD.findall(text)),
    }


def has_math_notation(text: str) -> bool:
    if not text:
        return False
    tex_hits = len(TEX_MARKERS.findall(text))
    symbol_hits = len(MATH_SYMBOLS.findall(text))
    equation_hits = len(EQUATION_LIKE.findall(text))
    operator_context = bool(MATH_OPERATOR_CONTEXT.search(text))
    equals_count = text.count("=")
    caret_count = text.count("^")

    if tex_hits or symbol_hits:
        return True
    if equation_hits >= 2:
        return True
    if equation_hits >= 1 and (operator_context or caret_count > 0):
        return True
    if equals_count >= 3 and operator_context:
        return True
    return False


def has_code(text: str) -> bool:
    if not text:
        return False
    code_indicators = 0
    if CODE_KEYWORDS.search(text):
        code_indicators += 1
    if API_HTTP.search(text):
        code_indicators += 1
    if MARKUP.search(text):
        code_indicators += 1
    if SQL.search(text):
        code_indicators += 1
    if API_DOC.search(text):
        code_indicators += 1
    if MAPLE_OR_CAS.search(text):
        code_indicators += 1

    semicolons = text.count(";")
    braces = text.count("{") + text.count("}")
    brackets = text.count("[") + text.count("]")
    symbolish_density = ratio(semicolons + braces + brackets, len(text))
    if semicolons >= 3 and symbolish_density > 0.01:
        code_indicators += 1
    if braces >= 4 and symbolish_density > 0.015:
        code_indicators += 1

    return code_indicators >= 2 or bool(MARKUP.search(text) or SQL.search(text))


def has_table_or_list(text: str) -> bool:
    if not text:
        return False
    if len(TABLE_LINE.findall(text)) >= 2:
        return True
    bullet_lines = len(BULLET_LINE.findall(text))
    if bullet_lines >= 3:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_structured = [
        line for line in lines if len(line) <= 80 and (":" in line or "|" in line or "\t" in line)
    ]
    return len(short_structured) >= 5 and len(short_structured) >= max(3, len(lines) // 3)


def repeated_line_score(text: str) -> float:
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return 0.0
    unique = len(set(lines))
    return round(1.0 - (unique / len(lines)), 6)


def surface_features(text: str) -> dict[str, Any]:
    char_count = len(text)
    letters = [ch for ch in text if ch.isalpha()]
    uppercase_letters = [ch for ch in letters if ch.isupper()]
    digits = sum(ch.isdigit() for ch in text)
    symbols = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)
    punctuation = sum(ch in string.punctuation for ch in text)

    return {
        "has_math_notation": has_math_notation(text),
        "has_code": has_code(text),
        "has_numbers": any(ch.isdigit() for ch in text),
        "has_table_or_list": has_table_or_list(text),
        "has_urls_or_links": bool(URLS.search(text)),
        "has_boilerplate_markers": bool(BOILERPLATE.search(text)),
        "symbol_density": ratio(symbols, char_count),
        "digit_density": ratio(digits, char_count),
        "uppercase_ratio": ratio(len(uppercase_letters), len(letters)),
        "punctuation_density": ratio(punctuation, char_count),
    }


def quality_features(text: str, stats: dict[str, Any], surface: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    char_count = int(stats["char_count"])
    word_count = int(stats["word_count_rough"])
    score = 0.0

    if not text.strip():
        return {
            "noise_level": "unknown",
            "noise_score": 1.0,
            "noise_reasons": ["missing_text"],
        }
    if char_count < 120:
        score += 0.25
        reasons.append("very_short_text")
    if surface["has_boilerplate_markers"]:
        score += 0.25
        reasons.append("boilerplate_markers")
    if surface["has_urls_or_links"]:
        score += 0.15
        reasons.append("urls_or_links")
    if surface["has_table_or_list"]:
        score += 0.1
        reasons.append("table_or_list_structure")
    if surface["symbol_density"] >= 0.12:
        score += 0.2
        reasons.append("high_symbol_density")
    if surface["punctuation_density"] >= 0.14:
        score += 0.15
        reasons.append("high_punctuation_density")
    if surface["uppercase_ratio"] >= 0.35 and word_count >= 20:
        score += 0.1
        reasons.append("high_uppercase_ratio")

    line_repetition = repeated_line_score(text)
    if line_repetition >= 0.25:
        score += 0.2
        reasons.append("repeated_lines")

    url_count = len(URLS.findall(text))
    if url_count >= 3:
        score += 0.2
        reasons.append("many_urls")

    normal_prose = word_count >= 80 and surface["symbol_density"] < 0.1
    if surface["has_boilerplate_markers"] and not normal_prose:
        score += 0.2
        reasons.append("boilerplate_with_little_prose")

    score = min(round(score, 3), 1.0)
    if score >= 0.7:
        level = "mostly_noise"
    elif score >= 0.25:
        level = "partial_noise"
    else:
        level = "clean"

    return {
        "noise_level": level,
        "noise_score": score,
        "noise_reasons": reasons,
    }


def enrich_record(record: dict[str, Any]) -> dict[str, Any]:
    text = get_text(record)
    stats = text_stats(text)
    surface = surface_features(text)
    quality = quality_features(text, stats, surface)
    enriched = dict(record)
    enriched["annotation_v2"] = {
        "provenance": build_provenance(record),
        "text_stats": stats,
        "surface": surface,
        "quality": quality,
        "schema_version": SCHEMA_VERSION,
    }
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract deterministic annotation schema v2 features from chunk JSONL."
    )
    parser.add_argument("--input", required=True, help="Input chunk JSONL.")
    parser.add_argument("--output", required=True, help="Output enriched JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    records = read_jsonl(input_path)
    enriched = [enrich_record(record) for record in records]
    write_jsonl(output_path, enriched)
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "records_read": len(records),
                "records_written": len(enriched),
                "schema_version": SCHEMA_VERSION,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
