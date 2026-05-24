#!/usr/bin/env python3
"""Small standard-library smoke test for rule-based classifier behavior."""

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from classify_chunks_rule_based import classify_record


CASES = [
    (
        "math text -> source_type math",
        "The derivative of f(x) is defined by a limit and represents the slope of the tangent line.",
        ("math", "stem", "mathematics", "calculus"),
    ),
    (
        "Python docs -> source_type code",
        "Parameters: path. Returns: parsed JSONL records. Example: ```python\ndef load(path): return []\n```",
        ("code", "software", "programming", "documentation"),
    ),
    (
        "Q&A mentioning cookies -> forum_qa",
        "Question: How do I ignore a cookie banner in a thread?\nAnswer: Keep the accepted answer and flag boilerplate separately.",
        ("forum_qa", "web", "forum_qa", "discussion"),
    ),
    (
        "product page -> commercial_product",
        "Product features include a warranty, free shipping, customer reviews, and a buy button.",
        ("commercial_product", "commercial", "product_page", "retail"),
    ),
    (
        "mixed English/Russian -> multilingual",
        "This note mixes English labels with русский текст про классификатор и локальный пайплайн.",
        ("unknown", "multilingual", "mixed_language", None),
    ),
    (
        "pure nav/cookie page -> boilerplate_or_noise",
        "Home | Search | Privacy | Terms | Subscribe | Contact | Sitemap. Accept cookies | Manage preferences | Footer links.",
        ("boilerplate_or_noise", "web", "boilerplate_or_navigation", "page_noise"),
    ),
    (
        "legal/government notice -> legal_government",
        "Public notice: the municipal agency will hold a public hearing for ordinance compliance.",
        ("legal_government", "government", "legal_notice", "public_information"),
    ),
    (
        "wiki/reference style -> wiki_reference",
        "Overview: A reservoir stores water. History: early reservoirs supported cities. Classification: service reservoirs. References: hydrology manuals.",
        ("wiki_reference", "reference", "encyclopedic_article", "general"),
    ),
    (
        "news-like article -> news",
        "Officials announced an update on Monday. According to the department, repairs continue. A spokesperson issued a statement.",
        ("news", "media", "news", "article"),
    ),
]


def main():
    failures = []
    for name, text, expected in CASES:
        record = {"chunk_id": name, "dataset": "smoke", "source_type": "unknown", "token_count": 100, "text": text}
        labeled = classify_record(record)
        actual = (labeled.get("source_type"), labeled.get("domain"), labeled.get("field"), labeled.get("subfield"))
        if actual != expected:
            failures.append((name, expected, actual))

    if failures:
        print("SMOKE TEST FAILED")
        for name, expected, actual in failures:
            print(f"- {name}: expected {expected}, got {actual}")
        raise SystemExit(1)

    print(f"SMOKE TEST PASSED: {len(CASES)} cases")


if __name__ == "__main__":
    main()
