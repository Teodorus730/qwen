#!/usr/bin/env python3
"""Smoke-test embedding baseline contract behavior without loading a model."""

from classify_chunks_embedding_baseline import EMBEDDING_METHOD, classify_records


class FakeEmbeddingModel:
    def encode(self, texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False, device=None):
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            if "calculus" in lowered or "derivative" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "python" in lowered or "api" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def main():
    labels = [
        {
            "domain": "stem",
            "field": "mathematics",
            "subfield": "calculus",
            "description": "Calculus derivative functions",
            "keywords": ["calculus", "derivative"],
        },
        {
            "domain": "software",
            "field": "programming",
            "subfield": "documentation",
            "description": "Python API documentation",
            "keywords": ["python", "api"],
        },
    ]
    records = [
        {
            "chunk_id": "sample_0",
            "dataset": "unit",
            "source_type": "math",
            "token_count": 10,
            "text": "A calculus note about the derivative.",
        },
        {
            "chunk_id": "sample_1",
            "dataset": "unit",
            "source_type": "code",
            "token_count": 10,
            "text": "Python API documentation for a function.",
        },
        {
            "chunk_id": "sample_2",
            "dataset": "unit",
            "source_type": "educational",
            "token_count": 10,
            "text": "Unrelated text should abstain at high threshold.",
        },
    ]

    output = classify_records(
        records=records,
        labels=labels,
        model=FakeEmbeddingModel(),
        model_name="fake-local-minilm",
        taxonomy_path="taxonomy/simple_domain_labels.json",
        min_confidence=0.9,
        text_chars=2000,
        batch_size=2,
        device_requested=None,
        device_actual=None,
        top_k=2,
        overwrite_existing_labels=False,
    )

    assert len(output) == 3
    assert output[0]["source_type"] == "math"
    assert output[1]["source_type"] == "code"
    assert output[2]["source_type"] == "educational"
    assert output[0]["domain"] == "stem"
    assert output[1]["domain"] == "software"
    assert output[2]["domain"] is None
    assert output[2]["low_confidence"] is True
    for record in output:
        assert record["label_method"] == EMBEDDING_METHOD
        assert record["embedding_model"] == "fake-local-minilm"
        assert record["taxonomy_path"] == "taxonomy/simple_domain_labels.json"
        assert record["min_confidence"] == 0.9
        assert record["batch_size"] == 2
        assert isinstance(record["top_k_labels"], list)
        assert record["top_k_labels"]

    print("SMOKE TEST PASSED: embedding baseline contract behavior")


if __name__ == "__main__":
    main()
