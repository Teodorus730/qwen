from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_representation_hypotheses import linear_cka  # noqa: E402
from evaluate_outputs import baseline_block_metrics  # noqa: E402
from run_attention_alignment import safe_correlation  # noqa: E402
from run_linear_probing import reconstruct_sentence  # noqa: E402


class OutputMetricTests(unittest.TestCase):
    def test_identical_logits_have_zero_divergence(self) -> None:
        generator = torch.Generator().manual_seed(7)
        logits = torch.randn(1, 5, 11, generator=generator)
        labels = torch.tensor([[1, 2, 3, 4, 5]])
        result, _, teacher_top1, student_top1 = baseline_block_metrics(
            logits,
            logits.clone(),
            labels,
            vocab_chunk=2,
            topk=3,
            temperatures=[1.0, 2.0],
        )
        self.assertAlmostEqual(result["top1_match"], 1.0, places=7)
        self.assertAlmostEqual(
            result["kl_teacher_student_t1"], 0.0, places=7
        )
        self.assertAlmostEqual(
            result["kl_teacher_student_t2"], 0.0, places=7
        )
        self.assertAlmostEqual(result["total_variation_t1"], 0.0, places=7)
        self.assertTrue(torch.equal(teacher_top1, student_top1))

    def test_error_counts_require_same_wrong_token(self) -> None:
        teacher = torch.tensor([[
            [0.0, 3.0, 1.0],
            [0.0, 3.0, 1.0],
        ]])
        student = torch.tensor([[
            [0.0, 3.0, 1.0],
            [0.0, 1.0, 3.0],
        ]])
        labels = torch.tensor([[0, 0]])
        result, _, _, _ = baseline_block_metrics(
            teacher,
            student,
            labels,
            vocab_chunk=2,
            topk=2,
            temperatures=[1.0, 2.0],
        )
        self.assertEqual(result["teacher_error"], 2.0)
        self.assertEqual(result["error_intersection"], 2.0)
        self.assertEqual(result["same_wrong_prediction"], 1.0)


class RepresentationMetricTests(unittest.TestCase):
    def test_linear_cka_is_rotation_invariant(self) -> None:
        generator = torch.Generator().manual_seed(11)
        x = torch.randn(128, 32, generator=generator)
        q, _ = torch.linalg.qr(
            torch.randn(32, 32, generator=generator)
        )
        self.assertAlmostEqual(linear_cka(x, x @ q), 1.0, places=5)


class StageFourAndFiveTests(unittest.TestCase):
    def test_ud_spacing_reconstruction_preserves_word_spans(self) -> None:
        text, spans = reconstruct_sentence(
            ["Hello", ",", "world"],
            ["SpaceAfter=No", "_", "_"],
        )
        self.assertEqual(text, "Hello, world")
        self.assertEqual(spans, [(0, 5), (5, 6), (7, 12)])

    def test_correlation_helper_reports_perfect_monotone_relation(self) -> None:
        result = safe_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        self.assertAlmostEqual(result["pearson_r"], 1.0, places=7)
        self.assertAlmostEqual(result["spearman_rho"], 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
