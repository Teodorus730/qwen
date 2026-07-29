from __future__ import annotations

import unittest
from pathlib import Path

from src.config import load_config


ROOT = Path(__file__).resolve().parent.parent


class ConfigTests(unittest.TestCase):
    def test_all_gpu_profiles_load(self):
        expected_batches = {
            "vast_12gb.yaml": 1,
            "vast_16gb.yaml": 4,
            "vast_24gb.yaml": 6,
        }
        for filename, micro_batch in expected_batches.items():
            with self.subTest(filename=filename):
                cfg = load_config(ROOT / "configs" / filename)
                self.assertEqual(cfg.training.micro_batch_size, micro_batch)
                self.assertGreaterEqual(cfg.total_updates, 1)
                self.assertGreaterEqual(cfg.tokens_per_update, 8192)

    def test_12gb_and_16gb_have_same_effective_tokens_per_update(self):
        small = load_config(ROOT / "configs" / "vast_12gb.yaml")
        medium = load_config(ROOT / "configs" / "vast_16gb.yaml")
        self.assertEqual(small.tokens_per_update, 8192)
        self.assertEqual(medium.tokens_per_update, 8192)
        self.assertEqual(small.total_updates, 1221)


if __name__ == "__main__":
    unittest.main()

