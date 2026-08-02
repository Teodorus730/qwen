from __future__ import annotations

import unittest
from pathlib import Path

from src.config import load_config


ROOT = Path(__file__).resolve().parent.parent


class ConfigTests(unittest.TestCase):
    def test_5090_profile_loads_with_comparable_effective_batch(self):
        cfg = load_config(ROOT / "configs" / "vast_5090_32gb.yaml")
        self.assertEqual(cfg.training.run_name, "vast_5090_32gb_10m")
        self.assertEqual(cfg.training.micro_batch_size, 8)
        self.assertEqual(cfg.training.gradient_accumulation_steps, 2)
        self.assertEqual(cfg.tokens_per_update, 8192)
        self.assertEqual(cfg.total_updates, 1221)
        self.assertEqual(
            cfg.benchmark.batch_sizes,
            (1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16),
        )


if __name__ == "__main__":
    unittest.main()
