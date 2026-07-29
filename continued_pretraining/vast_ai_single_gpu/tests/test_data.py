from __future__ import annotations

import unittest

import torch

from src.data import cyclic_batches, fixed_batches, pack_documents


class TinyTokenizer:
    eos_token_id = 99

    def __len__(self):
        return 128

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(character) - 96 for character in text]}


class DataTests(unittest.TestCase):
    def test_pack_documents_is_compact_and_inserts_eos(self):
        blocks = pack_documents(
            ["ab", "cde", "f"],
            TinyTokenizer(),
            sequence_length=4,
        )
        self.assertEqual(blocks.dtype, torch.int32)
        self.assertEqual(blocks.tolist(), [[1, 2, 99, 3], [4, 5, 99, 6]])

    def test_cyclic_batches_are_reproducible(self):
        blocks = torch.arange(30, dtype=torch.int32).reshape(10, 3)
        first = cyclic_batches(blocks, batch_size=4, seed=7)
        second = cyclic_batches(blocks, batch_size=4, seed=7)
        for _ in range(4):
            left = next(first)
            right = next(second)
            self.assertEqual(left.shape, (4, 3))
            self.assertTrue(torch.equal(left, right))

    def test_fixed_batches_respect_block_limit(self):
        blocks = torch.arange(30, dtype=torch.int32).reshape(10, 3)
        batches = list(fixed_batches(blocks, batch_size=2, max_blocks=5))
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertEqual(sum(len(batch) for batch in batches), 5)


if __name__ == "__main__":
    unittest.main()

