# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""accuracy.py's decode path against a fake runner: which positions it feeds
and which rows it scores."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import accuracy  # noqa: E402

VOCAB = 7


class FakeRunner:
    """Logits row r is one-hot at (input position % VOCAB) so rows can be traced."""

    def __init__(self):
        self.prefilled = None
        self.steps = []

    def _row(self, pos):
        row = np.zeros(VOCAB, np.float32)
        row[pos % VOCAB] = 1
        return row

    def prefill(self, toks):
        self.prefilled = len(toks)
        return SimpleNamespace(logits_at_pred=self._row(len(toks) - 1))

    def decode_step(self, tok, pos):
        self.steps.append(pos)
        return SimpleNamespace(lm_head_logits=self._row(pos))


class DecodePathTest(unittest.TestCase):
    def test_second_half_matches_llama_cpp_rows(self):
        r = FakeRunner()
        ctx, first = 16, 8
        rows = accuracy.decode_path_logits(r, np.arange(ctx), ctx, first, 2048)
        self.assertEqual(r.prefilled, first + 1)
        self.assertEqual(r.steps, list(range(first + 1, ctx - 1)))
        self.assertEqual(len(rows), ctx - 1 - first)
        self.assertEqual(
            list(rows.argmax(1)), [p % VOCAB for p in range(first, ctx - 1)]
        )

    def test_depth_fills_kv_with_decode_steps(self):
        r = FakeRunner()
        ctx, score, prefill_len = 64, 5, 10
        first = ctx - 1 - score
        rows = accuracy.decode_path_logits(r, np.arange(ctx), ctx, first, prefill_len)
        self.assertEqual(r.prefilled, prefill_len)
        self.assertEqual(r.steps, list(range(prefill_len, ctx - 1)))
        self.assertEqual(len(rows), score)
        self.assertEqual(
            list(rows.argmax(1)), [p % VOCAB for p in range(first, ctx - 1)]
        )


class CacheTest(unittest.TestCase):
    def test_drop_superseded_keeps_other_references(self):
        names = [
            "m.bf16.wikitext-2-test.c2048.n4.aaaaaaaa",  # superseded
            "m.bf16.wikitext-2-test.c2048.n4.bbbbbbbb",  # current
            "m.dequant-decode.wikitext-2-test.c2048.n4.aaaaaaaa",  # other kind
            "m.bf16.wikitext-2-test.c8192.s256.n2.aaaaaaaa",  # other window
        ]
        with tempfile.TemporaryDirectory() as d:
            for n in names:
                for ext in (".kld", ".json"):
                    Path(d, n + ext).write_text("")
            removed = accuracy.drop_superseded(Path(d, names[1] + ".kld"))
            self.assertEqual(
                sorted(p.name for p in removed), [names[0] + ".json", names[0] + ".kld"]
            )
            left = sorted(p.stem for p in Path(d).iterdir())
            self.assertEqual(left, sorted(n for n in names[1:] for _ in range(2)))


if __name__ == "__main__":
    unittest.main()
