# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""kld.py against exact float64 references on synthetic logits."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kld  # noqa: E402

N_VOCAB = 1001  # odd, to exercise the record padding


def _log_softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def _logits(rng, n, spread=4.0):
    return (rng.standard_normal((n, N_VOCAB)) * spread).astype(np.float32)


def _targets(rng, logits):
    """True next tokens drawn from the base distribution, as real text is.
    Uniform draws would mostly land below max-16, where the base format clamps."""
    p = np.exp(_log_softmax(logits))
    return np.array([rng.choice(N_VOCAB, p=row / row.sum()) for row in p])


class QuantizationTest(unittest.TestCase):
    def test_round_trip_within_half_step(self):
        logits = _logits(np.random.default_rng(0), 64)
        rec = kld.quantize_log_probs(logits)
        self.assertEqual(rec.shape, (64, kld.record_width(N_VOCAB)))
        back = kld.dequantize_log_probs(rec, N_VOCAB)
        exact = _log_softmax(logits)
        scale = np.ascontiguousarray(rec[:, :4]).view(np.float32)[:, 0]
        keep = exact > exact.max(axis=1, keepdims=True) - 16
        err = np.abs(back - exact) / scale[:, None]
        self.assertLessEqual(err[keep].max(), 0.5 + 1e-2)

    def test_file_round_trip(self):
        rng = np.random.default_rng(1)
        n_ctx, n_chunk = 16, 3
        tokens = rng.integers(0, N_VOCAB, size=n_ctx * n_chunk)
        per = kld.scored_per_chunk(n_ctx)
        chunks = [_logits(rng, per) for _ in range(n_chunk)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "base.kld"
            w = kld.BaseLogitsWriter(path, n_ctx, N_VOCAB, tokens)
            for c in chunks:
                w.add_chunk(c)
            w.close()
            r = kld.BaseLogitsReader(path)
            self.assertEqual((r.n_ctx, r.n_vocab, r.n_chunk), (n_ctx, N_VOCAB, n_chunk))
            np.testing.assert_array_equal(r.chunk_tokens(2), tokens[2 * n_ctx :])
            np.testing.assert_array_equal(r.chunk(1), kld.quantize_log_probs(chunks[1]))
            # Header layout: magic, then three int32.
            raw = path.read_bytes()
            self.assertEqual(raw[:8], b"_logits_")
            self.assertEqual(
                np.frombuffer(raw[8:20], dtype="<i4").tolist(),
                [n_ctx, N_VOCAB, n_chunk],
            )

    def test_incomplete_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            w = kld.BaseLogitsWriter(Path(d) / "b.kld", 8, N_VOCAB, np.zeros(16))
            w.add_chunk(_logits(np.random.default_rng(2), kld.scored_per_chunk(8)))
            with self.assertRaises(RuntimeError):
                w.close()


class StatsTest(unittest.TestCase):
    def _score(self, test, base, tokens):
        s = kld.KldStats()
        s.add(test, kld.quantize_log_probs(base), tokens)
        return s.summary()

    def test_identical_models(self):
        rng = np.random.default_rng(3)
        base = _logits(rng, 500)
        tokens = _targets(rng, base)
        s = self._score(base, base, tokens)
        self.assertLess(s["kld_mean"], 1e-6)
        self.assertEqual(s["same_top"], 1.0)
        self.assertAlmostEqual(s["ln_ppl_ratio"], 0.0, places=4)
        self.assertAlmostEqual(s["ln_ppl_cor"], 1.0, places=4)

    def test_matches_exact_kld_and_ppl(self):
        rng = np.random.default_rng(4)
        base = _logits(rng, 400)
        test = base + _logits(rng, 400, spread=0.3)
        tokens = _targets(rng, base)
        s = self._score(test, base, tokens)

        lp, lq = _log_softmax(base), _log_softmax(test)
        p = np.exp(lp)
        exact_kld = (p * (lp - lq)).sum(axis=1)
        rows = np.arange(400)
        nll, nll_base = -lq[rows, tokens], -lp[rows, tokens]
        self.assertAlmostEqual(s["kld_mean"], exact_kld.mean(), delta=1e-4)
        self.assertAlmostEqual(s["ppl"], np.exp(nll.mean()), delta=1e-9 * s["ppl"])
        self.assertAlmostEqual(s["ppl_base"], np.exp(nll_base.mean()), delta=1e-3)
        same = (test.argmax(1) == base.argmax(1)).mean()
        self.assertAlmostEqual(s["same_top"], same)
        self.assertAlmostEqual(
            s["same_top_se"], np.sqrt(same * (1 - same) / 399), places=12
        )
        self.assertAlmostEqual(
            s["kld_percentiles"]["median"], np.median(exact_kld), delta=1e-4
        )

    def test_block_split_does_not_change_result(self):
        rng = np.random.default_rng(5)
        base = _logits(rng, 300)
        test = base + _logits(rng, 300, spread=0.5)
        tokens = _targets(rng, base)
        whole = self._score(test, base, tokens)
        parts = kld.KldStats()
        rec = kld.quantize_log_probs(base)
        for lo, hi in ((0, 7), (7, 200), (200, 300)):
            parts.add(test[lo:hi], rec[lo:hi], tokens[lo:hi])
        split = parts.summary()
        merged = kld.KldStats()
        for lo, hi in ((0, 150), (150, 300)):
            part = kld.KldStats()
            part.add(test[lo:hi], rec[lo:hi], tokens[lo:hi])
            merged.merge(part)
        self.assertEqual(merged.summary()["kld_percentiles"], whole["kld_percentiles"])
        self.assertAlmostEqual(merged.summary()["ppl"], whole["ppl"], places=10)
        for k in ("ppl", "kld_mean", "same_top", "rms_dp", "ln_ppl_ratio_se"):
            self.assertAlmostEqual(whole[k], split[k], places=10)

    def test_compare_base_files(self):
        rng = np.random.default_rng(7)
        n_ctx, n_chunk = 256, 2
        per = kld.scored_per_chunk(n_ctx)
        tokens = rng.integers(0, N_VOCAB, size=n_ctx * n_chunk)
        base = [_logits(rng, per) for _ in range(n_chunk)]
        noisy = [b + _logits(rng, per, spread=0.5) for b in base]
        with tempfile.TemporaryDirectory() as d:
            paths = {}
            for name, chunks in (("base", base), ("noisy", noisy)):
                paths[name] = Path(d) / f"{name}.kld"
                w = kld.BaseLogitsWriter(paths[name], n_ctx, N_VOCAB, tokens)
                for c in chunks:
                    w.add_chunk(c)
                w.close()
            same = kld.compare_base_files(paths["base"], paths["base"])
            self.assertLess(abs(same["kld_mean"]), 1e-9)
            self.assertEqual(same["same_top"], 1.0)
            diff = kld.compare_base_files(paths["noisy"], paths["base"])
            self.assertGreater(diff["kld_mean"], 1e-2)

    def test_too_few_tokens(self):
        s = kld.KldStats()
        base = _logits(np.random.default_rng(6), 50)
        s.add(base, kld.quantize_log_probs(base), np.zeros(50, dtype=np.int64))
        with self.assertRaises(ValueError):
            s.summary()


if __name__ == "__main__":
    unittest.main()
