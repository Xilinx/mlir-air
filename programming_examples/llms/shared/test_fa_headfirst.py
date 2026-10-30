# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""fa_headfirst._center_keys: what it subtracts and that the softmax ignores it."""

import unittest

import numpy as np
from ml_dtypes import bfloat16

from shared.infra.fa_headfirst import _center_keys

SEQ, HEADS, D = 512, 2, 128


def softmax_rows(q, k):
    s = q @ k.T / np.sqrt(D)
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


class CenterKeysTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.q = rng.normal(0, 1, (SEQ, D)).astype(np.float32)
        k = rng.normal(0, 1, (SEQ, HEADS, D))
        # What a k projection bias leaves: a few channels every key shares.
        k[:, 0, :4] += 300.0
        self.k = k.astype(bfloat16)

    def test_shared_channels_are_subtracted_and_the_rest_kept(self):
        kc = _center_keys(self.k)
        self.assertLess(np.abs(kc[:, 0, :4].astype(np.float32)).max(), 10)
        np.testing.assert_array_equal(kc[:, 0, 4:], self.k[:, 0, 4:])
        np.testing.assert_array_equal(kc[:, 1], self.k[:, 1])

    def test_softmax_is_unchanged(self):
        kc = _center_keys(self.k)
        before = softmax_rows(self.q, self.k[:, 0].astype(np.float32))
        after = softmax_rows(self.q, kc[:, 0].astype(np.float32))
        np.testing.assert_allclose(after, before, atol=1e-3)

    def test_keys_without_a_shared_part_come_back_as_they_are(self):
        k = np.random.default_rng(1).normal(0, 1, (SEQ, HEADS, D)).astype(bfloat16)
        self.assertIs(_center_keys(k), k)


if __name__ == "__main__":
    unittest.main()
