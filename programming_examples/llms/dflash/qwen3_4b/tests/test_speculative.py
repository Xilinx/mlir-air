# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Greedy acceptance of B8 and B16 draft blocks, including EOS and output clipping."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import HERE, COMMON
from common.speculative import accept_greedy


class AcceptGreedyTests(unittest.TestCase):
    def test_every_rejection_point_remaining_budget_and_eos(self):
        for batch in (8, 16):
            for accepted in range(batch):
                # block[0] is the committed token; the target agrees with the
                # drafts up to the first 99.
                block = [100] + list(range(1, batch))
                target = list(range(1, batch + 1))
                if accepted < batch - 1:
                    block[accepted + 1] = 99
                sequence = list(range(1, accepted + 2))
                for remaining in range(batch + 2):
                    for eos in (None, *sequence):
                        with self.subTest(
                            batch=batch, accepted=accepted, remaining=remaining, eos=eos
                        ):
                            observed = accept_greedy(
                                block, target, remaining, () if eos is None else (eos,)
                            )
                            expected = sequence[:remaining]
                            stopped = eos is not None and eos in expected
                            if stopped:
                                expected = expected[: expected.index(eos) + 1]
                            self.assertEqual(observed.accepted_drafts, accepted)
                            self.assertEqual(observed.tokens, expected)
                            self.assertEqual(observed.stopped_on_eos, stopped)


if __name__ == "__main__":
    unittest.main()
