# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A point the host could not pin memory for is a host limit, not a failure of
the design, in all three sweepers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sweep_decode  # noqa: E402
import sweep_decode_runtime  # noqa: E402
import sweep_prefill  # noqa: E402

# bench_decode at a 128k KV cache past a 16 GiB locked-memory limit
PIN_FAIL = (
    "[XRT] ERROR: Failed to allocate host memory buffer (mmap(addr=0x7617d1800000, "
    "len=19327352832, prot=3, flags=8209, offset=9169715200) failed (err=-11): "
    "Resource temporarily unavailable), make sure host bank is enabled"
)


class HostMemoryTest(unittest.TestCase):
    def test_pinned_allocation_failure_is_host_memory(self):
        self.assertTrue(sweep_decode.HOST_OOM_RE.search(PIN_FAIL))
        self.assertFalse(sweep_decode.XRT_FAIL_RE.search(PIN_FAIL))
        self.assertEqual(sweep_decode_runtime._classify(PIN_FAIL, 1), "host_oom")
        self.assertTrue(sweep_prefill.HOST_OOM_RE.search(PIN_FAIL))

    def test_other_error_codes_are_not(self):
        for text in ("failed (err=-110)", "err=-1", "69.02 tok/s"):
            self.assertFalse(sweep_decode.HOST_OOM_RE.search(text), text)


if __name__ == "__main__":
    unittest.main()
