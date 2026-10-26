# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# RUN: %PYTHON %s %air_src_root | FileCheck %s
# CHECK: Affine-Q4 host tests passed.
"""Run the deployment codec, configuration and capture checks without a device."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(sys.argv[1]) / "programming_examples/llms"))
from shared import test_deployment

suite = unittest.defaultTestLoader.loadTestsFromModule(test_deployment)
if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
    raise SystemExit(1)
print("Affine-Q4 host tests passed.")
