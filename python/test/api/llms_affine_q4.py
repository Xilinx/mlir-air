# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# RUN: %PYTHON %s %air_src_root | FileCheck %s
# CHECK: Affine-Q4 host tests passed.
"""Run the deployment codec, configuration and capture checks without a device."""

from pathlib import Path
import sys
import unittest
import types

sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(sys.argv[1]) / "programming_examples")
]
from air_examples.llms.shared import test_deployment

suite = unittest.defaultTestLoader.loadTestsFromModule(test_deployment)
if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
    raise SystemExit(1)
print("Affine-Q4 host tests passed.")
