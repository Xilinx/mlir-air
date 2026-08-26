# ./python/test/api/naming.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""No test file may be named after a stdlib module.

lit puts the test's own directory on sys.path, so a test named e.g. select.py
can win the import of the stdlib module of that name. Whether it *does* depends
on how the running Python was built: `select` is statically built in on many
distro Pythons, so the clash is invisible there, but GitHub's hosted runners
ship it as a shared extension in lib-dynload and the path-based finder consults
sys.path first.

When that happens the damage is not local. numpy imports platform, which imports
subprocess, which imports select -- so the test file gets executed in the middle
of numpy's own import, tries to import air.api.types while air.api.types is
still importing numpy, and dies on a circular import. Every test in the
directory fails, not just the misnamed one.

This is cheap to check and expensive to debug, so it is checked.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

offenders = sorted(
    p.name
    for p in HERE.glob("*.py")
    if p.stem in sys.stdlib_module_names and p.stem != "naming"
)

print("stdlib-shadowing test files:", offenders if offenders else "none")

# CHECK: stdlib-shadowing test files: none
