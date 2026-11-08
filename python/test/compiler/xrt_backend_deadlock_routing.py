# ./python/test/compiler/xrt_backend_deadlock_routing.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""XRTBackend passes --allow-deadlock-prone-routing to aircc only when asked.

aircc is replaced by a script that records its arguments, so no toolchain or
device is needed.
"""

import os
import stat
import sys
import tempfile
from pathlib import Path

import air.backend.xrt as xrt
from air.backend.xrt import XRTBackend

work = Path(tempfile.mkdtemp())
os.chdir(work)
argv = work / "argv.txt"

fake = work / "fake_aircc"
fake.write_text(f"""#!{sys.executable}
import sys
args = sys.argv[1:]
for flag in ("-o", "-i"):
    if flag in args:
        open(args[args.index(flag) + 1], "w").write(flag)
open({str(argv)!r}, "w").write(" ".join(args))
""")
fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
xrt.resolve_tool = lambda name: fake
os.environ.pop("AIR_COMPILE_CACHE", None)

MODULE = "module { }"

for label, options in (
    ("default", {}),
    ("false", dict(allow_deadlock_prone_routing=False)),
    ("true", dict(allow_deadlock_prone_routing=True)),
):
    XRTBackend(target_device="npu2", **options).compile(MODULE, "design")
    passed = "--allow-deadlock-prone-routing" in argv.read_text().split()
    print(f"{label}: {passed}")

# CHECK: default: False
# CHECK: false: False
# CHECK: true: True
