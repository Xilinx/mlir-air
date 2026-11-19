# ./python/test/compiler/xrt_backend_elf_instance_name.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""An ELF build without instance_name loads the module's only public function.

aircc is replaced by a script that writes the files the backend expects, so no
toolchain or device is needed.
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

fake = work / "fake_aircc"
fake.write_text(f"""#!{sys.executable}
import sys
args = sys.argv[1:]
for flag in ("-o", "-i", "--elf-name"):
    if flag in args:
        open(args[args.index(flag) + 1], "w").write(flag)
open("params.txt", "w").write("")
""")
fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
xrt.resolve_tool = lambda name: fake
os.environ.pop("AIR_COMPILE_CACHE", None)


def kernel_of(module, **kwargs):
    backend = XRTBackend(output_format="elf", target_device="npu2", **kwargs)
    return backend.compile(module).kernel


KERN = "func.func private @kern(memref<4xi32>)\n"
ONE = "module {\n" + KERN + "func.func @copy(%a: memref<4xi32>) { return }\n}"
QUOTED = 'module { func.func public @"64_add"(%a: memref<4xi32>) { return } }'
TWO = (
    "module {\n"
    "func.func @copy(%a: memref<4xi32>) { return }\n"
    "func.func @copy.dma(%a: memref<4xi32>) { return }\n}"
)

# CHECK: one: main:copy
print("one:", kernel_of(ONE))
# CHECK: quoted: main:64_add
print("quoted:", kernel_of(QUOTED))
# Two candidates: nothing is guessed.
# CHECK: two: MLIR_AIE
print("two:", kernel_of(TWO))
# An explicit name is used as given.
# CHECK: named: main:copy.dma
print("named:", kernel_of(TWO, instance_name="copy.dma"))
