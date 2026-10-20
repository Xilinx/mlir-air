# ./python/test/api/kernel_rules.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s %air_src_root | FileCheck %s

"""programming_examples/kernel_rules.mk, through its two users.

external_kernels._compile_kernel recompiles a forced object only when its
flags, its source, a header the source includes or the compiler changed, and
never recompiles an unforced object that already exists.

fused_decode/decode_kernels.mk builds .o and .ll kernels with their own flags,
takes each source from the first directory that has it, rebuilds only the
kernels whose flags changed and leaves the including Makefile's default goal
alone.

A fake clang++ stands in for Peano: it logs each call, writes the directory of
the source and its -D/-O flags into the output and lists the source and its
header in the dependency file.
"""

import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

SRC = Path(sys.argv[1])
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(SRC / "programming_examples")
]
from air_examples.llms.shared.infra import external_kernels as ek  # noqa: E402

FAKE_CXX = """#!{python}
import sys
args = sys.argv[1:]
out = args[args.index("-o") + 1]
dep = args[args.index("-MF") + 1]
src = next(a for a in args if a.endswith(".cc"))
if "-DFAIL" in args:
    sys.stderr.write("kernel.cc:1:1: error: requested failure\\n")
    sys.exit(1)
with open("{log}", "a") as f:
    f.write(out + "\\n")
with open(out, "w") as f:
    f.write(src.split("/")[-2] + ": ")
    f.write(" ".join(a for a in args if a[:2] in ("-D", "-O")))
hdr = src[: -len(".cc")] + ".h"
with open(dep, "w") as f:
    f.write(out + ": " + src + " " + hdr + "\\n" + hdr + ":\\n")
"""

work = Path(tempfile.mkdtemp())
peano = work / "peano"
(peano / "bin").mkdir(parents=True)
inc = work / "aie" / "include" / "aie_api"
inc.mkdir(parents=True)
(inc / "aie.hpp").write_text("")
log = work / "calls.log"
cxx = peano / "bin" / "clang++"
cxx.write_text(FAKE_CXX.format(python=sys.executable, log=log))
cxx.chmod(0o755)
src = work / "kernel.cc"
src.write_text('#include "kernel.h"\n')
hdr = work / "kernel.h"
hdr.write_text("")

os.environ["PEANO_INSTALL_DIR"] = str(peano)
os.environ["MLIR_AIE_INSTALL_DIR"] = str(work / "aie")
os.environ["PATH"] = "/usr/bin:/bin"
# Called from a recipe of a make run with -q: the outer make's options must not
# reach the kernel build.
os.environ["MAKEFLAGS"] = "q"
os.environ["MAKELEVEL"] = "1"
build = work / "build"
build.mkdir()
os.chdir(build)


def step(name, flags, force=True):
    """Run one _compile_kernel call; report whether it compiled and the
    defines the object was built with."""
    before = log.read_text() if log.exists() else ""
    ek._compile_kernel(src, "k.o", extra_flags=flags, force=force)
    after = log.read_text() if log.exists() else ""
    compiled = len(after) > len(before)
    print(f"{name}: compiled={compiled} obj=[{Path('k.o').read_text()}]")


def make_newer(path):
    t = Path("k.o").stat().st_mtime + 10
    os.utime(path, (t, t))


step("first", ["-DA=1"])
# CHECK: first: compiled=True obj=[{{.*}}-DA=1]
step("same flags", ["-DA=1"])
# CHECK: same flags: compiled=False obj=[{{.*}}-DA=1]
step("new flags", ["-DA=2"])
# CHECK: new flags: compiled=True obj=[{{.*}}-DA=2]
step("unforced keeps", ["-DA=3"], force=False)
# CHECK: unforced keeps: compiled=False obj=[{{.*}}-DA=2]
make_newer(hdr)
step("header", ["-DA=2"])
# CHECK: header: compiled=True obj=[{{.*}}-DA=2]
make_newer(src)
step("source", ["-DA=2"])
# CHECK: source: compiled=True obj=[{{.*}}-DA=2]
make_newer(cxx)
step("compiler", ["-DA=2"])
# CHECK: compiler: compiled=True obj=[{{.*}}-DA=2]

# An unforced compile of a missing object records its flags too, so a later
# forced call with the old flags does not take that object for its own.
Path("k.o").unlink()
step("unforced missing", ["-DA=4"], force=False)
# CHECK: unforced missing: compiled=True obj=[{{.*}}-DA=4]
step("back to old flags", ["-DA=2"])
# CHECK: back to old flags: compiled=True obj=[{{.*}}-DA=2]

try:
    ek._compile_kernel(src, "k.o", extra_flags=["-DFAIL"], force=True)
    print("failure: no error")
except RuntimeError as e:
    print(f"failure: {e}")
# CHECK: failure: Failed to compile k.o: {{.*}}error: requested failure


# decode_kernels.mk, included the way the decode Makefiles include it.
for k in os.environ.copy():
    if k in ("MAKEFLAGS", "MAKELEVEL"):
        del os.environ[k]
for d, names in (("base", "abc"), ("over", "a")):
    (work / d).mkdir()
    for n in names:
        (work / d / f"{n}.cc").write_text(f'#include "{n}.h"\n')
        (work / d / f"{n}.h").write_text("")
dec = work / "dec"
# A compiler of its own: the one above was dated into the future.
peano_dec = work / "peano_dec"
(peano_dec / "bin").mkdir(parents=True)
(peano_dec / "bin" / "clang++").write_text(cxx.read_text())
(peano_dec / "bin" / "clang++").chmod(0o755)
os.environ["PEANO_INSTALL_DIR"] = str(peano_dec)
(work / "Makefile").write_text(f"""
help:
\t@echo default goal
KERNEL_DIR := {dec}
KERNEL_SRC_DIRS := {work / "over"} {work / "base"}
KERNEL_FLAGS = -DALL
O2_KERNELS := a b
LL_KERNELS := c
b_FLAGS = -DB=$(BV)
include {SRC / "programming_examples" / "fused_decode" / "decode_kernels.mk"}
""")


def decode(name, *args):
    """Run make on the decode Makefile; report which kernels compiled."""
    before = log.read_text().split()
    out = subprocess.run(
        ["make", "--no-print-directory", "-C", str(work), *args],
        capture_output=True,
        text=True,
    )
    built = [Path(o).name for o in log.read_text().split()[len(before) :]]
    print(f"{name}: rc={out.returncode} built={sorted(built)}")
    return out.stdout


print(decode("no goal").strip())
# CHECK: no goal: rc=0 built=[]
# CHECK-NEXT: default goal
decode("all", "decode-kernels", "BV=1")
# CHECK: all: rc=0 built=['a.o', 'b.o', 'c.ll']
for n in ("a.o", "b.o", "c.ll"):
    print(f"{n} = {(dec / n).read_text()}")
# CHECK: a.o = over: -DALL -O2
# CHECK: b.o = base: -DALL -DB=1 -O2
# CHECK: c.ll = base: -DALL -O1 -DDECODE_INLINE_ATTN
decode("again", "decode-kernels", "BV=1")
# CHECK: again: rc=0 built=[]
decode("one kernel's flags", "decode-kernels", "BV=2")
# CHECK: one kernel's flags: rc=0 built=['b.o']
decode("ll opt", "decode-kernels", "BV=2", "LL_OPT=O2")
# CHECK: ll opt: rc=0 built=['c.ll']
t = (dec / "a.o").stat().st_mtime + 10
os.utime(work / "over" / "a.h", (t, t))
decode("header", "decode-kernels", "BV=2", "LL_OPT=O2")
# CHECK: header: rc=0 built=['a.o']
