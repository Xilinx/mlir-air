# ./python/test/api/kernel_rules.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s %air_src_root | FileCheck %s

"""programming_examples/kernel_rules.mk, through its two users.

external_kernels._compile_kernel recompiles a forced object only when its
flags, its source, a header the source includes or the compiler changed, and
never recompiles an unforced object that already exists. Each flag reaches the
compiler as one unchanged argument.

fused_decode/decode_kernels.mk builds .o and .ll kernels with their own flags,
takes each source from the first directory that has it, rebuilds only the
kernels whose flags changed, expands KERNEL_FLAGS once per make run and leaves
the including Makefile's default goal alone.

A fake clang++ stands in for Peano: it logs each call, writes the source and
its -D/-O arguments, each in <>, into the output and lists the source and its
header in the dependency file.
"""

import os
import subprocess
import sys
import tempfile
import time
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
    f.write("/".join(src.split("/")[-2:]) + ": ")
    f.write(" ".join("<" + a + ">" for a in args if a[:2] in ("-D", "-O")))
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
log.write_text("")
cxx = peano / "bin" / "clang++"
cxx.write_text(FAKE_CXX.format(python=sys.executable, log=log))
cxx.chmod(0o755)
src = work / "kernel.cc"
other = work / "other.cc"
for cc in (src, other):
    cc.write_text(f'#include "{cc.stem}.h"\n')
    cc.with_suffix(".h").write_text("")

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


def calls():
    return log.read_text().split()


def date_before_now(files, newer=None):
    """Date files in the past and `newer`, if given, after them, so that it is
    the only input newer than the outputs."""
    t = time.time() - 100
    for f in files:
        if f.exists():
            os.utime(f, (t, t))
    if newer:
        os.utime(newer, (t + 10, t + 10))


def step(name, flags, force=True, source=src):
    """Run one _compile_kernel call; report whether it compiled and what the
    object was built from."""
    n = len(calls())
    ek._compile_kernel(source, "k.o", extra_flags=flags, force=force)
    print(f"{name}: compiled={len(calls()) > n} obj=[{Path('k.o').read_text()}]")


python_inputs = [
    src,
    other,
    src.with_suffix(".h"),
    other.with_suffix(".h"),
    cxx,
    Path("k.o"),
    Path("k.o.flags"),
]

step("first", ["-DA=1"])
# CHECK: first: compiled=True obj=[{{.*}}/kernel.cc: <-O2>{{.*}}<-DA=1>]
step("same flags", ["-DA=1"])
# CHECK: same flags: compiled=False
step("new flags", ["-DA=2"])
# CHECK: new flags: compiled=True obj=[{{.*}}<-DA=2>]
step("unforced keeps", ["-DA=3"], force=False)
# CHECK: unforced keeps: compiled=False obj=[{{.*}}<-DA=2>]

for name, newer in (
    ("header", src.with_suffix(".h")),
    ("source", src),
    ("compiler", cxx),
):
    date_before_now(python_inputs, newer)
    step(name, ["-DA=2"])
    step(f"{name} unchanged", ["-DA=2"])
# CHECK: header: compiled=True
# CHECK: header unchanged: compiled=False
# CHECK: source: compiled=True
# CHECK: source unchanged: compiled=False
# CHECK: compiler: compiled=True
# CHECK: compiler unchanged: compiled=False

# Another source, older than the object, with the same flags.
date_before_now(python_inputs)
step("other source", ["-DA=2"], source=other)
# CHECK: other source: compiled=True obj=[{{.*}}/other.cc: {{.*}}<-DA=2>]

# Quotes, spaces and $ arrive as they were given.
step("quoted", ['-DNAME="a b"', "-DCOST=$x"])
# CHECK: quoted: compiled=True obj=[{{.*}}<-DNAME="a b"> <-DCOST=$x>]
step("quoted unchanged", ['-DNAME="a b"', "-DCOST=$x"])
# CHECK: quoted unchanged: compiled=False

# An unforced compile of a missing object records its flags too, so a later
# forced call with the old flags does not take that object for its own.
Path("k.o").unlink()
step("unforced missing", ["-DA=4"], force=False)
# CHECK: unforced missing: compiled=True obj=[{{.*}}<-DA=4>]
step("back to old flags", ["-DA=2"])
# CHECK: back to old flags: compiled=True obj=[{{.*}}<-DA=2>]

try:
    ek._compile_kernel(src, "k.o", extra_flags=["-DFAIL"], force=True)
    print("failure: no error")
except RuntimeError as e:
    print(f"failure: {e}")
# CHECK: failure: Failed to compile k.o: {{.*}}error: requested failure

# decode_kernels.mk, included the way the decode Makefiles include it.
for k in ("MAKEFLAGS", "MAKELEVEL"):
    del os.environ[k]
for d, names in (("base", "abc"), ("over", "a")):
    (work / d).mkdir()
    for n in names:
        (work / d / f"{n}.cc").write_text(f'#include "{n}.h"\n')
        (work / d / f"{n}.h").write_text("")
dec = work / "dec"
queries = work / "queries.log"
queries.write_text("")
(work / "Makefile").write_text(f"""
help:
\t@echo default goal
KERNEL_DIR := {dec}
KERNEL_SRC_DIRS := {work / "over"} {work / "base"}
KERNEL_FLAGS = $(shell echo query >> {queries}; echo -DALL)
O2_KERNELS := a b
LL_KERNELS := c
b_FLAGS = -DB=$(BV)
include {SRC / "programming_examples" / "fused_decode" / "decode_kernels.mk"}
""")


def decode(name, *args):
    """Run make on the decode Makefile; report which kernels compiled and how
    many times KERNEL_FLAGS was expanded."""
    n, q = len(calls()), len(queries.read_text().split())
    out = subprocess.run(
        ["make", "--no-print-directory", "-C", str(work), *args],
        capture_output=True,
        text=True,
    )
    built = sorted(Path(o).name for o in calls()[n:])
    q = len(queries.read_text().split()) - q
    print(f"{name}: rc={out.returncode} built={built} queries={q}")
    return out.stdout


print(decode("no goal").strip())
# CHECK: no goal: rc=0 built=[] queries=0
# CHECK-NEXT: default goal
decode("all", "decode-kernels", "BV=1")
# CHECK: all: rc=0 built=['a.o', 'b.o', 'c.ll'] queries=1
for n in ("a.o", "b.o", "c.ll"):
    print(f"{n} = {(dec / n).read_text()}")
# CHECK: a.o = over/a.cc: <-DALL> <-O2>
# CHECK: b.o = base/b.cc: <-DALL> <-DB=1> <-O2>
# CHECK: c.ll = base/c.cc: <-DALL> <-O1> <-DDECODE_INLINE_ATTN>
decode("again", "decode-kernels", "BV=1")
# CHECK: again: rc=0 built=[] queries=1
decode("one kernel's flags", "decode-kernels", "BV=2")
# CHECK: one kernel's flags: rc=0 built=['b.o'] queries=1
decode("ll opt", "decode-kernels", "BV=2", "LL_OPT=O2")
# CHECK: ll opt: rc=0 built=['c.ll'] queries=1
decode_inputs = [p for p in work.glob("*/*") if p.suffix in (".cc", ".h", ".o")]
decode_inputs += list(dec.glob("*")) + [cxx]
date_before_now(decode_inputs, work / "over" / "a.h")
decode("header", "decode-kernels", "BV=2", "LL_OPT=O2")
# CHECK: header: rc=0 built=['a.o'] queries=1
decode("header unchanged", "decode-kernels", "BV=2", "LL_OPT=O2")
# CHECK: header unchanged: rc=0 built=[] queries=1
