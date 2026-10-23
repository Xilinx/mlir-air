# ./python/test/compiler/compile_cache_kernels.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""run_compiler reuses a kernel compile from AIR_COMPILE_CACHE only when the
preprocessed source, the arguments and the compiler all match, and leaves the
dependency file a build asked for either way.

The compiler is a script that "preprocesses" by inlining one header and counts
its real compiles, so no Peano install is needed.
"""

import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from air.backend import compile_cache
from air.backend.compile_cache import run_compiler

work = Path(tempfile.mkdtemp())
os.chdir(work)
compiles = work / "compiles.txt"

cc = work / "fake_cc"
SCRIPT = f"""#!{sys.executable}
import sys
args = sys.argv[1:]
src = next(a for a in args if a.endswith(".cc"))
out = args[args.index("-o") + 1]
text = open(src).read() + open("kernel.h").read() + " ".join(a for a in args if a.startswith("-D"))
if "-MF" in args:
    target = args[args.index("-MT") + 1] if "-MT" in args else out
    open(args[args.index("-MF") + 1], "w").write(target + ": " + src + " kernel.h\\n")
if "-E" in args:
    open(out, "w").write(text)
else:
    open(out, "w").write("object of " + text + "{{VERSION}}")
    open({str(compiles)!r}, "a").write("x")
"""
cc.write_text(SCRIPT.replace("{VERSION}", "v1"))
cc.chmod(cc.stat().st_mode | stat.S_IEXEC)

Path("kernel.cc").write_text("int k;")
Path("kernel.h").write_text("// header v1")


def compile(*extra, out="kernel.o", label=None, **kwargs):
    before = compiles.stat().st_size if compiles.exists() else 0
    cmd = [str(cc), "-O2", *extra, "-c", "kernel.cc", "-MMD", "-MP", "-MF", out + ".d"]
    run_compiler(cmd + ["-o", out], check=True, **kwargs)
    ran = (compiles.stat().st_size if compiles.exists() else 0) - before
    result = "compiled" if ran else "reused"
    print(f"{label}: {result}" if label else result)


# CHECK-LABEL: cache off
print("cache off")
# CHECK: compiled
compile()
# CHECK: compiled
compile()

os.environ["AIR_COMPILE_CACHE"] = str(work / "cache")

# CHECK-LABEL: first compile is stored, an identical one reused
print("first compile is stored, an identical one reused")
# CHECK: compiled
compile()
reference = Path("kernel.o").read_text()
Path("kernel.o").unlink()
Path("kernel.o.d").unlink()
# CHECK: compile cache hit {{[0-9a-f]+}}: kernel.o
# CHECK: reused
compile()
# CHECK: object restored: True
print("object restored:", Path("kernel.o").read_text() == reference)
# CHECK: dependency file names the object: True
print(
    "dependency file names the object:",
    Path("kernel.o.d").read_text().startswith("kernel.o:"),
)

# CHECK-LABEL: each component invalidates the entry
print("each component invalidates the entry")
# CHECK: header: compiled
Path("kernel.h").write_text("// header v2")
compile(label="header")
# CHECK: define: compiled
compile("-DTILE=64", label="define")
# CHECK: compiler: compiled
cc.write_text(SCRIPT.replace("{VERSION}", "v2"))
compile(label="compiler")
# CHECK: registered dependency: compiled
compile_cache.OBJECT_TOOLCHAIN["new-dependency"] = lambda ctx: "v1"
compile(label="registered dependency")
# CHECK: same again: reused
compile(label="same again")

# CHECK-LABEL: the output name and the working directory do not change the key
print("the output name and the working directory do not change the key")
# CHECK: renamed output: reused
compile(out="renamed.o", label="renamed output")
# CHECK: written under the new name: True
print(
    "written under the new name:",
    Path("renamed.o").read_text() == Path("kernel.o").read_text(),
)
sub = work / "sub"
sub.mkdir()
for f in ("kernel.cc", "kernel.h"):
    (sub / f).write_text((work / f).read_text())
# CHECK: other cwd: reused
compile(label="other cwd", cwd=sub)
# CHECK: written into cwd: True
print("written into cwd:", (sub / "kernel.o").exists())

# CHECK-LABEL: commands that are not one compile run as given
print("commands that are not one compile run as given")
r = run_compiler(
    [sys.executable, "-c", "print('linked')"], capture_output=True, text=True
)
# CHECK: linked
print(r.stdout.strip())

# CHECK-LABEL: the cc entry point
print("the cc entry point")
script = Path(compile_cache.__file__)
before = compiles.stat().st_size
cmd = [
    sys.executable,
    str(script),
    "cc",
    str(cc),
    "-O2",
    "-c",
    "kernel.cc",
    "-o",
    "cli.o",
]
subprocess.run(cmd, check=True)
subprocess.run(cmd, check=True)
# CHECK: compiles for two runs through cc: 1
print("compiles for two runs through cc:", compiles.stat().st_size - before)
