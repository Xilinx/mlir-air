# ./python/test/compiler/xrt_backend_compile_cache.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""XRTBackend reuses a build from AIR_COMPILE_CACHE only when every component
of its key matches, and rebuilds when any one of them changes.

aircc is replaced by a script that writes outputs derived from its input and
counts how often it ran, so no toolchain or device is needed.
"""

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

import air.backend.xrt as xrt
from air.backend import compile_cache
from air.backend.xrt import XRTBackend

work = Path(tempfile.mkdtemp())
os.chdir(work)
runs = work / "runs.txt"

fake = work / "fake_aircc"
fake.write_text(f"""#!{sys.executable}
import hashlib, os, sys
args = sys.argv[1:]
text = open("air.mlir").read() + " ".join(args)
for flag in ("-o", "-i", "--elf-name"):
    if flag in args:
        open(args[args.index(flag) + 1], "w").write(flag + hashlib.sha256(text.encode()).hexdigest())
if "--elf-name" in args:
    open("params.txt", "w").write("0\\n")
with open({str(runs)!r}, "a") as f:
    f.write("x")
""")
fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
xrt.resolve_tool = lambda name: fake

peano = work / "peano"
(peano / "bin").mkdir(parents=True)
for tool in ("clang", "clang++", "opt", "llc", "ld.lld"):
    (peano / "bin" / tool).write_text("peano-v1")
os.environ["PEANO_INSTALL_DIR"] = str(peano)

MODULE = 'module { func.func private @k() attributes {link_with = "k.o"} }'
Path("k.o").write_text("object-v1")


def build(module=MODULE, label=None, **options):
    before = runs.stat().st_size if runs.exists() else 0
    options.setdefault("target_device", "npu2")
    art = XRTBackend(**options).compile(module, "design")
    ran = (runs.stat().st_size if runs.exists() else 0) - before
    result = "ran aircc" if ran else "reused"
    print(f"{label}: {result}" if label else result)
    return art


# CHECK-LABEL: cache off
print("cache off")
# CHECK: ran aircc
build()
# CHECK: ran aircc
build()

cache = work / "cache"
os.environ["AIR_COMPILE_CACHE"] = str(cache)

# CHECK-LABEL: first build is stored, an identical one reused
print("first build is stored, an identical one reused")
# CHECK: ran aircc
build()
reference = Path("design.xclbin").read_text()
Path("design.xclbin").unlink()
# CHECK: compile cache hit
# CHECK: reused
build()
# CHECK: restored outputs match: True
print("restored outputs match:", Path("design.xclbin").read_text() == reference)
manifest = json.loads(next(cache.glob("??/*/manifest.json")).read_text())
# CHECK: components: format link_files module options toolchain
print("components:", " ".join(sorted(manifest["components"])))
# CHECK: peano recorded: True
print("peano recorded:", manifest["components"]["toolchain"]["peano"] is not None)
# CHECK: peano path not in key: True
print("peano path not in key:", str(peano) not in json.dumps(manifest["components"]))

# CHECK-LABEL: verbose output does not change the key
print("verbose output does not change the key")
# CHECK: reused
build(verbose=True)

# CHECK-LABEL: each component invalidates the entry
print("each component invalidates the entry")
# CHECK: module: ran aircc
build(module=MODULE.replace("@k", "@k2"), label="module")
# CHECK: options: ran aircc
build(stack_size=4096, label="options")
# CHECK: linked object: ran aircc
Path("k.o").write_text("object-v2")
build(label="linked object")
# CHECK: peano: ran aircc
(peano / "bin" / "llc").write_text("peano-v2")
build(label="peano")
# CHECK: registered dependency: ran aircc
compile_cache.TOOLCHAIN["new-dependency"] = lambda ctx: "v1"
build(label="registered dependency")
# CHECK: same again: reused
build(label="same again")

# CHECK-LABEL: a damaged entry is rebuilt, not restored
print("a damaged entry is rebuilt, not restored")
for blob in cache.glob("??/*/blobs/*"):
    blob.write_text("damaged")
# CHECK: ran aircc
build()

# CHECK-LABEL: unresolvable or unkeyable builds are never cached
print("unresolvable or unkeyable builds are never cached")
missing = MODULE.replace("k.o", "missing.o")
# CHECK: ran aircc
build(module=missing)
# CHECK: ran aircc
build(module=missing)
# CHECK: ran aircc
build(debug_ir=True)
# CHECK: ran aircc
build(debug_ir=True)

# CHECK-LABEL: elf builds restore params.txt
print("elf builds restore params.txt")
# CHECK: ran aircc
build(output_format="elf")
Path("params.txt").unlink()
# CHECK: reused
build(output_format="elf")
# CHECK: params.txt restored: True
print("params.txt restored:", Path("params.txt").exists())

# CHECK-LABEL: prune
print("prune")
compile_cache.prune(cache, 0)
# CHECK: entries left: 0
print("entries left:", len(list(cache.glob("??/*/manifest.json"))))
