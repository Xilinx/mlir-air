# ./python/test/api/fused_decode_channel_pairing.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s %air_src_root | FileCheck %s

"""Every DECODE_MODEL must emit air.channels that are in pairs.

programming_examples/fused_decode/fused_decode.py is one builder shared by ten
models. A feed ported to air.dma_memcpy_nd names its channel and lets
air-dma-to-channel derive the other endpoint, so the hand-written op on that
side must be deleted -- otherwise the feed is doubled. Delete it under a weaker
condition than the one the DMA is emitted under and the surviving half is
unpaired.

That has happened four times (@toAttnQ, @attnO, @appendK/@appendV, @ropeLUT),
each time caught only on the one model whose config exposed it, and twice only
on hardware: once as `'air.channel.get' op found channel op not in pairs`
followed by an assertion inside DirectedAdjacencyMap, once as a decode-pos-8
device timeout. The builder now asserts pairing itself; this drives that
assertion across every model's own config, which is the part a single-model
build cannot do.

Emit-only -- no aircc, no NPU -- so it gates per PR. Each model's environment
comes from its own Makefile, via make, so this cannot drift from what the
nightly actually builds.
"""

import os
import re
import subprocess
import sys

src_root = sys.argv[1]
examples = os.path.join(src_root, "programming_examples")
builder = os.path.join(examples, "fused_decode", "fused_decode.py")

# A tiny extra makefile gives us a target that prints the variable; passing both
# with -f concatenates them, so the model's own Makefile is what defines it.
printer = os.path.join(os.environ.get("TMPDIR", "/tmp"), "_fd_print_env.mk")
with open(printer, "w") as f:
    f.write("print-decode-env:\n\t@echo $(DECODE_ENV)\n")

llms = os.path.join(examples, "llms")
models = []
for name in sorted(os.listdir(llms)):
    mk = os.path.join(llms, name, "Makefile")
    if not os.path.isfile(mk):
        continue
    if not re.search(r"^DECODE_ENV", open(mk).read(), re.M):
        continue
    out = subprocess.run(
        [
            "make",
            "-s",
            "-C",
            os.path.join(llms, name),
            "-f",
            "Makefile",
            "-f",
            printer,
            "print-decode-env",
        ],
        capture_output=True,
        text=True,
    )
    env = out.stdout.strip().splitlines()[-1].strip() if out.stdout.strip() else ""
    if env:
        models.append((name, env))

assert models, "no DECODE_ENV models found -- the harness, not the builder, is broken"

bad = []
for name, envline in models:
    env = dict(os.environ)
    for tok in envline.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            env[k] = v
    env["FUSED_DECODE_EMIT_ONLY"] = "1"
    env["PYTHONPATH"] = examples + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, builder],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.join(examples, "fused_decode"),
    )
    if r.returncode == 0:
        status = "PAIRED"
    elif "channel op not in pairs" in r.stderr:
        detail = " ".join(
            re.findall(
                r"@[A-Za-z0-9_]+: \d+ put\(s\), \d+ get\(s\), \d+ dma\(s\)", r.stderr
            )
        )
        status = f"UNPAIRED {detail}"
        bad.append(name)
    else:
        # Anything else is this harness failing to build the model, not a
        # pairing result; report it but do not claim the builder is wrong.
        last = [l for l in r.stderr.splitlines() if l.strip()]
        status = f"DID NOT BUILD ({last[-1][:70] if last else 'no output'})"
    print(f"{name}: {status}")

# CHECK: {{.*}}: PAIRED
# CHECK-NOT: UNPAIRED
print(f"{len(models)} decode models checked, {len(bad)} with unpaired channels")
assert not bad, f"unpaired air.channels in: {bad}"
