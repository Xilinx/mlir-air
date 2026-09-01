#!/usr/bin/env python3
# ./python/test/api/fused_decode_port_attr_parity.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s %air_src_root | FileCheck %s

"""A channels->dma_memcpy_nd port must not change what buffers exist.

programming_examples/fused_decode/fused_decode.py has ten feeds whose data
movement is spelled as air.dma_memcpy_nd, letting air-dma-to-channel derive the
other endpoint. Each is behind a flag, and each port also has to re-home the
buffers the hand-written construct allocated. The DATA MOVEMENT is supposed to
change. The set of buffers, and the attributes on them, are not.

That is exactly what broke once already. #1969 pinned the q broadcast L2 buffer
with `air.memtile_col = 5`; the @ropeQ / @toAttnQ port allocates that buffer at
segment scope and returns before the hand-written site, so the ported path
never saw the pin and the three 4096xbf16 L2 buffers landed on columns 2, 5, 2
instead of all on 5. Nothing caught it: fused_decode_channel_pairing.py checks
put/get pairing, and a ten-model device sweep at ATTN_MAXL=2048 stayed green
because the derived column is right at that length -- the pin only matters at
the 128 the verify lits build. Four ports have now dropped something the
hand-written form carried: three guards, and this attribute.

So, per port flag: emit the module with the flag on and again with it forced
off, and compare the MULTISET of (attributes, memref type) over every alloc.
Order is deliberately not compared -- a port is allowed to move an alloc and to
renumber it; it is not allowed to change it.

Emit-only, no aircc, no NPU. Each model's environment comes from its own
Makefile via make, so this cannot drift from what the nightly builds.
"""

import os
import re
import subprocess
import sys
import tempfile

src_root = sys.argv[1]
examples = os.path.join(src_root, "programming_examples")
fd_dir = os.path.join(examples, "fused_decode")
builder = os.path.join(fd_dir, "fused_decode.py")

# The flags that select a ported feed. A name not assigned at module scope is
# reported rather than skipped silently: the point of this test is to notice
# when a port stops being covered.
PORT_FLAGS = [
    "ROPELUT_DMA",
    "RMSW_DMA",
    "APPEND_DMA",
    "ROPEQ_DMA",
    "TOATTNQ_DMA",
    "ATTNO_DMA",
    "TOKV_DMA",
]

# No single model exercises every port -- @ropeLUT is inert on qwen3_4b_q4nx and
# on both qwen3_8b_q4nx and qwen25_3b_q4, and is off entirely on the hybrid.
# Sweep until each flag has been exercised somewhere, then stop: a green row
# that means "not reached on this model" proves nothing, so the test must know
# the difference between covered and merely passing. qwen3_4b_q4nx leads
# because it is the model #1969 was measured on.
MODELS = ["qwen3_4b_q4nx", "llama32_3b_q4nx", "lfm2_1_2b_q4nx"]

printer = os.path.join(tempfile.gettempdir(), "_fd_parity_env.mk")
with open(printer, "w") as f:
    f.write("print-decode-env:\n\t@echo $(DECODE_ENV) DECODE_GOLDEN_L=$(LBUILD)\n")


def decode_env(model):
    """The model's own DECODE_ENV, at the LBUILD its verify LIT builds with."""
    mdir = os.path.join(examples, "llms", model)
    # LBUILD comes from the LIT, not the Makefile default: six models' lits pass
    # LBUILD=128 where the Makefile says 2048, and #1969 is a placement bug
    # visible only at the short length. Measuring at 2048 measures a different
    # program than CI runs.
    lit = os.path.join(mdir, "run_npu2_verify.lit")
    lb = ""
    if os.path.isfile(lit):
        m = re.search(r"LBUILD=(\d+)", open(lit).read())
        lb = m.group(1) if m else ""
    cmd = ["make", "-s", "-C", mdir, "-f", "Makefile", "-f", printer]
    if lb:
        cmd.append("LBUILD=" + lb)
    cmd.append("print-decode-env")
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = next(
        (l for l in reversed(out.stdout.splitlines()) if "DECODE_MODEL" in l), ""
    ).strip()
    assert line, f"no DECODE_ENV for {model} -- the harness, not the builder, is broken"
    env = dict(os.environ)
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            env[k] = v
    env["FUSED_DECODE_EMIT_ONLY"] = "1"
    env["PYTHONPATH"] = examples + os.pathsep + env.get("PYTHONPATH", "")
    return env


ALLOC = re.compile(r"memref\.alloc\(\)( \{[^}]*\})? : (memref<[^>]*>)")


def emit(source_path, env):
    """Emit the module and return the sorted (attrs, type) list of its allocs."""
    r = subprocess.run(
        [sys.executable, source_path],
        capture_output=True,
        text=True,
        env=env,
        cwd=fd_dir,
    )
    if r.returncode != 0:
        last = [l for l in r.stderr.splitlines() if l.strip()]
        return None, (last[-1][:90] if last else "no output")
    sigs = sorted(
        (a.strip() if a else "{}") + " :: " + t for a, t in ALLOC.findall(r.stdout)
    )
    return (sigs, r.stdout) if sigs else (None, "emitted no allocs")


pristine = open(builder).read()
missing = [f for f in PORT_FLAGS if not re.search(r"(?m)^%s = " % f, pristine)]
for f in missing:
    print(f"{f}: NOT A MODULE-SCOPE FLAG -- port no longer covered")

bad, covered = list(missing), set()
for model in MODELS:
    if len(covered) + len(missing) == len(PORT_FLAGS):
        break
    env = decode_env(model)
    on_sigs, on_ir = emit(builder, env)
    assert on_sigs, f"{model}: baseline emit failed: {on_ir}"

    for flag in PORT_FLAGS:
        if flag in covered or flag in missing:
            continue
        forced = re.sub(r"(?m)^%s = .*$" % flag, "%s = False" % flag, pristine, count=1)
        # Written BESIDE the real builder, never over it, so cwd and every
        # relative path the builder resolves stay as they are in a normal run.
        # An earlier version of this check edited the file in place and left it
        # modified when it failed part-way through.
        tmp = os.path.join(fd_dir, "_fd_parity_%s.py" % flag)
        try:
            with open(tmp, "w") as f:
                f.write(forced)
            off_sigs, off_ir = emit(tmp, env)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        if off_sigs is None:
            print(f"{model} {flag}: DID NOT BUILD with the port off ({off_ir})")
            bad.append(flag)
            covered.add(flag)
        elif off_ir == on_ir:
            # Forcing the flag off changed no IR, so nothing was compared.
            # Not an error -- just not coverage. Try the next model.
            continue
        elif off_sigs == on_sigs:
            print(f"{model} {flag}: PARITY OK ({len(on_sigs)} allocs)")
            covered.add(flag)
        else:
            only_off = [s for s in off_sigs if s not in on_sigs]
            only_on = [s for s in on_sigs if s not in off_sigs]
            print(f"{model} {flag}: PARITY BROKEN")
            for x in only_off[:4]:
                print(f"    hand-written only: {x}")
            for x in only_on[:4]:
                print(f"    ported only:       {x}")
            bad.append(flag)
            covered.add(flag)

for flag in PORT_FLAGS:
    if flag not in covered and flag not in missing:
        print(f"{flag}: NOT EXERCISED by any of {', '.join(MODELS)}")

# CHECK-NOT: PARITY BROKEN
# CHECK-NOT: NOT A MODULE-SCOPE FLAG
# CHECK-NOT: DID NOT BUILD
# CHECK: ports checked
print(
    f"{len(covered)}/{len(PORT_FLAGS)} ports exercised across "
    f"{len(MODELS)} models, {len(bad)} with changed buffers"
)
print("ports checked")
assert not bad, f"port changed the buffer set or its attributes: {bad}"
