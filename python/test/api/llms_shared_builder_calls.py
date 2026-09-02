# ./python/test/api/llms_shared_builder_calls.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s %air_src_root | FileCheck %s

"""The model builders under llms/ must be able to *call* the shared builders.

llms_builder_contract.py checks that the cross-directory builders llms/ imports
return a Module. This checks a different failure: a model file that calls a
shared builder it never imported.

That is not hypothetical. Converting the elementwise builders replaced each
model's private copy of the residual add with a call to
`shared.builders.o_ffn_multi.build_named_add` / `build_padded_add`, and four of
the qwen prefills ended up calling them with no import anywhere in the file --
nine undefined names across qwen25_0_5b, qwen25_1_5b, qwen25_3b and qwen3_4b.

Nothing caught it. The name resolves at *call* time, not import time, so an
import sweep passes; the return-type contract passes because these are the
models' own builders, not the six it lists; and llms/ is nightly-only, so the
PR would have been green and the nightly red hours later on main.

So the check is to actually call them, at the smallest shape each accepts. The
IR is not inspected -- a NameError is the failure being guarded against, and it
happens long before the module is worth looking at.
"""

import sys

sys.path[:0] = [
    sys.argv[1] + "/programming_examples",
    sys.argv[1] + "/programming_examples/llms",
]

import air.ir

# (model dir, module, builder, args). Every model-local builder that reaches
# for a symbol in shared/builders. Adding such a call means adding a row.
CALLS = [
    ("qwen25_0_5b", "qwen25_0_5b_prefill", "_build_ffn_add_2d_to_1d_ir", (128, 128)),
    (
        "qwen25_0_5b",
        "qwen25_0_5b_prefill",
        "_build_padded_residual_add_2d_ir",
        (128, 128, 256),
    ),
    (
        "qwen25_0_5b",
        "qwen25_0_5b_prefill",
        "_build_down_add_2d_padded_ir",
        (128, 128, 256),
    ),
    ("qwen25_1_5b", "qwen25_1_5b_prefill", "_build_residual_add_2d_ir", (128, 128)),
    ("qwen25_1_5b", "qwen25_1_5b_prefill", "_build_down_add_2d_to_1d_ir", (128, 128)),
    ("qwen25_3b", "qwen25_3b_prefill", "_build_residual_add_2d_ir", (128, 128)),
    ("qwen25_3b", "qwen25_3b_prefill", "_build_down_add_2d_to_1d_ir", (128, 128)),
    ("qwen3_4b", "qwen3_4b_prefill", "_build_residual_add_2d_ir", (128, 128)),
    ("qwen3_4b", "qwen3_4b_prefill", "_build_down_add_2d_to_1d_ir", (128, 128)),
]

ok = 0
for model, modname, fnname, args in CALLS:
    sys.path.insert(0, sys.argv[1] + f"/programming_examples/llms/{model}")
    try:
        mod = __import__(modname)
        text = str(getattr(mod, fnname)(*args))
        # Report through the status line rather than raising, so a failing run
        # names every broken builder instead of stopping at the first.
        why = "OK" if len(text.splitlines()) > 5 else "EMPTY"
    except Exception as e:  # noqa: BLE001 -- the message is the report
        why = f"{type(e).__name__}: {str(e).splitlines()[0][:60]}"
    ok += why == "OK"
    print(f"{modname}.{fnname}: {why}")

# CHECK: qwen25_0_5b_prefill._build_ffn_add_2d_to_1d_ir: OK
# CHECK: qwen25_0_5b_prefill._build_padded_residual_add_2d_ir: OK
# CHECK: qwen25_0_5b_prefill._build_down_add_2d_padded_ir: OK
# CHECK: qwen25_1_5b_prefill._build_residual_add_2d_ir: OK
# CHECK: qwen25_1_5b_prefill._build_down_add_2d_to_1d_ir: OK
# CHECK: qwen25_3b_prefill._build_residual_add_2d_ir: OK
# CHECK: qwen25_3b_prefill._build_down_add_2d_to_1d_ir: OK
# CHECK: qwen3_4b_prefill._build_residual_add_2d_ir: OK
# CHECK: qwen3_4b_prefill._build_down_add_2d_to_1d_ir: OK
print(f"\n{ok}/{len(CALLS)} model builders reached their shared builder")
# CHECK: 9/9 model builders reached their shared builder
