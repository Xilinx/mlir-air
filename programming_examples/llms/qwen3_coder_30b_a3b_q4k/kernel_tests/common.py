# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shared bits of the Qwen3-Coder kernel tests: the Peano compile of one kernel.
# The kernels are Qwen3.6's (and fused_decode_gdn's) at this model's sizes.
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
sys.path.insert(0, str(EXAMPLE))
Q36 = EXAMPLE.parent / "qwen36_35b_a3b_q4nx"

import qwen3_coder_30b_a3b_q4k_weights as W  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "q36_common", Q36 / "kernel_tests" / "common.py"
)
q36_common = importlib.util.module_from_spec(_spec)
_path = list(sys.path)
_spec.loader.exec_module(q36_common)
sys.path[:] = _path  # it puts Qwen3.6's example first

compare = q36_common.compare
SIZES = [f"-DMOE_N_EXPERTS={W.N_EXPERTS}", f"-DMOE_I={W.I_EXP}"]


def compile_kernel(name, extra=(), engine=False):
    """Qwen3.6's kernels/<name>.cc -> ./<name>.o at this model's sizes."""
    extra = [*SIZES, *extra]
    if engine:
        extra.append("-DMODEL_TYPE=QWEN3_CODER_30B_A3B")
    q36_common.compile_kernel(name, extra, engine)
