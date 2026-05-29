"""Single-launch standalone modules for the decode o_gemv_ffn kernel-group.

Exports a STANDALONES registry compatible with cells/common.py:compile_standalone_kernels.
The actual builder functions live in specs/o_gemv_ffn.py (alongside the SPEC); this
module is a thin derived registry that converts SPEC.sub_launches → list of tuples.
"""

from specs.o_gemv_ffn import SPEC

STANDALONES = [
    (sub.name, sub.builder_ref, sub.build_kwargs) for sub in SPEC.sub_launches
]
