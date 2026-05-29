"""Single-launch standalone modules for the prefill o_ffn kernel-group.

Exports a STANDALONES registry compatible with cells/common.py:compile_standalone_kernels.
"""

from specs.o_ffn import SPEC

STANDALONES = [
    (sub.name, sub.builder_ref, sub.build_kwargs) for sub in SPEC.sub_launches
]
