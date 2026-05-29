"""Single-launch standalone modules for the prefill rms_gemms_rope kernel-group.

Exports a STANDALONES registry compatible with cells/common.py:compile_standalone_kernels.
Each entry: (name, build_fn, build_kwargs).
"""

from specs.rms_gemms_rope import SPEC

STANDALONES = [
    (sub.name, sub.builder_ref, sub.build_kwargs) for sub in SPEC.sub_launches
]
