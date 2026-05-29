"""Shared helpers for prefill ablation cells.

Lifted (and extended for two-backend support) from Plan 1's
ablation/cells/common.py. The original Plan 1 file is read-only.

- compile_standalone_kernels(cache, group_name, registry, backend_preset):
    Compile every standalone in `registry` into `cache`, using the actual
    public func name extracted from the MLIR module as instance_name.
- _extract_public_func_name(mlir_text): regex over the module string.
- _share_bo(cache, src_key, src_slot, dst_key, dst_slot): alias cached BOs
  for Cell C's baton-pass.
- standalone_backend_kwargs(backend_preset, verbose): returns backend kwargs
  with instance_name removed (set per-kernel by compile_standalone_kernels).
"""

import re

from air.ir import Context as MLIRContext

from kernel_builder.cache import KernelCache


def _extract_public_func_name(mlir_text):
    """Find the first non-private `func.func @<name>` in the module text."""
    for line in mlir_text.split("\n"):
        if "func.func @" in line and "private" not in line:
            m = re.search(r"@(\w+)", line)
            if m:
                return m.group(1)
    raise ValueError("no public func.func found in module")


def standalone_backend_kwargs(backend_preset, verbose=False):
    """Backend kwargs with instance_name removed (set per-kernel by caller)."""
    base = {**backend_preset, "verbose": verbose}
    base.pop("instance_name", None)
    return base


def compile_standalone_kernels(
    cache: KernelCache, group_name: str, registry, backend_preset
):
    """Compile every standalone in `registry` into `cache` under names
    f"{group_name}__{name}". Skip any kernel already in cache.artifacts.

    Each registry entry: (name, build_fn, build_kwargs).
    """
    for name, build_fn, kwargs in registry:
        kernel_name = f"{group_name}__{name}"
        if kernel_name in cache.artifacts:
            continue
        with MLIRContext():
            mlir_module = build_fn(**kwargs)
            public_func = _extract_public_func_name(str(mlir_module))
        be = standalone_backend_kwargs(backend_preset, verbose=cache.verbose)
        be["instance_name"] = public_func
        cache.compile_and_cache(kernel_name, mlir_module, be)
    cache._save_manifest()


def _share_bo(cache, src_key, src_slot, dst_key, dst_slot):
    """Replace cached BO at (dst_key, dst_slot) with the same xrt.bo as
    (src_key, src_slot). Only valid after both kernels' first call has
    materialized BOs."""
    src_bos = cache._cached_bos[src_key]
    dst_bos = cache._cached_bos[dst_key]
    dst_bos[dst_slot] = src_bos[src_slot]


def main():
    """python3 -m cells.common — compile both kernel-groups' standalones."""
    from kernel_builder.backend_presets import RMS_GEMMS_ROPE_BACKEND, O_FFN_BACKEND
    from standalone_builders.rms_gemms_rope import STANDALONES as RMS_STD
    from standalone_builders.o_ffn import STANDALONES as O_STD

    cache = KernelCache(cache_dir="standalone_cache", verbose=True)
    cache.load_manifest()
    compile_standalone_kernels(cache, "rms_gemms_rope", RMS_STD, RMS_GEMMS_ROPE_BACKEND)
    compile_standalone_kernels(cache, "o_ffn", O_STD, O_FFN_BACKEND)
    print(f"Compiled {len(cache.artifacts)} standalone ELFs.")


if __name__ == "__main__":
    main()
