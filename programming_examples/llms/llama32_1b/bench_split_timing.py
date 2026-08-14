"""Verify the 330us 'regression' is XRT prep overhead inside cache.load_and_run's
t_kernel window, not a real kernel slowdown.

Monkey-patches cache.load_and_run to split t_kernel into:
  t_prep   = xrt.run() ctor + N set_arg calls
  t_npu    = run.start() + run.wait2()
Then runs one decode token and reports both per kernel.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from llama_kernel_builder import cache as _cache_mod

# Patch: split the t_kernel window of load_and_run into prep vs npu.
_orig_load_and_run = _cache_mod.KernelCache.load_and_run
_split = {"prep_ms": {}, "npu_ms": {}, "count": {}}


def _patched_load_and_run(self, name, backend_kwargs, *inputs, **kwargs):
    """Wrap load_and_run but ALSO recompute timing by replicating the same
    start/wait split that test_o_gemv_ffn.cpp uses."""
    import filelock
    import pyxrt as xrt
    import numpy as np
    from ml_dtypes import bfloat16
    from air.backend.xrt import XRTBackend

    if name not in self.artifacts:
        return _orig_load_and_run(self, name, backend_kwargs, *inputs, **kwargs)

    static_input_indices = kwargs.get("static_input_indices")
    intermediate_indices = kwargs.get("intermediate_indices")
    output_indices = kwargs.get("output_indices")
    bo_key = kwargs.get("bo_key")

    if name not in self._loaded:
        artifact = self.artifacts[name]
        backend = XRTBackend(**backend_kwargs)
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)
        self._loaded[name] = (backend, invoker)
    backend, _ = self._loaded[name]

    sizes_in_bytes = [a.size * a.itemsize for a in inputs]
    is_elf = self.artifacts[name].output_binary.endswith(".elf")
    static_indices = set(static_input_indices or [])
    intermediate_set = set(intermediate_indices or [])
    _bo_key = bo_key if bo_key is not None else name

    first_call = _bo_key not in self._cached_bos
    if first_call:
        bos = []
        for i, s in enumerate(sizes_in_bytes):
            if is_elf:
                bos.append(xrt.ext.bo(backend.device, s))
            else:
                bos.append(
                    xrt.bo(backend.device, s, xrt.bo.host_only, backend.kernel.group_id(i + 3))
                )
        self._cached_bos[_bo_key] = bos
        if not is_elf and backend.bo_instr is not None:
            backend.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bos = self._cached_bos[_bo_key]

    with filelock.FileLock("/tmp/npu.lock"):
        # Write inputs
        for i, a in enumerate(inputs):
            if i in static_indices and not first_call:
                continue
            if i in intermediate_set and not first_call:
                continue
            if a.dtype == bfloat16:
                a = a.view(np.int16)
            mv = bos[i].map()
            src = np.frombuffer(a, dtype=np.uint8)
            dst = np.frombuffer(mv, dtype=np.uint8, count=len(src))
            np.copyto(dst, src, casting="no")
            bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # SPLIT: prep vs npu
        if is_elf:
            t_prep = time.perf_counter()
            run = xrt.run(backend.kernel)
            for i, bo in enumerate(bos):
                run.set_arg(i, bo)
            t_npu_start = time.perf_counter()
            run.start()
            run.wait2()
            t_npu_end = time.perf_counter()
        else:
            t_prep = time.perf_counter()
            t_npu_start = time.perf_counter()
            h = backend.kernel(3, backend.bo_instr, len(backend.instr_v), *bos)
            h.wait()
            t_npu_end = time.perf_counter()

        prep_ms = (t_npu_start - t_prep) * 1000
        npu_ms = (t_npu_end - t_npu_start) * 1000

        _split["prep_ms"].setdefault(name, []).append(prep_ms)
        _split["npu_ms"].setdefault(name, []).append(npu_ms)
        _split["count"][name] = _split["count"].get(name, 0) + 1

        # Read back
        if output_indices is None:
            readback_set = {len(inputs) - 1}
        else:
            readback_set = set(output_indices)
        for idx in readback_set:
            bos[idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        results = tuple(
            (
                np.frombuffer(bos[i].map(), dtype=inputs[i].dtype, count=inputs[i].size)
                if i in readback_set
                else np.empty(0, dtype=inputs[i].dtype)
            )
            for i, s in enumerate(sizes_in_bytes)
        )
    return results


_cache_mod.KernelCache.load_and_run = _patched_load_and_run


# Argv shim
parser = argparse.ArgumentParser()
parser.add_argument("--use-int4-decode", action="store_true")
parser.add_argument("--n-tokens", type=int, default=30)
args_user = parser.parse_args()


class _Args:
    compile_only = False
    run_only = True
    n_tokens = args_user.n_tokens
    profile = False
    verify = False
    cpu_attn = False
    verbose = False
    prompt = "Write a long story about a dragon flying through forests and mountains."
    synthetic_weights = False
    model = "instruct"
    use_int4_decode = args_user.use_int4_decode
    interactive = False


from llama32_1b_inference import build_session, run_once

session = build_session(_Args)

# Reset stats so weight preload doesn't pollute decode-loop numbers
for k in list(_split["prep_ms"].keys()):
    _split["prep_ms"][k] = []
    _split["npu_ms"][k] = []
    _split["count"][k] = 0

run_once(session, _Args.prompt, n_tokens=_Args.n_tokens,
         profile=False, verify=False, cpu_attn=False, on_token=None)

print(f"\n{'='*70}")
print(f"PER-KERNEL SPLIT: prep (xrt.run + set_arg) vs npu (start+wait2)")
print(f"  int4={_Args.use_int4_decode}, decode-loop only ({_Args.n_tokens} tokens)")
print(f"{'='*70}")
n_gen = _Args.n_tokens  # approximation
for name in sorted(_split["prep_ms"].keys()):
    prep_list = _split["prep_ms"][name][-n_gen * 16:]  # decode-loop last entries
    npu_list = _split["npu_ms"][name][-n_gen * 16:]
    if not prep_list:
        continue
    n = len(prep_list)
    avg_prep = sum(prep_list) / n
    avg_npu = sum(npu_list) / n
    total = avg_prep + avg_npu
    print(
        f"  {name:25s} prep={avg_prep:5.3f}ms  npu={avg_npu:5.3f}ms  total={total:5.3f}ms  (x{n})"
    )
