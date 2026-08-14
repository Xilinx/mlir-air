"""Time o_gemv_ffn_int4 in a tight loop (no CPU work between iters), via
the same XRT path cache.load_and_run uses, but with the SAME timing window
test_o_gemv_ffn.cpp uses (start+wait2 only). If this matches the 1352us
standalone number, then the 1653us we see in e2e is genuinely due to
inter-kernel idle (e.g. CPU attention between calls)."""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from ml_dtypes import bfloat16
import filelock
import pyxrt as xrt
from air.backend.xrt import XRTBackend

from llama_kernel_builder.cache import KernelCache
from llama_kernel_builder.backend_presets import OGF_INT4_BACKEND

cache = KernelCache("decode_kernel_cache", verbose=False)
cache.load_manifest()

backend_kwargs = OGF_INT4_BACKEND
artifact = cache.artifacts["o_gemv_ffn_int4"]
backend = XRTBackend(**backend_kwargs)
with filelock.FileLock("/tmp/npu.lock"):
    backend.load(artifact)

# Build dummy BOs matching the o_gemv_ffn_int4 ABI (15 args, sizes for emb=2048, hidden=8192).
emb, hidden, gs, m_tile, k_chunk, n_cores = 2048, 8192, 128, 8, 2048, 8


def packed_bytes(M, K):
    n_gpc = k_chunk // gs
    tile_bytes = m_tile * (k_chunk // 2) + n_gpc * m_tile * 2 + n_gpc * m_tile
    M_per_core = M // n_cores
    M_div = M_per_core // m_tile
    K_div = K // k_chunk
    total_tiles = n_cores * M_div * K_div
    return total_tiles * tile_bytes


sizes_bytes = [
    packed_bytes(emb, emb),                   # arg0 wo_packed
    emb * 2,                                  # arg1 attn_out (bf16)
    emb * 2,                                  # arg2 dead
    emb * 2,                                  # arg3 x_residual
    emb * 2,                                  # arg4 dead
    emb * 2,                                  # arg5 dead
    2 * emb * 2,                              # arg6 packed_rms
    packed_bytes(2 * hidden, emb),            # arg7 gateup_packed
    hidden * 2,                               # arg8 dead
    hidden * emb * 2,                         # arg9 dead
    hidden * 2,                               # arg10 dead
    hidden * 2,                               # arg11 swiglu
    packed_bytes(emb, hidden),                # arg12 wdown_packed
    emb * 2,                                  # arg13 dead
    emb * 2,                                  # arg14 output
]
print(f"Allocating {len(sizes_bytes)} BOs (total {sum(sizes_bytes)/1024/1024:.1f} MB)...")
bos = [xrt.ext.bo(backend.device, s) for s in sizes_bytes]
print(f"  allocated.")

N_WARMUP = 20
N_ITERS = 100

# Tight-loop bench: replicate test_o_gemv_ffn.cpp pattern exactly.
times_us = []
with filelock.FileLock("/tmp/npu.lock"):
    # Warmup
    for _ in range(N_WARMUP):
        run = xrt.run(backend.kernel)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)
        run.start()
        run.wait2()
    # Timed iters: time only start+wait2 (like the standalone bench)
    for _ in range(N_ITERS):
        run = xrt.run(backend.kernel)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)
        t0 = time.perf_counter()
        run.start()
        run.wait2()
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)

import statistics
print(f"\nTight-loop o_gemv_ffn_int4 (start+wait2 only, no CPU work between):")
print(f"  avg = {statistics.mean(times_us):.1f} us")
print(f"  min = {min(times_us):.1f} us")
print(f"  max = {max(times_us):.1f} us")
print(f"  p50 = {statistics.median(times_us):.1f} us")
print(f"  p95 = {sorted(times_us)[int(len(times_us)*0.95)]:.1f} us")
print(f"\nReference points:")
print(f"  standalone (test_o_gemv_ffn.cpp, no CPU work between): 1352 us")
print(f"  e2e (decode loop, CPU attn between calls):            1653 us")

# Second test: with a tiny CPU-busy delay between iters to mimic CPU attention.
import numpy as np
DELAY_MS = 1.5
fake_k = np.zeros((8, 2048, 64), dtype=np.float32)
fake_q = np.zeros((32, 64), dtype=np.float32)

times_with_delay_us = []
with filelock.FileLock("/tmp/npu.lock"):
    for _ in range(N_WARMUP):
        run = xrt.run(backend.kernel)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)
        run.start()
        run.wait2()
    for _ in range(N_ITERS):
        # Simulate CPU attention work between launches (matmul-ish)
        _ = (fake_q @ fake_k[0].T)  # tiny CPU compute
        run = xrt.run(backend.kernel)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)
        t0 = time.perf_counter()
        run.start()
        run.wait2()
        t1 = time.perf_counter()
        times_with_delay_us.append((t1 - t0) * 1e6)

print(f"\nWith CPU work (~tiny matmul) between iters:")
print(f"  avg = {statistics.mean(times_with_delay_us):.1f} us")
print(f"  min = {min(times_with_delay_us):.1f} us")
print(f"  p50 = {statistics.median(times_with_delay_us):.1f} us")

# Third: explicit sleep between iters
times_sleep_us = []
with filelock.FileLock("/tmp/npu.lock"):
    for _ in range(N_WARMUP):
        run = xrt.run(backend.kernel)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)
        run.start()
        run.wait2()
    for _ in range(N_ITERS):
        time.sleep(0.0015)  # 1.5 ms sleep mimics CPU attn latency at full ctx
        run = xrt.run(backend.kernel)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)
        t0 = time.perf_counter()
        run.start()
        run.wait2()
        t1 = time.perf_counter()
        times_sleep_us.append((t1 - t0) * 1e6)

print(f"\nWith time.sleep(1.5ms) between iters (mimics NPU idle):")
print(f"  avg = {statistics.mean(times_sleep_us):.1f} us")
print(f"  min = {min(times_sleep_us):.1f} us")
print(f"  p50 = {statistics.median(times_sleep_us):.1f} us")
