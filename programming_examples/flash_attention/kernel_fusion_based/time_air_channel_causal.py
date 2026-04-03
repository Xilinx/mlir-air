#!/usr/bin/env python3
"""
Timing wrapper for air_channel causal attention.
Loads the pre-built ELF from build_peano_causal and times 12 execution runs.
"""
import time
import os
import sys
import numpy as np
from math import sqrt

# Must be run from the build_peano_causal directory (so air.elf is found relative)
# Or pass ELF path as argument

ELF_PATH = os.path.join(os.path.dirname(__file__), "build_peano_causal", "air.elf")
KERNEL_NAME = "main:attention_bf16"

# Causal lit test params
LK = 2048
LKP = 64
LQ = 2048
LQP = 256
DK = 64
DV = 64
NUM_HEADS = 2
NUM_KV_HEADS = 2
N_RUNS = 12
N_WARMUP = 2

from ml_dtypes import bfloat16
INPUT_DATATYPE = OUTPUT_DATATYPE = bfloat16

rng = np.random.default_rng(42)
input_q = (rng.uniform(0, 1, (NUM_HEADS, LQ, DK)) * 0.5 + 0.5).astype(INPUT_DATATYPE)
input_k = (rng.uniform(0, 1, (NUM_KV_HEADS, LK, DK)) * 0.5 + 0.5).astype(INPUT_DATATYPE)
input_v = (rng.uniform(0, 1, (NUM_KV_HEADS, LK, DV)) * 0.5 + 0.5).astype(INPUT_DATATYPE)
input_m = np.zeros((NUM_HEADS, LQ, LK), dtype=INPUT_DATATYPE)
input_q_scaled = (input_q / sqrt(DK)).astype(INPUT_DATATYPE)
output_placeholder = np.zeros((NUM_HEADS, LQ, DV), dtype=OUTPUT_DATATYPE)

# All tensors passed to kernel
all_tensors = [input_q_scaled, input_k, input_v, input_m, output_placeholder]

import pyxrt as xrt

print(f"Loading ELF: {ELF_PATH}")
device = xrt.device(0)
elf = xrt.elf(ELF_PATH)
context = xrt.hw_context(device, elf)
kernel = xrt.ext.kernel(context, KERNEL_NAME)
print(f"ELF loaded OK, kernel: {KERNEL_NAME}")

def run_once(tensors):
    sizes_in_bytes = [a.size * a.itemsize for a in tensors]
    bos = [xrt.ext.bo(device, s) for s in sizes_in_bytes]
    for i, a in enumerate(tensors):
        arr = a.view(np.int16) if a.dtype == bfloat16 else a
        bos[i].write(arr, 0)
        bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    run = xrt.run(kernel)
    for i, bo in enumerate(bos):
        run.set_arg(i, bo)

    t0 = time.perf_counter()
    run.start()
    run.wait2()
    t1 = time.perf_counter()

    for i, a in enumerate(tensors):
        bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    return t1 - t0

print(f"Warming up ({N_WARMUP} runs)...")
for i in range(N_WARMUP):
    t = run_once(all_tensors)
    print(f"  Warmup {i+1}: {t*1000:.3f} ms")

print(f"\nTiming {N_RUNS} runs...")
timings = []
for i in range(N_RUNS):
    t = run_once(all_tensors)
    timings.append(t * 1000)  # convert to ms
    print(f"  Run {i+1}: {t*1000:.3f} ms")

timings_sorted = sorted(timings)
median = timings_sorted[N_RUNS // 2]
mean = sum(timings) / len(timings)
print(f"\n--- air_channel causal timing summary (LK={LK}, LQ={LQ}, LKP={LKP}, LQP={LQP}, NUM_HEADS={NUM_HEADS}) ---")
print(f"All timings (ms): {[f'{t:.3f}' for t in timings]}")
print(f"Min:    {min(timings):.3f} ms")
print(f"Max:    {max(timings):.3f} ms")
print(f"Median: {median:.3f} ms")
print(f"Mean:   {mean:.3f} ms")
