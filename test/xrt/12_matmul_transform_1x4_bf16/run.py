# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
import pyxrt as xrt
import sys
import time

M = 128
N = 128
K = 256

in_a_size = (M, K)
in_b_size = (K, N)
out_size = (M, N)

in_a_size_bytes = (in_a_size[0] * in_a_size[1]) * 2
in_b_size_bytes = (in_b_size[0] * in_b_size[1]) * 2
out_size_bytes = (out_size[0] * out_size[1]) * 4

with open("insts.txt", "r") as f:
    instr_text = f.read().split("\n")
    instr_text = [l for l in instr_text if l != ""]
    instr_v = np.array([int(i, 16) for i in instr_text], dtype=np.uint32)

opts_xclbin = sys.argv[1]
opts_kernel = "MLIR_AIE"

device = xrt.device(0)
xclbin = xrt.xclbin(opts_xclbin)
kernels = xclbin.get_kernels()
try:
    xkernel = [k for k in kernels if opts_kernel in k.get_name()][0]
except:
    print(f"Kernel '{opts_kernel}' not found in '{opts_xclbin}'")
    exit(-1)

print("Running...")

device.register_xclbin(xclbin)
context = xrt.hw_context(device, xclbin.get_uuid())
kernel = xrt.kernel(context, xkernel.get_name())

bo_instr = xrt.bo(device, len(instr_v) * 4, xrt.bo.cacheable, kernel.group_id(1))
bo_a = xrt.bo(device, in_a_size_bytes, xrt.bo.host_only, kernel.group_id(3))
bo_b = xrt.bo(device, in_b_size_bytes, xrt.bo.host_only, kernel.group_id(4))
bo_c = xrt.bo(device, out_size_bytes, xrt.bo.host_only, kernel.group_id(5))

bo_instr.write(instr_v, 0)
bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

input_a = np.random.rand(*in_a_size).astype(bfloat16)
bo_a.write(input_a.view(np.int16), 0)
bo_a.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

macs = 2.0 * M * K * N

input_b = np.random.rand(*in_b_size).astype(bfloat16)
bo_b.write(input_b.view(np.int16), 0)
bo_b.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

niter = 1
npu_total_time = 0
for i in range(0, niter):

    start = time.time()
    opcode = 3
    h = kernel(opcode, bo_instr, len(instr_v), bo_a, bo_b, bo_c)
    h.wait()
    stop = time.time()

    t = stop - start
    print("Execution time:", t)
    npu_total_time = npu_total_time + t

    bo_c.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    output_buffer = bo_c.read(out_size_bytes, 0).view(np.float32)
print("macs:", macs)
print("Avg NPU gflops:", macs / (1e9 * npu_total_time / niter))

print("input a:", input_a)
print("input b:", input_b)
print("output:", output_buffer)

ref = (input_a @ input_b).reshape(-1)
print("reference:", ref)

err = 0
for i in range(0, len(ref)):
    if not np.allclose(output_buffer[i], ref[i], 0.01):
        # print(i, output_buffer[i], "!=", ref[i])
        err = err + 1
if not err:
    print("PASS!")
    exit(0)
else:
    print("failed. errors =", err)
    exit(-1)
