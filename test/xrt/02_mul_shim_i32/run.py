# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
import pyxrt as xrt
import sys

in_size = 32*32
out_size = 32*32

in_size_bytes = in_size * 4
out_size_bytes = out_size * 4


opts_xclbin = sys.argv[1]
opts_kernel = 'MLIR_AIE'
opts_insts = opts_xclbin.removesuffix('.xclbin') + ".insts.txt"

with open(opts_insts, 'r') as f:
    instr_text = f.read().split('\n')
    instr_text = [l for l in instr_text if l != '']
    instr_v = np.array([int(i,16) for i in instr_text], dtype=np.uint32)

device = xrt.device(0)
xclbin = xrt.xclbin(opts_xclbin)
kernels = xclbin.get_kernels()
try:
    xkernel = [k for k in kernels if opts_kernel in k.get_name()][0]
except:
    print(f"Kernel '{opts_kernel}' not found in '{opts_xclbin}'")
    exit(-1)

device.register_xclbin(xclbin)
context = xrt.hw_context(device, xclbin.get_uuid())
kernel = xrt.kernel(context, xkernel.get_name())

bo_instr = xrt.bo(device, len(instr_v)*4, xrt.bo.cacheable, kernel.group_id(0))
bo_a = xrt.bo(device, in_size_bytes, xrt.bo.host_only, kernel.group_id(2))
bo_b = xrt.bo(device, in_size_bytes, xrt.bo.host_only, kernel.group_id(3))
bo_c = xrt.bo(device, out_size_bytes, xrt.bo.host_only, kernel.group_id(4))

bo_instr.write(instr_v, 0)
bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

input_a = np.arange(in_size, dtype=np.uint32)
bo_a.write(input_a, 0)
bo_a.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

input_b = np.arange(in_size, dtype=np.uint32)
bo_b.write(input_b, 0)
bo_b.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

h = kernel(bo_instr, len(instr_v), bo_a, bo_b, bo_c)
h.wait()

bo_c.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
output_buffer = bo_c.read(out_size_bytes, 0).view(np.uint32)
print("input:", input_a)
print("input:", input_b)
print("output:", output_buffer)

ref = input_a * input_b
if np.equal(ref, output_buffer).all():
    print("PASS!")
    exit(0)
else:
    print("failed.")
    exit(-1)
