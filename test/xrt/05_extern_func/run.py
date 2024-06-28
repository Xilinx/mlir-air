# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
import pyxrt as xrt

out_size = 1024

out_size_bytes = out_size * 4

with open("insts.txt", "r") as f:
    instr_text = f.read().split("\n")
    instr_text = [l for l in instr_text if l != ""]
    instr_v = np.array([int(i, 16) for i in instr_text], dtype=np.uint32)

opts_xclbin = "aie.xclbin"
opts_kernel = "MLIR_AIE"

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

bo_instr = xrt.bo(device, len(instr_v) * 4, xrt.bo.cacheable, kernel.group_id(1))
bo_c = xrt.bo(device, out_size_bytes, xrt.bo.host_only, kernel.group_id(3))

bo_instr.write(instr_v, 0)
bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

opcode = 3
h = kernel(opcode, bo_instr, len(instr_v), bo_c)
h.wait()

bo_c.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
output_buffer = bo_c.read(out_size_bytes, 0).view(np.uint32)

errors = 0
for i in range(0, 1024, 4):
    err = output_buffer[i + 0] != 0xDEADBEEF
    err = output_buffer[i + 1] != 0xCAFECAFE
    err = output_buffer[i + 2] != 0x000DECAF
    err = output_buffer[i + 3] != 0x5A1AD000 + i
    if err:
        errors = errors + 1
        print("error at", i)
        print([hex(d) for d in output_buffer[i : i + 4]])

if not errors:
    print("PASS!")
    exit(0)
else:
    print("failed.")
    exit(-1)
