# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from bfloat16 import bfloat16
import pyxrt as xrt
import sys
import time
import os
import os.path
import argparse
from shim_dma_2d import *

KERNEL_NAME = "MLIR_AIE"

INOUT_DATATYPE = np.uint32
INOUT_ELEM_SIZE = np.dtype(INOUT_DATATYPE).itemsize
INOUT_SIZE = IMAGE_SIZE[0] * IMAGE_SIZE[1]
INOUT_SIZE_BYTES = INOUT_SIZE * INOUT_ELEM_SIZE


def parse_and_check_args():
    parser = argparse.ArgumentParser(
        prog="run.py", description="Runs the program on the npu and checks the output."
    )
    parser.add_argument("xclbin")
    parser.add_argument("insts")

    args = parser.parse_args()

    if os.path.isfile(args.xclbin) and os.access(args.xclbin, os.R_OK):
        print("Found xcl bin...")
    else:
        print(f"Failed to find xclbin file: {args.xclbin}")
        exit(-1)

    if os.path.isfile(args.insts) and os.access(args.insts, os.R_OK):
        print("Found insts...")
    else:
        print(f"Failed to find insts file: {args.insts}")
        exit(-1)
    return (args.xclbin, args.insts)


def main():
    (xclbin_path, insts_path) = parse_and_check_args()

    with open(insts_path, "r") as f:
        instr_text = f.read().split("\n")
    instr_text = [l for l in instr_text if l != ""]
    instr_v = np.array([int(i, 16) for i in instr_text], dtype=INOUT_DATATYPE)

    device = xrt.device(0)
    xclbin = xrt.xclbin(xclbin_path)
    kernels = xclbin.get_kernels()
    try:
        xkernel = [k for k in kernels if KERNEL_NAME in k.get_name()][0]
    except:
        print(f"Kernel '{KERNEL_NAME}' not found in '{xclbin}'")
        exit(-1)

    print("Running...")

    device.register_xclbin(xclbin)
    context = xrt.hw_context(device, xclbin.get_uuid())
    kernel = xrt.kernel(context, xkernel.get_name())

    bo_instr = xrt.bo(device, len(instr_v) * 4, xrt.bo.cacheable, kernel.group_id(0))
    bo_in = xrt.bo(device, INOUT_SIZE_BYTES, xrt.bo.host_only, kernel.group_id(2))
    bo_out = xrt.bo(device, INOUT_SIZE_BYTES, xrt.bo.host_only, kernel.group_id(2))

    bo_instr.write(instr_v, 0)
    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    input_a = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    output_a = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    for i in range(INOUT_SIZE):
        input_a[i] = i + 0x1000
        output_a[i] = 0x00DEFACED
    bo_in.write(input_a, 0)
    bo_in.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    bo_out.write(output_a, 0)
    bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    h = kernel(bo_instr, len(instr_v), bo_in, bo_out)
    h.wait()

    bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    output_buffer = bo_out.read(INOUT_SIZE_BYTES, 0).view(INOUT_DATATYPE)

    # check output, should have the top left filled in
    errors = 0
    for i in range(INOUT_SIZE):
        rb = output_buffer[i]

        row = i / IMAGE_WIDTH
        col = i % IMAGE_WIDTH

        if row < TILE_HEIGHT and col < TILE_WIDTH:
            # value should have been updated
            if not (rb == 0x1000 + i):
                print(f"IM {i} [{col}, {row}] should be 0x{i:x}, is 0x{rb:x}\n")
                errors += 1
        else:
            # value should stay unchanged
            if rb != 0x00DEFACED:
                print(
                    f"IM {i} [{col}, {row}] should be 0xdefaced, is 0x{rb:x}\n",
                    i,
                    col,
                    row,
                    rb,
                )
                errors += 1

    if errors == 0:
        print("PASS!")
        exit(0)
    else:
        print("failed. errors=", errors)
        exit(-1)


if __name__ == "__main__":
    main()
