import numpy as np
import air.backend.xrt as xrt_backend
import os
import os.path
import filelock
import argparse

from data_config import *
import single_core_dma
import multi_core_dma

KERNEL_NAME = "MLIR_AIE"

INOUT_DATATYPE = np.uint32
INOUT_ELEM_SIZE = np.dtype(INOUT_DATATYPE).itemsize
INOUT_SIZE = IMAGE_SIZE[0] * IMAGE_SIZE[1]
INOUT_SIZE_BYTES = INOUT_SIZE * INOUT_ELEM_SIZE

verbose = False

if __name__ == "__main__":
    input_a = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    input_b = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    for i in range(INOUT_SIZE):
        input_a[i] = i + 0x1000
        input_b[i] = 0x00DEFACED

    backend = xrt_backend.XRTBackend(verbose=verbose, xclbin="aie.xclbin", insts="insts.txt")

    print("=========================== INPUT START ===========================")
    for i in range(IMAGE_HEIGHT):
        row = input_b[i * IMAGE_WIDTH : (i + 1) * IMAGE_WIDTH]
        for val in row:
            val = val & 0xFFFF
            print(f"{val:04x}", end=" ")
        print("")
    print("=========================== INPUT END ===========================")

    # run the module
    with filelock.FileLock("/tmp/npu.lock"):
        addone = backend.load(None)
        (_, output_b) = addone(input_a, input_b)

    backend.unload()

    print("=========================== OUTPUT START ===========================")
    for i in range(IMAGE_HEIGHT):
        row = output_b[i * IMAGE_WIDTH : (i + 1) * IMAGE_WIDTH]
        for val in row:
            val = val & 0xFFFF
            print(f"{val:04x}", end=" ")
        print("")
    print("=========================== OUTPUT END ===========================")

    # check output, should have all values incremented
    errors = 0
    for i in range(INOUT_SIZE):
        rb = output_b[i]

        row = i / IMAGE_WIDTH
        col = i % IMAGE_WIDTH

        # value should have been updated
        if not (rb == 0x1000 + i + 1):
            # print(f"IM {i} [{col}, {row}] should be 0x{i:x}, is 0x{rb:x}\n")
            errors += 1

    if errors == 0:
        print("PASS!")
        exit(0)
    else:
        print("failed. errors=", errors)
        exit(-1)