#!/usr/bin/env python3
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Local IR testing workflow for flash attention 4x2 broadcast design.

Takes a .mlir IR file, compiles it with aiecc, and runs on NPU2 via pyxrt.
This bypasses the Python module builder so you can iterate directly on the IR.

Usage:
    # 1. Edit the IR file (npu.air.mlir or a copy)
    # 2. Compile + run:
    python3 test_ir.py npu_4x2_broadcast.mlir

    # Compile only (no device run):
    python3 test_ir.py npu_4x2_broadcast.mlir --compile-only

    # Run with a previously compiled ELF:
    python3 test_ir.py --elf build_cascade/air.elf

    # Custom ELF output name:
    python3 test_ir.py npu_4x2_broadcast.mlir --elf-name my_test.elf
"""

import argparse
import os
import subprocess
import sys
from math import sqrt

import numpy as np
from ml_dtypes import bfloat16
import filelock

# Flash attention dimensions
TSQ = 64
DK = 64
DV = 64
LKP = 64
NQ = 4
NS = 2
CHUNKS = 2


def corr(a, b):
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])


def compile_ir(mlir_file, tmpdir="air_project", elf_name="air.elf", aiecc_path=None):
    """Compile an MLIR IR file using aiecc."""
    peano = os.environ.get("PEANO_INSTALL_DIR", "")
    if not peano:
        print("ERROR: PEANO_INSTALL_DIR not set")
        sys.exit(1)

    if not aiecc_path:
        print("ERROR: aiecc not found in PATH")
        sys.exit(1)

    cmd = [
        aiecc_path, "-v",
        "--no-aiesim", "--no-xchesscc", "--no-xbridge", "--no-compile-host",
        f"--tmpdir={tmpdir}",
        "--generate-full-elf", "--expand-load-pdis",
        f"--full-elf-name={elf_name}",
        "-O", "3",
        "--peano", peano,
        mlir_file,
    ]
    print(f"Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: aiecc failed with exit code {result.returncode}")
        sys.exit(1)
    print(f"Compilation succeeded: {elf_name}")
    return elf_name


def run_on_device(elf_path, kernel_name="main:full_4x2_direct", val_range=3.0):
    """Load ELF and run on NPU2 via pyxrt."""
    import pyxrt as xrt

    total_lk = LKP * CHUNKS * NS

    np.random.seed(42)
    Q = np.random.uniform(0, val_range, (NQ * TSQ, DK)).astype(bfloat16)
    K = np.random.uniform(0, val_range, (total_lk, DK)).astype(bfloat16)
    V = np.random.uniform(0, val_range, (total_lk, DV)).astype(bfloat16)
    Gp_out = np.zeros((NQ * TSQ, DK), dtype=bfloat16)

    args = [Q, K, V, Gp_out]

    print(f"Loading ELF: {elf_path}")
    print(f"Kernel: {kernel_name}")
    print(f"Input shapes: Q={Q.shape}, K={K.shape}, V={V.shape}, Gp_out={Gp_out.shape}")

    with filelock.FileLock("/tmp/npu.lock"):
        device = xrt.device(0)
        elf = xrt.elf(elf_path)
        context = xrt.hw_context(device, elf)
        kernel = xrt.ext.kernel(context, kernel_name)

        sizes_in_bytes = [a.size * a.itemsize for a in args]
        bos = [xrt.ext.bo(device, s) for s in sizes_in_bytes]

        for i, a in enumerate(args):
            data = a.view(np.int16) if a.dtype == bfloat16 else a
            bos[i].write(data, 0)
            bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        run = xrt.run(kernel)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)

        print("Starting kernel...")
        run.start()
        try:
            run.wait2()
            print("Kernel completed successfully!")
        except RuntimeError as e:
            print(f"DEADLOCK: {e}")
            return None

        for i in range(len(args)):
            bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

        results = [bos[i].read(s, 0).view(args[i].dtype) for i, s in enumerate(sizes_in_bytes)]

    npu = results[3].reshape(NQ * TSQ, DK).astype(np.float32)

    # Compute reference
    sqrt_dk = sqrt(DK)
    ref = np.zeros_like(npu)
    for qt in range(NQ):
        Qs = Q[qt * TSQ:(qt + 1) * TSQ].astype(np.float32)
        Kf = K.astype(np.float32)
        Vf = V.astype(np.float32)
        scores = Qs @ Kf.T
        mx = np.max(scores, axis=-1, keepdims=True)
        P = np.exp((scores - mx) / sqrt_dk)
        P = P / np.sum(P, axis=-1, keepdims=True)
        ref[qt * TSQ:(qt + 1) * TSQ] = (P @ Vf).astype(bfloat16).astype(np.float32)

    c = corr(npu, ref)
    print(f"\n  4x2 broadcast correlation: {c:.6f}")
    for qt in range(NQ):
        n = npu[qt * TSQ:(qt + 1) * TSQ]
        r = ref[qt * TSQ:(qt + 1) * TSQ]
        cq = corr(n, r)
        print(f"    q_tile {qt}: corr={cq:.6f}")
    passed = c > 0.99
    print(f"  {'PASS' if passed else 'FAIL'} (threshold: corr > 0.99)")
    return c


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mlir_file", nargs="?", default=None,
                        help="MLIR IR file to compile (e.g. npu_4x2_broadcast.mlir)")
    parser.add_argument("--elf", default=None,
                        help="Pre-compiled ELF to run (skips compilation)")
    parser.add_argument("--elf-name", default="air.elf",
                        help="Output ELF filename (default: air.elf)")
    parser.add_argument("--tmpdir", default="air_project",
                        help="aiecc tmpdir (default: air_project)")
    parser.add_argument("--compile-only", action="store_true",
                        help="Compile only, don't run on device")
    parser.add_argument("--kernel", default="main:full_4x2_direct",
                        help="Kernel name for pyxrt (default: main:full_4x2_direct)")
    parser.add_argument("--val-range", type=float, default=3.0,
                        help="Input value range (default: 3.0)")
    args = parser.parse_args()

    if args.mlir_file is None and args.elf is None:
        parser.error("Must specify either an MLIR file or --elf")

    # Resolve aiecc path before changing directory
    import shutil
    aiecc_path = shutil.which("aiecc") or shutil.which("aiecc.py")

    # Work from build_cascade directory
    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_cascade")
    os.makedirs(build_dir, exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir(build_dir)

    try:
        if args.elf:
            elf_path = args.elf if os.path.isabs(args.elf) else os.path.join(orig_dir, args.elf)
        else:
            mlir_path = args.mlir_file if os.path.isabs(args.mlir_file) else os.path.join(orig_dir, args.mlir_file)
            compile_ir(mlir_path, tmpdir=args.tmpdir, elf_name=args.elf_name, aiecc_path=aiecc_path)
            elf_path = args.elf_name

        if not args.compile_only:
            run_on_device(elf_path, kernel_name=args.kernel, val_range=args.val_range)
    finally:
        os.chdir(orig_dir)


if __name__ == "__main__":
    main()
