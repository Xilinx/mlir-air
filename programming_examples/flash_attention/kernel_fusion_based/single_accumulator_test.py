# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""NPU regression for transpose, overwrite and row-scaled attention GEMMs."""

import argparse
import os
from pathlib import Path
import subprocess

import numpy as np
from ml_dtypes import bfloat16
from air import api as air
from air.api import ops
from air.backend.xrt_runner import XRTRunner


def tiled(matrix):
    # Column-major blocks; each 8x8 intrinsic tile is row-major.
    return matrix.reshape(4, 8, 4, 8).transpose(2, 0, 1, 3).reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfp16", action="store_true")
    args = parser.parse_args()
    flags = ["-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"] if args.bfp16 else []
    source = Path(__file__).resolve().with_suffix(".cc")
    obj = Path("attention_contract.o").resolve()
    subprocess.run(
        [
            str(Path(os.environ["PEANO_INSTALL_DIR"]) / "bin/clang++"),
            "-std=c++20",
            "--target=aie2p-none-unknown-elf",
            "-O2",
            "-c",
            "-I",
            str(Path(os.environ["MLIR_AIE_INSTALL_DIR"]) / "include"),
            "-D__AIE_API_AIE_ADF_HPP__",
            "-DATTENTION_SINGLE_ACCUMULATOR",
            "-DROUND_CONV_EVEN",
            "-Dlqp=32",
            "-Dlkp=32",
            "-Ddk=32",
            "-Ddk_full=64",
            "-Ddv=32",
            str(source),
            "-o",
            str(obj),
        ]
        + flags,
        check=True,
    )
    width = 3 * 32 * 32 + 32
    x = air.tensor([4, width], air.bf16)
    y = air.tensor([4, 1024], air.bf16)
    fn = air.extern("attention_contract", link_with=str(obj), scalars=[air.i32])
    with air.launch(name="attention_contract_test") as launch:

        @launch.body
        def _():
            with air.herd([range(1)], shape=(1,)) as herd:

                @herd.body
                def _(_tx):
                    a = air.alloc([width], air.bf16, scope=herd.private())
                    b = air.alloc([1024], air.bf16, scope=herd.private())
                    for mode in air.sequential(4):
                        ops.load(a, x[mode : mode + 1, :])
                        fn(a, b, mode)
                        ops.store(b, y[mode : mode + 1, :])

    rng = np.random.default_rng(27)
    inputs, expected = [], []
    for mode in range(4):
        a = (rng.integers(-1, 2, (32, 32)) / 4).astype(bfloat16)
        b = (rng.integers(-1, 2, (32, 32)) / 4).astype(bfloat16)
        c = (rng.integers(-2, 3, (32, 32)) / 4).astype(bfloat16)
        rows = np.resize(np.array([0.5, 1, 2, 0], dtype=bfloat16), 32)
        product = a.astype(np.float32) @ b.astype(np.float32)
        if mode == 2:
            c[:] = np.nan  # zero_c must never propagate old destination values.
            result = product
        elif mode == 3:
            result = product + c.astype(np.float32) * rows[:, None].astype(np.float32)
        else:
            result = product + c.astype(np.float32)
        # QK arrives as K^T in n-major tiles. Pretransposition changes only
        # the 8x8 tile contents, while G@V uses ordinary column-major blocks.
        bt = tiled(b.T)
        if mode in (1, 2):
            bt = bt.reshape(-1, 8, 8).transpose(0, 2, 1).reshape(-1)
        elif mode == 3:
            bt = tiled(b)
        inputs.append(np.concatenate([tiled(a), bt, tiled(c), rows]))
        expected.append(tiled(result.astype(bfloat16)))
    runner = XRTRunner(
        omit_while_true_loop=False,
        instance_name="attention_contract_test",
        stack_size=4096,
    )
    return runner.run_test(
        launch.build(target="npu2"),
        inputs=[np.stack(inputs)],
        expected_outputs=[np.stack(expected)],
        rtol=0,
        atol=0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
