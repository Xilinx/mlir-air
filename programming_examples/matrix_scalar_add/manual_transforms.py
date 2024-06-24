# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import air
import air.compiler.util
from air.ir import *
import air.passmanager

import multi_core_dma


if __name__ == "__main__":
    air_module = multi_core_dma.build_module()
    context = air_module.context

    ################################################
    ## Tiling
    ################################################

    pm = air.passmanager.PassManager.parse(
        air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE,
        context=context,
    )
    pm.run(air_module.operation)
    with open("air_input.mlir", "w") as f:
        f.write(str(air_module))

    ################################################
    ## Binding scf.paralell to air hierarchies
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                #"buffer-results-to-out-params",
                #"air-par-to-herd{depth=1}",
                #"air-par-to-launch{has-air-segment=true}",
                #"air-copy-to-dma",
                #"canonicalize",
                #"cse",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline, context=context)
    pm.run(air_module.operation)

    with open("air_sync.mlir", "w") as f:
        f.write(str(air_module))

    ################################################
    ## Extract event dependency and optimize schedule
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                #"air-dependency",
                #"air-dependency-schedule-opt",
                #"air-specialize-dma-broadcast",
                #"air-dma-to-channel",
                #"canonicalize",
                #"cse",
                #"air-dependency-canonicalize",
                #"canonicalize",
                #"cse",
                #"air-label-scf-for-to-ping-pong",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline, context=context)
    pm.run(air_module.operation)
    # Not sure why parsing the ir solves the segmentation fault...
    air_module = Module.parse(str(air_module), context=context)
    pipeline = (
        "builtin.module("
        + ",".join(
            [
                #"air-ping-pong-transform{keep-memref-dealloc=true}",
                #"air-dealias-memref",
                #"canonicalize",
                #"cse",
                #"air-isolate-async-dma-loop-nests",
                #"air-specialize-channel-wrap-and-stride",
                #"canonicalize",
                #"cse",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline, context=context)
    pm.run(air_module.operation)
    with open("aircc_input.mlir", "w") as f:
        f.write(str(air_module))

    ################################################
    ## Place herd to segment
    ################################################

    air_async_module = Module.parse(str(air_module), context=context)
    col_anchor = 0
    pipeline = (
        "builtin.module("
        + ",".join(
            [
                #"func.func(air-collapse-herd)",
                #"canonicalize",
                #"cse",
                "air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0}",
                "canonicalize",
                "cse",
                "func.func(air-renumber-dma)",
                #"func.func(convert-linalg-to-loops)",
            ]
        )
        + ")"
    )

    pm = air.passmanager.PassManager.parse(pipeline, context=context)
    pm.run(air_module.operation)
    with open("air_placed.mlir", "w") as f:
        f.write(str(air_module))

    # ################################################
    # ## MLIR-AIR to MLIR-AIE
    # ################################################

    air_to_aie_pass = (
        "air-to-aie{row-offset=2 col-offset=0 device=npu1_4col emit-while-loop=false"
    )
    air_to_aie_pass = air_to_aie_pass + "}"
    pipeline = (
        "builtin.module("
        + ",".join(
            [
                air_to_aie_pass,
                "canonicalize",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline, context=context)
    pm.run(air_module.operation)
    with open("aircc_decomp_aiecc.mlir", "w") as f:
        f.write(str(air_module))

    ################################################
    ## MLIR-AIR runtime lowering
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "air-to-std",
                "airrt-to-npu{"
                + "}",
                "canonicalize",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline, context=context)
    pm.run(air_module.operation)
    with open("aie.mlir", "w") as f:
        f.write(str(air_module))