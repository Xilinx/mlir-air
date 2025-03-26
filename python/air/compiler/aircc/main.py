# ./python/air/compiler/aircc/main.py -*- Python -*-
#
# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
aircc - AIR compiler driver for MLIR tools
"""

import os
import platform
import sys
import subprocess
from joblib import Parallel, delayed
import shutil

from air.passmanager import PassManager
from air.ir import Module, Context, Location
from air.dialects import air as airdialect

import air.compiler.aircc.cl_arguments as cl_arguments
from air.compiler.aircc.configure import *

import aie.compiler.aiecc.main as aiecc


def get_L2_splitting_analysis_pass():
    L2_SPLITTING_PASSES = [
        "func.func(air-split-l2-memref)",
        "canonicalize",
        "cse",
        "air-isolate-async-dma-loop-nests",
        "canonicalize",
        "cse",
    ]
    return L2_SPLITTING_PASSES


def get_air_optimization_pass(
    device,
    omit_pingpong=False,
    lower_linalg_to_func=None,
    air_loop_fusion=False,
    omit_auto_broadcast=False,
    channel_multiplexing=[],
):
    OPTIMIZATION_PASSES = [
        "air-dependency",
    ]
    if not omit_auto_broadcast:
        OPTIMIZATION_PASSES += [
            "air-dependency-schedule-opt",
            "air-specialize-dma-broadcast",
        ]
    OPTIMIZATION_PASSES += [
        "air-dma-to-channel",
        "canonicalize",
        "cse",
        "air-dependency-canonicalize",
        "canonicalize",
        "cse",
        "air-isolate-async-dma-loop-nests",
        "canonicalize",
        "cse",
    ]
    if len(channel_multiplexing) != 0:
        OPTIMIZATION_PASSES += [
            "air-fuse-channels{aggressive-mode="
            + ",".join(s for s in channel_multiplexing)
            + "}",
        ]
    else:
        OPTIMIZATION_PASSES += [
            "air-fuse-channels",
        ]
    OPTIMIZATION_PASSES += [
        "canonicalize",
        "cse",
    ]
    if "npu_1col" not in device:
        OPTIMIZATION_PASSES += get_L2_splitting_analysis_pass()
    if air_loop_fusion:
        OPTIMIZATION_PASSES += [
            "func.func(air-loop-fusion)",
        ]
    else:
        OPTIMIZATION_PASSES += [
            "func.func(air-fuse-alloc-dealloc)",
            "func.func(air-shrink-memref-sizes-by-access)",
        ]
    if not omit_pingpong:
        OPTIMIZATION_PASSES += [
            "air-label-scf-for-to-ping-pong",
            "air-ping-pong-transform",
            "canonicalize",
            "cse",
        ]
    if lower_linalg_to_func != None:
        OPTIMIZATION_PASSES += [
            "air-linalg-to-func{link-with=" + f"{lower_linalg_to_func}" + "}",
        ]
    else:
        OPTIMIZATION_PASSES += [
            "func.func(convert-linalg-to-loops)",
        ]

    OPTIMIZATION_PASSES += [
        "func.func(air-opt-memtile-dma-bds{" + f"device={device}" + "})",
        "canonicalize",
        "cse",
    ]
    return OPTIMIZATION_PASSES


def emit_wrapper(herd_name="segment", include_name="aie.inc"):
    s = """// generated by aircc, do not edit
#include "stdio.h"
#include "assert.h"
#include "air_host.h"
#include "air_host_impl.h"

namespace air {
namespace segments {
"""
    s = s + f"namespace {herd_name} {{\n"
    s = s + f'#include "{include_name}"'
    s = (
        s
        + """
}
}
}
"""
    )
    s = s + f"using namespace air::segments::{herd_name};"
    s = (
        s
        + """
extern "C" {
"""
    )
    s = s + f"air_rt_aie_functions_t __airrt_{herd_name}_aie_functions {{"
    s = (
        s
        + """
  .configure_cores = &mlir_aie_configure_cores,
  .configure_switchboxes = &mlir_aie_configure_switchboxes,
  .initialize_locks = &mlir_aie_initialize_locks,
  .configure_dmas = &mlir_aie_configure_dmas,
  .start_cores = &mlir_aie_start_cores
};
}
"""
    )
    return s


def do_call(command):
    global opts
    if opts.verbose:
        print(" ".join(command))
    ret = subprocess.call(command)
    if ret != 0:
        print("Error encountered while running: " + " ".join(command))
        sys.exit(1)


def do_run(command):
    global opts
    if opts.verbose:
        print(" ".join(command))
    ret = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    return ret


def run_passes(pass_pipeline, mlir_module, opts, outputfile=None):
    if opts.verbose:
        print("Running:", pass_pipeline)
    PassManager.parse(pass_pipeline).run(mlir_module.operation)
    if outputfile:
        with open(outputfile, "w") as g:
            g.write(str(mlir_module))


def lower_airrt_to_airhost(air_to_aie_module, air_placed_module, air_mlir_filename):
    pass_pipeline = "air-split-devices{"
    pass_pipeline = pass_pipeline + f"output-prefix={opts.tmpdir}/" + "}"
    run_passes("builtin.module(" + pass_pipeline + ")", air_to_aie_module, opts)

    # lower the airrt control program to llvm dialect

    airrt_module = Module.parse(str(air_to_aie_module))
    aie_ctrl_airrt = opts.tmpdir + "/airrt." + air_mlir_filename
    pass_pipeline = ",".join(
        [
            "convert-vector-to-llvm",
            "convert-math-to-llvm",
            "func.func(air-label-broadcast-channel-with-tile)",
            "lower-affine",
            "func.func(air-opt-shim-dma-bds{device=" + opts.device + "})",
            "air-to-std",
            "air-lower-linalg-tensors",
            "canonicalize",
            "cse",
        ]
    )
    run_passes(
        "builtin.module(" + pass_pipeline + ")", airrt_module, opts, aie_ctrl_airrt
    )

    aie_ctrl = opts.tmpdir + "/aie_ctrl." + air_mlir_filename
    pass_pipeline = ",".join(["airrt-to-llvm", "one-shot-bufferize"])
    run_passes("builtin.module(" + pass_pipeline + ")", airrt_module, opts, aie_ctrl)

    aie_ctrl_refback = opts.tmpdir + "/refback." + air_mlir_filename
    pass_pipeline = ",".join(
        [
            "convert-vector-to-llvm",
            "convert-math-to-llvm",
            "func.func(air-label-broadcast-channel-with-tile)",
            "lower-affine",
            "func.func(air-opt-shim-dma-bds{device=" + opts.device + "})",
            "air-to-std",
            "air-lower-linalg-tensors",
            "canonicalize",
            "cse",
            "airrt-to-llvm",
            "canonicalize",
            "cse",
        ]
    )
    run_passes(
        "builtin.module(" + pass_pipeline + ")",
        Module.parse(str(air_placed_module)),
        opts,
        aie_ctrl_refback,
    )

    aie_ctrl_llvm = opts.tmpdir + "/llvm." + air_mlir_filename
    pass_pipeline = ",".join(
        [
            "expand-strided-metadata",
            "lower-affine",
            "convert-scf-to-cf",
            "finalize-memref-to-llvm",
            "convert-func-to-llvm",
            "convert-arith-to-llvm",
            "convert-cf-to-llvm",
            "canonicalize",
            "cse",
        ]
    )
    run_passes(
        "builtin.module(" + pass_pipeline + ")", airrt_module, opts, aie_ctrl_llvm
    )

    # compile the llvm dialect into a .o object file

    aie_ctrl_llvm_ir = opts.tmpdir + "/" + air_mlir_filename + ".ll"
    do_call(
        ["aie-translate", "--mlir-to-llvmir", aie_ctrl_llvm, "-o", aie_ctrl_llvm_ir]
    )

    aie_ctrl_llvm_opt_bc = opts.tmpdir + "/" + air_mlir_filename + ".opt.bc"
    do_call(["opt", "-O3", aie_ctrl_llvm_ir, "-o", aie_ctrl_llvm_opt_bc])

    aie_ctrl_llvm_opt_ir = opts.tmpdir + "/" + air_mlir_filename + ".opt.ll"
    do_call(["llvm-dis", aie_ctrl_llvm_opt_bc, "-o", aie_ctrl_llvm_opt_ir])

    aie_ctrl_obj = opts.tmpdir + "/" + air_mlir_filename + ".o"
    llc_target = None
    if "x86_64" in opts.host_target:
        llc_target = "x86-64"
    elif "aarch64" in opts.host_target:
        llc_target = "aarch64"
    elif opts.host_target:
        print("Unhandled llc host target: '" + opts.host_target + "'")
    do_call(
        ["llc", "-O3", "--filetype=obj", "--relocation-model=pic"]
        + (["-march=" + llc_target] if llc_target else [])
        + [aie_ctrl_llvm_opt_ir, "-o", aie_ctrl_obj]
    )

    # make aie elf files and host .o files for each herd in the program

    t = do_run(["air-translate", "--airrt-generate-json", aie_ctrl_airrt])
    module_meta = eval(t.stdout)
    segments = [module_meta[segment]["sym_name"] for segment in module_meta]
    obj_files = [aie_ctrl_obj]
    for segment in segments:
        if opts.verbose:
            print("Compiling segment:", segment)

        # build the elf files for the segment

        # herd_file = opts.tmpdir+'/aie.'+herd+'.mlir'
        segment_file = opts.tmpdir + "/aie." + segment + ".mlir"
        aiecc_file = opts.tmpdir + "/aiecc." + segment + ".mlir"
        aiecc_dir = opts.tmpdir + "/" + segment

        do_call(
            [
                "air-opt",
                segment_file,
                "-air-lower-linalg-tensors",
                "-lower-affine",
                "-canonicalize",
                "-cse",
                "-o",
                aiecc_file,
            ]
        )

        # set host target for aiecc
        if "x86_64" in platform.uname()[5]:
            aiecc_target = "x86_64-amd-linux-gnu"
        else:
            aiecc_target = "aarch64-linux-gnu"
        aiecc_target = opts.host_target if opts.host_target else aiecc_target

        # run aiecc to make the elf and configuration files
        sysroot = opts.sysroot if opts.sysroot else "/"
        do_call(
            ["aiecc.py"]
            + (["-v"] if opts.verbose else [])
            + ["--sysroot", sysroot]
            + ["--host-target", aiecc_target]
            + ["--tmpdir", aiecc_dir]
            + ["--no-aiesim"]
            + ["--compile-host"]
            + ["--xbridge" if opts.xbridge else "--no-xbridge"]
            + ["--xchesscc" if opts.xchesscc else "--no-xchesscc"]
            + [aiecc_file]
        )

        inc_file = opts.tmpdir + "/" + air_mlir_filename + "." + segment + ".inc"
        cpp_file = opts.tmpdir + "/" + air_mlir_filename + "." + segment + ".cpp"
        obj_file = opts.tmpdir + "/" + air_mlir_filename + "." + segment + ".o"

        # compile the libxaie configuration functions generated by aie-translate

        do_call(["cp", aiecc_dir + "/aie_inc.cpp", inc_file])

        with open(cpp_file, "w") as f:
            f.write(emit_wrapper(segment, inc_file))

        cmd = [opts.cc, "-std=c++11", "-g", "-I."]

        # set flags for cross-compilation
        cmd += ["--sysroot=%s" % opts.sysroot] if opts.sysroot else []
        if opts.sysroot and "aarch64-linux-gnu" in opts.host_target:
            cmd += ["--gcc-toolchain=%s/usr" % opts.sysroot]
        cmd += ["--target=%s" % opts.host_target] if opts.host_target else []

        # air runtime include path
        thispath = os.path.dirname(os.path.realpath(__file__))
        cmd += [f"-I{thispath}/../../../../runtime_lib/airhost/include"]

        # aie runtime include path
        if "x86_64" in aiecc_target:
            cmd += [f"-I{aiecc_path}/runtime_lib/x86_64/test_lib/include"]
        if "aarch64" in aiecc_target:
            cmd += [f"-I{aiecc_path}/runtime_lib/aarch64/test_lib/include"]

        # libxaie include path
        cmd += [f"-I{libxaie_path}/include"]
        cmd += [f"-I{rocm_path}/../../../include"]
        cmd += ["-DLIBXAIENGINEV2"]
        cmd += ["-DAIE_LIBXAIE_ENABLE", "-fPIC", "-c"]
        cmd += ["-o", obj_file, cpp_file]
        do_call(cmd)

        obj_files.append(obj_file)

    # combine the host side .o files generated above into a single library

    lib_file = air_mlir_filename + (".so" if opts.shared else ".a")
    lib_file = opts.tmpdir + "/" + lib_file
    if opts.shared:
        cmd = ["clang", "-shared"]
        cmd += ["--sysroot", opts.sysroot] if opts.sysroot else []
        cmd += ["-target", opts.host_target] if opts.host_target else []
        cmd += ["-fuse-ld=lld", "-o", lib_file] + obj_files
    else:
        cmd = ["llvm-ar", "rc", lib_file] + obj_files
    do_call(cmd)

    if opts.output_file:
        do_call(["cp", lib_file, opts.output_file])


def run(mlir_module, args=None):
    global opts
    global aiecc_path
    if args is not None:
        opts = cl_arguments.parse_args(args)

    if opts.tmpdir:
        tmpdirname = opts.tmpdir
        try:
            os.mkdir(tmpdirname)
        except FileExistsError:
            pass
        if opts.verbose:
            print("created temporary directory", tmpdirname)

    if not opts.num_cols:
        opts.num_cols = 4 if "npu" in opts.device else 10

    if not opts.col_offset:
        opts.col_offset = 0 if "npu" in opts.device else 7

    if not opts.num_rows:
        opts.num_rows = 6

    if not opts.row_offset:
        opts.row_offset = 2

    if opts.verbose:
        print("compiling %s for %s\n" % (opts.air_mlir_file, opts.device))

    aiecc_path = shutil.which("aiecc.py")
    if aiecc_path == None:
        print("Error: could not find aiecc.py")
        sys.exit(1)

    aiecc_path = os.path.dirname(os.path.realpath(aiecc_path)) + "/.."
    if opts.verbose:
        print("Using aiecc.py from: ", aiecc_path)

    with mlir_module.context as ctx:
        _, air_mlir_filename = os.path.split(opts.air_mlir_file)
        # num_tile_rows = 4
        air_collapse_herd_to_cols_pass = (
            "func.func(air-collapse-herd{" + f"max-col-size={4} " + "})"
        )
        trace_col_offset = 1 if int(opts.trace_size) > 0 else 0
        air_place_pass = (
            "air-place-herds{"
            + f"num-rows={opts.num_rows} "
            + f"num-cols={opts.num_cols} "
            + f"row-anchor={opts.row_offset} "
            + f"col-anchor={int(opts.col_offset) + trace_col_offset}"
            + "}"
        )

        air_placed = opts.tmpdir + "/placed." + air_mlir_filename
        pass_pipeline = ",".join(
            [
                "air-insert-launch-and-segment-around-herd",
                "func.func(air-lower-herd-parallel)",
            ]
            + (
                get_air_optimization_pass(
                    opts.device,
                    opts.omit_pingpong,
                    opts.lower_linalg_to_func,
                    opts.air_loop_fusion,
                    opts.omit_auto_broadcast,
                    opts.channel_multiplexing,
                )
                if "npu" in opts.device
                else []
            )
            + [
                air_collapse_herd_to_cols_pass,
                "canonicalize",
                "cse",
                air_place_pass,
                "canonicalize",
                "cse",
                "func.func(air-renumber-dma)",
            ]
        )
        air_placed_module = Module.parse(str(mlir_module))
        run_passes(
            "builtin.module(" + pass_pipeline + ")", air_placed_module, opts, air_placed
        )

        air_to_aie_pass = "air-to-aie{"
        air_to_aie_pass = (
            air_to_aie_pass
            + f"emit-while-loop={str(not opts.omit_while_true_loop).lower()}"
        )
        air_to_aie_pass = air_to_aie_pass + f" row-offset={opts.row_offset}"
        air_to_aie_pass = air_to_aie_pass + f" col-offset={opts.col_offset}"
        air_to_aie_pass = air_to_aie_pass + f" device={opts.device}"
        if int(opts.trace_size) > 0:
            air_to_aie_pass = air_to_aie_pass + " insert-trace-packet-flow=true"
        air_to_aie_pass = air_to_aie_pass + "}"
        pass_pipeline = ",".join([air_to_aie_pass])

        air_to_aie_file = opts.tmpdir + "/aie." + air_mlir_filename
        air_to_aie_module = Module.parse(str(air_placed_module))
        run_passes(
            "builtin.module(" + pass_pipeline + ")",
            air_to_aie_module,
            opts,
            air_to_aie_file,
        )

        if "npu" in opts.device:
            air_opt_shim_dma_bds_pass = "func.func(air-opt-shim-dma-bds{device="
            air_opt_shim_dma_bds_pass = air_opt_shim_dma_bds_pass + opts.device
            air_opt_shim_dma_bds_pass = air_opt_shim_dma_bds_pass + (
                " shim-dma-tile-sizes="
                + ",".join(str(s) for s in opts.runtime_loop_tiling_sizes)
                if opts.runtime_loop_tiling_sizes
                else ""
            )
            air_opt_shim_dma_bds_pass = air_opt_shim_dma_bds_pass + "})"

            airrt_to_npu_pass = "airrt-to-npu{"
            airrt_to_npu_pass = airrt_to_npu_pass + f" trace-size={opts.trace_size}"
            airrt_to_npu_pass = (
                airrt_to_npu_pass + f" trace-offset={opts.trace_offset}" + "}"
            )

            air_to_npu_file = opts.tmpdir + "/npu." + air_mlir_filename
            air_to_npu_module = Module.parse(str(air_to_aie_module))
            air_to_npu_passes = (
                "builtin.module("
                + ",".join(
                    [
                        air_opt_shim_dma_bds_pass,
                        "air-to-std",
                        "symbol-dce",
                        "affine-expand-index-ops",
                        "canonicalize",
                        "cse",
                        airrt_to_npu_pass,
                        "canonicalize",
                        "cse",
                    ]
                )
                + ")"
            )
            run_passes(air_to_npu_passes, air_to_npu_module, opts, air_to_npu_file)
            xclbin_file = "aie.xclbin"
            if opts.output_file:
                xclbin_file = opts.output_file
            if opts.insts_file:
                insts_file = opts.insts_file
            else:
                assert xclbin_file.endswith(".xclbin")
                insts_file = opts.output_file.removesuffix(".xclbin") + ".insts.txt"
            aiecc_output_file_options = [
                (
                    "--xclbin-name=" + xclbin_file
                    if opts.output_format == "xclbin"
                    else ""
                ),
                (
                    "--xclbin-kernel-name=" + opts.kernel_name
                    if opts.kernel_name
                    else ""
                ),
                (
                    "--xclbin-instance-name=" + opts.instance_name
                    if opts.instance_name
                    else ""
                ),
                ("--xclbin-kernel-id=" + opts.kernel_id if opts.kernel_id else ""),
            ]
            if opts.output_format == "xclbin":
                aiecc_output_file_options = aiecc_output_file_options + [
                    "--aie-generate-xclbin"
                ]
            elif opts.output_format == "txn":
                aiecc_output_file_options = aiecc_output_file_options + [
                    "--aie-generate-txn"
                ]
            else:
                print("Error: unknown output-format")
                sys.exit(1)
            aiecc_existing_xclbin_options = [
                ("--xclbin-input=" + opts.xclbin_input if opts.xclbin_input else "")
            ]
            aiecc_options = (
                (["-v"] if opts.verbose else [])
                + [
                    "--no-aiesim",
                    "--xchesscc" if opts.xchesscc else "--no-xchesscc",
                    "--xbridge" if opts.xbridge else "--no-xbridge",
                    "--aie-generate-npu",
                    "--no-compile-host",
                    "--npu-insts-name=" + insts_file,
                ]
                + aiecc_output_file_options
                + aiecc_existing_xclbin_options
                + [air_to_npu_file]
            )
            print(aiecc_options)
            aiecc.run(air_to_npu_module, aiecc_options)
        else:
            lower_airrt_to_airhost(
                air_to_aie_module, air_placed_module, air_mlir_filename
            )


def main():
    global opts
    opts = cl_arguments.parse_args()
    is_windows = platform.system() == "Windows"

    with Context() as ctx, Location.unknown():
        with open(opts.air_mlir_file, "r") as f:
            module = Module.parse(f.read())
            run(module)
