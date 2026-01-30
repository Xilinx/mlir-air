# ./python/air/compiler/aircc/cl_arguments.py -*- Python -*-

# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import sys

from air.compiler.aircc.configure import *


def parse_args(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="aircc")
    parser.add_argument(
        "air_mlir_file",
        metavar="air_mlir_file",
        default="air.mlir",
        help="AIR Dialect mlir file",
    )
    parser.add_argument("-o", dest="output_file", default="", help="Output filename")
    parser.add_argument(
        "-i",
        dest="insts_file",
        default="",
        help="Output insts file name. Only used for compilation on an NPU.",
    )
    parser.add_argument(
        "--tmpdir",
        metavar="tmpdir",
        default="air_project",
        help="directory used for temporary file storage",
    )
    parser.add_argument(
        "-v",
        dest="verbose",
        default=False,
        action="store_true",
        help="Trace commands as they are executed",
    )
    parser.add_argument(
        "-row-offset",
        dest="row_offset",
        help="Default row offset for generated segments",
    )
    parser.add_argument(
        "-col-offset",
        dest="col_offset",
        help="Default column offset for generated segments",
    )
    parser.add_argument(
        "-num-rows",
        dest="num_rows",
        help="Default number of rows for generated segments",
    )
    parser.add_argument(
        "-num-cols",
        dest="num_cols",
        help="Default number of rows for generated segments",
    )
    parser.add_argument(
        "-trace-size",
        dest="trace_size",
        default=0,
        help="Create packet routed traces for cores and memtiles",
    )
    parser.add_argument(
        "-trace-offset",
        dest="trace_offset",
        default=0,
        help="Trace buffer offset appended to output",
    )
    parser.add_argument("-cc", dest="cc", default="clang", help="Compiler to use")
    parser.add_argument(
        "--sysroot", metavar="sysroot", default="", help="sysroot for cross-compilation"
    )
    parser.add_argument(
        "--host-target",
        metavar="host_target",
        default="",
        help="Target architecture of the host program",
    )
    parser.add_argument(
        "--shared",
        dest="shared",
        default=False,
        action="store_true",
        help="Generate a shared library (.so) instead of the default of a static library (.a)",
    )
    parser.add_argument(
        "--xbridge",
        dest="xbridge",
        default=air_link_with_xchesscc,
        action="store_true",
        help="Link using xbridge",
    )
    parser.add_argument(
        "--no-xbridge",
        dest="xbridge",
        default=air_link_with_xchesscc,
        action="store_false",
        help="Link using peano",
    )
    parser.add_argument(
        "--xchesscc",
        dest="xchesscc",
        default=air_compile_with_xchesscc,
        action="store_true",
        help="Compile using xchesscc",
    )
    parser.add_argument(
        "--no-xchesscc",
        dest="xchesscc",
        default=air_compile_with_xchesscc,
        action="store_false",
        help="Compile using peano",
    )
    parser.add_argument(
        "--peano",
        dest="peano_install_dir",
        default="",
        help="Root directory where peano compiler is installed",
    )
    parser.add_argument(
        "--device",
        metavar="target_device",
        default="xcvc1902",
        help="Target AIE device",
    )
    parser.add_argument(
        "--omit-while-true-loop",
        dest="omit_while_true_loop",
        default=False,
        action="store_true",
        help="By default, aircc may output a while(true) loop around per-core logic. If this option is specified, a while(true) loop will not be added.",
    )
    parser.add_argument(
        "--omit-ping-pong-transform",
        dest="omit_pingpong",
        default="",
        type=str,
        nargs="?",
        const="all",
        help="Omit ping-pong buffering transformation for specific memory levels. Supported values: '', 'L1', 'L2', 'all'. Empty string means no omission (default). For backward compatibility, using the flag without a value is equivalent to 'all'.",
    )
    parser.add_argument(
        "--lower-linalg-to-func",
        type=str,
        dest="lower_linalg_to_func",
        default=None,
        help="Whether to run pass which lowers linalg.generic ops to function calls. If a string is passed in, then register the string value as the object file name to link with.",
    )
    parser.add_argument(
        "--air-loop-fusion",
        dest="air_loop_fusion",
        default=False,
        action="store_true",
        help="Adds air-loop-fusion pass to the compiler pipeline. It is an experimental pass which tries to enforce loop fusion for lowring to efficient DMA BDs",
    )
    parser.add_argument(
        "--air-runtime-loop-tiling-sizes",
        type=int,
        nargs="*",  # Accept zero or more integers
        dest="runtime_loop_tiling_sizes",
        default=[4, 4],
        help="Adds tiling factors to be applied to the runtime host affine loop nest. It is an experimental pass which enforces extra innermost tilings at runtime, to comply with constraints of certain hardware. If this option is omitted, the default tiling factors [4, 4] are used; specifying the flag without any values disables shim-dma-tile-sizes; providing one or more integers overrides the default tiling factors.",
    )
    parser.add_argument(
        "--omit-auto-broadcast",
        dest="omit_auto_broadcast",
        default=False,
        action="store_true",
        help="Omits the air-dependency-schedule-opt pass, which detects and lowers broadcasts",
    )
    parser.add_argument(
        "--air-channel-multiplexing",
        type=str,
        nargs="*",  # Accept zero or more strings
        dest="channel_multiplexing",
        default=[],
        help="Adds memory spaces to which air channels shall get time-multiplexed, if operating on them",
    )
    parser.add_argument(
        "--use-lock-race-condition-fix",
        dest="use_lock_race_condition_fix",
        default=False,
        action="store_true",
        help="Switch to enable a fix for lock race condition, which protects against the risk of race condition, at the cost of inserting extra dummy DMA BDs.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "txn", "elf", "none"],
        dest="output_format",
        default="xclbin",
        help="File format for the generated binary. Use 'none' for compile-only mode without XRT dependencies (generates intermediate artifacts only).",
    )
    parser.add_argument(
        "--xclbin-kernel-name",
        dest="kernel_name",
        default="",
        help="Kernel name in xclbin file",
    )
    parser.add_argument(
        "--xclbin-instance-name",
        dest="instance_name",
        default="",
        help="Instance name in xclbin metadata",
    )
    parser.add_argument(
        "--xclbin-kernel-id",
        dest="kernel_id",
        default="",
        help="Kernel id in xclbin file",
    )
    parser.add_argument(
        "--xclbin-input",
        dest="xclbin_input",
        default=None,
        help="Generate kernel into existing xclbin file",
    )
    parser.add_argument(
        "--elf-name",
        dest="elf_name",
        default="aie.elf",
        help="Output filename for full ELF when using --output-format=elf (default: aie.elf)",
    )
    parser.add_argument(
        "--emit-main-device",
        dest="emit_main_device",
        default=False,
        action="store_true",
        help="Always generate a main aie.device wrapper with configure/run ops, even for single-device designs. This enables reconfiguration mode for designs with a single aie.device.",
    )

    opts = parser.parse_args(args)
    return opts
