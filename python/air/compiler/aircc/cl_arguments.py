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
        "-xbridge",
        dest="xbridge",
        default=air_link_with_xchesscc,
        action="store_true",
        help="pass --xbridge to aiecc, otherwise pass --no-xbridge",
    )
    parser.add_argument(
        "-xchesscc",
        dest="xchesscc",
        default=air_compile_with_xchesscc,
        action="store_true",
        help="pass --xchesscc to aiecc, otherwise pass --no-xchesscc",
    )
    parser.add_argument(
        "--device",
        metavar="target_device",
        default="xcvc1902",
        help="Target AIE device",
    )
    parser.add_argument(
        "-e",
        "--experimental-passes",
        dest="experimental_passes",
        default=False,
        action="store_true",
        help="Whether to run experimental passes or not. This will only change the behavior for this program for npu devices",
    )

    opts = parser.parse_args(args)
    return opts
