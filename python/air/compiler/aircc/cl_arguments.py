# ./python/air/compiler/aircc/cl_arguments.py -*- Python -*-

# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import sys

def parse_args(args=None):
    if (args is None):
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(prog='aircc')
    parser.add_argument('air_mlir_file',
            metavar="air_mlir_file",
            default="air.mlir",
            help='AIR Dialect mlir file')
    parser.add_argument('-o',
            dest="output_file",
            default="",
            help='Output filename')
    parser.add_argument('--tmpdir',
            metavar="tmpdir",
            default="air_project",
            help='directory used for temporary file storage')
    parser.add_argument('-v',
            dest="verbose",
            default=False,
            action='store_true',
            help='Trace commands as they are executed')
    parser.add_argument('-row-offset',
            dest="row_offset",
            default=2,
            help='Default row offset for generated segments')
    parser.add_argument('-col-offset',
            dest="col_offset",
            default=7,
            help='Default column offset for generated segments')
    parser.add_argument('-num-rows',
            dest="num_rows",
            default=6,
            help='Default number of rows for generated segments')
    parser.add_argument('-num-cols',
            dest="num_cols",
            default=10,
            help='Default number of rows for generated segments')
    parser.add_argument('-cc',
            dest="cc",
            default="clang",
            help="Compiler to use")
    parser.add_argument('--sysroot',
            metavar="sysroot",
            default="",
            help='sysroot for cross-compilation')
    parser.add_argument('--host-target',
            metavar="host_target",
            default="",
            help='Target architecture of the host program')
    parser.add_argument('--shared',
            dest="shared",
            default=False,
            action='store_true',
            help='Generate a shared library (.so) instead of the default of a static library (.a)')
    parser.add_argument('-xbridge',
            dest="xbridge",
            default=False,
            action='store_true',
            help='pass --xbridge to aiecc, otherwise pass --no-xbridge')
    parser.add_argument('-xchesscc',
            dest="xchesscc",
            default=False,
            action='store_true',
            help='pass --xchesscc to aiecc, otherwise pass --no-xchesscc')

    opts = parser.parse_args(args)
    return opts

