# ./python/air/compiler/aircc/cl_arguments.py -*- Python -*-

# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import sys

def parse_args(args=None):
    if (args is None):
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(prog='aircc')
    parser.add_argument('air_mlir_file',
                        metavar="air_mlir_file",
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
            default=1,
            help='Herd physical row offset')
    parser.add_argument('-col-offset',
            dest="col_offset",
            default=1,
            help='Herd physical column offset')
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

    opts = parser.parse_args(args)
    return opts

