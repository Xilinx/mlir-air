import argparse
import sys

def parse_args():
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
    parser.add_argument('--shared',
            dest="shared",
            default=False,
            action='store_true',
            help='Generate a shared library (.so) instead of the default of a static library (.a)')

    opts = parser.parse_args(sys.argv[1:])
    return opts

