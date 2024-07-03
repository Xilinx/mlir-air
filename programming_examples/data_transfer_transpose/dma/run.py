# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
import argparse
import sys
from pathlib import Path  # if you haven't already done so

# Python paths are a bit complex. Taking solution from : https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from dma.transpose import build_module
from common import test_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the matrix_scalar_add/single_core_channel example",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        type=int,
        default=64,
        help="The matrix to transpose will be of size M x K, this parameter sets the M value",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=32,
        help="The matrix to transpose will be of size M x K, this parameter sets the k value",
    )
    args = parser.parse_args()
    test_main(build_module, m=args.m, k=args.k, verbose=args.verbose)
