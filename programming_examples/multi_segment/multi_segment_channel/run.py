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

from multi_segment_channel.multi_segment import build_module
from common import test_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the multi-segment channel example",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()
    test_main(build_module, verbose=args.verbose)
