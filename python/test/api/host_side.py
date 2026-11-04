# ./python/test/api/host_side.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""What the XRT backend passes besides the caller's buffers.

A sequence that patches descriptors at run time takes aircc's control-packet
table as its last buffer, and its scalars as scratchpad parameters, named after
the argument they come from or listed in air_params.txt.
"""

import os
import tempfile

import numpy as np

from air.backend.xrt import _HostSide

with tempfile.TemporaryDirectory() as d:
    host = os.path.join(d, "host")
    os.makedirs(host)
    np.arange(12, dtype=np.int32).tofile(os.path.join(host, "air_ctrl_table.bin"))
    with open(os.path.join(host, "air_params.txt"), "w") as f:
        f.write("__air_param_fwd_0 2 1 0\n")
    params = os.path.join(d, "params.txt")
    with open(params, "w") as f:
        f.write("3 3\n")
        f.write("__air_param_argoff_2_x6_m6 0 i32 addr 0 42\n")
        f.write("__air_param_arglen_2 1 i32 addr 0 8\n")
        f.write("__air_param_fwd_0 2 i32 core - -\n")

    side = _HostSide(host)
    x, y = np.zeros(4, np.int32), np.ones(4, np.int32)
    buffers, scalars = side.split((x, y, 5))

    # The table goes last, after the caller's buffers; the int is a scalar.
    # CHECK: buffers 3 table 12 scalars {2: 5}
    print("buffers", len(buffers), "table", buffers[-1].size, "scalars", scalars)

    # Each parameter is scale * L + addend, from its name or air_params.txt.
    # CHECK: __air_param_argoff_2_x6_m6 24
    # CHECK: __air_param_arglen_2 5
    # CHECK: __air_param_fwd_0 5
    for name, value in side.parameters(params, scalars):
        print(name, value)

    # Results come back in the caller's argument positions.
    # CHECK: results ['ndarray', 'ndarray', 'int']
    out = side.results((x, y, 5), [x + 1, y + 1, buffers[-1]])
    print("results", [type(o).__name__ for o in out])

# No host directory: buffers pass through unchanged.
# CHECK: plain 2
print("plain", len(_HostSide(None).split((np.zeros(2), np.zeros(2)))[0]))
