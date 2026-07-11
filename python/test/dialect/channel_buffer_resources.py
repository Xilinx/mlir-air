# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

# Exercises the `buffer_resources` knob on the placed Channel wrapper. The
# attribute is consumed by AIRToAIEPass as the depth of the lowered
# aie.objectfifo (default 1); see
# test/Conversion/AIRToAIE/air_channel_to_objectfifo_buffer_resources.mlir.

from air.ir import *
from air.dialects.air import *


@module_builder
def build_module():
    # Default: no buffer_resources attribute is emitted (depth stays 1). The
    # CHECK-NOT guards the region up to the next channel, so a stray attribute
    # would fail the test rather than slip through the prefix match.
    # CHECK: air.channel @chan_default []
    # CHECK-NOT: buffer_resources
    Channel("chan_default")

    # Explicit depth-2 FIFO via the python knob.
    # CHECK: air.channel @chan_depth2 [] {buffer_resources = 2 : i64}
    Channel("chan_depth2", buffer_resources=2)

    # Explicit depth-4 FIFO.
    # CHECK: air.channel @chan_depth4 [] {buffer_resources = 4 : i64}
    Channel("chan_depth4", buffer_resources=4)


print(build_module())
