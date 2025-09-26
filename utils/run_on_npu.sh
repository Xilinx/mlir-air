#!/bin/bash

# (c) Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

XRT_DIR=/opt/xilinx/xrt
source $XRT_DIR/setup.sh

"$@"
err=$?

exit $err
