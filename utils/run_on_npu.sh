#!/bin/bash

# (c) Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

XRT_DIR=/opt/xilinx/xrt
source $XRT_DIR/setup.sh
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1

"$@"
err=$?

exit $err
