#!/usr/bin/env bash

##===- utils/clone-rocm-air-platforms.sh ---------------------*- Script -*-===##
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

##===----------------------------------------------------------------------===##
#
# This script checks out the ROCm-air-platforms repository which contains the
# driver, hardware, and firmware of the AIR ROCm platform.
#
##===----------------------------------------------------------------------===##

target_dir=ROCm-air-platforms

if [[ ! -d $target_dir ]]; then
  git clone https://github.com/Xilinx/ROCm-air-platforms $target_dir
fi
