# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# AIE_TARGET can be set to 'aie2' (default) or 'aie2p' to select the target architecture.
# Usage examples:
#   make run                  # uses aie2 target
#   make run AIE_TARGET=aie2p # uses aie2p target
AIE_TARGET ?= aie2

all: run

print:
	${powershell} python3 ${srcdir}/scalar_reciprocal.py -p --arch $(AIE_TARGET)

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/scalar_reciprocal.py --arch $(AIE_TARGET)

clean:
	rm -rf $(BUILD_DIR) __pycache__
