# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# AIE target: aie2 (NPU1) or aie2p (NPU2, default)
AIE_TARGET ?= aie2p

# Default herd_x for the multi-tile `run` target. NPU2 (aie2p) has enough
# compute rows to fan out to 8 tiles; NPU1 (aie2) is single-tile.
ifeq ($(AIE_TARGET),aie2p)
  HERD_X ?= 8
else
  HERD_X ?= 1
endif

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

ITERATIONS ?= 5

all: run

print:
	${powershell} python3 ${srcdir}/weighted_rms_norm.py $(OUTPUT_FORMAT_FLAG) -p

# Default: multi-tile execution (8 tiles on NPU2, configurable via HERD_X)
run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/weighted_rms_norm.py $(OUTPUT_FORMAT_FLAG) --herd-x $(HERD_X)

# Original single-tile baseline (herd_x=1)
run_single_tile:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/weighted_rms_norm.py $(OUTPUT_FORMAT_FLAG) --herd-x 1

# Profile multi-tile execution (timing + bandwidth; correctness via `make run`)
profile:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/weighted_rms_norm.py $(OUTPUT_FORMAT_FLAG) --herd-x $(HERD_X) --profile --iterations $(ITERATIONS)

clean:
	rm -rf $(BUILD_DIR) __pycache__
