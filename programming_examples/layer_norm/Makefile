# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

# Which NPU generation the herd default is sized for: aie2 (NPU1) or aie2p
# (NPU2, default). This picks the HERD_X default below and nothing else -- the
# example takes its compilation target from XRT device auto-detection, not from
# a flag, so setting this does not retarget the build.
AIE_TARGET ?= aie2p

# Default herd_x for the multi-tile `run` target. NPU2 (aie2p) fans out to 8
# tiles (full chip width); NPU1 (aie2) is single-tile.
ifeq ($(AIE_TARGET),aie2p)
  HERD_X ?= 8
else
  HERD_X ?= 1
endif

# SigLIP vision-encoder LayerNorm shape (num_patches=1024, hidden=768)
M ?= 1024
N ?= 768

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Output format: xclbin (default) or elf
OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

all: run

print:
	${powershell} $(PYTHON) ${srcdir}/layer_norm.py $(OUTPUT_FORMAT_FLAG) -p

# Default: multi-tile execution (8 tiles on NPU2, configurable via HERD_X)
run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} $(PYTHON) ${srcdir}/layer_norm.py $(OUTPUT_FORMAT_FLAG) --M $(M) --N $(N) --herd-x $(HERD_X)

# Original single-tile baseline (herd_x=1)
run_single_tile:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} $(PYTHON) ${srcdir}/layer_norm.py $(OUTPUT_FORMAT_FLAG) --M $(M) --N $(N) --herd-x 1

clean:
	rm -rf $(BUILD_DIR) __pycache__
