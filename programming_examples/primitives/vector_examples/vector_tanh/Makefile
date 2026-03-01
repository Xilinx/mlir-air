# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

BUILD_DIR := build_peano

# Output format: xclbin (default) or elf
OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

# AIE_TARGET: aie2p (default) — hardware tanh is AIE2P-only
AIE_TARGET ?= aie2p

# VECTOR_SIZE: 16 (default) or 32
VECTOR_SIZE ?= 16

all: run

print:
	${powershell} python3 ${srcdir}/vector_tanh.py $(OUTPUT_FORMAT_FLAG) -p --vector-size $(VECTOR_SIZE) --arch $(AIE_TARGET)

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/vector_tanh.py $(OUTPUT_FORMAT_FLAG) --vector-size $(VECTOR_SIZE) --arch $(AIE_TARGET)

clean:
	rm -rf $(BUILD_DIR) __pycache__
