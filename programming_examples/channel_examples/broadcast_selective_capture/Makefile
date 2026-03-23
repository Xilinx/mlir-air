# (c) Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

BUILD_DIR := build_peano

# Output format: xclbin (default) or elf
OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

all: run

print:
	${powershell} python3 ${srcdir}/broadcast_selective_capture.py $(OUTPUT_FORMAT_FLAG) -p

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/broadcast_selective_capture.py $(OUTPUT_FORMAT_FLAG)

clean:
	rm -rf $(BUILD_DIR) __pycache__
