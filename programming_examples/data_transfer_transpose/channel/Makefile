# (c) Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Output format: xclbin (default) or elf
OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

M ?= 64
K ?= 32

all: run_int run_float

print:
	${powershell} python3 ${srcdir}/transpose.py $(OUTPUT_FORMAT_FLAG) -p

run_int:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/transpose.py $(OUTPUT_FORMAT_FLAG) -m ${M} -k ${K} -t uint32

run_float:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/transpose.py $(OUTPUT_FORMAT_FLAG) -m ${M} -k ${K} -t float32

clean:
	rm -rf $(BUILD_DIR) __pycache__