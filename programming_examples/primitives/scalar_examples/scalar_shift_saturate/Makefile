# Copyright (C) 2026, Advanced Micro Devices, Inc.
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

# SHIFT_AMOUNT: right shift amount for quantization (default: 4)
SHIFT_AMOUNT ?= 4

all: run

print:
	${powershell} python3 ${srcdir}/scalar_shift_saturate.py -p --shift-amount $(SHIFT_AMOUNT)

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/scalar_shift_saturate.py $(OUTPUT_FORMAT_FLAG) --shift-amount $(SHIFT_AMOUNT)

clean:
	rm -rf $(BUILD_DIR) __pycache__
