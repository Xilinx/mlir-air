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

# VECTOR_SIZE can be set to configure the vector size for SIMD operations (default: 16)
# Usage examples:
#   make run                      # uses default vector size of 16
#   make run VECTOR_SIZE=32       # uses vector size of 32
VECTOR_SIZE ?= 16

# BF16_EMULATION can be set to enable f32 input with bf16 emulation
# Usage examples:
#   make run                       # uses default bf16 data type
#   make run BF16_EMULATION=1      # uses f32 with bf16 emulation
BF16_EMULATION ?=

all: run

print:
	${powershell} python3 ${srcdir}/vector_select.py $(OUTPUT_FORMAT_FLAG) -p --vector-size $(VECTOR_SIZE) $(if $(BF16_EMULATION),--bf16-emulation,)

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/vector_select.py $(OUTPUT_FORMAT_FLAG) --vector-size $(VECTOR_SIZE) $(if $(BF16_EMULATION),--bf16-emulation,)
clean:
	rm -rf $(BUILD_DIR) __pycache__
