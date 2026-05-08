# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

# 16 lanes of i32 = 512 bits = the size that hits the int<->fp scalarize
# path fixed by mlir-aie #3058. Override e.g. VECTOR_SIZE=32 for 1024-bit ops.
VECTOR_SIZE ?= 16

all: run

print:
	${powershell} python3 ${srcdir}/vector_sitofp.py $(OUTPUT_FORMAT_FLAG) -p --vector-size $(VECTOR_SIZE)

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/vector_sitofp.py $(OUTPUT_FORMAT_FLAG) --vector-size $(VECTOR_SIZE)

clean:
	rm -rf $(BUILD_DIR) __pycache__
