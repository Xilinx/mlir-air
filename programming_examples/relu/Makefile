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

# Extra Python flags (used by LIT tests to override defaults)
EXTRA_PY_FLAGS ?=

all: run

N ?= 65536
TILE_N ?= 1024
DTYPE ?= bf16
VECTOR_SIZE ?=
PERF_ITERS ?= 0

VEC_FLAG = $(if $(VECTOR_SIZE),--vector-size $(VECTOR_SIZE),)
# The device is auto-detected: one invocation has to run on whichever NPU is
# installed. Set TARGET=npu1|npu2 to compile for a generation that is not
# plugged in.
TARGET ?= auto
PY_FLAGS = --n $(N) --tile-n $(TILE_N) --dtype $(DTYPE) $(VEC_FLAG) \
	--target $(TARGET) $(OUTPUT_FORMAT_FLAG) $(EXTRA_PY_FLAGS)

print:
	${powershell} python3 ${srcdir}/relu.py $(PY_FLAGS) --print-ir

compile:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/relu.py $(PY_FLAGS) --compile-only

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/relu.py $(PY_FLAGS) \
		--perf-iters $(PERF_ITERS)

clean:
	rm -rf $(BUILD_DIR) __pycache__
