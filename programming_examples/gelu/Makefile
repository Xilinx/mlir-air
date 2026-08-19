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

# bf16 only: f32 math.tanh does not legalize on either generation. npu2 only:
# on npu1 bf16 math.tanh lowers to emitc, which peano cannot translate.
N ?= 65536
TILE_N ?= 1024
VECTOR_SIZE ?=
HERD_SHAPE ?=
PERF_ITERS ?= 0

VEC_FLAG = $(if $(VECTOR_SIZE),--vector-size $(VECTOR_SIZE),)
HERD_FLAG = $(if $(HERD_SHAPE),--herd-shape $(HERD_SHAPE),)
TARGET ?= auto
PY_FLAGS = --n $(N) --tile-n $(TILE_N) $(VEC_FLAG) $(HERD_FLAG) \
	--target $(TARGET) $(OUTPUT_FORMAT_FLAG) $(EXTRA_PY_FLAGS)

print:
	${powershell} python3 ${srcdir}/gelu.py $(PY_FLAGS) --print-ir

compile:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/gelu.py $(PY_FLAGS) --compile-only

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/gelu.py $(PY_FLAGS) \
		--perf-iters $(PERF_ITERS)

# ---------------------------------------------------------------------------
# The raw-bindings predecessor, kept until the npu2 A/B is done.
# ---------------------------------------------------------------------------

DIALECT_PY_FLAGS = --n $(N) --tile-n $(TILE_N) $(OUTPUT_FORMAT_FLAG)

run-dialect:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/gelu_dialect.py $(DIALECT_PY_FLAGS)

clean:
	rm -rf $(BUILD_DIR) __pycache__
