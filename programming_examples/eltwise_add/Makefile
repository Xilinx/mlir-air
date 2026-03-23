# Copyright (C) 2025, Advanced Micro Devices, Inc.
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

# AIE target: aie2 (NPU1) or aie2p (NPU2, default)
AIE_TARGET ?= aie2p

# Device-dependent defaults
ifeq ($(AIE_TARGET),aie2p)
  # NPU2: BF16 vectorized, 8-column config
  DTYPE ?= bf16
  VECTOR_SIZE ?= 16
  HERD_X ?= 8
  HERD_Y ?= 1
  TILE_N ?= 2048
  N ?= 4194304
else
  # NPU1: F32 scalar, 2-tile config
  DTYPE ?= f32
  VECTOR_SIZE ?= 0
  HERD_X ?= 1
  HERD_Y ?= 2
  TILE_N ?= 1024
  N ?= 65536
endif

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)

# Auto-detect XILINX_XRT if not set
XILINX_XRT ?= $(wildcard /opt/xilinx/xrt)

# Extra Python flags (used by LIT tests to override defaults)
EXTRA_PY_FLAGS ?=

all: run

print:
	${powershell} python3 ${srcdir}/eltwise_add.py $(OUTPUT_FORMAT_FLAG) \
		--n $(N) --dtype $(DTYPE) --vector-size $(VECTOR_SIZE) \
		--herd-x $(HERD_X) --herd-y $(HERD_Y) \
		--tile-n $(TILE_N) $(EXTRA_PY_FLAGS) -p

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/eltwise_add.py \
		$(OUTPUT_FORMAT_FLAG) --n $(N) --dtype $(DTYPE) --vector-size $(VECTOR_SIZE) \
		--herd-x $(HERD_X) --herd-y $(HERD_Y) --tile-n $(TILE_N) $(EXTRA_PY_FLAGS)

profile: build-test-exe
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		python3 ${srcdir}/eltwise_add.py --n $(N) --tile-n $(TILE_N) \
		--dtype $(DTYPE) --vector-size $(VECTOR_SIZE) \
		--herd-x $(HERD_X) --herd-y $(HERD_Y) \
		--compile-mode compile-only --output-format xclbin
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin \
		-S $(N)

build-test-exe:
	@GPP=$$( \
		for bin in /usr/bin/g++-*; do \
			ver=$$(echo $$bin | grep -oE '[0-9]+$$'); \
			if [ "$$ver" -ge 13 ] 2>/dev/null; then \
				echo "$$ver $$bin"; \
			fi; \
		done | sort -nr | head -n1 | awk '{print $$2}' \
	); \
	if [ -z "$$GPP" ]; then \
		echo "Error: No g++ version >= 13 found in /usr/bin."; \
		exit 1; \
	fi; \
	if [ -z "$(XILINX_XRT)" ]; then \
		echo "Error: XILINX_XRT not set and /opt/xilinx/xrt not found."; \
		exit 1; \
	fi; \
	if [ -z "$(AIEOPT_DIR)" ]; then \
		echo "Error: AIEOPT_DIR environment variable not set. Please make sure to have sourced utils/env_setup.sh."; \
		exit 1; \
	fi; \
	echo "Using compiler: $$GPP"; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP ${srcdir}/test.cpp -o test.exe -std=c++23 -Wall \
		-I$(XILINX_XRT)/include -L$(XILINX_XRT)/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

clean:
	rm -rf $(BUILD_DIR) __pycache__
