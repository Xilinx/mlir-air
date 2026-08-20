# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Output format: xclbin (default) or elf
OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

AIEOPT_DIR ?= $(if $(MLIR_AIE_INSTALL_DIR),$(subst \,/,$(MLIR_AIE_INSTALL_DIR)),$(shell realpath $(dir $(shell which aie-opt))/..))
# Auto-detect XILINX_XRT if not set
XILINX_XRT ?= $(wildcard /opt/xilinx/xrt)

# Extra Python flags (used by LIT tests to override defaults)
EXTRA_PY_FLAGS ?=

all: run

# SHAPE is 1-D (N) or 2-D (M N); TILE and VECTOR_SIZE default per rank and
# dtype. See eltwise_add.py.
SHAPE ?= 1024 1024
DTYPE ?= bf16
TILE ?=
VECTOR_SIZE ?=
PERF_ITERS ?= 0

TILE_FLAG = $(if $(TILE),--tile $(TILE),)
VEC_FLAG = $(if $(VECTOR_SIZE),--vector-size $(VECTOR_SIZE),)
# The device is auto-detected: one invocation has to run on whichever NPU is
# installed. Set TARGET=npu1|npu2 to compile for a generation that is not
# plugged in.
TARGET ?= auto
PY_FLAGS = --shape $(SHAPE) --dtype $(DTYPE) $(TILE_FLAG) $(VEC_FLAG) \
	--target $(TARGET) $(OUTPUT_FORMAT_FLAG) $(EXTRA_PY_FLAGS)

# Total element count, for the -S argument of test.exe.
SHAPE_ELEMENTS = $(shell $(PYTHON) -c "import math,sys;print(math.prod([int(x) for x in '$(SHAPE)'.split()]))")

print:
	${powershell} $(PYTHON) ${srcdir}/eltwise_add.py $(PY_FLAGS) --print-ir

compile:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} $(PYTHON) ${srcdir}/eltwise_add.py $(PY_FLAGS) --compile-only

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} $(PYTHON) ${srcdir}/eltwise_add.py $(PY_FLAGS) \
		--perf-iters $(PERF_ITERS)

# Bandwidth profiling via the shared test.cpp harness. bf16 only -- test.cpp
# fixes DATATYPE to std::bfloat16_t -- but rank-agnostic, since it takes a
# total element count.
profile: build-test-exe
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		$(PYTHON) ${srcdir}/eltwise_add.py $(PY_FLAGS) --compile-only
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin \
		-S $(SHAPE_ELEMENTS)

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
