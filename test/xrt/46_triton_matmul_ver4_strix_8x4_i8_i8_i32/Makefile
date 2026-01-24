# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Architecture selection: aie2 or aie2p (default)
AIE_TARGET ?= aie2p

# Transform script path: can be overridden via command line
# Example: make run TRANSFORM_SCRIPT_PATH=/path/to/custom/transform.mlir
TRANSFORM_SCRIPT_PATH ?=

# Select transform script based on target (used only if TRANSFORM_SCRIPT_PATH is not set)
ifeq ($(TRANSFORM_SCRIPT_PATH),)
  ifeq ($(AIE_TARGET),aie2p)
    TRANSFORM_SCRIPT := $(srcdir)/transform_aie2p.mlir
  else
    TRANSFORM_SCRIPT := $(srcdir)/transform_aie2.mlir
  endif
else
  TRANSFORM_SCRIPT := $(TRANSFORM_SCRIPT_PATH)
endif

INPUT_IR_PATH := $(srcdir)/asm_src.mlir

# Matrix dimensions (configurable via command line, e.g., make profile M=1024 K=1024 N=1024)
M ?= 1024
K ?= 1024
N ?= 1024

# Verbosity level for test output (0 = minimal, 1+ = verbose)
VERBOSITY ?= 0

# Debug IR output path (used with debug_ir target)
DEBUG_IR_PATH ?= ir_debug/final.mlir

# Build directory
BUILD_DIR := build

# Environment setup
AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)

.PHONY: all run run-aie2 run-aie2p compile-xclbin profile build-test-exe debug_ir clean

all: run

# Run correctness validation (original behavior)
run: run-$(AIE_TARGET)

run-aie2:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python $(srcdir)/run.py --input-ir ${INPUT_IR_PATH} --transform-script ${TRANSFORM_SCRIPT}

run-aie2p:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python $(srcdir)/run.py --input-ir ${INPUT_IR_PATH} --transform-script ${TRANSFORM_SCRIPT}

# Compile xclbin only (no execution)
compile-xclbin:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python $(srcdir)/run.py --input-ir ${INPUT_IR_PATH} --transform-script $(TRANSFORM_SCRIPT) --compile-only

# Profile using C++ executable
profile: compile-xclbin build-test-exe
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin -M $(M) -K $(K) -N $(N) -v $(VERBOSITY)

# Build the C++ test executable
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
	if [ -z "$$XILINX_XRT" ]; then \
		echo "Error: XILINX_XRT environment variable not set. Please make sure to have sourced xrt/setup.sh."; \
		exit 1; \
	fi; \
	if [ -z "$(AIEOPT_DIR)" ]; then \
		echo "Error: AIEOPT_DIR environment variable not set. Please make sure to have sourced utils/env_setup.sh."; \
		exit 1; \
	fi; \
	echo "Using compiler: $$GPP"; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP $(srcdir)/test.cpp -o test.exe -std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

# Debug IR: dump transformed IR to file and exit
debug_ir:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python $(srcdir)/run.py --input-ir ${INPUT_IR_PATH} --transform-script $(TRANSFORM_SCRIPT) --debug-ir $(DEBUG_IR_PATH)

clean:
	rm -rf $(BUILD_DIR) __pycache__
