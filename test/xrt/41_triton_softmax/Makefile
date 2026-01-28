# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Target selection: npu1 (Phoenix) or npu2 (Strix)
# Override with: make TARGET=npu1
TARGET ?= npu2

# Build directory based on target and compiler
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Softmax dimensions (can be overridden)
M ?= 256
N ?= 256

# Verbosity level for test output (0 = minimal, 1+ = verbose)
VERBOSITY ?= 0

# Debug IR output path (used with debug_ir target)
DEBUG_IR_PATH ?= ir_debug/final.mlir

# AIE target architecture
ifeq ($(TARGET),npu1)
  AIE_TARGET := aie2-none-unknown-elf
else ifeq ($(TARGET),npu2)
  AIE_TARGET := aie2p-none-unknown-elf
else
  $(error Unknown TARGET: $(TARGET). Use npu1 or npu2)
endif

# Find aie-opt to get include directory
AIEOPT_DIR := $(shell realpath $(dir $(shell which aie-opt))/..)

# Compiler flags for Peano
WARNING_FLAGS := -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANO_FLAGS := -O2 -std=c++20 --target=$(AIE_TARGET) $(WARNING_FLAGS) -DNDEBUG -I$(AIEOPT_DIR)/include

# PowerShell detection for WSL
TEST_POWERSHELL := $(shell command -v powershell.exe >/dev/null 2>&1 && echo yes || echo no)
ifeq ($(TEST_POWERSHELL),yes)
  powershell := powershell.exe
else
  powershell :=
endif

.PHONY: all run compile-kernel compile-xclbin profile build-test-exe debug_ir clean help

all: run

help:
	@echo "Triton Softmax Example for AMD AI Engine"
	@echo ""
	@echo "Usage:"
	@echo "  make [TARGET=npu1|npu2] [M=256] [N=256]"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build and run (default)"
	@echo "  run            - Compile kernel and run correctness test"
	@echo "  compile-kernel - Compile extern_func.cc only"
	@echo "  compile-xclbin - Compile xclbin without running tests"
	@echo "  profile        - Run performance profiling with C++ executable"
	@echo "  build-test-exe - Build the C++ test executable only"
	@echo "  debug_ir       - Dump transformed IR to file and exit"
	@echo "  clean          - Remove build artifacts"
	@echo "  help           - Show this message"
	@echo ""
	@echo "Variables:"
	@echo "  TARGET         - npu1 (Phoenix) or npu2 (Strix, default)"
	@echo "  M              - Parallel dimension size (default: 256)"
	@echo "  N              - Reduction dimension size (default: 256)"
	@echo "  VERBOSITY      - Output verbosity level (default: 0)"
	@echo "  DEBUG_IR_PATH  - Path for debug IR output (default: ir_debug/final.mlir)"
	@echo "  PEANO_INSTALL_DIR - Path to Peano compiler (auto-detected if in PATH)"
	@echo ""
	@echo "Examples:"
	@echo "  make TARGET=npu2                    # Run on NPU2 with defaults"
	@echo "  make TARGET=npu1 M=512 N=256        # Run on NPU1 with custom size"
	@echo "  make profile M=256 N=256            # Profile performance"
	@echo "  make compile-kernel                 # Just compile the kernel"

compile-kernel:
	@mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Using Peano compiler from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)"; \
		echo "Target: $(AIE_TARGET)"; \
		$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_FLAGS) \
			-c $(srcdir)/extern_func.cc \
			-o $(BUILD_DIR)/extern_func.o; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH"; \
		echo "Target: $(TARGET)"; \
		cd $(BUILD_DIR) && $(powershell) xchesscc_wrapper aie2 \
			-c $(srcdir)/extern_func.cc \
			-o extern_func.o; \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		echo ""; \
		echo "To fix this, either:"; \
		echo "  1. Set PEANO_INSTALL_DIR environment variable"; \
		echo "  2. Activate the mlir-air sandbox: source sandbox/bin/activate"; \
		echo "  3. Install Vitis for xchesscc_wrapper"; \
		exit 1; \
	fi
	@echo "Kernel compiled successfully: $(BUILD_DIR)/extern_func.o"

# Run correctness validation (original behavior)
run: compile-kernel
	@echo "Running softmax test with M=$(M), N=$(N) on $(TARGET)..."
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--transform-script $(srcdir)/transform.mlir \
			--M $(M) \
			--N $(N)

# Compile xclbin only (no execution)
compile-xclbin: compile-kernel
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--transform-script $(srcdir)/transform.mlir \
			--M $(M) \
			--N $(N) \
			--compile-only

# Profile using C++ executable
profile: compile-xclbin build-test-exe
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin -M $(M) -N $(N) -v $(VERBOSITY)

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
debug_ir: compile-kernel
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--transform-script $(srcdir)/transform.mlir \
			--M $(M) \
			--N $(N) \
			--debug-ir $(DEBUG_IR_PATH)

clean:
	rm -rf build_* __pycache__
	@echo "Cleaned build directories"
