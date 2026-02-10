# Copyright (C) 2026, Advanced Micro Devices, Inc.
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
N ?= 1024

# Input MLIR file (can be overridden)
INPUT_MLIR ?= $(srcdir)/input_ir/4x1024_mul_inv_bf16mul.mlir

# Verbose mode (0 = minimal, 1 = enabled for pass logging and test output)
VERBOSE ?= 0
ifeq ($(VERBOSE),1)
  VERBOSE_FLAG := --verbose
else
  VERBOSE_FLAG :=
endif

# Debug AIRCC mode (0 = disabled, 1 = emit IR after each individual pass)
DEBUG_AIRCC ?= 0
ifeq ($(DEBUG_AIRCC),1)
  DEBUG_AIRCC_FLAG := --debug-aircc
else
  DEBUG_AIRCC_FLAG :=
endif

# Debug IR output path (used with debug_ir target)
DEBUG_IR_PATH ?= ir_debug/final.mlir

# AIE target architecture and transform script selection
# NPU1 (aie2) requires extern_func.cc compilation for getExpBf16
# NPU2 (aie2p) uses native intrinsics, no external kernel needed
ifeq ($(TARGET),npu1)
  AIE_TARGET := aie2-none-unknown-elf
  TRANSFORM_SCRIPT ?= $(srcdir)/transform_aie2.mlir
  NEEDS_KERNEL := yes
else ifeq ($(TARGET),npu2)
  AIE_TARGET := aie2p-none-unknown-elf
  TRANSFORM_SCRIPT ?= $(srcdir)/transform_aie2p.mlir
  NEEDS_KERNEL := no
else
  $(error Unknown TARGET: $(TARGET). Use npu1 or npu2)
endif

# Find aie-opt to get include directory
AIEOPT_DIR := $(shell realpath $(dir $(shell which aie-opt))/..)

# Auto-detect XILINX_XRT if not provided (for profile/build-test-exe targets)
XILINX_XRT ?= $(shell echo $$XILINX_XRT)
ifeq ($(XILINX_XRT),)
  XILINX_XRT := /opt/xilinx/xrt
endif

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

.PHONY: all run compile-kernel compile-xclbin profile build-test-exe debug_ir debug_aircc clean help

all: run

help:
	@echo "Triton Softmax BF16 Example for AMD AI Engine"
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
	@echo "  debug_aircc    - Compile with per-pass IR output (fine-grained debugging)"
	@echo "  clean          - Remove build artifacts"
	@echo "  help           - Show this message"
	@echo ""
	@echo "Variables:"
	@echo "  TARGET         - npu1 (Phoenix) or npu2 (Strix, default)"
	@echo "  M              - Parallel dimension size (default: 256)"
	@echo "  N              - Reduction dimension size (default: 1024)"
	@echo "  INPUT_MLIR     - Input MLIR file path (default: input_ir/4x1024_mul_inv_fp32mul.mlir)"
	@echo "  VERBOSE        - Enable verbose mode for pass logging and profiling (1 = enabled, default: 0)"
	@echo "  DEBUG_AIRCC    - Enable per-pass IR output for fine-grained debugging (1 = enabled, default: 0)"
	@echo "  DEBUG_IR_PATH  - Path for debug IR output (default: ir_debug/final.mlir)"
	@echo "  TRANSFORM_SCRIPT - Override transform script path (default: based on TARGET)"
	@echo "  PEANO_INSTALL_DIR - Path to Peano compiler (auto-detected if in PATH)"
	@echo ""
	@echo "Transform scripts:"
	@echo "  NPU1 (aie2):  transform_aie2.mlir"
	@echo "  NPU2 (aie2p): transform_aie2p.mlir"
	@echo ""
	@echo "Examples:"
	@echo "  make TARGET=npu2                    # Run on NPU2 with defaults"
	@echo "  make TARGET=npu1 M=512 N=256        # Run on NPU1 with custom size"
	@echo "  make profile M=256 N=256            # Profile performance"
	@echo "  make compile-xclbin VERBOSE=1       # Compile with pass logging"
	@echo "  make debug_aircc                    # Compile with per-pass IR output"
	@echo "  make run DEBUG_AIRCC=1              # Run with per-pass IR output"
	@echo "  make compile-kernel                 # Just compile the kernel"
	@echo "  make INPUT_MLIR=4x256.mlir          # Use custom input MLIR file"
	@echo "  make TRANSFORM_SCRIPT=my_transform.mlir  # Use custom transform script"

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

# Run correctness validation
# NPU1 requires kernel compilation, NPU2 doesn't
ifeq ($(NEEDS_KERNEL),yes)
run: compile-kernel
else
run:
endif
	@mkdir -p $(BUILD_DIR)
	@echo "Running softmax bf16 test with M=$(M), N=$(N) on $(TARGET)..."
	@echo "Using transform script: $(TRANSFORM_SCRIPT)"
	@echo "Using input MLIR: $(INPUT_MLIR)"
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--input-mlir $(INPUT_MLIR) \
			--transform-script $(TRANSFORM_SCRIPT) \
			--M $(M) \
			--N $(N) \
			$(VERBOSE_FLAG) \
			$(DEBUG_AIRCC_FLAG)

# Compile xclbin only (no execution)
# NPU1 requires kernel compilation, NPU2 doesn't
ifeq ($(NEEDS_KERNEL),yes)
compile-xclbin: compile-kernel
else
compile-xclbin:
endif
	@mkdir -p $(BUILD_DIR)
	@echo "Using transform script: $(TRANSFORM_SCRIPT)"
	@echo "Using input MLIR: $(INPUT_MLIR)"
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--input-mlir $(INPUT_MLIR) \
			--transform-script $(TRANSFORM_SCRIPT) \
			--M $(M) \
			--N $(N) \
			--compile-only \
			$(VERBOSE_FLAG) \
			$(DEBUG_AIRCC_FLAG)

# Profile using C++ executable
profile: compile-xclbin build-test-exe
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin -M $(M) -N $(N) -v $(VERBOSE)

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
	if [ ! -d "$(XILINX_XRT)" ]; then \
		echo "Error: XILINX_XRT directory not found: $(XILINX_XRT)"; \
		echo "Please set XILINX_XRT or source xrt/setup.sh."; \
		exit 1; \
	fi; \
	if [ -z "$(AIEOPT_DIR)" ]; then \
		echo "Error: AIEOPT_DIR environment variable not set. Please make sure to have sourced utils/env_setup.sh."; \
		exit 1; \
	fi; \
	echo "Using compiler: $$GPP"; \
	echo "Using XRT: $(XILINX_XRT)"; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP $(srcdir)/test.cpp -o test.exe -std=c++23 -Wall \
		-I$(XILINX_XRT)/include -L$(XILINX_XRT)/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

# Debug IR: dump transformed IR to file and exit
# NPU1 requires kernel compilation, NPU2 doesn't
ifeq ($(NEEDS_KERNEL),yes)
debug_ir: compile-kernel
else
debug_ir:
endif
	@mkdir -p $(BUILD_DIR)
	@echo "Using transform script: $(TRANSFORM_SCRIPT)"
	@echo "Using input MLIR: $(INPUT_MLIR)"
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--input-mlir $(INPUT_MLIR) \
			--transform-script $(TRANSFORM_SCRIPT) \
			--M $(M) \
			--N $(N) \
			--debug-ir $(DEBUG_IR_PATH)

# Debug AIRCC: compile with per-pass IR output for fine-grained debugging
# NPU1 requires kernel compilation, NPU2 doesn't
ifeq ($(NEEDS_KERNEL),yes)
debug_aircc: compile-kernel
else
debug_aircc:
endif
	@mkdir -p $(BUILD_DIR)
	@echo "Debug AIRCC mode: emitting IR after each individual pass..."
	@echo "Using transform script: $(TRANSFORM_SCRIPT)"
	@echo "Using input MLIR: $(INPUT_MLIR)"
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--input-mlir $(INPUT_MLIR) \
			--transform-script $(TRANSFORM_SCRIPT) \
			--M $(M) \
			--N $(N) \
			--compile-only \
			--verbose \
			--debug-aircc
	@echo ""
	@echo "Debug IR files saved to: $(BUILD_DIR)/air_project/debug_ir/"
	@echo "Pass log saved to: $(BUILD_DIR)/air_project/pass.log"

clean:
	rm -rf build_* __pycache__
	@echo "Cleaned build directories"
