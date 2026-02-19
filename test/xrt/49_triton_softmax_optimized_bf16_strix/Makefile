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

# AIE target architecture and transform script selection
# NPU1 (aie2) requires extern_func.cc compilation for getExpBf16
# NPU2 (aie2p) uses native intrinsics, no external kernel needed
ifeq ($(TARGET),npu1)
  AIE_TARGET := aie2-none-unknown-elf
  TRANSFORM_SCRIPT_1D ?= $(srcdir)/transform_aie2.mlir
  TRANSFORM_SCRIPT_2D ?= $(srcdir)/transform_aie2_2d.mlir
  NEEDS_KERNEL := yes
else ifeq ($(TARGET),npu2)
  AIE_TARGET := aie2p-none-unknown-elf
  TRANSFORM_SCRIPT_1D ?= $(srcdir)/transform_aie2p.mlir
  TRANSFORM_SCRIPT_2D ?= $(srcdir)/transform_aie2p_2d.mlir
  NEEDS_KERNEL := no
else
  $(error Unknown TARGET: $(TARGET). Use npu1 or npu2)
endif

# Input MLIR files for 1D and 2D
INPUT_MLIR_1D ?= $(srcdir)/input_ir/initial/4x1024_1d.mlir
INPUT_MLIR_2D ?= $(srcdir)/input_ir/initial/4x4x1024_2d.mlir

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

.PHONY: all help clean compile-kernel build-test-exe \
        run-1d compile-xclbin-1d profile-1d debug-aircc-1d \
        run-2d compile-xclbin-2d profile-2d debug-aircc-2d \
        run-pretransform-1d-baseline run-pretransform-1d-optimized \
        profile-pretransform-1d-baseline profile-pretransform-1d-optimized \
        run-pretransform-2d-optimized profile-pretransform-2d-optimized

# Default target: 2D (16 cores, best performance)
all: run-2d

help:
	@echo "Triton Softmax BF16 Example for AMD AI Engine"
	@echo ""
	@echo "Usage:"
	@echo "  make [TARGET=npu1|npu2] [M=256] [N=1024]"
	@echo ""
	@echo "============================================================"
	@echo "1D HERD (1×4, 4 cores)"
	@echo "============================================================"
	@echo "  run-1d                          Run validation (transform script)"
	@echo "  compile-xclbin-1d               Compile to xclbin only"
	@echo "  profile-1d                      Profile performance"
	@echo "  debug-aircc-1d                  Per-pass IR output for debugging"
	@echo ""
	@echo "  Pre-transformed IR (skip transform script):"
	@echo "  run-pretransform-1d-baseline    Run baseline_transformed.mlir"
	@echo "  run-pretransform-1d-optimized   Run optimized_transformed.mlir"
	@echo "  profile-pretransform-1d-baseline   Profile baseline (scalar reduction)"
	@echo "  profile-pretransform-1d-optimized  Profile optimized (vector accumulation)"
	@echo ""
	@echo "============================================================"
	@echo "2D HERD (4×4, 16 cores) - DEFAULT"
	@echo "============================================================"
	@echo "  run-2d                          Run validation (transform script)"
	@echo "  compile-xclbin-2d               Compile to xclbin only"
	@echo "  profile-2d                      Profile performance (~123us)"
	@echo "  debug-aircc-2d                  Per-pass IR output for debugging"
	@echo ""
	@echo "  Pre-transformed IR (skip transform script):"
	@echo "  run-pretransform-2d-optimized   Run optimized_transformed_2d.mlir"
	@echo "  profile-pretransform-2d-optimized  Profile optimized (~75us, 39%% faster)"
	@echo ""
	@echo "============================================================"
	@echo "UTILITY"
	@echo "============================================================"
	@echo "  compile-kernel                  Compile extern_func.cc (NPU1 only)"
	@echo "  build-test-exe                  Build C++ profiler executable"
	@echo "  clean                           Remove build artifacts"
	@echo "  help                            Show this message"
	@echo ""
	@echo "============================================================"
	@echo "VARIABLES"
	@echo "============================================================"
	@echo "  TARGET       npu1 (Phoenix) or npu2 (Strix, default)"
	@echo "  M            Parallel dimension size (default: 256)"
	@echo "  N            Reduction dimension size (default: 1024)"
	@echo "  VERBOSE      Enable verbose mode (1 = enabled, default: 0)"
	@echo "  DEBUG_AIRCC  Enable per-pass IR output (1 = enabled, default: 0)"
	@echo ""
	@echo "Examples:"
	@echo "  make                            # Default: run-2d"
	@echo "  make run-1d                     # Run 1D validation"
	@echo "  make profile-2d                 # Profile 2D baseline"
	@echo "  make profile-pretransform-2d-optimized  # Best performance (75us)"

# ============================================================
# UTILITY TARGETS
# ============================================================

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

clean:
	rm -rf build_* __pycache__ debug_output air_project
	@echo "Cleaned build directories"

# ============================================================
# 1D HERD (1×4, 4 cores) TARGETS
# ============================================================

# Run validation (1D, 4 cores)
ifeq ($(NEEDS_KERNEL),yes)
run-1d: compile-kernel
else
run-1d:
endif
	@mkdir -p $(BUILD_DIR)
	@echo "Running 1D softmax (1×4 herd, 4 cores)..."
	@echo "Using transform script: $(TRANSFORM_SCRIPT_1D)"
	@echo "Using input MLIR: $(INPUT_MLIR_1D)"
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--input-mlir $(INPUT_MLIR_1D) \
			--transform-script $(TRANSFORM_SCRIPT_1D) \
			--M $(M) \
			--N $(N) \
			--tile-rows 4 \
			--herd-shape 1x4 \
			$(VERBOSE_FLAG) \
			$(DEBUG_AIRCC_FLAG)

# Compile xclbin only (1D)
ifeq ($(NEEDS_KERNEL),yes)
compile-xclbin-1d: compile-kernel
else
compile-xclbin-1d:
endif
	@mkdir -p $(BUILD_DIR)
	@echo "Compiling 1D softmax (1×4 herd, 4 cores)..."
	@echo "Using transform script: $(TRANSFORM_SCRIPT_1D)"
	@echo "Using input MLIR: $(INPUT_MLIR_1D)"
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--input-mlir $(INPUT_MLIR_1D) \
			--transform-script $(TRANSFORM_SCRIPT_1D) \
			--M $(M) \
			--N $(N) \
			--tile-rows 4 \
			--herd-shape 1x4 \
			--compile-only \
			$(VERBOSE_FLAG) \
			$(DEBUG_AIRCC_FLAG)

# Profile (1D)
profile-1d: compile-xclbin-1d build-test-exe
	@echo ""
	@echo "Profiling 1D softmax..."
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin -M $(M) -N $(N) -v $(VERBOSE)

# Debug AIRCC (1D)
ifeq ($(NEEDS_KERNEL),yes)
debug-aircc-1d: compile-kernel
else
debug-aircc-1d:
endif
	@mkdir -p $(BUILD_DIR)
	@echo "Debug AIRCC mode (1D): emitting IR after each individual pass..."
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--input-mlir $(INPUT_MLIR_1D) \
			--transform-script $(TRANSFORM_SCRIPT_1D) \
			--M $(M) \
			--N $(N) \
			--tile-rows 4 \
			--herd-shape 1x4 \
			--compile-only \
			--verbose \
			--debug-aircc
	@echo ""
	@echo "Debug IR files saved to: $(BUILD_DIR)/air_project/debug_ir/"

# ============================================================
# 1D PRE-TRANSFORMED IR TARGETS
# ============================================================

# Run validation with baseline_transformed.mlir (1D)
run-pretransform-1d-baseline:
	@mkdir -p $(BUILD_DIR)
	@echo "Running 1D BASELINE (pre-transformed IR)..."
	@echo "Using: baseline_transformed.mlir"
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--pre-transformed-ir $(srcdir)/input_ir/pre-transformed/4x1024_baseline_1d.mlir \
			--M $(M) \
			--N $(N) \
			--tile-rows 4 \
			--herd-shape 1x4 \
			$(VERBOSE_FLAG) \
			$(DEBUG_AIRCC_FLAG)

# Run validation with optimized_transformed.mlir (1D)
run-pretransform-1d-optimized:
	@mkdir -p $(BUILD_DIR)
	@echo "Running 1D OPTIMIZED (pre-transformed IR)..."
	@echo "Using: 4x1024_optimized_1d.mlir"
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--pre-transformed-ir $(srcdir)/input_ir/pre-transformed/4x1024_optimized_1d.mlir \
			--M $(M) \
			--N $(N) \
			--tile-rows 4 \
			--herd-shape 1x4 \
			$(VERBOSE_FLAG) \
			$(DEBUG_AIRCC_FLAG)

# Profile baseline_transformed.mlir (1D)
profile-pretransform-1d-baseline: build-test-exe
	@mkdir -p $(BUILD_DIR)
	@echo "Profiling 1D BASELINE (scalar reduction inside loop)..."
	@echo "Using: 4x1024_baseline_1d.mlir"
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--pre-transformed-ir $(srcdir)/input_ir/pre-transformed/4x1024_baseline_1d.mlir \
			--M $(M) \
			--N $(N) \
			--tile-rows 4 \
			--herd-shape 1x4 \
			--compile-only \
			$(VERBOSE_FLAG)
	@echo ""
	@echo "Running profiling..."
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin -M $(M) -N $(N) -v $(VERBOSE)

# Profile optimized_transformed.mlir (1D)
profile-pretransform-1d-optimized: build-test-exe
	@mkdir -p $(BUILD_DIR)
	@echo "Profiling 1D OPTIMIZED (vector accumulation, single post-loop reduction)..."
	@echo "Using: 4x1024_optimized_1d.mlir"
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--pre-transformed-ir $(srcdir)/input_ir/pre-transformed/4x1024_optimized_1d.mlir \
			--M $(M) \
			--N $(N) \
			--tile-rows 4 \
			--herd-shape 1x4 \
			--compile-only \
			$(VERBOSE_FLAG)
	@echo ""
	@echo "Running profiling..."
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin -M $(M) -N $(N) -v $(VERBOSE)

# ============================================================
# 2D HERD (4×4, 16 cores) TARGETS
# ============================================================

# Run validation (2D, 16 cores)
run-2d:
	@mkdir -p $(BUILD_DIR)
	@echo "Running 2D softmax (4×4 herd, 16 cores)..."
	@echo "Using transform script: $(TRANSFORM_SCRIPT_2D)"
	@echo "Using input MLIR: $(INPUT_MLIR_2D)"
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--input-mlir $(INPUT_MLIR_2D) \
			--transform-script $(TRANSFORM_SCRIPT_2D) \
			--M $(M) \
			--N $(N) \
			--tile-rows 16 \
			--herd-shape 4x4 \
			$(VERBOSE_FLAG) \
			$(DEBUG_AIRCC_FLAG)

# Compile xclbin only (2D)
compile-xclbin-2d:
	@mkdir -p $(BUILD_DIR)
	@echo "Compiling 2D softmax (4×4 herd, 16 cores)..."
	@echo "Using transform script: $(TRANSFORM_SCRIPT_2D)"
	@echo "Using input MLIR: $(INPUT_MLIR_2D)"
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--input-mlir $(INPUT_MLIR_2D) \
			--transform-script $(TRANSFORM_SCRIPT_2D) \
			--M $(M) \
			--N $(N) \
			--tile-rows 16 \
			--herd-shape 4x4 \
			--compile-only \
			$(VERBOSE_FLAG) \
			$(DEBUG_AIRCC_FLAG)

# Profile (2D)
profile-2d: compile-xclbin-2d build-test-exe
	@echo ""
	@echo "Profiling 2D softmax..."
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin -M $(M) -N $(N) -v $(VERBOSE)

# Debug AIRCC (2D)
debug-aircc-2d:
	@mkdir -p $(BUILD_DIR)
	@echo "Debug AIRCC mode (2D): emitting IR after each individual pass..."
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--input-mlir $(INPUT_MLIR_2D) \
			--transform-script $(TRANSFORM_SCRIPT_2D) \
			--M $(M) \
			--N $(N) \
			--tile-rows 16 \
			--herd-shape 4x4 \
			--compile-only \
			--verbose \
			--debug-aircc
	@echo ""
	@echo "Debug IR files saved to: $(BUILD_DIR)/air_project/debug_ir/"

# ============================================================
# 2D PRE-TRANSFORMED IR TARGETS
# ============================================================

# Run validation with optimized_transformed_2d.mlir (2D)
run-pretransform-2d-optimized:
	@mkdir -p $(BUILD_DIR)
	@echo "Running 2D OPTIMIZED (pre-transformed IR)..."
	@echo "Using: 4x4x1024_optimized_2d.mlir (hoisted reductions)"
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--pre-transformed-ir $(srcdir)/input_ir/pre-transformed/4x4x1024_optimized_2d.mlir \
			--M $(M) \
			--N $(N) \
			--tile-rows 16 \
			--herd-shape 4x4 \
			$(VERBOSE_FLAG) \
			$(DEBUG_AIRCC_FLAG)

# Profile optimized_transformed_2d.mlir (2D)
profile-pretransform-2d-optimized: build-test-exe
	@mkdir -p $(BUILD_DIR)
	@echo "Profiling 2D OPTIMIZED (4×4 herd, hoisted reductions)..."
	@echo "Using: 4x4x1024_optimized_2d.mlir"
	@cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) \
		$(powershell) python3 $(srcdir)/run.py \
			--pre-transformed-ir $(srcdir)/input_ir/pre-transformed/4x4x1024_optimized_2d.mlir \
			--M $(M) \
			--N $(N) \
			--tile-rows 16 \
			--herd-shape 4x4 \
			--compile-only \
			$(VERBOSE_FLAG)
	@echo ""
	@echo "Running profiling..."
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin -M $(M) -N $(N) -v $(VERBOSE)
