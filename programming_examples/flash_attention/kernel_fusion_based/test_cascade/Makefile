# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Cascade connection verification test for AIE2P.
#
# Usage:
#   make compile-kernel    # Compile both kernel .o files
#   make run               # Compile kernels + run tests on NPU
#   make print             # Print generated MLIR (no compilation)
#   make clean

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

BUF_SIZE ?= 1024
BUILD_DIR := build_cascade

# Flash attention merge dimensions
TILE_SIZE_Q ?= 64
LKP ?= 64
DK ?= 64
DV ?= 64

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

all: run

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Compiling cascade_kernel.o (simple tests)..."; \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
			-c ${srcdir}/cascade_kernel.cc \
			-o $(BUILD_DIR)/cascade_kernel.o \
			-DBUF_SIZE=$(BUF_SIZE); \
		echo "Compiling cascade_merge_kernel.o (merge test)..."; \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
			-c ${srcdir}/cascade_merge_kernel.cc \
			-o $(BUILD_DIR)/cascade_merge_kernel.o \
			-Dlqp=$(TILE_SIZE_Q) -Dlkp=$(LKP) -Ddk=$(DK) -Ddv=$(DV) \
			-DROUND_CONV_EVEN; \
	else \
		echo "Error: PEANO_INSTALL_DIR not set."; \
		exit 1; \
	fi

run: compile-kernel
	cd ${srcdir} && python3 cascade_test.py $(EXTRA_PY_FLAGS)

print:
	cd ${srcdir} && python3 cascade_test.py -p

clean:
	rm -rf $(BUILD_DIR) __pycache__
