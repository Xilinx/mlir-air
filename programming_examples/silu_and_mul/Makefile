# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# NPU2 (Strix / AIE2P) only. The kernel links an external C++ vector
# microkernel (silu_and_mul.cc -> silu_and_mul.o) that needs the hardware
# aie::tanh, so there is no NPU1 / direct-codegen path.
BUILD_DIR := build_peano

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include -D__AIE_API_AIE_ADF_HPP__

OUTPUT_FORMAT ?= xclbin

# Best config (see kernel_registry/details/SiLU_Mul_bf16.md): full chip width
# at the fastest placeable tile. N = 2048*8192 is llama-3.2-1B prefill scale.
N ?= 16777216
TILE_N ?= 4096
HERD_X ?= 8
HERD_Y ?= 1
ITERATIONS ?= 30

all: run

# Compile the external SwiGLU C++ microkernel to silu_and_mul.o. aircc picks
# it up from the cwd (BUILD_DIR) and copies it into air_project/.
compile-kernel:
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-include aie_kernels/aie_kernel_utils.h \
		-c $(srcdir)/silu_and_mul.cc -o $(BUILD_DIR)/silu_and_mul.o

# Print the MLIR module (no kernel .o / NPU needed)
print:
	${powershell} python3 ${srcdir}/silu_and_mul.py \
		--n $(N) --tile-n $(TILE_N) --herd-x $(HERD_X) --herd-y $(HERD_Y) -p

# Compile + run correctness check on NPU2
run: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/silu_and_mul.py \
		--output-format $(OUTPUT_FORMAT) \
		--n $(N) --tile-n $(TILE_N) --herd-x $(HERD_X) --herd-y $(HERD_Y)

# Compile + run with latency/bandwidth profiling (correctness still checked)
profile: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/silu_and_mul.py \
		--output-format $(OUTPUT_FORMAT) \
		--n $(N) --tile-n $(TILE_N) --herd-x $(HERD_X) --herd-y $(HERD_Y) \
		--perf-iters $(ITERATIONS)

clean:
	rm -rf $(BUILD_DIR) __pycache__
