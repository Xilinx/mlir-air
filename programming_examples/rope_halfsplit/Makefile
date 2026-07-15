# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# NPU2 (Strix / AIE2P) only. The kernel links an external C++ vector
# microkernel (rope_halfsplit.cc -> rope.o, the half-split RoPE that
# llama-3.2-1B links via external_kernels.py:compile_rope). No NPU1 path.
BUILD_DIR := build_peano

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include -D__AIE_API_AIE_ADF_HPP__

OUTPUT_FORMAT ?= xclbin

# Best config (see kernel_registry/details/RoPE_bf16.md): full chip width.
# ROWS = 65536 = n_heads(32) * seq(2048) is llama-3.2-1B prefill RoPE Q scale.
ROWS ?= 65536
HEAD_DIM ?= 64
HERD_X ?= 8
HERD_Y ?= 1
ITERATIONS ?= 30

all: run

# Compile the external half-split RoPE microkernel to rope.o. aircc picks it
# up from the cwd (BUILD_DIR) and copies it into air_project/.
compile-kernel:
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-c $(srcdir)/rope_halfsplit.cc -o $(BUILD_DIR)/rope.o

# Print the MLIR module (no kernel .o / NPU needed)
print:
	${powershell} python3 ${srcdir}/rope_halfsplit.py \
		--rows $(ROWS) --head-dim $(HEAD_DIM) --herd-x $(HERD_X) \
		--herd-y $(HERD_Y) -p

# Compile + run correctness check on NPU2
run: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/rope_halfsplit.py \
		--output-format $(OUTPUT_FORMAT) \
		--rows $(ROWS) --head-dim $(HEAD_DIM) --herd-x $(HERD_X) \
		--herd-y $(HERD_Y)

# Compile + run with latency profiling (correctness still checked)
profile: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/rope_halfsplit.py \
		--output-format $(OUTPUT_FORMAT) \
		--rows $(ROWS) --head-dim $(HEAD_DIM) --herd-x $(HERD_X) \
		--herd-y $(HERD_Y) \
		--perf-iters $(ITERATIONS)

clean:
	rm -rf $(BUILD_DIR) __pycache__
