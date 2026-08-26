# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

# NPU2 (Strix / AIE2P) only. The kernel links an external C++ vector
# microkernel (conv1d_depthwise.cc -> conv1d_depthwise.o), so there is no
# NPU1 / direct-codegen path.
BUILD_DIR := build_peano

AIEOPT_DIR ?= $(if $(MLIR_AIE_INSTALL_DIR),$(subst \,/,$(MLIR_AIE_INSTALL_DIR)),$(shell realpath $(dir $(shell which aie-opt))/..))
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include -D__AIE_API_AIE_ADF_HPP__

OUTPUT_FORMAT ?= xclbin

# LFM2-1.2B ShortConv prefill scale: seq = 2048, conv_dim = 2048.
SEQ ?= 2048
CHANNELS ?= 2048
TILE_S ?= 32
HERD_X ?= 8
HERD_Y ?= 1
ITERATIONS ?= 30
EXTRA ?=

all: run

# Compile the external depthwise-conv C++ microkernel. aircc picks it up from
# the cwd (BUILD_DIR) and copies it into air_project/.
compile-kernel:
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-include aie_kernels/aie_kernel_utils.h \
		-c $(srcdir)/conv1d_depthwise.cc -o $(BUILD_DIR)/conv1d_depthwise.o

# Print the MLIR module (no kernel .o / NPU needed)
print:
	${powershell} $(PYTHON) ${srcdir}/conv1d_depthwise.py \
		--seq $(SEQ) --channels $(CHANNELS) --tile-s $(TILE_S) \
		--herd-x $(HERD_X) --herd-y $(HERD_Y) -p

# Compile + run correctness check on NPU2
run: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} $(PYTHON) ${srcdir}/conv1d_depthwise.py \
		--output-format $(OUTPUT_FORMAT) \
		--seq $(SEQ) --channels $(CHANNELS) --tile-s $(TILE_S) \
		--herd-x $(HERD_X) --herd-y $(HERD_Y) $(EXTRA)

# Compile + run with latency/bandwidth profiling (correctness still checked)
profile: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} $(PYTHON) ${srcdir}/conv1d_depthwise.py \
		--output-format $(OUTPUT_FORMAT) \
		--seq $(SEQ) --channels $(CHANNELS) --tile-s $(TILE_S) \
		--herd-x $(HERD_X) --herd-y $(HERD_Y) \
		--perf-iters $(ITERATIONS)

clean:
	rm -rf $(BUILD_DIR) __pycache__
