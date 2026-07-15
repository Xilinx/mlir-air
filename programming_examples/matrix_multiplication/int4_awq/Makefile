# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# int4-AWQ GEMM example. The kernel .cc lives in the int4-AWQ GEMV sibling
# directory and is shared (matmul_int4_bf16_packed + zero_vectorized_bf16_mn).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
INT4_SRCDIR := $(srcdir)/../../matrix_vector_multiplication/int4_awq

# TILE_K_L2 must equal K: the segment-level K loop has 1 iter so the per-PE
# L1 C accumulator survives across all K_CHUNK iterations.
M       ?= 64
K       ?= 128
N       ?= 128
GS      ?= 128
TILE_M  ?= 16
TILE_N  ?= 16
TILE_K_L1 ?= 128
TILE_K_L2 ?= $(K)
HERD_M  ?= 2
HERD_N  ?= 4

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include -D__AIE_API_AIE_ADF_HPP__

PY_ARGS = --m $(M) --k $(K) --n $(N) --gs $(GS) \
          --tile-m $(TILE_M) --tile-n $(TILE_N) \
          --tile-k-l1 $(TILE_K_L1) --tile-k-l2 $(TILE_K_L2) \
          --herd-m $(HERD_M) --herd-n $(HERD_N)

all: run_packed

print:
	${powershell} python3 ${srcdir}/matmul_int4_packed.py $(PY_ARGS) -p

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -z "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Error: PEANO_INSTALL_DIR not set (source utils/env_setup.sh)."; \
		exit 1; \
	fi
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DDIM_M=$(TILE_M) -DDIM_N=$(TILE_N) -DDIM_K_CHUNK=$(TILE_K_L1) -DDIM_GS=$(GS) \
		-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 \
		-c $(INT4_SRCDIR)/mv_int4_bf16.cc -o $(BUILD_DIR)/mv_int4_bf16.o

run_packed: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 ${srcdir}/matmul_int4_packed.py $(PY_ARGS)

run1x1: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 ${srcdir}/matmul_int4_packed.py \
	    --m 16 --k 128 --n 16 --gs $(GS) \
	    --tile-m 16 --tile-n 16 --tile-k-l1 128 --tile-k-l2 128 \
	    --herd-m 1 --herd-n 1

run2x4: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 ${srcdir}/matmul_int4_packed.py \
	    --m 64 --k 128 --n 128 --gs $(GS) \
	    --tile-m $(TILE_M) --tile-n $(TILE_N) \
	    --tile-k-l1 $(TILE_K_L1) --tile-k-l2 128 \
	    --herd-m 2 --herd-n 4

run8x4: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 ${srcdir}/matmul_int4_packed.py \
	    --m 256 --k 128 --n 128 --gs $(GS) \
	    --tile-m $(TILE_M) --tile-n $(TILE_N) \
	    --tile-k-l1 $(TILE_K_L1) --tile-k-l2 128 \
	    --herd-m 8 --herd-n 4

# Llama Q-proj scale: M=N=K=2048, herd 8x4, TILE_K_L2=K (no re-zero).
run_llama_qproj: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 ${srcdir}/matmul_int4_packed.py \
	    --m 2048 --k 2048 --n 2048 --gs $(GS) \
	    --tile-m 16 --tile-n 16 --tile-k-l1 128 --tile-k-l2 2048 \
	    --herd-m 8 --herd-n 4

clean:
	rm -rf $(BUILD_DIR) __pycache__
