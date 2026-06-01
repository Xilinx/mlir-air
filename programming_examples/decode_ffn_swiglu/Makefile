# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Single-token GEMV with fused weighted RMSNorm input and SwiGLU output.
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

# Default shape: M = 2*hidden, K = emb_dim. Tuned for an 8-col herd at
# emb_dim=2048 / hidden_dim=8192.
M ?= 16384
K ?= 2048
TILE_M ?= 32
M_INPUT ?= 4
HERD_COLS ?= 8
N_CASCADE ?= 4

all: run

print:
	${powershell} python3 ${srcdir}/matvec_swiglu_rms.py $(OUTPUT_FORMAT_FLAG) -p \
		--m $(M) --k $(K) --tile-m $(TILE_M) --m-input $(M_INPUT) \
		--herd-cols $(HERD_COLS) --n-cascade $(N_CASCADE)

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 ${srcdir}/matvec_swiglu_rms.py $(OUTPUT_FORMAT_FLAG) \
		--m $(M) --k $(K) --tile-m $(TILE_M) --m-input $(M_INPUT) \
		--herd-cols $(HERD_COLS) --n-cascade $(N_CASCADE)

# int4-AWQ variant: M = 2*hidden, K = emb. Reuses mv_int4_bf16.o from the
# int4_awq examples (same M_TILE / K_CHUNK / GS macros).
INT4_SRCDIR := $(srcdir)/../matrix_vector_multiplication/int4_awq
INT4_M_TILE  ?= 8
INT4_K_CHUNK ?= 2048
INT4_GS      ?= 128

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf \
	-Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
	-DNDEBUG -I ${AIEOPT_DIR}/include

compile-kernel-int4:
	mkdir -p $(BUILD_DIR)
	@if [ -z "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Error: PEANO_INSTALL_DIR not set (source utils/env_setup.sh)."; \
		exit 1; \
	fi
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DDIM_M=$(INT4_M_TILE) -DDIM_K=$(INT4_K_CHUNK) -DDIM_GS=$(INT4_GS) \
		-c $(INT4_SRCDIR)/mv_int4_bf16.cc -o $(BUILD_DIR)/mv_int4_bf16.o

run_int4: compile-kernel-int4
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 ${srcdir}/matvec_int4_swiglu_rms.py $(OUTPUT_FORMAT_FLAG) \
		--m $(M) --k $(K) --gs $(INT4_GS) \
		--m-tile $(INT4_M_TILE) --k-chunk $(INT4_K_CHUNK)

clean:
	rm -rf $(BUILD_DIR) __pycache__
