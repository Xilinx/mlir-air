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

clean:
	rm -rf $(BUILD_DIR) __pycache__
