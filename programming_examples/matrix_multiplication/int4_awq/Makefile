# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# int4 AWQ GEMM example (packed Q+S+Z BO). Reuses mv_int4_bf16.cc from the
# matrix_vector_multiplication/int4_awq examples — the GEMM symbol is in the
# same .o file as the GEMV/zero/partial_plus_r helpers.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
INT4_SRCDIR := $(srcdir)/../../matrix_vector_multiplication/int4_awq

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= elf
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

# Shapes / tiling (overridable). Defaults match the smallest correctness test.
M       ?= 32
K       ?= 128
N       ?= 64
GS      ?= 128
M_TILE  ?= 16
N_TILE  ?= 16
K_CHUNK ?= 128
N_CORES ?= 4

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

PY_ARGS = --m $(M) --k $(K) --n $(N) --gs $(GS) --m-tile $(M_TILE) --n-tile $(N_TILE) --k-chunk $(K_CHUNK) --n-cores $(N_CORES)

all: run_packed

print_packed:
	${powershell} python3 ${srcdir}/matmul_int4_packed.py $(OUTPUT_FORMAT_FLAG) -p $(PY_ARGS)

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -z "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Error: PEANO_INSTALL_DIR not set (source utils/env_setup.sh)."; \
		exit 1; \
	fi
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DDIM_M=$(M_TILE) -DDIM_N=$(N_TILE) -DDIM_K=$(K_CHUNK) -DDIM_GS=$(GS) \
		-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 \
		-c $(INT4_SRCDIR)/mv_int4_bf16.cc -o $(BUILD_DIR)/mv_int4_bf16.o

run_packed: compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 ${srcdir}/matmul_int4_packed.py $(OUTPUT_FORMAT_FLAG) $(PY_ARGS)

clean:
	rm -rf $(BUILD_DIR) __pycache__
