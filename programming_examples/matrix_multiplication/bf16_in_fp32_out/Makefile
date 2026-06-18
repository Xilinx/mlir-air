# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# bf16-in / f32-out GEMM. Output is f32 → the accumulator IS the output, so the
# K reduction is FP32 the whole way with NO intermediate bf16 truncation. Always
# "high precision". Two codegen backends (same math): external (mm.o, fast) and
# direct (compiler, self-contained).
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Tile config (registry-best for large GEMMs on NPU2, full-chip 8x4 herd).
TILE_M ?= 64
TILE_K_L2 ?= 256
TILE_K_L1 ?= 32
TILE_N ?= 128
HERD_M ?= 8
HERD_N ?= 4
M ?= 2048
K ?= 8192
N ?= 2048

# external (default) needs mm.o; direct needs only PEANO. CODEGEN={external,direct}
CODEGEN ?= external
AIE_TARGET ?= aie2p
COMPILE_MODE ?= compile-and-run
PERF_ITERS ?= 20

TILE_M_DIV_4 := $(shell echo $$(( $(TILE_M) / 4 )))
TILE_N_DIV_4 := $(shell echo $$(( $(TILE_N) / 4 )))
TILE_M_DIV_8 := $(shell echo $$(( $(TILE_M) / 8 )))
TILE_N_DIV_8 := $(shell echo $$(( $(TILE_N) / 8 )))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

all: run

# Compile the external microkernel mm.o for the current tile config (external only).
compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ "$(CODEGEN)" = "external" ] && [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c ${srcdir}/mm_aie2p.cc -o $(BUILD_DIR)/mm.o \
			-DDIM_M=$(TILE_M) -DDIM_N=$(TILE_N) -DDIM_K=$(TILE_K_L1) \
			-DDIM_N_DIV_4=$(TILE_N_DIV_4) -DDIM_M_DIV_4=$(TILE_M_DIV_4) \
			-DDIM_N_DIV_8=$(TILE_N_DIV_8) -DDIM_M_DIV_8=$(TILE_M_DIV_8) \
			-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16; \
	fi

run: compile-kernel
	cd $(BUILD_DIR) && python3 ${srcdir}/run.py --arch $(AIE_TARGET) --codegen $(CODEGEN) \
		--m $(M) --k $(K) --n $(N) \
		--tile-m $(TILE_M) --tile-k-l2 $(TILE_K_L2) --tile-k-l1 $(TILE_K_L1) --tile-n $(TILE_N) \
		--herd-m $(HERD_M) --herd-n $(HERD_N) --compile-mode $(COMPILE_MODE) --perf-iters $(PERF_ITERS)

clean:
	rm -rf build_peano build_chess
