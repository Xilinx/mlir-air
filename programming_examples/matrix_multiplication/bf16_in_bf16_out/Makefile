# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# bf16-in / bf16-out GEMM. Primary knob HIGH_PRECISION:
#   true  (default) = FP32 accumulate across the whole K reduction + single
#                     epilogue cast to bf16 (GPU standard). METHOD picks how:
#                       auto (default) = fused-cast for large shapes, drain for small.
#                       fused-cast     = external GEMM f32-out (tile_m=64) + cast launch.
#                       drain          = in-GEMM drain cast (single launch, tile_m=32).
#   false = direct-codegen bf16 (per-L2-tile bf16 truncation; faster, lower precision).
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

TILE_M ?= 64
TILE_K_L2 ?= 256
TILE_K_L1 ?= 32
TILE_N ?= 128
HERD_M ?= 8
HERD_N ?= 4
M ?= 2048
K ?= 8192
N ?= 2048

HIGH_PRECISION ?= true
METHOD ?= auto
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

# mm.o is needed for high_precision=true (fused-cast / drain both use the external
# GEMM). DIM_M must match TILE_M (fused-cast=64, drain forces 32 — pass TILE_M=32).
compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c ${srcdir}/mm_aie2p.cc -o $(BUILD_DIR)/mm.o \
			-DDIM_M=$(TILE_M) -DDIM_N=$(TILE_N) -DDIM_K=$(TILE_K_L1) \
			-DDIM_N_DIV_4=$(TILE_N_DIV_4) -DDIM_M_DIV_4=$(TILE_M_DIV_4) \
			-DDIM_N_DIV_8=$(TILE_N_DIV_8) -DDIM_M_DIV_8=$(TILE_M_DIV_8) \
			-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16; \
	fi

run: compile-kernel
	cd $(BUILD_DIR) && python3 ${srcdir}/run.py --arch $(AIE_TARGET) \
		--high-precision $(HIGH_PRECISION) --method $(METHOD) \
		--m $(M) --k $(K) --n $(N) \
		--tile-m $(TILE_M) --tile-k-l2 $(TILE_K_L2) --tile-k-l1 $(TILE_K_L1) --tile-n $(TILE_N) \
		--herd-m $(HERD_M) --herd-n $(HERD_N) --compile-mode $(COMPILE_MODE) --perf-iters $(PERF_ITERS)

clean:
	rm -rf build_peano build_chess
