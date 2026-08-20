# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Tile sizes.
# bfp16ebs8 constrains these more than bf16 does: the host shuffle requires
# TILE_K_L1 % 64 == 0 and TILE_N % 64 == 0 (tile width in elements), and
# mm_bfp.cc requires TILE_M % 16 == 0.
TILE_M ?= 64
TILE_K_L2 ?= 128
TILE_K_L1 ?= 64
TILE_N ?= 64

# bfp16ebs8 is an AIE2P-only type and has no Chess codegen path, so unlike the
# bf16 example there is no aie2 / xchesscc branch here. run.py exits with a
# diagnostic if --arch aie2 is requested.
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include -D__AIE_API_AIE_ADF_HPP__

# The GEMM kernel is mlir-aie's reference design, used unmodified. It ships in
# the mlir-aie install tree, so resolve it through AIEOPT_DIR rather than
# assuming a source checkout is present next to mlir-air.
KERNEL_SRC ?= ${AIEOPT_DIR}/include/aie_kernels/aie2p/mm_bfp.cc

# AIE_TARGET is accepted for symmetry with the bf16 example, but aie2p is the
# only legal value; run.py rejects aie2 with a diagnostic.
# COMPILE_MODE can be 'compile-and-run' (default), 'compile-and-xclbin', or
# 'compile-only'.
# Usage examples:
#   make run4x4                            # compile-and-run (default)
#   make run4x4 OUTPUT_DTYPE=bf16          # bf16 instead of f32 output
#   make run4x4 COMPILE_MODE=compile-only  # no XRT needed
#   make run4x4 TILE_M=128 TILE_K_L2=256
AIE_TARGET ?= aie2p
COMPILE_MODE ?= compile-and-run
MAX_HERD_M = 8
MAX_HERD_N = 4

OUTPUT_DTYPE ?=
ifneq ($(OUTPUT_DTYPE),)
  OUTPUT_DTYPE_FLAG = --output-dtype $(OUTPUT_DTYPE)
else
  OUTPUT_DTYPE_FLAG =
endif

COMMON_ARGS = --tile-m $(TILE_M) --tile-k-l2 $(TILE_K_L2) --tile-k-l1 $(TILE_K_L1) --tile-n $(TILE_N) --compile-mode $(COMPILE_MODE) --arch $(AIE_TARGET) $(OUTPUT_DTYPE_FLAG)

all: run4x4

print:
	${powershell} python3 ${srcdir}/run.py -p --herd-m 8 --herd-n 4 --m 2048 --n 1024 --k 512 \
		--tile-m $(TILE_M) --tile-k-l2 $(TILE_K_L2) --tile-k-l1 $(TILE_K_L1) --tile-n $(TILE_N) --arch $(AIE_TARGET)

run8x4: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/run.py --herd-m 8 --herd-n 4 --m 1024 --n 512 --k 512 -v \
		$(COMMON_ARGS)

run4x4: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/run.py --herd-m 4 --herd-n 4 --m 512 --n 512 --k 512 \
		$(COMMON_ARGS)

run2x4: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/run.py --herd-m 2 --herd-n 4 --m 512 --n 512 --k 512 \
		$(COMMON_ARGS)

run2x2: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/run.py --herd-m 2 --herd-n 2 --m 512 --n 512 --k 512 \
		$(COMMON_ARGS)

# 576 = 9 * 64, so the 64-element tile floor still divides it 3 ways.
run3x3: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/run.py --herd-m 3 --herd-n 3 --m 576 --n 576 --k 576 \
		--tile-m 64 --tile-k-l2 192 --tile-k-l1 64 --tile-n 64 --compile-mode $(COMPILE_MODE) --arch $(AIE_TARGET) $(OUTPUT_DTYPE_FLAG)

# Smallest single-PE config. TILE_K_L1/TILE_N cannot go below 64 for bfp16ebs8
# (host shuffle granularity), so this is 64x64x64 rather than bf16's 32x32x16.
run1x1: compile-kernel-1x1
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/run.py --herd-m 1 --herd-n 1 --m 64 --n 64 --k 64 \
		--tile-m 64 --tile-k-l2 64 --tile-k-l1 64 --tile-n 64 --compile-mode $(COMPILE_MODE) --arch $(AIE_TARGET) $(OUTPUT_DTYPE_FLAG)

compile-kernel-1x1:
	$(MAKE) -f $(srcdir)/Makefile compile-kernel TILE_M=64 TILE_N=64 TILE_K_L1=64

# LLAMA-3.2-1B GEMM shapes (M=128)
# Q/O: 128x2048x2048, K/V: 128x2048x512, Gate/Up: 128x2048x8192, Down: 128x8192x2048
#
# These are 4x4 (16 cores), not bf16's 8x4. M=128 with herd_m=8 forces
# TILE_M=16, and mm_bfp.cc runs out of registers on AIE2P below TILE_M=32
# (rowA=2 vs its AIE_LOOP_MIN_ITERATION_COUNT(4)). TILE_M=32 => herd_m=4.
# TILE_N is 64 (not bf16's 32) because of the bfp16ebs8 shuffle granularity.
LLAMA_TILE_M = 32
LLAMA_TILE_N = 64
LLAMA_TILE_K_L1 = 64
LLAMA_HERD_M = 4
LLAMA_HERD_N = 4

compile-kernel-llama:
	$(MAKE) -f $(srcdir)/Makefile compile-kernel TILE_M=$(LLAMA_TILE_M) TILE_N=$(LLAMA_TILE_N) TILE_K_L1=$(LLAMA_TILE_K_L1)

LLAMA_ARGS = --tile-m $(LLAMA_TILE_M) --tile-k-l2 128 --tile-k-l1 $(LLAMA_TILE_K_L1) --tile-n $(LLAMA_TILE_N) --compile-mode $(COMPILE_MODE) --arch $(AIE_TARGET) $(OUTPUT_DTYPE_FLAG)

# _down (K=8192) is deliberately excluded: mm_bfp.cc keeps C in bfp16ebs8
# between K-l1 chunks, so 8192/64 = 128 re-quantizations leave the result
# unusable (mean_rel_L1 ~1.2). The target is kept below for experimentation.
run_llama_4x4: run_llama_4x4_qo run_llama_4x4_kv run_llama_4x4_gate_up

run_llama_4x4_qo: compile-kernel-llama
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && python3 ${srcdir}/run.py --herd-m $(LLAMA_HERD_M) --herd-n $(LLAMA_HERD_N) --m 128 --k 2048 --n 2048 \
		$(LLAMA_ARGS)

run_llama_4x4_kv: compile-kernel-llama
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && python3 ${srcdir}/run.py --herd-m $(LLAMA_HERD_M) --herd-n $(LLAMA_HERD_N) --m 128 --k 2048 --n 512 \
		$(LLAMA_ARGS)

run_llama_4x4_gate_up: compile-kernel-llama
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && python3 ${srcdir}/run.py --herd-m $(LLAMA_HERD_M) --herd-n $(LLAMA_HERD_N) --m 128 --k 2048 --n 8192 \
		$(LLAMA_ARGS)

run_llama_4x4_down: compile-kernel-llama
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && python3 ${srcdir}/run.py --herd-m $(LLAMA_HERD_M) --herd-n $(LLAMA_HERD_N) --m 128 --k 8192 --n 2048 \
		$(LLAMA_ARGS)

# Report latency + GFLOPs alongside the correctness check. Uses XRTRunner's
# built-in timing (--perf-iters), so no separate C++ harness is needed.
profile: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/run.py --m 1024 --k 1024 --n 1024 \
		--herd-m $(MAX_HERD_M) --herd-n $(MAX_HERD_N) --perf-iters 50 $(COMMON_ARGS)

# Measure the e2e latencies across a range of problem shapes.
sweep4x4: compile-kernel
	@for M in 256 512 1024 2048; do \
		for N in 256 512 1024 2048; do \
			for K in 256 512 1024 2048; do \
				echo "Running with M=$$M, N=$$N, K=$$K"; \
				cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/run.py --m $$M --n $$N --k $$K \
					--herd-m 4 --herd-n 4 --perf-iters 20 $(COMMON_ARGS); \
			done; \
		done; \
	done

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -z "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Error: PEANO_INSTALL_DIR not set. The Chess toolchain has no"\
		     "bfp16ebs8 codegen path; source utils/env_setup.sh."; \
		exit 1; \
	fi
	@if [ ! -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
		echo "Error: invalid PEANO_INSTALL_DIR, clang++ not found."; \
		exit 1; \
	fi
	@if [ ! -f "$(KERNEL_SRC)" ]; then \
		echo "Error: reference kernel not found at $(KERNEL_SRC)."; \
		echo "Point KERNEL_SRC at mlir-aie's aie_kernels/aie2p/mm_bfp.cc."; \
		exit 1; \
	fi
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) \
		-DDIM_M=$(TILE_M) -DDIM_K=$(TILE_K_L1) -DDIM_N=$(TILE_N) \
		-c $(KERNEL_SRC) -o $(BUILD_DIR)/mm_bfp_gemm.o
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) \
		-DDIM_M=$(TILE_M) -DDIM_K=$(TILE_K_L1) -DDIM_N=$(TILE_N) \
		-c $(srcdir)/bfp_cvt.cc -o $(BUILD_DIR)/mm_bfp_cvt.o
	$(PEANO_INSTALL_DIR)/bin/ld.lld -r -o $(BUILD_DIR)/mm_bfp.o \
		$(BUILD_DIR)/mm_bfp_gemm.o $(BUILD_DIR)/mm_bfp_cvt.o

clean:
	rm -rf $(BUILD_DIR) __pycache__

.PHONY: all print run8x4 run4x4 run2x4 run2x2 run3x3 run1x1 compile-kernel-1x1 \
        compile-kernel-llama run_llama_4x4 run_llama_4x4_qo run_llama_4x4_kv \
        run_llama_4x4_gate_up run_llama_4x4_down profile sweep4x4 compile-kernel clean
