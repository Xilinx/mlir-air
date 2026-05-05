# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Output format: xclbin (default) or elf
OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2_FLAGS = -O2 -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include
# -Os (aggressive size opt) keeps the AIE2P core ELF under the 16 KB
# program memory limit when rms shares a tile with vecmat + attn +
# softmax + RoPE (-O2 builds ~18 KB for N_IN=2048 TILE_K=128).
PEANOWRAP2P_FLAGS = -Os -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -I ${AIEOPT_DIR}/include

all: run

print:
	${powershell} python3 ${srcdir}/mha.py $(OUTPUT_FORMAT_FLAG) -p

run: compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/mha.py $(OUTPUT_FORMAT_FLAG) --pos 2 --k 288 --n 48 --compile-mode compile-and-run

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Detected PEANO_INSTALL_DIR from environment: $(PEANO_INSTALL_DIR)"; \
		if [ -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
			echo "Using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)"; \
			$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c ${srcdir}/mha.cc -o $(BUILD_DIR)/mha.o; \
		else \
			echo "Error: invalid PEANO_INSTALL_DIR, clang++ not found."; \
			exit 1; \
		fi; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH"; \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper aie2 -c ${srcdir}/mha.cc -o mha.o; \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

# ============================================================================
# Step 3/4: GQA + multi-col NKV>=2, optional rms front-end
# ============================================================================

# Configurable for profiling and verification.
# Defaults: full LLaMA-3.2-1B decode config (NKV=8 GROUP_SIZE=4 N_IN=2048
# LK=2048 head_dim=64). TILE_K must be 256 at N_IN >= 512 — smaller TILE_K
# at large N_IN produces wrong output (suspected memtile DMA streaming
# limit at small packet × many fires). TILE_K=256 caps L1 weight buf at
# 32 KB which fits alongside attn_l1=16 KB and the rest of the herd state.
NKV ?= 8
GROUP_SIZE ?= 4
LK ?= 2048
LKP ?= 64
DK ?= 64
DV ?= 64
N_IN ?= 2048
TILE_K ?= 128
TILE_N ?= 64
POS ?= 2046
WARMUP ?= 10
ITERS ?= 20

print-multicol:
	${powershell} python3 ${srcdir}/mha_gqa_multicol.py $(OUTPUT_FORMAT_FLAG) -p --nkv $(NKV) --pos $(POS) --k $(N_IN) --n $(DK) --tile-k $(TILE_K) --tile-n $(TILE_N) --seq-len $(LK)

run-multicol: compile-kernel-gqa
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/mha_gqa_multicol.py $(OUTPUT_FORMAT_FLAG) --pos $(POS) --k $(N_IN) --n $(DK) --tile-k $(TILE_K) --tile-n $(TILE_N) --seq-len $(LK) --nkv $(NKV) --compile-mode compile-and-run

run-multicol-nkv8: compile-kernel-gqa
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/mha_gqa_multicol.py $(OUTPUT_FORMAT_FLAG) --pos $(POS) --k $(N_IN) --n $(DK) --tile-k $(TILE_K) --tile-n $(TILE_N) --seq-len $(LK) --nkv 8 --compile-mode compile-and-run

# ============================================================================
# Step 2b: dk=dv=64 + GQA group_size=4 (4 Q heads share 1 K, 1 V)
# ============================================================================

print-gqa:
	${powershell} python3 ${srcdir}/mha_gqa.py $(OUTPUT_FORMAT_FLAG) -p

run-gqa: compile-kernel-gqa
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/mha_gqa.py $(OUTPUT_FORMAT_FLAG) --pos 2 --k 64 --n 64 --tile-k 64 --tile-n 64 --seq-len 128 --compile-mode compile-and-run

compile-kernel-gqa:
	mkdir -p $(BUILD_DIR)
	@if [ -z "$(PEANO_INSTALL_DIR)" ] || [ ! -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
		echo "Error: NPU2 build requires PEANO_INSTALL_DIR with clang++."; \
		exit 1; \
	fi
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -DSEQ_LEN=$(LK) -DGROUP_SIZE=$(GROUP_SIZE) -DDIM_K=$(N_IN) -DTILE_K=$(TILE_K) -DDIM_N=$(DK) -DHEAD_SIZE=$(DK) -c ${srcdir}/mha_gqa.cc -o $(BUILD_DIR)/mha_gqa.o

# ============================================================================
# Profile target for the decode design at LLaMA configuration
# Usage:
#   make profile-decode [NKV=8] [LK=2048] [POS=2046] [N_IN=64] [DK=64] [GROUP_SIZE=4] [WARMUP=10] [ITERS=20]
# ============================================================================
profile-decode: compile-kernel-gqa build-test-exe-decode
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/mha_gqa_multicol.py \
		--nkv $(NKV) --pos $(POS) --k $(N_IN) --n $(DK) --tile-k $(TILE_K) --tile-n $(TILE_N) --seq-len $(LK) \
		--compile-mode compile-only --output-format xclbin
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ./test_xclbin_decode.exe \
		-x air.xclbin -k "MLIR_AIE" -i air.insts.bin \
		--nkv $(NKV) --group-size $(GROUP_SIZE) --n-in $(N_IN) --head-dim $(DK) --lk $(LK) --tile-k $(TILE_K) --tile-n $(TILE_N) \
		--warmup $(WARMUP) --iterations $(ITERS)

# ELF profile target — preferred for multi-iteration profiling. ELF format
# has lightweight reset device that resets BD state across host invocations
# (xclbin format does not, so the design's repeat_count BDs hang on iter 2).
profile-decode-elf: compile-kernel-gqa build-test-exe-decode-elf
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/mha_gqa_multicol.py \
		--nkv $(NKV) --pos $(POS) --k $(N_IN) --n $(DK) --tile-k $(TILE_K) --tile-n $(TILE_N) --seq-len $(LK) \
		--compile-mode compile-only --output-format elf
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ./test_elf_npu2_decode.exe \
		-e air.elf -k "main:mha_bf16" \
		--nkv $(NKV) --group-size $(GROUP_SIZE) --n-in $(N_IN) --head-dim $(DK) --lk $(LK) --tile-k $(TILE_K) --tile-n $(TILE_N) \
		--warmup $(WARMUP) --iterations $(ITERS)

build-test-exe-decode-elf:
	@GPP=$$( \
		for bin in /usr/bin/g++-*; do \
			ver=$$(echo $$bin | grep -oE '[0-9]+$$'); \
			if [ "$$ver" -ge 13 ] 2>/dev/null; then \
				echo "$$ver $$bin"; \
			fi; \
		done | sort -nr | head -n1 | awk '{print $$2}' \
	); \
	if [ -z "$$GPP" ]; then echo "Error: No g++ >= 13 in /usr/bin."; exit 1; fi; \
	if [ -z "$$XILINX_XRT" ]; then echo "Error: source xrt/setup.sh."; exit 1; fi; \
	if [ -z "$(AIEOPT_DIR)" ]; then echo "Error: source utils/env_setup.sh."; exit 1; fi; \
	echo "Using compiler: $$GPP"; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP ${srcdir}/test_elf_npu2_decode.cpp -o test_elf_npu2_decode.exe \
		-std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

build-test-exe-decode:
	@GPP=$$( \
		for bin in /usr/bin/g++-*; do \
			ver=$$(echo $$bin | grep -oE '[0-9]+$$'); \
			if [ "$$ver" -ge 13 ] 2>/dev/null; then \
				echo "$$ver $$bin"; \
			fi; \
		done | sort -nr | head -n1 | awk '{print $$2}' \
	); \
	if [ -z "$$GPP" ]; then \
		echo "Error: No g++ version >= 13 found in /usr/bin."; \
		exit 1; \
	fi; \
	if [ -z "$$XILINX_XRT" ]; then \
		echo "Error: XILINX_XRT not set. source /opt/xilinx/xrt/setup.sh first."; \
		exit 1; \
	fi; \
	if [ -z "$(AIEOPT_DIR)" ]; then \
		echo "Error: AIEOPT_DIR not set. source utils/env_setup.sh first."; \
		exit 1; \
	fi; \
	echo "Using compiler: $$GPP"; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP ${srcdir}/test_xclbin_decode.cpp -o test_xclbin_decode.exe \
		-std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

# ============================================================================
# Step 2a: dk=dv=64 (LLaMA head_size), single Q head per call
# ============================================================================

print-dk64:
	${powershell} python3 ${srcdir}/mha_dk64.py $(OUTPUT_FORMAT_FLAG) -p

run-dk64: compile-kernel-dk64
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/mha_dk64.py $(OUTPUT_FORMAT_FLAG) --pos 2 --k 64 --n 64 --tile-k 64 --tile-n 64 --seq-len 128 --compile-mode compile-and-run

compile-kernel-dk64:
	mkdir -p $(BUILD_DIR)
	@if [ -z "$(PEANO_INSTALL_DIR)" ] || [ ! -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
		echo "Error: NPU2 build requires PEANO_INSTALL_DIR with clang++."; \
		exit 1; \
	fi
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -DSEQ_LEN=128 -c ${srcdir}/mha_dk64.cc -o $(BUILD_DIR)/mha_dk64.o

# ============================================================================
# NPU2 (AIE2P) targets — uses mha_npu2.py + mha_npu2.cc
# ============================================================================

print-npu2:
	${powershell} python3 ${srcdir}/mha_npu2.py $(OUTPUT_FORMAT_FLAG) -p

run-npu2: compile-kernel-npu2
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/mha_npu2.py $(OUTPUT_FORMAT_FLAG) --pos 2 --k 288 --n 48 --compile-mode compile-and-run

compile-kernel-npu2:
	mkdir -p $(BUILD_DIR)
	@if [ -z "$(PEANO_INSTALL_DIR)" ] || [ ! -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
		echo "Error: NPU2 build requires PEANO_INSTALL_DIR with clang++."; \
		exit 1; \
	fi
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -c ${srcdir}/mha_npu2.cc -o $(BUILD_DIR)/mha_npu2.o

clean:
	rm -rf $(BUILD_DIR) __pycache__
