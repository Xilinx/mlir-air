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
# -Os (aggressive size opt) keeps the AIE2P core ELF under the 16 KB
# program memory limit when rms shares a tile with vecmat + attn +
# softmax + RoPE (-O2 builds ~18 KB for N_IN=2048 TILE_K=128).
PEANOWRAP2P_FLAGS = -Os -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -I ${AIEOPT_DIR}/include

# ============================================================================
# attn_decode_npu2: per-token attention block (RMSNorm + QKV GEMV + RoPE +
# Q.K + softmax + softmax.V) on NPU2. One herd of NKV tiles, one KV head
# per tile, GROUP_SIZE Q heads per KV head.
# ============================================================================

# Defaults: full LLaMA-3.2-1B decode config.
NKV ?= 8
GROUP_SIZE ?= 4
LK ?= 2048
DK ?= 64
DV ?= 64
N_IN ?= 2048
TILE_K ?= 128
TILE_N ?= 64
POS ?= 2046
WARMUP ?= 10
ITERS ?= 20

all: run-attn-decode

print-attn-decode:
	${powershell} python3 ${srcdir}/attn_decode_npu2.py $(OUTPUT_FORMAT_FLAG) -p --nkv $(NKV) --pos $(POS) --k $(N_IN) --n $(DK) --tile-k $(TILE_K) --tile-n $(TILE_N) --seq-len $(LK)

run-attn-decode: compile-attn-decode-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/attn_decode_npu2.py $(OUTPUT_FORMAT_FLAG) --pos $(POS) --k $(N_IN) --n $(DK) --tile-k $(TILE_K) --tile-n $(TILE_N) --seq-len $(LK) --nkv $(NKV) --compile-mode compile-and-run

compile-attn-decode-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -z "$(PEANO_INSTALL_DIR)" ] || [ ! -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
		echo "Error: NPU2 build requires PEANO_INSTALL_DIR with clang++."; \
		exit 1; \
	fi
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -DSEQ_LEN=$(LK) -DGROUP_SIZE=$(GROUP_SIZE) -DDIM_K=$(N_IN) -DTILE_K=$(TILE_K) -DDIM_N=$(DK) -DHEAD_SIZE=$(DK) -c ${srcdir}/attn_decode_npu2.cc -o $(BUILD_DIR)/attn_decode_npu2.o

# DATA_MOVEMENT_ONLY build: every extern-C kernel function returns
# immediately (compute disabled). Locks/DMA still cycle through the herd
# body, so per-token latency reflects pure data-movement + sync cost.
# Use to measure DMA vs compute share of the per-token time.
compile-attn-decode-kernel-nocompute:
	mkdir -p $(BUILD_DIR)
	@if [ -z "$(PEANO_INSTALL_DIR)" ] || [ ! -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
		echo "Error: NPU2 build requires PEANO_INSTALL_DIR with clang++."; \
		exit 1; \
	fi
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -DSEQ_LEN=$(LK) -DGROUP_SIZE=$(GROUP_SIZE) -DDIM_K=$(N_IN) -DTILE_K=$(TILE_K) -DDIM_N=$(DK) -DHEAD_SIZE=$(DK) -DDATA_MOVEMENT_ONLY -c ${srcdir}/attn_decode_npu2.cc -o $(BUILD_DIR)/attn_decode_npu2.o

profile-decode-nocompute: compile-attn-decode-kernel-nocompute build-test-exe-decode
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/attn_decode_npu2.py \
		--nkv $(NKV) --pos $(POS) --k $(N_IN) --n $(DK) --tile-k $(TILE_K) --tile-n $(TILE_N) --seq-len $(LK) \
		--compile-mode compile-only --output-format xclbin
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ./test_xclbin_decode.exe \
		-x air.xclbin -k "MLIR_AIE" -i air.insts.bin \
		--nkv $(NKV) --group-size $(GROUP_SIZE) --n-in $(N_IN) --head-dim $(DK) --lk $(LK) --tile-k $(TILE_K) --tile-n $(TILE_N) \
		--warmup $(WARMUP) --iterations $(ITERS)

# ============================================================================
# Profile target for the decode design at LLaMA configuration
# Usage:
#   make profile-decode [NKV=8] [LK=2048] [POS=2046] [N_IN=64] [DK=64] [GROUP_SIZE=4] [WARMUP=10] [ITERS=20]
# ============================================================================
profile-decode: compile-attn-decode-kernel build-test-exe-decode
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/attn_decode_npu2.py \
		--nkv $(NKV) --pos $(POS) --k $(N_IN) --n $(DK) --tile-k $(TILE_K) --tile-n $(TILE_N) --seq-len $(LK) \
		--compile-mode compile-only --output-format xclbin
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ./test_xclbin_decode.exe \
		-x air.xclbin -k "MLIR_AIE" -i air.insts.bin \
		--nkv $(NKV) --group-size $(GROUP_SIZE) --n-in $(N_IN) --head-dim $(DK) --lk $(LK) --tile-k $(TILE_K) --tile-n $(TILE_N) \
		--warmup $(WARMUP) --iterations $(ITERS)

# ELF profile target — alternative profiler using the ELF runtime path.
# Both xclbin and ELF formats run multi-iteration cleanly for this design.
profile-decode-elf: compile-attn-decode-kernel build-test-exe-decode-elf
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/attn_decode_npu2.py \
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

clean:
	rm -rf $(BUILD_DIR) __pycache__
