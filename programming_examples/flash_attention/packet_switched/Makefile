# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Packet-switched flash attention example.
# Uses dma_packet channels for Q and K routing through shared
# compute tile S2MM DMA channels.
# Supports both NPU2 (AIE2P) and NPU1 (AIE2).
#
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Attention parameters
LK ?= 512
LKP ?= 64
LQ ?= 512
LQP ?= 256
DK ?= 64
DV ?= 64
NUM_HEADS ?= 2
NUM_KV_HEADS ?= $(NUM_HEADS)
VAL_RANGE ?= 3

# Derived: kernel tile size = LQP / num_q_tiles (4)
NUM_Q_TILES ?= 4
LQP_TILE := $(shell echo $$(($(LQP) / $(NUM_Q_TILES))))

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include
PEANOWRAP2_FLAGS = -O2 -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

# ============================================================================
# NPU2 (AIE2P) targets — default
# ============================================================================

all: run

print:
	${powershell} python3 ${srcdir}/attn.py -p --lk $(LK) --lkp $(LKP) --lq $(LQ) --lqp $(LQP) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS) --num-kv-heads $(NUM_KV_HEADS)

run: compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/attn.py --lk $(LK) --lkp $(LKP) --lq $(LQ) --lqp $(LQP) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS) --num-kv-heads $(NUM_KV_HEADS) --val-range $(VAL_RANGE) $(EXTRA_PY_FLAGS)

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Detected PEANO_INSTALL_DIR from environment: $(PEANO_INSTALL_DIR)"; \
		if [ -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
			echo "Using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)"; \
			$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c ${srcdir}/attn_pkt.cc -o $(BUILD_DIR)/attn_pkt.o -Dlqp=$(LQP_TILE) -Dlkp=$(LKP) -Ddk=$(DK) -Ddv=$(DV) -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -DROUND_CONV_EVEN $(EXTRA_KERNEL_FLAGS); \
		else \
			echo "Error: invalid PEANO_INSTALL_DIR, clang++ not found."; \
			exit 1; \
		fi; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH"; \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper aie2p -c ${srcdir}/attn_pkt.cc -o attn_pkt.o -Dlqp=$(LQP_TILE) -Dlkp=$(LKP) -Ddk=$(DK) -Ddv=$(DV); \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

# ============================================================================
# NPU1 (AIE2) targets
# ============================================================================

print-npu1:
	${powershell} python3 ${srcdir}/attn_npu1.py -p --lk $(LK) --lkp $(LKP) --lq $(LQ) --lqp $(LQP) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS) --num-kv-heads $(NUM_KV_HEADS)

run-npu1: compile-kernel-npu1
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/attn_npu1.py --lk $(LK) --lkp $(LKP) --lq $(LQ) --lqp $(LQP) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS) --num-kv-heads $(NUM_KV_HEADS) --val-range $(VAL_RANGE) $(EXTRA_PY_FLAGS)

compile-kernel-npu1:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Detected PEANO_INSTALL_DIR from environment: $(PEANO_INSTALL_DIR)"; \
		if [ -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
			echo "Using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) for AIE2 (NPU1)"; \
			$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c ${srcdir}/attn_npu1.cc -o $(BUILD_DIR)/attn_npu1.o -I${srcdir}/../dataflow_based -Dlqp=$(LQP_TILE) -Dlkp=$(LKP) -Ddk=$(LKP) -Ddk_full=$(DK) -Ddv=$(LKP) -Ddv_full=$(DV) $(EXTRA_KERNEL_FLAGS); \
		else \
			echo "Error: invalid PEANO_INSTALL_DIR, clang++ not found."; \
			exit 1; \
		fi; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH for AIE2 (NPU1)"; \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper aie2 -c ${srcdir}/attn_npu1.cc -o attn_npu1.o -I${srcdir}/../dataflow_based -Dlqp=$(LQP_TILE) -Dlkp=$(LKP) -Ddk=$(LKP) -Ddk_full=$(DK) -Ddv=$(LKP) -Ddv_full=$(DV); \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR) __pycache__
