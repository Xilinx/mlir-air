# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
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

all: run

print:
	${powershell} python3 ${srcdir}/attn_npu2.py -p --lk $(LK) --lkp $(LKP) --lq $(LQ) --lqp $(LQP) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS) --num-kv-heads $(NUM_KV_HEADS)

run: compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/attn_npu2.py --lk $(LK) --lkp $(LKP) --lq $(LQ) --lqp $(LQP) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS) --num-kv-heads $(NUM_KV_HEADS) $(EXTRA_PY_FLAGS)

# Profile ELF: compile elf and run with C++ test executable for elf format
# Usage: make profile [LK=...] [LQ=...] etc.
profile: compile-kernel build-test-exe
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/attn_npu2.py \
		--lk $(LK) --lkp $(LKP) --lq $(LQ) --lqp $(LQP) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS) --num-kv-heads $(NUM_KV_HEADS) \
		--compile-mode compile-only
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ./test_elf_npu2.exe -e air.elf -k "main:attention_bf16" \
		--lq $(LQ) --lk $(LK) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS)

# Profile xclbin on NPU2: compile xclbin and run with C++ test executable
# Usage: make profile-xclbin [LK=...] [LQ=...] etc.
profile-xclbin: compile-kernel build-test-exe-xclbin
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/attn_npu2.py \
		--lk $(LK) --lkp $(LKP) --lq $(LQ) --lqp $(LQP) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS) --num-kv-heads $(NUM_KV_HEADS) \
		--compile-mode compile-only --output-format xclbin
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ./test_xclbin_npu2.exe \
		-x air.xclbin -k "MLIR_AIE" -i air.insts.bin \
		--lq $(LQ) --lk $(LK) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS)

build-test-exe-xclbin:
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
		echo "Error: XILINX_XRT environment variable not set. Please make sure to have sourced xrt/setup.sh."; \
		exit 1; \
	fi; \
	if [ -z "$(AIEOPT_DIR)" ]; then \
		echo "Error: AIEOPT_DIR environment variable not set. Please make sure to have sourced utils/env_setup.sh."; \
		exit 1; \
	fi; \
	echo "Using compiler: $$GPP"; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP ${srcdir}/test_xclbin_npu2.cpp -o test_xclbin_npu2.exe -std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

build-test-exe:
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
		echo "Error: XILINX_XRT environment variable not set. Please make sure to have sourced xrt/setup.sh."; \
		exit 1; \
	fi; \
	if [ -z "$(AIEOPT_DIR)" ]; then \
		echo "Error: AIEOPT_DIR environment variable not set. Please make sure to have sourced utils/env_setup.sh."; \
		exit 1; \
	fi; \
	echo "Using compiler: $$GPP"; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP ${srcdir}/test_elf_npu2.cpp -o test_elf_npu2.exe -std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Detected PEANO_INSTALL_DIR from environment: $(PEANO_INSTALL_DIR)"; \
		if [ -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
			echo "Using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)"; \
			$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c ${srcdir}/attn_npu2.cc -o $(BUILD_DIR)/attn_npu2.o -Dlqp=$(LQP_TILE) -Dlkp=$(LKP) -Ddk=$(LKP) -Ddk_full=$(DK) -Ddv=$(LKP) -Ddv_full=$(DV) -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -DROUND_CONV_EVEN $(EXTRA_KERNEL_FLAGS); \
		else \
			echo "Error: invalid PEANO_INSTALL_DIR, clang++ not found."; \
			exit 1; \
		fi; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH"; \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper aie2p -c ${srcdir}/attn_npu2.cc -o attn_npu2.o -Dlqp=$(LQP_TILE) -Dlkp=$(LKP) -Ddk=$(LKP) -Ddk_full=$(DK) -Ddv=$(LKP) -Ddv_full=$(DV); \
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
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/attn_npu1.py --lk $(LK) --lkp $(LKP) --lq $(LQ) --lqp $(LQP) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS) --num-kv-heads $(NUM_KV_HEADS) $(EXTRA_PY_FLAGS)

# Profile xclbin: compile xclbin and run with C++ test executable
# Usage: make profile-npu1 [LK=...] [LQ=...] [NUM_HEADS=...] etc.
profile-npu1: compile-kernel-npu1 build-test-exe-npu1
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/attn_npu1.py \
		--lk $(LK) --lkp $(LKP) --lq $(LQ) --lqp $(LQP) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS) --num-kv-heads $(NUM_KV_HEADS) \
		--compile-mode compile-only
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ./test_xclbin_npu1.exe \
		-x air.xclbin -k "MLIR_AIE" -i air.insts.bin \
		--lq $(LQ) --lk $(LK) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS)

build-test-exe-npu1:
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
		echo "Error: XILINX_XRT environment variable not set. Please make sure to have sourced xrt/setup.sh."; \
		exit 1; \
	fi; \
	if [ -z "$(AIEOPT_DIR)" ]; then \
		echo "Error: AIEOPT_DIR environment variable not set. Please make sure to have sourced utils/env_setup.sh."; \
		exit 1; \
	fi; \
	echo "Using compiler: $$GPP"; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP ${srcdir}/test_xclbin_npu1.cpp -o test_xclbin_npu1.exe -std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

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
