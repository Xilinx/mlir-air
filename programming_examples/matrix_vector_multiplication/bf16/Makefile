# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# GEMV BF16: C[M] = A[M,K] @ B[K]
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
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

# AIE_TARGET: aie2p (default for NPU2/Strix)
AIE_TARGET ?= aie2p
PEANO_FLAGS = ${PEANOWRAP2P_FLAGS}

# GEMV dimensions: C[M] = A[M,K] @ B[K]
# Note: tile_m_l2 * K * 2 must fit in MemTile L2 (~256KB usable).
# For K=8192: tile_m_l2 <= 4 (4*8192*2=64KB, leaving room for B and C)
# For K=2048: tile_m_l2 <= 16 (16*2048*2=64KB)
M ?= 2048
K ?= 8192
TILE_M_L2 ?= 4
M_INPUT ?= 1

all: run

print:
	${powershell} python3 ${srcdir}/matvec.py $(OUTPUT_FORMAT_FLAG) -p --m $(M) --k $(K) --tile-m-l2 $(TILE_M_L2) --m-input $(M_INPUT)

run: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/matvec.py $(OUTPUT_FORMAT_FLAG) --m $(M) --k $(K) --tile-m-l2 $(TILE_M_L2) --m-input $(M_INPUT)

profile: compile-kernel build-test-exe
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && python3 ${srcdir}/matvec.py $(OUTPUT_FORMAT_FLAG) --compile-mode compile-and-xclbin --m $(M) --k $(K) --tile-m-l2 $(TILE_M_L2) --m-input $(M_INPUT)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin -M $(M) -K $(K)

# Convenience targets matching IRON GEMV test shapes
# Case 1: M=2048, K=8192 (tile_m_l2=4 fits in L2: 4*8192*2=64KB)
run1:
	$(MAKE) run M=2048 K=8192 TILE_M_L2=4 M_INPUT=1

profile1:
	$(MAKE) profile M=2048 K=8192 TILE_M_L2=4 M_INPUT=1

# Case 2: M=8192, K=2048 (tile_m_l2=16 fits in L2: 16*2048*2=64KB)
run2:
	$(MAKE) run M=8192 K=2048 TILE_M_L2=16 M_INPUT=4

profile2:
	$(MAKE) profile M=8192 K=2048 TILE_M_L2=16 M_INPUT=4

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
	cd $(BUILD_DIR) && $$GPP ${srcdir}/test.cpp -o test.exe -std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Detected PEANO_INSTALL_DIR from environment: $(PEANO_INSTALL_DIR)"; \
		if [ -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
			echo "Using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) with target $(AIE_TARGET)"; \
			$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANO_FLAGS} -DDIM_M_OUTPUT=$(TILE_M_L2) -c ${srcdir}/mv.cc -o $(BUILD_DIR)/mv.o; \
		else \
			echo "Error: invalid PEANO_INSTALL_DIR, clang++ not found."; \
			exit 1; \
		fi; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH with target $(AIE_TARGET)"; \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper ${AIE_TARGET} -DDIM_M_OUTPUT=$(TILE_M_L2) -c ${srcdir}/mv.cc -o mv.o; \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR) __pycache__
