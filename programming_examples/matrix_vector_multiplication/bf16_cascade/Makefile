# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Cascade GEMV BF16: C[M] = A[M,K] @ B[K]
# K-reduction split across n_cascade tiles per column via cascade channels.
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
# tile_m=2 with 8 cols and K=8192: A_L2 = 8*2*8192*2 = 256KB < 512KB
M ?= 2048
K ?= 8192
TILE_M ?= 2
M_INPUT ?= 1
HERD_COLS ?= 8
N_CASCADE ?= 4

all: run

print:
	${powershell} python3 ${srcdir}/matvec_cascade.py $(OUTPUT_FORMAT_FLAG) -p --m $(M) --k $(K) --tile-m $(TILE_M) --m-input $(M_INPUT) --herd-cols $(HERD_COLS) --n-cascade $(N_CASCADE)

run: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/matvec_cascade.py $(OUTPUT_FORMAT_FLAG) --m $(M) --k $(K) --tile-m $(TILE_M) --m-input $(M_INPUT) --herd-cols $(HERD_COLS) --n-cascade $(N_CASCADE) $(if $(DEBUG_IR),--debug-ir) || \
	  (echo "=== npu.air.mlir (packet_flow and aie.flow lines) ===" >&2 && \
	   grep -n "packet_flow\|aie.flow" air_project/npu.air.mlir >&2 2>/dev/null; \
	   echo "=== input_with_addresses.mlir (packet_flow lines) ===" >&2 && \
	   grep -n "packet_flow" air_project/input_with_addresses.mlir >&2 2>/dev/null; \
	   echo "=== aiecc_failure IR (first 100 lines) ===" >&2 && \
	   head -100 air_project/aiecc_failure_*.mlir >&2 2>/dev/null; \
	   false)

profile: compile-kernel build-test-exe
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && python3 ${srcdir}/matvec_cascade.py $(OUTPUT_FORMAT_FLAG) --compile-mode compile-and-xclbin --m $(M) --k $(K) --tile-m $(TILE_M) --m-input $(M_INPUT) --herd-cols $(HERD_COLS) --n-cascade $(N_CASCADE)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && LD_LIBRARY_PATH=$$XILINX_XRT/lib:$$LD_LIBRARY_PATH ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin -M $(M) -K $(K)

# Default: 8 columns x 4 cascade tiles, M=2048, K=8192, tile_m=2
run1:
	$(MAKE) run M=2048 K=8192 TILE_M=2 M_INPUT=1 HERD_COLS=8 N_CASCADE=4 OUTPUT_FORMAT=elf

profile1:
	$(MAKE) profile M=2048 K=8192 TILE_M=2 M_INPUT=1 HERD_COLS=8 N_CASCADE=4

# 4 columns x 4 cascade tiles (half-device, tile_m=4 fits L2)
run_4col:
	$(MAKE) run M=2048 K=8192 TILE_M=4 M_INPUT=1 HERD_COLS=4 N_CASCADE=4 OUTPUT_FORMAT=elf

profile_4col:
	$(MAKE) profile M=2048 K=8192 TILE_M=4 M_INPUT=1 HERD_COLS=4 N_CASCADE=4

# Smaller test: 2 columns x 4 cascade tiles
run_small:
	$(MAKE) run M=256 K=512 TILE_M=4 M_INPUT=1 HERD_COLS=2 N_CASCADE=4

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
			$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANO_FLAGS} -DDIM_M_OUTPUT=$(TILE_M) -c ${srcdir}/mv_cascade.cc -o $(BUILD_DIR)/mv_cascade.o; \
		else \
			echo "Error: invalid PEANO_INSTALL_DIR, clang++ not found."; \
			exit 1; \
		fi; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH with target $(AIE_TARGET)"; \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper ${AIE_TARGET} -DDIM_M_OUTPUT=$(TILE_M) -c ${srcdir}/mv_cascade.cc -o mv_cascade.o; \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR) __pycache__
