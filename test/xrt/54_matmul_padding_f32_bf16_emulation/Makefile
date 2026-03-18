# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Makefile for f32 matmul with bf16 emulation (test 54).
# Correctness test uses non-tile-aligned dimensions (M=500, N=500).
# Profiling uses tile-aligned dimensions (no padding overhead).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
BUILD_DIR := build
AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)

# Correctness test dimensions (non-tile-aligned, exercises padding)
RUN_M ?= 500
RUN_N ?= 500
RUN_K ?= 784
K_L2_TILE ?= 16

# Profiling dimensions (tile-aligned: M=1024, N=512, K=784)
PROFILE_M ?= 1024
PROFILE_N ?= 512
PROFILE_K ?= 784

all: run

run:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 ${srcdir}/run.py \
		--transform-script ${srcdir}/transform_aie2p.mlir \
		--M $(RUN_M) --N $(RUN_N) --K $(RUN_K) --k-l2-tile $(K_L2_TILE) -v

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
		echo "Error: XILINX_XRT not set. Source xrt/setup.sh."; \
		exit 1; \
	fi; \
	echo "Using compiler: $$GPP"; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP ${srcdir}/test.cpp -o test.exe -std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

profile: build-test-exe
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 ${srcdir}/run.py \
		--transform-script ${srcdir}/transform_aie2p.mlir \
		--M $(PROFILE_M) --N $(PROFILE_N) --K $(PROFILE_K) \
		--k-l2-tile $(K_L2_TILE) --compile-mode compile-only
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin \
		-M $(PROFILE_M) -K $(PROFILE_K) -N $(PROFILE_N)

clean:
	rm -rf $(BUILD_DIR)
