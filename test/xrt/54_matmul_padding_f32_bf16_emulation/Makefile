# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
BUILD_DIR := build
AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)

# AIE_TARGET: aie2p (NPU2) only — npu1 not supported (requires ELF reload)
AIE_TARGET ?= aie2p

TILE_M ?= 64
TILE_N ?= 32
M ?= 500
N ?= 500
K ?= 784
K_L2_TILE ?= 16

HERD_M ?= 4
HERD_N ?= 4

COMMON_FLAGS = --k-l2-tile $(K_L2_TILE) --herd-m $(HERD_M) --herd-n $(HERD_N)

# Pad M/N to launch-tile-aligned for profile
LAUNCH_TILE_M = $(shell echo $$(( $(TILE_M) * $(HERD_M) )))
LAUNCH_TILE_N = $(shell echo $$(( $(TILE_N) * $(HERD_N) )))
M_PADDED = $(shell python3 -c "import math; print(math.ceil($(M)/$(LAUNCH_TILE_M))*$(LAUNCH_TILE_M))")
N_PADDED = $(shell python3 -c "import math; print(math.ceil($(N)/$(LAUNCH_TILE_N))*$(LAUNCH_TILE_N))")

all: run

run:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 ${srcdir}/run.py $(COMMON_FLAGS) \
		--M $(M) --N $(N) --K $(K) -v

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
		echo "Error: No g++ version >= 13 found."; \
		exit 1; \
	fi; \
	if [ -z "$$XILINX_XRT" ]; then \
		echo "Error: XILINX_XRT not set."; \
		exit 1; \
	fi; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP ${srcdir}/test.cpp -o test.exe -std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

profile: build-test-exe
	@echo "Profiling M=$(M) N=$(N) K=$(K) (padded to $(M_PADDED)x$(N_PADDED))"
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 ${srcdir}/run.py $(COMMON_FLAGS) \
		--M $(M_PADDED) --N $(N_PADDED) --K $(K) --compile-mode compile-only
	cd $(BUILD_DIR) && ./test.exe -e air.elf -k main:matmul_padding_kernel \
		-M $(M) -K $(K) -N $(N) --alloc_m $(M_PADDED) --alloc_n $(N_PADDED)

clean:
	rm -rf $(BUILD_DIR)
