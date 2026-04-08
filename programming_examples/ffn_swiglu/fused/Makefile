# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Fused SwiGLU: output = SiLU(x @ W_gate) * (x @ W_up)
# Single launch with time-multiplexed herds on NPU2 (AIE2P).
#
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Tile sizes
TILE_M ?= 64
TILE_K_L2 ?= 128
TILE_K_L1 ?= 32
TILE_N ?= 64

# Problem sizes
M ?= 512
K ?= 512
N ?= 512

# Herd dimensions (4 rows x 4 cols = 16 tiles on NPU2)
HERD_M ?= 4
HERD_N ?= 4

# Derived constants for kernel compilation
TILE_M_DIV_8 := $(shell echo $$(( $(TILE_M) / 8 )))
TILE_N_DIV_8 := $(shell echo $$(( $(TILE_N) / 8 )))

# Output format
OUTPUT_FORMAT ?= elf

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

COMPILE_MODE ?= compile-and-run

all: run

print:
	${powershell} python3 ${srcdir}/swiglu_fused.py -p \
		--m $(M) --k $(K) --n $(N) \
		--tile-m $(TILE_M) --tile-k-l2 $(TILE_K_L2) --tile-k-l1 $(TILE_K_L1) --tile-n $(TILE_N) \
		--herd-m $(HERD_M) --herd-n $(HERD_N)

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Compiling swiglu_fused.cc with Peano for AIE2P"; \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
			-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 \
			-DDIM_M=$(TILE_M) -DDIM_K=$(TILE_K_L1) -DDIM_N=$(TILE_N) \
			-DDIM_M_DIV_8=$(TILE_M_DIV_8) -DDIM_N_DIV_8=$(TILE_N_DIV_8) \
			-c ${srcdir}/swiglu_fused.cc \
			-o $(BUILD_DIR)/swiglu_fused.o; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Compiling swiglu_fused.cc with xchesscc for AIE2P"; \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper aie2p \
			-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 \
			-DDIM_M=$(TILE_M) -DDIM_K=$(TILE_K_L1) -DDIM_N=$(TILE_N) \
			-DDIM_M_DIV_8=$(TILE_M_DIV_8) -DDIM_N_DIV_8=$(TILE_N_DIV_8) \
			-c ${srcdir}/swiglu_fused.cc \
			-o swiglu_fused.o; \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

run: compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/swiglu_fused.o $(BUILD_DIR)/air_project/swiglu_fused.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/swiglu_fused.py \
			--m $(M) --k $(K) --n $(N) \
			--tile-m $(TILE_M) --tile-k-l2 $(TILE_K_L2) --tile-k-l1 $(TILE_K_L1) --tile-n $(TILE_N) \
			--herd-m $(HERD_M) --herd-n $(HERD_N) \
			--output-format $(OUTPUT_FORMAT) \
			--compile-mode $(COMPILE_MODE)

# Smaller config for quick testing / CI
run4x4: compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/swiglu_fused.o $(BUILD_DIR)/air_project/swiglu_fused.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/swiglu_fused.py \
			--m 256 --k 256 --n 256 \
			--tile-m $(TILE_M) --tile-k-l2 $(TILE_K_L2) --tile-k-l1 $(TILE_K_L1) --tile-n $(TILE_N) \
			--herd-m 4 --herd-n 4 \
			--output-format $(OUTPUT_FORMAT) \
			--compile-mode $(COMPILE_MODE)

# Compile-only (no XRT needed)
compile-only: compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/swiglu_fused.o $(BUILD_DIR)/air_project/swiglu_fused.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/swiglu_fused.py \
			--m $(M) --k $(K) --n $(N) \
			--tile-m $(TILE_M) --tile-k-l2 $(TILE_K_L2) --tile-k-l1 $(TILE_K_L1) --tile-n $(TILE_N) \
			--herd-m $(HERD_M) --herd-n $(HERD_N) \
			--compile-mode compile-only --output-format none

# Compile xclbin (for profile / C++ test usage)
compile-xclbin: compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/swiglu_fused.o $(BUILD_DIR)/air_project/swiglu_fused.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/swiglu_fused.py \
			--m $(M) --k $(K) --n $(N) \
			--tile-m $(TILE_M) --tile-k-l2 $(TILE_K_L2) --tile-k-l1 $(TILE_K_L1) --tile-n $(TILE_N) \
			--herd-m $(HERD_M) --herd-n $(HERD_N) \
			--compile-mode compile-only --output-format xclbin

# Profile: compile + run with Python-based timing
# Usage: make profile [M=...] [K=...] [N=...]
profile: compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/swiglu_fused.o $(BUILD_DIR)/air_project/swiglu_fused.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/swiglu_fused.py \
			--m $(M) --k $(K) --n $(N) \
			--tile-m $(TILE_M) --tile-k-l2 $(TILE_K_L2) --tile-k-l1 $(TILE_K_L1) --tile-n $(TILE_N) \
			--herd-m $(HERD_M) --herd-n $(HERD_N) \
			--compile-mode profile

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
		echo "Error: XILINX_XRT environment variable not set. Please source xrt/setup.sh."; \
		exit 1; \
	fi; \
	if [ -z "$(AIEOPT_DIR)" ]; then \
		echo "Error: aie-opt not found on PATH. Please source utils/env_setup.sh."; \
		exit 1; \
	fi; \
	echo "Using compiler: $$GPP"; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP ${srcdir}/test.cpp -o test.exe -std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

clean:
	rm -rf build_peano build_chess __pycache__
