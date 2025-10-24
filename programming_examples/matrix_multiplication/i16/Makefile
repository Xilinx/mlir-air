# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
  # DIRECT_CODEGEN_FLAG can be overridden from command line (e.g., make DIRECT_CODEGEN_FLAG= run4x4)
  DIRECT_CODEGEN_FLAG ?= --direct-codegen
else
  BUILD_DIR := build_chess
  DIRECT_CODEGEN_FLAG ?=
endif

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2_FLAGS = -O2 -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

# AIE_TARGET can be set to 'aie2' (default) or 'aie2p' to select the target architecture.
# Usage examples:
#   make run4x4                  # uses aie2 target
#   make run4x4 AIE_TARGET=aie2p # uses aie2p target
AIE_TARGET ?= aie2

# Select compiler flags and kernel dependency based on AIE_TARGET
# Matmul kernels with i16 inputs are currently incompatible with aie2p, so KERNEL_DEP is empty for aie2p
ifeq ($(AIE_TARGET),aie2p)
  PEANO_FLAGS = ${PEANOWRAP2P_FLAGS}
  KERNEL_DEP =
else
  PEANO_FLAGS = ${PEANOWRAP2_FLAGS}
  KERNEL_DEP = compile-kernel
endif

all: run4x4

print:
	${powershell} python3 ${srcdir}/run.py -p --arch $(AIE_TARGET) $(DIRECT_CODEGEN_FLAG)

run4x4: $(KERNEL_DEP)
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/run.py --herd-m 4 --herd-n 4 --m 512 --n 512 --k 512 --compile-mode compile-and-run --arch $(AIE_TARGET) $(DIRECT_CODEGEN_FLAG)

run2x4: $(KERNEL_DEP)
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/run.py --herd-m 2 --herd-n 4 --m 512 --n 512 --k 512 --compile-mode compile-and-run --arch $(AIE_TARGET) $(DIRECT_CODEGEN_FLAG)

run2x2: $(KERNEL_DEP)
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/run.py --herd-m 2 --herd-n 2 --m 512 --n 512 --k 512 --compile-mode compile-and-run --arch $(AIE_TARGET) $(DIRECT_CODEGEN_FLAG)

run3x3: $(KERNEL_DEP)
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/run.py --herd-m 3 --herd-n 3 --m 576 --n 576 --k 576 --tile-k-l2 288 --tile-k-l1 48 --compile-mode compile-and-run --arch $(AIE_TARGET) $(DIRECT_CODEGEN_FLAG)

# Measure the e2e latencies across a range of problem shapes.
sweep4x4: $(KERNEL_DEP) build-test-exe
	@for M in 256 512 1024 2048; do \
		for N in 256 512 1024 2048; do \
			for K in 256 512 1024 2048; do \
			echo "Running with M=$$M, N=$$N, K=$$K"; \
			cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/run.py --m $$M --n $$N --k $$K --compile-mode compile-only --arch $(AIE_TARGET) $(DIRECT_CODEGEN_FLAG); \
			./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin -M $$M -N $$N -K $$K; \
			done; \
		done; \
	done

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Detected PEANO_INSTALL_DIR from environment: $(PEANO_INSTALL_DIR)"; \
		if [ -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
			echo "Using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) with target $(AIE_TARGET)"; \
			$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANO_FLAGS} -DBIT_WIDTH=8 -c ${srcdir}/mm.cc -o $(BUILD_DIR)/mm.o; \
		else \
			echo "Error: invalid PEANO_INSTALL_DIR, clang++ not found."; \
			exit 1; \
		fi; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH with target $(AIE_TARGET)"; \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper ${AIE_TARGET} -c ${srcdir}/mm.cc -o mm.o; \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

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

clean:
	rm -rf $(BUILD_DIR) __pycache__
