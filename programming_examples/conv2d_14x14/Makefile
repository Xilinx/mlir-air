# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

DEVICE ?= npu2

ifeq ($(DEVICE),npu2)
  AIE_TARGET := aie2p
  AIE_KERNEL_DIR := aie2p
else ifeq ($(DEVICE),npu1)
  AIE_TARGET := aie2
  AIE_KERNEL_DIR := aie2p
else
  $(error Unsupported DEVICE='$(DEVICE)'. Use DEVICE=npu1 or DEVICE=npu2)
endif

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano_$(DEVICE)
else
  BUILD_DIR := build_chess_$(DEVICE)
endif

OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)
DEVICE_FLAG = --target-device $(DEVICE)

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2_FLAGS  = -O2 -std=c++20 --target=aie2-none-unknown-elf  ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

ifeq ($(AIE_TARGET),aie2p)
  PEANO_FLAGS = ${PEANOWRAP2P_FLAGS}
else
  PEANO_FLAGS = ${PEANOWRAP2_FLAGS}
endif

# The aie2p source compiles for both aie2 and aie2p targets; on aie2 the
# mmul<8,8,8> specialization is emulated via two mmul<4,8,8> macs.
KERNEL_SRC = ${AIEOPT_DIR}/include/aie_kernels/$(AIE_KERNEL_DIR)/conv2dk14.cc

all: run

print:
	${powershell} python3 ${srcdir}/conv2d_14x14.py $(OUTPUT_FORMAT_FLAG) $(DEVICE_FLAG) -p

run: compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/conv2d_14x14.py $(OUTPUT_FORMAT_FLAG) $(DEVICE_FLAG)

profile: compile-kernel build-test-exe
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/conv2d_14x14.py $(OUTPUT_FORMAT_FLAG) $(DEVICE_FLAG)
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin --device $(DEVICE)

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
	if [ -z "$(AIEOPT_DIR)" ]; then \
		echo "Error: AIEOPT_DIR not set. Source utils/env_setup.sh."; \
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
			$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANO_FLAGS} -c ${KERNEL_SRC} -o $(BUILD_DIR)/conv2dk14.o; \
		else \
			echo "Error: invalid PEANO_INSTALL_DIR, clang++ not found."; \
			exit 1; \
		fi; \
	else \
		echo "Error: PEANO_INSTALL_DIR not set."; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR) __pycache__
