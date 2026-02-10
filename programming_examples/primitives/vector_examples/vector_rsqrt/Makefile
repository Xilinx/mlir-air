# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

BUILD_DIR := build_peano

# Output format: xclbin (default) or elf
OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2_FLAGS = -O2 -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

# AIE_TARGET can be set to 'aie2' (default) or 'aie2p' to select the target architecture.
# Usage examples:
#   make run                  # uses aie2 target
#   make run AIE_TARGET=aie2p # uses aie2p target
AIE_TARGET ?= aie2

all: run

# Default target (Version 1)
print:
	${powershell} python3 ${srcdir}/vector_rsqrt_v1.py $(OUTPUT_FORMAT_FLAG) -p --arch $(AIE_TARGET)

run: run-v1

# Version 1: f32 vector rsqrt
print-v1:
	${powershell} python3 ${srcdir}/vector_rsqrt_v1.py $(OUTPUT_FORMAT_FLAG) -p --arch $(AIE_TARGET)

run-v1: compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/vector_rsqrt_v1.py $(OUTPUT_FORMAT_FLAG) --arch $(AIE_TARGET)

# Version 2: f32 scalar rsqrt in loop
print-v2:
	${powershell} python3 ${srcdir}/vector_rsqrt_v2.py $(OUTPUT_FORMAT_FLAG) -p --arch $(AIE_TARGET)

run-v2: compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/vector_rsqrt_v2.py $(OUTPUT_FORMAT_FLAG) --arch $(AIE_TARGET)

# Version 3: bf16 vector rsqrt with f32 conversion
print-v3:
	${powershell} python3 ${srcdir}/vector_rsqrt_v3.py $(OUTPUT_FORMAT_FLAG) -p --arch $(AIE_TARGET)

run-v3: compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/vector_rsqrt_v3.py $(OUTPUT_FORMAT_FLAG) --arch $(AIE_TARGET)

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ "$(AIE_TARGET)" = "aie2" ]; then \
		if [ -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
			echo "Compiling extern_func.o for aie2 using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)"; \
			$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2_FLAGS} -c ${srcdir}/extern_func.cc -o $(BUILD_DIR)/extern_func.o; \
		else \
			echo "Error: PEANO_INSTALL_DIR not set or clang++ not found."; \
			exit 1; \
		fi; \
	else \
		echo "Skipping extern_func.o compilation for $(AIE_TARGET)"; \
	fi

clean:
	rm -rf $(BUILD_DIR) __pycache__
