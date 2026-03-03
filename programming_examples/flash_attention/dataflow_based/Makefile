# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Attention parameters
LK ?= 12288
LKP ?= 96
LQ ?= 64
DK ?= 64
DV ?= 64

LQP := $(LQ)

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

# AIE_TARGET can be set to 'aie2' (default) or 'aie2p' to select the target architecture.
AIE_TARGET ?= aie2

ifeq ($(AIE_TARGET),aie2p)
  PEANOWRAP_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include
  ATTN_CC = ${srcdir}/attn_aie2p.cc
  BFP16_FLAG = -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
else
  PEANOWRAP_FLAGS = -O2 -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include
  ATTN_CC = ${srcdir}/attn.cc
  BFP16_FLAG =
endif

all: run

print:
	${powershell} python3 ${srcdir}/attn.py -p --arch $(AIE_TARGET)

run: compile-kernel
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/attn.o $(BUILD_DIR)/air_project/attn.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/attn.py --lk $(LK) --lkp $(LKP) --lq $(LQ) --dk $(DK) --dv $(DV) --arch $(AIE_TARGET) $(OUTPUT_FORMAT_FLAG)

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) with target $(AIE_TARGET)"; \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP_FLAGS} -DBIT_WIDTH=8 -c $(ATTN_CC) -o $(BUILD_DIR)/attn.o -Dlqp=$(LQP) -Dlkp=$(LKP) -Ddk=$(DK) -Ddv=$(DV) $(BFP16_FLAG); \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH with target $(AIE_TARGET)"; \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper $(AIE_TARGET) -c $(ATTN_CC) -o attn.o -Dlqp=$(LQP) -Dlkp=$(LKP) -Ddk=$(DK) -Ddv=$(DV) $(BFP16_FLAG); \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR) __pycache__
