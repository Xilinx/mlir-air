# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

M ?= 64
K ?= 32

WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG

all: run

print:
	${powershell} python3 ${srcdir}/transpose_bf16.py $(OUTPUT_FORMAT_FLAG) -p -m $(M) -k $(K)

compile-kernel:
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DDIM_M=$(M) -DDIM_N=$(K) \
		-c ${srcdir}/transpose.cc -o $(BUILD_DIR)/transpose.o

run: compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/transpose.o $(BUILD_DIR)/air_project/transpose.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/transpose_bf16.py $(OUTPUT_FORMAT_FLAG) -m $(M) -k $(K)

clean:
	rm -rf $(BUILD_DIR) __pycache__
