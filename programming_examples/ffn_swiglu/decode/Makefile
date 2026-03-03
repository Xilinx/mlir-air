# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)

DIM ?= 128
NUM_COLS ?= 4
DIM_M_PER_COL = $(shell echo $$(( $(DIM) / $(NUM_COLS) )))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

all: run

print:
	${powershell} python3 ${srcdir}/ffn_decode.py $(OUTPUT_FORMAT_FLAG) -p --dim $(DIM) --num-cols $(NUM_COLS)

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -c ${srcdir}/ffn_kernels.cc -o $(BUILD_DIR)/ffn_kernels.o -DDIM_M=$(DIM_M_PER_COL) -DDIM_K=$(DIM); \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper aie2p -c ${srcdir}/ffn_kernels.cc -o ffn_kernels.o -DDIM_M=$(DIM_M_PER_COL) -DDIM_K=$(DIM); \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

run: compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/ffn_kernels.o $(BUILD_DIR)/air_project/ffn_kernels.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/ffn_decode.py --dim $(DIM) --num-cols $(NUM_COLS) $(OUTPUT_FORMAT_FLAG)

clean:
	rm -rf $(BUILD_DIR) __pycache__
