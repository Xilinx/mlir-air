# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= elf
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

SEQ_LEN ?= 16
DIM ?= 128
NUM_COLS ?= 4
DIM_N = $(shell echo $$(( $(DIM) / $(NUM_COLS) )))

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

all: run

print:
	${powershell} python3 ${srcdir}/ffn_prefill.py $(OUTPUT_FORMAT_FLAG) -p \
		--seq-len $(SEQ_LEN) --dim $(DIM) --num-cols $(NUM_COLS)

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
			-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 \
			-DDIM_M=$(SEQ_LEN) -DDIM_K=$(DIM) -DDIM_N=$(DIM_N) \
			-c ${srcdir}/ffn_kernels.cc -o $(BUILD_DIR)/ffn_kernels.o; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper aie2p -c ${srcdir}/ffn_kernels.cc \
			-o ffn_kernels.o -DDIM_M=$(SEQ_LEN) -DDIM_K=$(DIM) -DDIM_N=$(DIM_N); \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

run: compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/ffn_kernels.o $(BUILD_DIR)/air_project/ffn_kernels.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/ffn_prefill.py $(OUTPUT_FORMAT_FLAG) \
		--seq-len $(SEQ_LEN) --dim $(DIM) --num-cols $(NUM_COLS)

clean:
	rm -rf $(BUILD_DIR) __pycache__
