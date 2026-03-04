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

NUM_Q_HEADS ?= 4
NUM_KV_HEADS ?= 2
LQ ?= 16
LK ?= 16
D ?= 16

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

all: run

print:
	${powershell} python3 ${srcdir}/gqa.py $(OUTPUT_FORMAT_FLAG) -p \
		--num-q-heads $(NUM_Q_HEADS) --num-kv-heads $(NUM_KV_HEADS) \
		--lq $(LQ) --lk $(LK) --d $(D)

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
			-DDIM_LQ=$(LQ) -DDIM_LK=$(LK) -DDIM_D=$(D) \
			-c ${srcdir}/gqa_kernels.cc -o $(BUILD_DIR)/gqa_kernels.o; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper aie2p -c ${srcdir}/gqa_kernels.cc \
			-o gqa_kernels.o -DDIM_LQ=$(LQ) -DDIM_LK=$(LK) -DDIM_D=$(D); \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

run: compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/gqa_kernels.o $(BUILD_DIR)/air_project/gqa_kernels.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/gqa.py $(OUTPUT_FORMAT_FLAG) \
		--num-q-heads $(NUM_Q_HEADS) --num-kv-heads $(NUM_KV_HEADS) \
		--lq $(LQ) --lk $(LK) --d $(D)

clean:
	rm -rf $(BUILD_DIR) __pycache__
