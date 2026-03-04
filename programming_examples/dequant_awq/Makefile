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

N ?= 1024
GROUP_SIZE ?= 128

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

all: run

print:
	${powershell} python3 ${srcdir}/dequant_awq.py $(OUTPUT_FORMAT_FLAG) -p --n $(N) --group-size $(GROUP_SIZE)

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
			-DDIM_N=$(N) -DGROUP_SIZE=$(GROUP_SIZE) \
			-c ${srcdir}/dequant.cc -o $(BUILD_DIR)/dequant.o; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper aie2p -c ${srcdir}/dequant.cc \
			-o dequant.o -DDIM_N=$(N) -DGROUP_SIZE=$(GROUP_SIZE); \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

run: compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/dequant.o $(BUILD_DIR)/air_project/dequant.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/dequant_awq.py $(OUTPUT_FORMAT_FLAG) --n $(N) --group-size $(GROUP_SIZE)

clean:
	rm -rf $(BUILD_DIR) __pycache__
