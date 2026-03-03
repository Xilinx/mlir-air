# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2_FLAGS = -O2 -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

# AIE_TARGET can be set to 'aie2' (default) or 'aie2p' to select the target architecture.
AIE_TARGET ?= aie2

# Select compiler flags based on AIE_TARGET
ifeq ($(AIE_TARGET),aie2p)
  PEANO_FLAGS = ${PEANOWRAP2P_FLAGS}
else
  PEANO_FLAGS = ${PEANOWRAP2_FLAGS}
endif

all: run

print:
	${powershell} python3 ${srcdir}/bottleneck.py -p

compile-kernels:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) with target $(AIE_TARGET)"; \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANO_FLAGS} -DINT8_ACT -c ${srcdir}/conv2dk1.cc -o $(BUILD_DIR)/conv2dk1.o; \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANO_FLAGS} -DUINT8_ACT -c ${srcdir}/conv2dk3.cc -o $(BUILD_DIR)/conv2dk3.o; \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANO_FLAGS} -DINT8_ACT -c ${srcdir}/conv2dk1_skip.cc -o $(BUILD_DIR)/conv2dk1_skip.o; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH with target $(AIE_TARGET)"; \
		cd $(BUILD_DIR) && \
		xchesscc_wrapper ${AIE_TARGET} -DINT8_ACT -c ${srcdir}/conv2dk1.cc -o conv2dk1.o && \
		xchesscc_wrapper ${AIE_TARGET} -DUINT8_ACT -c ${srcdir}/conv2dk3.cc -o conv2dk3.o && \
		xchesscc_wrapper ${AIE_TARGET} -DINT8_ACT -c ${srcdir}/conv2dk1_skip.cc -o conv2dk1_skip.o; \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

run: compile-kernels
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) ${powershell} python3 ${srcdir}/bottleneck.py

# All kernels are inline MLIR in bottleneck_mlir.py — no .o files needed
run-mlir:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) ${powershell} python3 ${srcdir}/bottleneck_mlir.py

clean:
	rm -rf $(BUILD_DIR) __pycache__
