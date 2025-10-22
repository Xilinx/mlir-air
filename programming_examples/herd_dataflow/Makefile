# Copyright (C) 2025, Advanced Micro Devices, Inc.
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
# Usage examples:
#   make run                  # uses aie2 target
#   make run AIE_TARGET=aie2p # uses aie2p target
AIE_TARGET ?= aie2

# Select compiler flags based on AIE_TARGET
ifeq ($(AIE_TARGET),aie2p)
  PEANO_FLAGS = ${PEANOWRAP2P_FLAGS}
else
  PEANO_FLAGS = ${PEANOWRAP2_FLAGS}
endif

# Parametrizable sizes
# M_SIZE must be a multiple of 64, N_SIZE must be a multiple of 256 (partial tiles not supported)
M_SIZE ?= 256
N_SIZE ?= 256

# MLIR_SOURCE can be set to 'python' (default) or 'file' to select how MLIR is obtained.
# Usage examples:
#   make print                # uses python bindings
#   make print MLIR_SOURCE=file  # uses air.mlir file
#   make run                  # uses python bindings
#   make run MLIR_SOURCE=file    # uses air.mlir file
MLIR_SOURCE ?= python

all: run

.PHONY: check_sizes

check_sizes:
	@if [ $$(( $(M_SIZE) % 64 )) -ne 0 ]; then \
		echo "Error: M_SIZE ($(M_SIZE)) must be a multiple of 64."; exit 1; \
	fi
	@if [ $$(( $(N_SIZE) % 256 )) -ne 0 ]; then \
		echo "Error: N_SIZE ($(N_SIZE)) must be a multiple of 256, assuming that 4 columns of cores are used."; exit 1; \
	fi

print:
	$(MAKE) -f $(srcdir)/Makefile check_sizes
	${powershell} python3 ${srcdir}/run.py --m-size $(M_SIZE) --n-size $(N_SIZE) -p --mlir-source $(MLIR_SOURCE)

run: compile-kernel
	$(MAKE) -f $(srcdir)/Makefile check_sizes
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/run.py --m-size $(M_SIZE) --n-size $(N_SIZE) --mlir-source $(MLIR_SOURCE)

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Detected PEANO_INSTALL_DIR from environment: $(PEANO_INSTALL_DIR)"; \
		if [ -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
			echo "Using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) with target $(AIE_TARGET)"; \
			$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANO_FLAGS} -DBIT_WIDTH=8 -c ${srcdir}/extern_func.cc -o $(BUILD_DIR)/extern_func.o; \
		else \
			echo "Error: invalid PEANO_INSTALL_DIR, clang++ not found."; \
			exit 1; \
		fi; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH with target $(AIE_TARGET)"; \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper ${AIE_TARGET} -c ${srcdir}/extern_func.cc -o extern_func.o; \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR) __pycache__
