# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
progdir := $(realpath $(srcdir)/../..)

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Peano compiler flags
AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

all: run

# Compile the SwiGLU C++ kernel
compile-kernel:
	mkdir -p $(BUILD_DIR)/air_project
	$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-include aie_kernels/aie_kernel_utils.h \
		-c $(srcdir)/silu_and_mul.cc -o $(BUILD_DIR)/air_project/silu_and_mul.o

# Print combined MLIR module
print:
	${powershell} python3 ${srcdir}/run.py -p

# Print each sub-kernel's MLIR separately
print-kernels:
	${powershell} python3 ${srcdir}/run.py --print-kernels -p

# Compile + run correctness test
run: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/run.py

# Compile + run with profiling
profile: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/run.py --profile

clean:
	rm -rf $(BUILD_DIR) __pycache__
