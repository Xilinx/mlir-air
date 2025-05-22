# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

all: run

print:
	${powershell} python3 ${srcdir}/single_core.py -p

run: compile-kernel
	cd build && ${powershell} python3 ${srcdir}/single_core.py

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2_FLAGS = -O2 -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

compile-kernel:
	mkdir -p build
	@if [ -n "$$LLVM_AIE_INSTALL_DIR" ]; then \
		echo "Using clang++ from LLVM_AIE_INSTALL_DIR"; \
		$$LLVM_AIE_INSTALL_DIR/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c ${srcdir}/vm.cc -o build/vm.o; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH"; \
		cd build && ${powershell} xchesscc_wrapper aie2 -c ${srcdir}/vm.cc -o vm.o; \
	else \
		echo "Error: Neither LLVM_AIE_INSTALL_DIR nor xchesscc_wrapper found." >&2; \
		exit 1; \
	fi

clean:
	rm -rf build __pycache__
