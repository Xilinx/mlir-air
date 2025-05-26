# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Allow user to override BACKEND (default: auto-detect)
BACKEND ?= auto

# Extract PEANO_INSTALL_DIR from the installed llvm-aie pip package
PEANO_INSTALL_DIR := $(shell python3 -m pip show llvm-aie 2>/dev/null | grep ^Location: | awk '{print $$2}')/llvm-aie
AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2_FLAGS = -O2 -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

all: run4x4

print:
	${powershell} python3 ${srcdir}/run.py -p

run4x4: compile-kernel
	cd build && ${powershell} python3 ${srcdir}/run.py --herd-m 4 --herd-n 4 --m 512 --n 512 --k 512 --compile-mode compile-and-run

run2x4: compile-kernel
	cd build && ${powershell} python3 ${srcdir}/run.py --herd-m 2 --herd-n 4 --m 512 --n 512 --k 512 --compile-mode compile-and-run

run2x2: compile-kernel
	cd build && ${powershell} python3 ${srcdir}/run.py --herd-m 2 --herd-n 2 --m 512 --n 512 --k 512 --compile-mode compile-and-run

compile-kernel:
	mkdir -p build
	@if [ "$(BACKEND)" = "peano" ]; then \
		if [ -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
			echo "Using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)"; \
			$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c ${srcdir}/mm.cc -o build/mm.o; \
		else \
			echo "Error: llvm-aie pip package not found or invalid."; \
			exit 1; \
		fi \
	elif [ "$(BACKEND)" = "xchesscc" ]; then \
		if command -v xchesscc_wrapper >/dev/null 2>&1; then \
			echo "Using xchesscc_wrapper from PATH"; \
			cd build && ${powershell} xchesscc_wrapper aie2 -c ${srcdir}/mm.cc -o mm.o; \
		else \
			echo "Error: xchesscc_wrapper not found in PATH."; \
			exit 1; \
		fi \
	else \
		if [ -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
			echo "Auto-selected peano backend"; \
			$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c ${srcdir}/mm.cc -o build/mm.o; \
		elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
			echo "Auto-selected xchesscc backend"; \
			cd build && ${powershell} xchesscc_wrapper aie2 -c ${srcdir}/mm.cc -o mm.o; \
		else \
			echo "Error: no usable backend found."; \
			exit 1; \
		fi \
	fi

clean:
	rm -rf build __pycache__
