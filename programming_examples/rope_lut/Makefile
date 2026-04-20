# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

SEQ_LEN ?= 64
EMBED_DIM ?= 64
HERD_X ?= 1

WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

all: run

print:
	${powershell} python3 ${srcdir}/rope_lut.py $(OUTPUT_FORMAT_FLAG) \
		--seq-len $(SEQ_LEN) --embed-dim $(EMBED_DIM) --herd-x $(HERD_X) -p

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -c ${srcdir}/rope.cc -o $(BUILD_DIR)/rope.o; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper aie2p -c ${srcdir}/rope.cc -o rope.o; \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

compile-xclbin: compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/rope.o $(BUILD_DIR)/air_project/rope.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/rope_lut.py --output-format xclbin --compile-mode compile-only \
		--seq-len $(SEQ_LEN) --embed-dim $(EMBED_DIM) --herd-x $(HERD_X)

run: compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/rope.o $(BUILD_DIR)/air_project/rope.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/rope_lut.py $(OUTPUT_FORMAT_FLAG) \
		--seq-len $(SEQ_LEN) --embed-dim $(EMBED_DIM) --herd-x $(HERD_X)

# C++ profiling harness (10 warmup + 20 measured iterations, microsecond precision)
profile: compile-xclbin build-test-exe
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin \
		-R $(SEQ_LEN) -D $(EMBED_DIM)

# LLAMA shapes: correctness (make run) + profiling (C++ harness) for Q and K
run_llama: compile-kernel build-test-exe
	@echo "============================================"
	@echo "  Prefill RoPE Q: 65536 rows x 64, herd=[8,1]"
	@echo "============================================"
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/rope.o $(BUILD_DIR)/air_project/rope.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		python3 ${srcdir}/rope_lut.py --output-format xclbin \
		--seq-len 65536 --embed-dim 64 --herd-x 8
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin \
		-R 65536 -D 64
	@echo ""
	@echo "============================================"
	@echo "  Prefill RoPE K: 16384 rows x 64, herd=[8,1]"
	@echo "============================================"
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		python3 ${srcdir}/rope_lut.py --output-format xclbin \
		--seq-len 16384 --embed-dim 64 --herd-x 8
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin \
		-R 16384 -D 64

# Default XRT path; uses ?= so XILINX_XRT from environment (xrt/setup.sh)
# takes precedence. Fallback needed for LIT tests where env may not propagate.
XILINX_XRT ?= /opt/xilinx/xrt

build-test-exe:
	@GPP=$$( \
		for bin in /usr/bin/g++-*; do \
			ver=$$(echo $$bin | grep -oE '[0-9]+$$'); \
			if [ "$$ver" -ge 13 ] 2>/dev/null; then \
				echo "$$ver $$bin"; \
			fi; \
		done | sort -nr | head -n1 | awk '{print $$2}' \
	); \
	if [ -z "$$GPP" ]; then \
		echo "Error: No g++ version >= 13 found in /usr/bin."; \
		exit 1; \
	fi; \
	echo "Using compiler: $$GPP"; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP ${srcdir}/test.cpp -o test.exe -std=c++23 -Wall \
		-I$(XILINX_XRT)/include -L$(XILINX_XRT)/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

clean:
	rm -rf $(BUILD_DIR) __pycache__
