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
HERD_N ?= 4

# Per-tile N -- the C kernel processes one tile per call.
N_TILE := $(shell expr $(N) / $(HERD_N))

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

PY_ARGS = --n $(N) --group-size $(GROUP_SIZE) --herd-n $(HERD_N)

all: run

print:
	${powershell} python3 ${srcdir}/dequant_awq.py $(OUTPUT_FORMAT_FLAG) -p $(PY_ARGS)

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} \
			-DDIM_N=$(N_TILE) -DGROUP_SIZE=$(GROUP_SIZE) \
			-c ${srcdir}/dequant.cc -o $(BUILD_DIR)/dequant.o; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper aie2p -c ${srcdir}/dequant.cc \
			-o dequant.o -DDIM_N=$(N_TILE) -DGROUP_SIZE=$(GROUP_SIZE); \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

run: compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/dequant.o $(BUILD_DIR)/air_project/dequant.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/dequant_awq.py $(OUTPUT_FORMAT_FLAG) $(PY_ARGS)

# ELF profile harness (compile module ELF, then run XRT benchmark loop)
build-test-dequant-exe:
	@$(MAKE) build-test-exe-impl SRC=test_dequant.cpp OUT=test_dequant.exe

profile: build-test-dequant-exe compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/dequant.o $(BUILD_DIR)/air_project/dequant.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/dequant_awq.py --compile-mode compile-only --output-format elf $(PY_ARGS)
	cd $(BUILD_DIR) && ./test_dequant.exe -e air.elf -k "main:dequant" \
		-N $(N) -G $(GROUP_SIZE) -H $(HERD_N)

build-test-exe-impl:
	@GPP=$$( \
		for bin in /usr/bin/g++-*; do \
			ver=$$(echo $$bin | grep -oE '[0-9]+$$'); \
			if [ "$$ver" -ge 13 ] 2>/dev/null; then \
				echo "$$ver $$bin"; \
			fi; \
		done | sort -nr | head -n1 | awk '{print $$2}' \
	); \
	if [ -z "$$GPP" ]; then \
		echo "Error: No g++ version >= 13 found in /usr/bin."; exit 1; \
	fi; \
	if [ -z "$$XILINX_XRT" ]; then \
		echo "Error: XILINX_XRT not set (source xrt/setup.sh)."; exit 1; \
	fi; \
	if [ -z "$(AIEOPT_DIR)" ]; then \
		echo "Error: AIEOPT_DIR unset (source utils/env_setup.sh)."; exit 1; \
	fi; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP ${srcdir}/$(SRC) -o $(OUT) -std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

clean:
	rm -rf $(BUILD_DIR) __pycache__
