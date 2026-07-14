# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Target NPU device. npu2 = AIE2P (Strix), npu1 = AIE2 (Phoenix). npu1 only
# supports --output-format=xclbin (ELF output requires npu2 or later).
DEVICE ?= npu2

OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

# DIRECT_CODEGEN_FLAG: pass --direct-codegen to emit the per-tile dequant
# body as inline MLIR ops instead of calling dequant.o. AIE2P only.
#   make run                                # extern .o (default)
#   make run DIRECT_CODEGEN_FLAG=--direct-codegen   # inline MLIR
DIRECT_CODEGEN_FLAG ?=

N ?= 1024
GROUP_SIZE ?= 128
HERD_N ?= 4

# Per-tile N -- the C kernel processes one tile per call.
N_TILE := $(shell expr $(N) / $(HERD_N))

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include -D__AIE_API_AIE_ADF_HPP__
PEANOWRAP2_FLAGS  = -O2 -std=c++20 --target=aie2-none-unknown-elf  ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include -D__AIE_API_AIE_ADF_HPP__ -mllvm -aie-disable-fold-imm

ifeq ($(DEVICE),npu1)
  PEANO_FLAGS = $(PEANOWRAP2_FLAGS)
  XCHESS_TARGET = aie2
else
  PEANO_FLAGS = $(PEANOWRAP2P_FLAGS)
  XCHESS_TARGET = aie2p
endif

PY_ARGS = --device $(DEVICE) --n $(N) --group-size $(GROUP_SIZE) --herd-n $(HERD_N)

all: run

print:
	${powershell} python3 ${srcdir}/dequant_awq.py $(OUTPUT_FORMAT_FLAG) -p $(PY_ARGS) $(DIRECT_CODEGEN_FLAG)

# Build dequant.o for the extern path. No-op when --direct-codegen is set
# (the inline path has no external kernel object).
compile-kernel:
ifeq ($(DIRECT_CODEGEN_FLAG),)
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANO_FLAGS} \
			-DDIM_N=$(N_TILE) -DGROUP_SIZE=$(GROUP_SIZE) \
			-c ${srcdir}/dequant.cc -o $(BUILD_DIR)/dequant.o; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper $(XCHESS_TARGET) -c ${srcdir}/dequant.cc \
			-o dequant.o -DDIM_N=$(N_TILE) -DGROUP_SIZE=$(GROUP_SIZE); \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi
else
	@echo "DIRECT_CODEGEN_FLAG set: skipping dequant.o build (inline path)."
endif

run: compile-kernel
	mkdir -p $(BUILD_DIR)/air_project
ifeq ($(DIRECT_CODEGEN_FLAG),)
	cp $(BUILD_DIR)/dequant.o $(BUILD_DIR)/air_project/dequant.o
endif
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/dequant_awq.py $(OUTPUT_FORMAT_FLAG) $(DIRECT_CODEGEN_FLAG) $(PY_ARGS)

# ELF profile harness (compile module ELF, then run XRT benchmark loop)
build-test-dequant-exe:
	@$(MAKE) build-test-exe-impl SRC=test_dequant.cpp OUT=test_dequant.exe

profile: build-test-dequant-exe compile-kernel
ifeq ($(DEVICE),npu1)
	$(error `make profile` requires DEVICE=npu2: the ELF profile harness compiles with --output-format=elf, which is only supported on npu2 and later. Use `make run` for npu1 correctness.)
endif
	mkdir -p $(BUILD_DIR)/air_project
ifeq ($(DIRECT_CODEGEN_FLAG),)
	cp $(BUILD_DIR)/dequant.o $(BUILD_DIR)/air_project/dequant.o
endif
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/dequant_awq.py --compile-mode compile-only --output-format elf $(DIRECT_CODEGEN_FLAG) $(PY_ARGS)
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
