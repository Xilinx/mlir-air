# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
# The microkernels are kernel_fusion_based's.
fusiondir := $(srcdir)/../kernel_fusion_based

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

# Attention parameters. Blocks are 64x64, so DK = DV = 64 and LQ = LK.
LK ?= 512
LQ ?= 512
DK ?= 64
DV ?= 64
NUM_HEADS ?= 2
BLOCK := 64
# Operator for `make print`: qk, sm or pv.
OP ?= qk

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

AIEOPT_DIR ?= $(if $(MLIR_AIE_INSTALL_DIR),$(subst \,/,$(MLIR_AIE_INSTALL_DIR)),$(shell realpath $(dir $(shell which aie-opt))/..))
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include -D__AIE_API_AIE_ADF_HPP__
KERNEL_DEFS = -Dlqp=$(BLOCK) -Dlkp=$(BLOCK) -Ddk=$(BLOCK) -Ddk_full=$(DK) -Ddv=$(BLOCK) -Ddv_full=$(DV)
PY_ARGS = --lk $(LK) --lq $(LQ) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS)
BENCH_ARGS = --lq $(LQ) --lk $(LK) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS)

all: run

print:
	${powershell} $(PYTHON) ${srcdir}/attn_npu2.py -p --op $(OP) $(PY_ARGS) $(EXTRA_PY_FLAGS)

run: compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} $(PYTHON) ${srcdir}/attn_npu2.py $(PY_ARGS) $(EXTRA_PY_FLAGS)

# Profile ELF: compile the three operators as ELF and time qk -> sm -> pv
# Usage: make profile [LK=...] [LQ=...] etc.
profile: compile-kernel build-test-exe
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} $(PYTHON) ${srcdir}/attn_npu2.py \
		$(PY_ARGS) $(EXTRA_PY_FLAGS) --compile-mode compile-only
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ./test_npu2.exe --format elf $(BENCH_ARGS)

# Profile xclbin: compile the three operators as xclbin and time qk -> sm -> pv
# Usage: make profile-xclbin [LK=...] [LQ=...] etc.
profile-xclbin: compile-kernel build-test-exe
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} $(PYTHON) ${srcdir}/attn_npu2.py \
		$(PY_ARGS) $(EXTRA_PY_FLAGS) --compile-mode compile-only --output-format xclbin
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ./test_npu2.exe --format xclbin $(BENCH_ARGS)

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
	if [ -z "$$XILINX_XRT" ]; then \
		echo "Error: XILINX_XRT environment variable not set. Please make sure to have sourced xrt/setup.sh."; \
		exit 1; \
	fi; \
	if [ -z "$(AIEOPT_DIR)" ]; then \
		echo "Error: AIEOPT_DIR environment variable not set. Please make sure to have sourced utils/env_setup.sh."; \
		exit 1; \
	fi; \
	echo "Using compiler: $$GPP"; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP ${srcdir}/test_npu2.cpp -o test_npu2.exe -std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)"; \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c ${fusiondir}/attn_npu2.cc -o $(BUILD_DIR)/attn_npu2.o $(KERNEL_DEFS) -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -DROUND_CONV_EVEN; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH"; \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper aie2p -c ${fusiondir}/attn_npu2.cc -o attn_npu2.o $(KERNEL_DEFS); \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR) __pycache__
