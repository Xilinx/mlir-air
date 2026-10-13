# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
# The microkernels and the host benchmarks are kernel_fusion_based's.
fusiondir := $(srcdir)/../kernel_fusion_based

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

# Attention parameters. Blocks are 64x64, so DK = DV = 64 and LQ = LK.
LK ?= 512
LQ ?= 512
DK ?= 64
DV ?= 64
NUM_HEADS ?= 2
BLOCK := 64

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

AIEOPT_DIR ?= $(if $(MLIR_AIE_INSTALL_DIR),$(subst \,/,$(MLIR_AIE_INSTALL_DIR)),$(shell realpath $(dir $(shell which aie-opt))/..))
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include -D__AIE_API_AIE_ADF_HPP__
KERNEL_DEFS = -I${fusiondir} -Dlqp=$(BLOCK) -Dlkp=$(BLOCK) -Ddk=$(BLOCK) -Ddk_full=$(DK) -Ddv=$(BLOCK) -Ddv_full=$(DV)
PY_ARGS = --lk $(LK) --lq $(LQ) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS)

all: run

print:
	${powershell} $(PYTHON) ${srcdir}/attn_npu2.py -p $(PY_ARGS) $(EXTRA_PY_FLAGS)

run: compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} $(PYTHON) ${srcdir}/attn_npu2.py $(PY_ARGS) $(EXTRA_PY_FLAGS)

# Profile ELF: compile elf and run with C++ test executable for elf format
# Usage: make profile [LK=...] [LQ=...] etc.
profile: compile-kernel build-test-exe
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} $(PYTHON) ${srcdir}/attn_npu2.py \
		$(PY_ARGS) $(EXTRA_PY_FLAGS) --compile-mode compile-only
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ./test_elf_npu2.exe -e air.elf -k "main:attention_bf16" \
		--lq $(LQ) --lk $(LK) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS)

# Profile xclbin: compile xclbin and run with C++ test executable
# Usage: make profile-xclbin [LK=...] [LQ=...] etc.
profile-xclbin: compile-kernel build-test-exe-xclbin
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} $(PYTHON) ${srcdir}/attn_npu2.py \
		$(PY_ARGS) $(EXTRA_PY_FLAGS) --compile-mode compile-only --output-format xclbin
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ./test_xclbin.exe \
		-x air.xclbin -k "MLIR_AIE" -i air.insts.bin \
		--lq $(LQ) --lk $(LK) --dk $(DK) --dv $(DV) --num-heads $(NUM_HEADS)

build-test-exe-xclbin:
	$(MAKE) -f $(fusiondir)/Makefile build-test-exe-xclbin BUILD_DIR=$(abspath $(BUILD_DIR))

build-test-exe:
	$(MAKE) -f $(fusiondir)/Makefile build-test-exe BUILD_DIR=$(abspath $(BUILD_DIR))

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)"; \
		$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c ${srcdir}/attn_stages.cc -o $(BUILD_DIR)/attn_stages.o $(KERNEL_DEFS) -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -DROUND_CONV_EVEN; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH"; \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper aie2p -c ${srcdir}/attn_stages.cc -o attn_stages.o $(KERNEL_DEFS); \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR) __pycache__
