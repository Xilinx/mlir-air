# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

M       ?= 64
K       ?= 128
N       ?= 128
TILE_M  ?= 32
TILE_N  ?= 32
TILE_K_L1 ?= 128
TILE_K_L2 ?= $(K)
HERD_M  ?= 2
HERD_N  ?= 4

# bfp16ebs8 MMUL is Peano-only; xchesscc has no bfp16ebs8 codegen path.
# compile-kernel errors at recipe time if PEANO_INSTALL_DIR is unset.
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

PY_ARGS = --m $(M) --k $(K) --n $(N) \
          --tile-m $(TILE_M) --tile-n $(TILE_N) \
          --tile-k-l1 $(TILE_K_L1) --tile-k-l2 $(TILE_K_L2) \
          --herd-m $(HERD_M) --herd-n $(HERD_N)

all: run

print:
	${powershell} python3 $(srcdir)/matmul_bf16_x_bfp16.py $(PY_ARGS) -p

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -z "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Error: PEANO_INSTALL_DIR not set. Chess toolchain is not"\
		     "supported for bfp16ebs8; source utils/env_setup.sh."; \
		exit 1; \
	fi
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) \
		-DDIM_M=$(TILE_M) -DDIM_N=$(TILE_N) -DDIM_K=$(TILE_K_L1) \
		-c $(srcdir)/mm_bf16_x_bfp16.cc -o $(BUILD_DIR)/mm_bf16_x_bfp16.o

run: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 $(srcdir)/matmul_bf16_x_bfp16.py $(PY_ARGS)

compile-only: compile-kernel
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 $(srcdir)/matmul_bf16_x_bfp16.py $(PY_ARGS) --compile-mode compile-only

# Build the C++ XRT timing harness. Mirrors bf16/Makefile:build-test-exe.
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
		echo "Error: XILINX_XRT not set. Source xrt/setup.sh first."; \
		exit 1; \
	fi; \
	if [ -z "$(AIEOPT_DIR)" ]; then \
		echo "Error: AIEOPT_DIR not set. Source utils/env_setup.sh first."; \
		exit 1; \
	fi; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP $(srcdir)/test.cpp -o test.exe -std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(AIEOPT_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

# End-to-end latency profile: build xclbin then run the C++ harness.
profile: compile-kernel build-test-exe
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
	  ${powershell} python3 $(srcdir)/matmul_bf16_x_bfp16.py $(PY_ARGS) \
	    --compile-mode compile-and-xclbin
	cd $(BUILD_DIR) && ./test.exe -x air.xclbin -k MLIR_AIE \
	  -i air.insts.bin -M $(M) -K $(K) -N $(N)

clean:
	rm -rf $(BUILD_DIR) __pycache__
