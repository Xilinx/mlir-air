# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Attention parameters
LK ?= 12288
LKP ?= 96
LQ ?= 128
DK ?= 64
DV ?= 64

# Derived parameter (lqp = lq / num_q_tiles, where num_q_tiles = 4)
LQP := $(shell echo $$(( $(LQ) / 4 )))

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

all: run

print:
	${powershell} python3 ${srcdir}/attn.py -p --lk $(LK) --lkp $(LKP) --lq $(LQ) --dk $(DK) --dv $(DV)

run: compile-kernel
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/attn.py --lk $(LK) --lkp $(LKP) --lq $(LQ) --dk $(DK) --dv $(DV)

compile-kernel:
	mkdir -p $(BUILD_DIR)
	@if [ -n "$(PEANO_INSTALL_DIR)" ]; then \
		echo "Detected PEANO_INSTALL_DIR from environment: $(PEANO_INSTALL_DIR)"; \
		if [ -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
			echo "Using clang++ from PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)"; \
			$(PEANO_INSTALL_DIR)/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c ${srcdir}/attn.cc -o $(BUILD_DIR)/attn.o -Dlqp=$(LQP) -Dlkp=$(LKP) -Ddk=$(DK) -Ddv=$(DV) -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16; \
		else \
			echo "Error: invalid PEANO_INSTALL_DIR, clang++ not found."; \
			exit 1; \
		fi; \
	elif command -v xchesscc_wrapper >/dev/null 2>&1; then \
		echo "Using xchesscc_wrapper from PATH"; \
		cd $(BUILD_DIR) && ${powershell} xchesscc_wrapper aie2p -c ${srcdir}/attn.cc -o attn.o -Dlqp=$(LQP) -Dlkp=$(LKP) -Ddk=$(DK) -Ddv=$(DV); \
	else \
		echo "Error: Neither PEANO_INSTALL_DIR nor xchesscc_wrapper found."; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR) __pycache__
